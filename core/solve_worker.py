# SPDX-License-Identifier: GPL-3.0-or-later
"""
Single linear-elastic FEA solve, run in a separate process.

Threads sharing Blender's C data structures are a known crash source, so
the actual solve runs in a subprocess instead. The job itself is simple:
one voxelization pass, one linear solve, one von-Mises/safety-factor field,
no iterative schedule or per-iteration meshing. Steps:

  1. inside tests (part / exclusions / bearings / loads) -> grid masks
  2. grid assembly (active voxels, fixed DOFs, force vector)
  3. ONE linear-elastic FEA solve with the real material's E/nu
  4. real (Pa) von Mises stress + a safety-factor field (yield / von Mises)

The worker streams status to a work directory (``status.json``) and writes
``final.pkl`` on completion; the parent (``SolverClient``) polls those files
from the modal timer and never blocks. Only numpy + stdlib at import time, so
importing this module into the add-on (for ``SolverClient``) is cheap and
bpy-free; the actual numeric core (``core.fea``) is imported lazily inside
the worker process, which has no ``bpy`` at all.
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

# Percentile (of active-element von Mises stress) used as the color-scale
# ceiling instead of the raw max, so the handful of singular elements that
# always sit right at point loads/supports on a voxel grid don't wash out
# the rest of the heatmap. The true max is still reported separately as
# ``max_stress_pa`` (unclamped) for the actual engineering number.
_STRESS_PERCENTILE = 99.0


# ===========================================================================
# Parent side: launch + non-blocking poll (runs inside Blender, bpy-free)
# ===========================================================================

class SolverClient:
    """Launch the worker and poll it without blocking.

    Events from :meth:`poll`:
      ('voxel', frac)   - still building the grid (0..1)
      ('solve', info)   - voxelization done, CG solve in progress. info is
                          None until the worker's first throttled progress
                          write, then a dict {'iter', 'resid', 'tol'} with
                          the latest CG iteration/residual-ratio, for a live
                          convergence plot (see ui.py).
      ('done', dict)    - finished; dict has stress3d/safety3d/active3d/
                          origin/vsize/scale info + summary numbers
      ('error', message)
      ('running', frac) - working, nothing new to show yet
    """

    def __init__(self):
        self._proc = None
        self._dir = None
        self._logf = None
        self._finished = False

    def start(self, job):
        self._dir = tempfile.mkdtemp(prefix="blendfea_solve_")
        job_path = os.path.join(self._dir, "job.pkl")
        with open(job_path, "wb") as fh:
            pickle.dump(job, fh)

        cmd = [sys.executable, os.path.abspath(__file__), job_path, self._dir]
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000   # CREATE_NO_WINDOW
        # The worker imports sibling modules as `core.fea` etc, which only
        # resolves if `blendfea/` (this file's grandparent) is on the
        # *child's* sys.path. Handed via PYTHONPATH in the subprocess's own
        # environment -- Blender's own process/sys.path is never touched.
        env = dict(os.environ)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (pkg_parent, env.get("PYTHONPATH", "")) if p)
        self._logf = open(os.path.join(self._dir, "worker.log"), "wb")
        self._proc = subprocess.Popen(
            cmd, stdout=self._logf, stderr=self._logf, env=env, **kwargs)

    def _status(self):
        try:
            with open(os.path.join(self._dir, "status.json")) as fh:
                return json.load(fh)
        except Exception:
            return None

    def _log_tail(self):
        try:
            with open(os.path.join(self._dir, "worker.log"), "rb") as fh:
                return fh.read().decode("utf-8", "replace").strip()
        except Exception:
            return ""

    def poll(self):
        if self._finished:
            return ("running", 1.0)

        st = self._status()
        if st is None:
            if self._proc is not None and self._proc.poll() is not None:
                return ("error", self._log_tail() or "worker exited early")
            return ("running", 0.0)

        phase = st.get("phase")
        if phase == "error":
            return ("error", st.get("error") or self._log_tail() or "worker error")
        if phase == "voxel":
            return ("voxel", float(st.get("frac", 0.0)))
        if phase == "solve":
            info = None
            if "cg_iter" in st:
                info = {"iter": int(st.get("cg_iter", 0)),
                       "resid": float(st.get("cg_resid", float("nan"))),
                       "tol": float(st.get("cg_tol", 1e-6))}
            return ("solve", info)
        if phase == "done":
            try:
                with open(os.path.join(self._dir, "final.pkl"), "rb") as fh:
                    final = pickle.load(fh)
            except Exception as exc:
                return ("error", f"could not read solver result: {exc}")
            self._finished = True
            return ("done", final)

        return ("running", 0.0)

    def cancel(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.cleanup()

    def cleanup(self):
        if self._logf is not None:
            try:
                self._logf.close()
            except Exception:
                pass
            self._logf = None
        if self._dir and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None


def solver_available():
    """Whether the out-of-process solver can be launched."""
    try:
        return bool(sys.executable) and os.path.exists(os.path.abspath(__file__))
    except Exception:
        return False


# ===========================================================================
# Child side: the actual worker (runs in the subprocess, no bpy)
# ===========================================================================

def _replace_with_retry(tmp, path, attempts=10, delay=0.05):
    """``os.replace`` that tolerates transient Windows PermissionError if a
    reader (the parent's polling ``open()``, an AV scanner, ...) briefly has
    the destination open."""
    last_exc = None
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay * (i + 1))
    raise last_exc


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    _replace_with_retry(tmp, path)


def _atomic_write_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    _replace_with_retry(tmp, path)


def _worker_main(job_path, work_dir):
    import numpy as np
    from core import inside_worker
    from core import fea
    from core.fea import VoxelFEA, build_edof

    status_path = os.path.join(work_dir, "status.json")

    def set_status(**kw):
        _atomic_write_json(status_path, kw)

    set_status(phase="voxel", frac=0.0)

    with open(job_path, "rb") as fh:
        job = pickle.load(fh)

    g = job["grid"]
    dims = tuple(g["dims"])
    origin = np.asarray(g["origin"], dtype=float)
    vsize = float(g["vsize"])
    nx, ny, nz = dims
    direction = job.get("direction")

    centers = inside_worker.voxel_centers(dims, origin, vsize)
    nodes = inside_worker.node_coords(dims, origin, vsize)
    point_sets = {"centers": centers, "nodes": nodes}

    # --- inside tests (voxelization), with progress -----------------------
    queries = job["queries"]
    descr = job["descr"]
    total_pts = sum(len(point_sets[q["target"]]) for q in queries) or 1
    done_pts = [0]
    masks = []
    for q in queries:
        pts = point_sets[q["target"]]
        base = done_pts[0]

        def _cb(done_in_query, _base=base):
            set_status(phase="voxel", frac=(_base + done_in_query) / total_pts)

        m = inside_worker.inside_mask(q["verts"], q["faces"], pts,
                                      direction=direction, progress_cb=_cb)
        masks.append(np.asarray(m, dtype=bool))
        done_pts[0] += len(pts)
        set_status(phase="voxel", frac=done_pts[0] / total_pts)

    # --- assemble grid ------------------------------------------------------
    inside = None
    excl = np.zeros(nx * ny * nz, dtype=bool)
    fixed = []
    ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    force = np.zeros(ndof)
    for (role, meta), mask in zip(descr, masks):
        if role == "build":
            inside = mask
        elif role == "exclude":
            excl = excl | mask
        elif role == "bearing":
            fix_x, fix_y, fix_z = meta
            for n in np.where(mask)[0]:
                if fix_x:
                    fixed.append(3 * n)
                if fix_y:
                    fixed.append(3 * n + 1)
                if fix_z:
                    fixed.append(3 * n + 2)
        elif role == "load":
            nin = np.where(mask)[0]
            if len(nin):
                fv = np.asarray(meta, dtype=float) / len(nin)
                for n in nin:
                    force[3 * n:3 * n + 3] += fv
    if inside is None:
        inside = np.zeros(nx * ny * nz, dtype=bool)
    active = inside & ~excl
    fixed_dofs = (np.unique(np.asarray(fixed, dtype=np.int64))
                  if fixed else np.zeros(0, dtype=np.int64))

    # Drop voxel islands that never reach a support: unconstrained material
    # is a zero-energy rigid-body mode that makes K (numerically) singular
    # and is the real reason CG can burn thousands of iterations without
    # converging even on a coarse grid -- see core.fea.drop_unsupported_islands
    # for the full reasoning. Handled silently here (no popup/extra prompt);
    # the removed count is only surfaced as a short status-line note, same
    # place the CG convergence note already lives.
    n_islands_removed = 0
    if fixed_dofs.size and active.any():
        active3d_pre = active.reshape(nx, ny, nz)
        kept3d, n_islands_removed = fea.drop_unsupported_islands(
            active3d_pre, fixed_dofs)
        if n_islands_removed:
            active = kept3d.ravel()

    if int(active.sum()) == 0:
        set_status(phase="error",
                   error="No material inside the part at this resolution "
                         "(check the Part object / resolution)")
        return
    if fixed_dofs.size == 0:
        set_status(phase="error",
                   error="No supports (bearings) reach any grid node at this "
                         "resolution -- move/enlarge the support mesh or "
                         "raise the resolution")
        return
    if not np.any(force):
        # Not fatal (a self-check with zero external load still "solves" to
        # zero displacement everywhere) but almost certainly not what the
        # user meant -- surface it as an error rather than a silently
        # trivial result.
        set_status(phase="error",
                   error="No load reaches any grid node -- check the Load "
                         "mesh(es) and force vector(s)")
        return

    # --- one linear-elastic solve -------------------------------------------
    set_status(phase="solve", frac=0.0)
    mat = job["material"]
    E = float(mat["youngs_modulus"])
    nu = float(mat["poisson"])
    yield_strength = float(mat.get("yield_strength") or 0.0)

    active3d = active.reshape(nx, ny, nz)
    nelem = nx * ny * nz
    vfea = VoxelFEA(nx, ny, nz, nu=nu, active_elems=active3d,
                    compute_mode=job.get("compute_mode", "AUTO"),
                    cpu_threads=int(job.get("cpu_threads", 0)),
                    verbose=bool(job.get("verbose", False)))

    # Stream CG convergence to status.json so the UI can draw a *live*
    # residual plot (see blendfea/ui.py's convergence overlay), throttled by
    # wall-clock time (not iteration count) so a fast-diverging or a
    # thousands-of-iterations solve both write status.json a handful of
    # times a second rather than on every single iteration.
    _last_write = [0.0]
    _MIN_INTERVAL = 0.15

    def _cg_progress(it, resid_ratio, tol):
        now = time.time()
        if now - _last_write[0] < _MIN_INTERVAL:
            return
        _last_write[0] = now
        set_status(phase="solve", frac=0.0, cg_iter=it,
                   cg_resid=resid_ratio, cg_tol=tol)

    try:
        vfea.set_fixed(fixed_dofs)
        # core.fea._hex8_KE builds the element stiffness for a unit cube
        # (hardcoded J = 0.5*I, i.e. it assumes each voxel has edge length
        # 1). Real stiffness scales linearly with the true edge length, so
        # the per-element modulus fed to solve() must be E*vsize, not E, or
        # displacement/stress come out wrong by a factor of ~1/vsize. The
        # stress recovery matrix _hex8_B_C is the same unit-cube B, so the
        # strain (and therefore stress) it computes from the real
        # displacement field is too small by a factor of vsize --
        # element_von_mises_stress is called with the real E (not E*vsize)
        # and the result divided by vsize below to correct for that.
        # Verified against analytical cantilever beam theory (tip
        # deflection, bending stress).
        Evec_stiffness = np.full(nelem, E * vsize, dtype=float)
        Evec_stress = np.full(nelem, E, dtype=float)
        u_full = vfea.solve(Evec_stiffness, force, tol=1e-6,
                            progress_cb=_cg_progress)
        vm = vfea.element_von_mises_stress(u_full, Evec_stress) / vsize
    finally:
        resid_history = list(vfea.resid_history)
        vfea.close()

    active_vals = vm[active]
    max_stress_pa = float(active_vals.max()) if active_vals.size else 0.0
    scale_vmax = (float(np.percentile(active_vals, _STRESS_PERCENTILE))
                 if active_vals.size else 0.0)
    scale_vmax = max(scale_vmax, 1e-6)

    # Node displacement magnitude, for a "max deflection" report number and
    # a per-element displacement field for the Displacement result view.
    # Displacement lives naturally on nodes, not element centers, but the
    # coloring/sampling pipeline expects a voxel-centered field shaped
    # (nx,ny,nz) like stress3d/safety3d, so each element's displacement is
    # approximated as the mean of its 8 corner nodes' displacement
    # magnitude (build_edof gives, per element, the 24 global DOF indices =
    # 8 nodes x 3 DOFs each; dividing any of a node's 3 DOF indices by 3
    # recovers its node id).
    u_nodal = u_full.reshape(-1, 3)
    disp_mag = np.sqrt((u_nodal ** 2).sum(axis=1))
    max_disp_m = float(disp_mag.max()) if disp_mag.size else 0.0
    edof_full = build_edof(nx, ny, nz)
    node_ids = edof_full[:, 0::3] // 3          # (nelem, 8)
    disp3d = disp_mag[node_ids].mean(axis=1).reshape(nx, ny, nz)

    # Safety factor field (yield / von Mises), only meaningful where there is
    # material and a real yield strength; capped so a near-zero-stress
    # element (safety factor -> infinity) doesn't blow out the color scale.
    safety_cap = 10.0
    safety3d = np.zeros(nelem)
    min_safety_factor = None
    if yield_strength > 0 and active_vals.size:
        eps = max(1e-9, 1e-9 * yield_strength)
        sf_active = np.clip(yield_strength / np.maximum(active_vals, eps),
                            0.0, safety_cap)
        min_safety_factor = float(
            (yield_strength / np.maximum(active_vals, eps)).min())
        safety3d[active] = sf_active

    disp_vmax = float(disp3d[active3d].max()) if active3d.any() else 0.0
    disp_vmax = max(disp_vmax, 1e-9)

    # Sanity check independent of "did CG say it converged": on a real
    # small-deformation linear solve, the tip deflection should never
    # approach the part's own size. If it does, the result is numerical
    # noise -- flagged for the UI to show plainly rather than silently
    # drawn as if it were trustworthy.
    bbox_diag_m = float(np.sqrt(((np.asarray(dims) * vsize) ** 2).sum()))
    plausible_disp = max_disp_m <= 0.5 * bbox_diag_m
    reliable = bool(vfea.last_converged) and plausible_disp

    final = {
        "dims": dims, "origin": origin, "vsize": vsize,
        "active3d": active3d,
        "stress3d": vm.reshape(nx, ny, nz).astype(np.float32),
        "stress_vmax": scale_vmax,
        "max_stress_pa": max_stress_pa,
        "safety3d": safety3d.reshape(nx, ny, nz).astype(np.float32),
        "safety_cap": safety_cap,
        "min_safety_factor": min_safety_factor,
        "yield_strength_pa": yield_strength,
        "displacement3d": disp3d.astype(np.float32),
        "displacement_vmax_m": disp_vmax,
        "max_displacement_m": max_disp_m,
        "cg_iters": int(vfea.last_iters),
        "cg_converged": bool(vfea.last_converged),
        "cg_resid_history": resid_history,
        "cg_tol": 1e-6,
        "reliable": reliable,
        "n_islands_removed": int(n_islands_removed),
        "compute_label": vfea.plan.label,
    }
    _atomic_write_pickle(os.path.join(work_dir, "final.pkl"), final)
    set_status(phase="done", frac=1.0)


def _main(argv):
    job_path, work_dir = argv[1], argv[2]
    try:
        _worker_main(job_path, work_dir)
    except Exception as exc:
        import traceback
        try:
            _atomic_write_json(
                os.path.join(work_dir, "status.json"),
                {"phase": "error", "error": f"{exc}\n{traceback.format_exc()}"})
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
