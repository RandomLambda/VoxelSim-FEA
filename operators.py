# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators: list management, material presets, and the modal solve run.

Preferred path: voxelization + the linear FEA solve run in a plain-Python
subprocess (see core.solve_worker) so Blender's main thread does the bare
minimum and the viewport stays interactive; the modal timer just polls it and
applies the finished result (four dedicated result objects) into the
"BlenderFEA Results" collection.

Fallback (no subprocess available): voxelization runs as a chunked in-process
generator so that phase doesn't freeze the UI. The solve itself is a single
CG call with no natural yield points to chunk, so in this fallback path only
it may pause the UI for its duration; this is surfaced in the tooltip/status
text rather than hidden.

No background threads either way (a known Blender crash source) -- only
subprocesses, which cannot touch or crash Blender state.

Units: FEA_Settings.youngs_modulus is stored (and shown in the UI) in GPa;
yield_strength in MPa, matching how datasheets are normally quoted. The
solver works in SI Pascals; the *1e9 / *1e6 conversions happen once, at the
two places a job is handed to the solver (_build_job for the subprocess path,
_run_inprocess_solve for the fallback).

Results: every solve builds/updates FOUR objects in the "BlenderFEA Results"
collection -- BlenderFEA_Result_Stress, BlenderFEA_Result_Displacement,
BlenderFEA_Result_Safety (each an independent copy of the part's mesh, see
core.extract.duplicate_result_object) and BlenderFEA_StressCloud (the voxel
cross-section). FEA_Settings.result_view picks which ONE is visible at a
time; switching it (properties.py's update callback -> apply_result_visibility
below) is instant and never re-solves, since all four already exist.
"""

import time

import numpy as np

import traceback

import bpy
from bpy.types import Operator
from bpy.props import IntProperty, StringProperty

from . import ui, properties
from .core import voxelize, extract, solve_worker, materials
from .core import fea as fea_core
from .core.fea import VoxelFEA, build_edof

RESULTS_COLLECTION = "BlenderFEA Results"
RESULT_STRESS_NAME = "BlenderFEA_Result_Stress"
RESULT_DISPLACEMENT_NAME = "BlenderFEA_Result_Displacement"
RESULT_SAFETY_NAME = "BlenderFEA_Result_Safety"
STRESS_CLOUD_NAME = "BlenderFEA_StressCloud"

_RESULT_OBJECT_NAMES = {
    'STRESS': RESULT_STRESS_NAME,
    'DISPLACEMENT': RESULT_DISPLACEMENT_NAME,
    'SAFETY': RESULT_SAFETY_NAME,
    'CROSS_SECTION': STRESS_CLOUD_NAME,
}

_GPA_TO_PA = 1.0e9
_MPA_TO_PA = 1.0e6

_RUN_COUNTER = 0

# The last finished solve's raw fields, kept only so switching FEA_Settings.
# result_view (no re-solve) can still refresh the on-screen legend. Never
# written to disk / never part of any .blend save -- purely a live cache.
_LAST_PAYLOAD = None


def _set_wire(obj):
    if obj is not None:
        try:
            obj.display_type = 'WIRE'
        except Exception:
            pass


def _set_textured(obj):
    if obj is not None:
        try:
            obj.display_type = 'TEXTURED'
        except Exception:
            pass


def force_helpers_solid(s):
    """Reset every assigned support/load helper object to solid display.
    Called when the "Wireframe Helper-Objects" checkbox is turned off (see
    properties._on_wireframe_helpers_update)."""
    for item in list(s.bearings) + list(s.loads):
        _set_textured(item.obj)


def _results_collection(context):
    coll = bpy.data.collections.get(RESULTS_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(RESULTS_COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


def apply_result_visibility(context, s):
    """Show exactly the object matching ``s.result_view``, hide the other
    three. Called after every solve and from properties.py's update callback
    when the dropdown changes with no new solve; purely visibility
    toggling on objects that already exist."""
    selected = s.result_view
    for key, name in _RESULT_OBJECT_NAMES.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        visible = (key == selected)
        try:
            obj.hide_set(not visible)
        except Exception:
            obj.hide_viewport = not visible

    if selected == 'NONE' or _LAST_PAYLOAD is None:
        ui.clear_legend()
        return
    _refresh_legend(s, _LAST_PAYLOAD)


def apply_inputs_visibility(context, s):
    """Show/hide the part + every support/load helper object, driven by
    FEA_Settings.inputs_visible. Called from properties.py's update callback
    (user toggles the checkbox) and once automatically right after a solve
    finishes (see _apply_result)."""
    visible = bool(s.inputs_visible)
    objs = []
    if s.part is not None:
        objs.append(s.part)
    for b in s.bearings:
        if b.obj is not None:
            objs.append(b.obj)
    for ld in s.loads:
        if ld.obj is not None:
            objs.append(ld.obj)
    for obj in objs:
        try:
            obj.hide_set(not visible)
        except Exception:
            try:
                obj.hide_viewport = not visible
            except Exception:
                pass


def _rebuild_result_objects(context, payload, coll):
    """Build/update all FOUR result objects from ``payload`` (a solve's
    cached field arrays) and refresh their visibility/legend/convergence
    plot. Does not include the operator-only status text/report at the end
    of FEA_OT_run._apply_result, which wraps this for the "just finished a
    solve" case.

    Cheap (coloring existing mesh copies, no re-solve), so it doubles as the
    handler for a pure display-setting change (Pixel/Contour shading, band
    count) via refresh_result_colors() below -- that toggle is instant
    instead of requiring "Run FEA" again just to redraw the same data.
    """
    global _LAST_PAYLOAD
    s = context.scene.blendfea
    part = s.part
    if part is None:
        return
    _LAST_PAYLOAD = payload

    properties.ensure_material_preview_shading(context)

    # None = old, fully continuous "Pixel" gradient; an int = discrete
    # "Contour" bands (see extract._apply_contour_colors) -- resolved once
    # here and threaded into every coloring call below so the surface
    # results AND the cross-section cloud always agree on which mode is
    # active.
    bands = int(s.contour_bands) if s.color_mode == 'BANDED' else None

    # Pixel mode gets its resolution from voxel-grid bisecting
    # (extract._bisect_to_voxel_grid) inside apply_stress_colors, so
    # pre-subdividing here is skipped (vsize=None). Contour mode samples
    # smoothly per-vertex (see _apply_contour_colors) and needs a much finer
    # mesh so band edges look smooth rather than following triangle edges
    # (see extract._CONTOUR_SUBDIV_FACTOR).
    stress_safety_subdiv_vsize = payload["vsize"] if bands is not None else None
    contour_subdiv_factor = extract._CONTOUR_SUBDIV_FACTOR if bands is not None else None

    # Build/update all FOUR result objects every run -- cheap (coloring
    # an existing mesh copy, no re-solve), so switching which one is
    # shown afterwards (result_view dropdown) is instant. Each failure
    # is independent and non-fatal: one broken result must never hide
    # the other three or abort a finished solve.
    try:
        stress_obj = extract.duplicate_result_object(
            part, RESULT_STRESS_NAME, collection=coll,
            vsize=stress_safety_subdiv_vsize, subdiv_factor=contour_subdiv_factor)
        extract.apply_stress_colors(
            stress_obj, payload["stress3d"], payload["origin"],
            payload["vsize"], payload["stress_vmax"], 0.0,
            active3d=payload["active3d"], bands=bands)
    except Exception as exc:  # noqa: BLE001
        print(f"[BlenderFEA] stress result build failed: {exc}")
        traceback.print_exc()

    try:
        disp_obj = extract.duplicate_result_object(
            part, RESULT_DISPLACEMENT_NAME, collection=coll,
            vsize=payload["vsize"], subdiv_factor=contour_subdiv_factor)
        extract.apply_stress_colors(
            disp_obj, payload["displacement3d"], payload["origin"],
            payload["vsize"], payload["displacement_vmax_m"], 0.0,
            bands=bands)
    except Exception as exc:  # noqa: BLE001
        print(f"[BlenderFEA] displacement result build failed: {exc}")
        traceback.print_exc()

    try:
        safety_obj = extract.duplicate_result_object(
            part, RESULT_SAFETY_NAME, collection=coll,
            vsize=stress_safety_subdiv_vsize, subdiv_factor=contour_subdiv_factor)
        cap = payload.get("safety_cap", 10.0)
        # Field is negated so t=1 (red) always means "worst" regardless of
        # sign, matching the cross-section legend (see extract._jet_rgb).
        extract.apply_stress_colors(
            safety_obj, -payload["safety3d"], payload["origin"],
            payload["vsize"], 0.0, -cap,
            active3d=payload["active3d"], bands=bands)
    except Exception as exc:  # noqa: BLE001
        print(f"[BlenderFEA] safety result build failed: {exc}")
        traceback.print_exc()

    try:
        cloud = extract.voxel_cloud_to_object(
            payload["stress3d"], payload["active3d"], payload["origin"],
            payload["vsize"], payload["stress_vmax"], 0.0,
            STRESS_CLOUD_NAME, collection=coll, bands=bands)
        extract.ensure_stress_cloud_geonodes(
            cloud, collection=coll, enable_clip=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[BlenderFEA] cross-section result build failed: {exc}")

    apply_result_visibility(context, s)

    # Auto-hide the part/supports/loads now that a result exists to look at.
    # Assigned through the property (not hide_set directly) so it goes
    # through _on_inputs_visible_update in properties.py, keeping the
    # N-panel checkbox in sync.
    if s.inputs_visible:
        s.inputs_visible = False

    ui.set_convergence_history(payload.get("cg_resid_history"),
                               payload.get("cg_tol", 1e-6))


def refresh_result_colors(context):
    """Re-bake every result object's colors from the LAST solve's cached
    payload, without re-solving -- see _rebuild_result_objects. A no-op if
    no solve has finished yet (nothing cached to redraw from)."""
    if _LAST_PAYLOAD is None:
        return
    _rebuild_result_objects(context, _LAST_PAYLOAD, _results_collection(context))


def _refresh_legend(s, payload):
    rv = s.result_view
    # Stress and Displacement share one gradient bar (see ui.set_legend):
    # whichever field is coloring the mesh gets the normal right-hand
    # labels; the other field's 0..max is drawn as a second label column
    # on the left of the same bar.
    stress_top = f"{payload['stress_vmax'] / _MPA_TO_PA:.3g}"
    disp_top = f"{payload['displacement_vmax_m'] * 1000.0:.3g}"
    if rv == 'STRESS':
        ui.set_legend("Von Mises Stress", "MPa", "0", stress_top,
                     left_unit="mm", left_bottom="0", left_top=disp_top)
    elif rv == 'DISPLACEMENT':
        ui.set_legend("Displacement", "mm", "0", disp_top,
                     left_unit="MPa", left_bottom="0", left_top=stress_top)
    elif rv == 'SAFETY':
        cap = payload.get("safety_cap", 10.0)
        ui.set_legend("Safety Factor", "",
                     f">= {cap:g} (safe)", "0 (at/above yield)")
    elif rv == 'CROSS_SECTION':
        ui.set_legend("Von Mises Stress (cross-section)", "MPa", "0",
                     stress_top, left_unit="mm", left_bottom="0",
                     left_top=disp_top)
    else:
        ui.clear_legend()


# ---------------------------------------------------------------------------
# Assignment / list management
# ---------------------------------------------------------------------------

class FEA_OT_set_part(Operator):
    bl_idname = "blendfea.set_part"
    bl_label = "Set Part from Active"
    bl_description = "Use the active object as the part to analyze"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        context.scene.blendfea.part = obj
        _set_textured(obj)
        return {'FINISHED'}


class _ListAdd(Operator):
    """Base: add the active object to a named collection."""
    collection_name = ""

    def execute(self, context):
        s = context.scene.blendfea
        item = getattr(s, self.collection_name).add()
        if context.active_object and context.active_object.type == 'MESH':
            item.obj = context.active_object
            if s.wireframe_helpers:
                _set_wire(context.active_object)
        return {'FINISHED'}


class FEA_OT_add_bearing(_ListAdd):
    bl_idname = "blendfea.add_bearing"
    bl_label = "Add Support"
    collection_name = "bearings"


class FEA_OT_add_load(_ListAdd):
    bl_idname = "blendfea.add_load"
    bl_label = "Add Load"
    collection_name = "loads"


class FEA_OT_remove_item(Operator):
    bl_idname = "blendfea.remove_item"
    bl_label = "Remove"
    collection_name: StringProperty()
    index: IntProperty()

    def execute(self, context):
        s = context.scene.blendfea
        coll = getattr(s, self.collection_name)
        if 0 <= self.index < len(coll):
            coll.remove(self.index)
        return {'FINISHED'}


class FEA_OT_apply_material_preset(Operator):
    bl_idname = "blendfea.apply_material_preset"
    bl_label = "Apply Preset"
    bl_description = ("Copy the selected preset's reference E (GPa) / "
                      "Poisson / yield strength (MPa) into the editable "
                      "fields below")

    # Used to take an explicit preset key so the UI could offer one
    # confirm-checkmark button PER material (4 buttons next to the
    # dropdown). Reads the dropdown's own current selection instead now --
    # one button confirms whatever s.material_preset is already set to, see
    # ui.py's FEA_PT_material.draw.
    def execute(self, context):
        s = context.scene.blendfea
        preset = materials.get(s.material_preset)
        if preset is None:
            self.report({'ERROR'}, f"Unknown preset: {s.material_preset}")
            return {'CANCELLED'}
        s.youngs_modulus = preset["youngs_modulus"]
        s.poisson = preset["poisson"]
        s.yield_strength = preset["yield_strength"]
        self.report({'INFO'}, f"Applied {preset['label']}: "
                              f"{preset['note']}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# The modal solve run
# ---------------------------------------------------------------------------

class FEA_OT_run(Operator):
    bl_idname = "blendfea.run"
    bl_label = "Run FEA"
    bl_description = "Voxelize the part and solve in the background"

    def invoke(self, context, event):
        s = context.scene.blendfea
        if s.part is None:
            self.report({'ERROR'}, "Set a part first")
            return {'CANCELLED'}
        if len(s.loads) == 0 or len(s.bearings) == 0:
            self.report({'ERROR'}, "Need at least one support and one load")
            return {'CANCELLED'}
        try:
            # Cheap (O(corners)) preflight: catches bearings/loads that sit
            # outside the Part's bounding box -- and thus can never reach a
            # grid node at any resolution -- before voxelizing the whole part.
            voxelize.check_reach(s)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        global _RUN_COUNTER
        _RUN_COUNTER += 1
        self._run_id = _RUN_COUNTER
        self._settings = s
        self._coll = _results_collection(context)
        self._depsgraph = context.evaluated_depsgraph_get()
        self._solver = None
        self._voxgen = None
        self._voxizer = None
        self._grid = None
        self._error = ""
        self._all_done = False
        self._interrupted = False

        self._mode = 'solver' if solve_worker.solver_available() else 'inproc'

        s.running = True
        s.status = "voxelizing..."
        ui.reset_convergence()

        try:
            self._start(context)
        except Exception as exc:  # noqa: BLE001
            s.running = False
            s.status = "Stopped due to error: " + str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _start(self, context):
        s = self._settings
        if self._mode == 'solver':
            try:
                job = self._build_job(context)
                self._solver = solve_worker.SolverClient()
                self._solver.start(job)
                s.status = f"voxelizing (res {s.resolution})... [bg]"
                return
            except Exception as exc:  # noqa: BLE001 - degrade, don't fail
                print(f"[BlenderFEA] subprocess solver unavailable, using "
                      f"in-process path: {exc}")
                self._mode = 'inproc'
                self._solver = None

        # In-process fallback: chunked voxelization, then one blocking solve.
        if voxelize.subprocess_available():
            try:
                vx = voxelize.AsyncVoxelizer()
                vx.start(s, self._depsgraph)
                self._voxizer = vx
            except Exception as exc:  # noqa: BLE001
                print(f"[BlenderFEA] async voxelize unavailable: {exc}")
                self._voxizer = None
        if self._voxizer is None:
            self._voxgen = voxelize.build_grid_steps(s, self._depsgraph)
        s.status = f"voxelizing (res {s.resolution})..."

    def _build_job(self, context):
        s = self._settings
        grid, _reach = voxelize._grid_and_reach(s)
        self._depsgraph.update()

        queries, descr = [], []
        v, f = voxelize._object_triangles(s.part, self._depsgraph)
        queries.append({'verts': v, 'faces': f, 'target': 'centers'})
        descr.append(('build', None))
        for b in s.bearings:
            if b.obj is None:
                continue
            v, f = voxelize._object_triangles(b.obj, self._depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'nodes'})
            descr.append(('bearing', (bool(b.fix_x), bool(b.fix_y), bool(b.fix_z))))
        for ld in s.loads:
            if ld.obj is None:
                continue
            v, f = voxelize._object_triangles(ld.obj, self._depsgraph)
            queries.append({'verts': v, 'faces': f, 'target': 'nodes'})
            descr.append(('load', np.asarray(ld.force, dtype=float)))

        return {
            'direction': voxelize.inside_worker._RAY_DIR,
            'grid': {'dims': (grid.nx, grid.ny, grid.nz),
                     'origin': np.asarray(grid.origin, dtype=float),
                     'vsize': float(grid.vsize)},
            'queries': queries, 'descr': descr,
            # GPa/MPa (UI/property units) -> Pa (solver units), see module docstring.
            'material': {'youngs_modulus': s.youngs_modulus * _GPA_TO_PA,
                        'poisson': s.poisson,
                        'yield_strength': s.yield_strength * _MPA_TO_PA},
            'compute_mode': s.compute_mode,
            'cpu_threads': int(s.cpu_threads),
            'verbose': bool(s.verbose_log),
        }

    def modal(self, context, event):
        s = context.scene.blendfea
        if event.type == 'ESC' and event.value == 'PRESS':
            self._interrupted = True
            s.status = "cancelling..."
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            try:
                self._step(context)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                self.report({'ERROR'}, str(exc))
                self._all_done = True
            if self._all_done:
                return self._finish(context)

        return {'PASS_THROUGH'}

    def _step(self, context):
        if self._interrupted:
            self._all_done = True
            return

        if self._mode == 'solver':
            self._step_solver(context)
            return

        # in-process fallback
        if self._grid is None:
            self._step_voxelize(context)
            return
        # Grid is ready: run the (blocking) solve in one tick.
        self._run_inprocess_solve(context, self._grid)
        self._all_done = True

    # -- solver (subprocess) mode --------------------------------------------

    def _step_solver(self, context):
        s = context.scene.blendfea
        ev, payload = self._solver.poll()

        if ev == 'error':
            raise RuntimeError(payload)
        elif ev == 'voxel':
            pct = 100.0 * float(payload)
            s.status = f"voxelizing (res {s.resolution})  {pct:.0f}% [bg]"
        elif ev == 'solve':
            if payload:
                s.status = (f"solving... [bg]  CG it {payload['iter']}, "
                           f"residual {payload['resid']:.2e} "
                           f"(target {payload['tol']:.1e})")
                ui.push_convergence_point(payload['iter'], payload['resid'],
                                          payload['tol'])
            else:
                s.status = "solving... [bg]"
        elif ev == 'done':
            self._apply_result(context, payload)
            self._solver.cleanup()
            self._solver = None
            self._all_done = True

        for area in context.screen.areas:
            area.tag_redraw()

    # -- in-process fallback --------------------------------------------------

    def _step_voxelize(self, context):
        s = context.scene.blendfea
        if self._voxizer is not None:
            tag, payload = self._voxizer.poll()
            if tag == 'grid':
                self._voxizer = None
                self._grid = payload
            elif tag == 'error':
                self._voxizer = None
                raise RuntimeError(payload)
            else:
                pct = 100.0 * float(payload)
                s.status = f"voxelizing (res {s.resolution})  {pct:.0f}% [bg]"
            for area in context.screen.areas:
                area.tag_redraw()
            return

        budget = 0.02
        t0 = time.perf_counter()
        while True:
            try:
                tag, *rest = next(self._voxgen)
            except StopIteration:
                raise RuntimeError("voxelization produced no grid")
            if tag == 'grid':
                self._voxgen = None
                self._grid = rest[0]
                for area in context.screen.areas:
                    area.tag_redraw()
                return
            if time.perf_counter() - t0 >= budget:
                done_pts, total_pts = rest
                pct = (100.0 * done_pts / total_pts) if total_pts else 0.0
                s.status = f"voxelizing (res {s.resolution})  {pct:.0f}% [main]"
                for area in context.screen.areas:
                    area.tag_redraw()
                return

    def _run_inprocess_solve(self, context, grid):
        s = context.scene.blendfea
        if grid.active.sum() == 0:
            raise RuntimeError("No material inside the part at this "
                               "resolution -- check the Part object / "
                               "resolution")
        if grid.fixed_dofs.size == 0:
            raise RuntimeError("No supports (bearings) reach any grid node "
                               "at this resolution")
        if not np.any(grid.force):
            raise RuntimeError("No load reaches any grid node -- check the "
                               "Load mesh(es) and force vector(s)")

        s.status = "solving..."
        for area in context.screen.areas:
            area.tag_redraw()

        # GPa/MPa (UI/property units) -> Pa (solver units), see module docstring.
        E_pa = s.youngs_modulus * _GPA_TO_PA
        yield_pa = s.yield_strength * _MPA_TO_PA

        nx, ny, nz = grid.nx, grid.ny, grid.nz
        active3d = grid.active.reshape(nx, ny, nz)

        # Voxels with no connected path to a support are a zero-energy
        # rigid-body mode and a common cause of CG failing to converge, so
        # they're dropped here (see core.fea.drop_unsupported_islands),
        # matching core/solve_worker.py's subprocess path.
        n_islands_removed = 0
        if grid.fixed_dofs.size and active3d.any():
            active3d, n_islands_removed = fea_core.drop_unsupported_islands(
                active3d, grid.fixed_dofs)
            grid.active = active3d.ravel()

        nelem = nx * ny * nz
        vfea = VoxelFEA(nx, ny, nz, nu=s.poisson,
                        active_elems=active3d, compute_mode=s.compute_mode,
                        cpu_threads=int(s.cpu_threads),
                        verbose=bool(s.verbose_log))
        try:
            vfea.set_fixed(grid.fixed_dofs)
            # See core/solve_worker.py's matching comment: core.fea's element
            # stiffness assumes a unit-size voxel, so the stiffness solve
            # needs E*vsize, and the raw stress recovery (also unit-cube
            # based) needs dividing back down by vsize afterwards.
            Evec_stiffness = np.full(nelem, E_pa * grid.vsize, dtype=float)
            Evec_stress = np.full(nelem, E_pa, dtype=float)
            u_full = vfea.solve(Evec_stiffness, grid.force, tol=1e-6)
            vm = vfea.element_von_mises_stress(u_full, Evec_stress) / grid.vsize
        finally:
            resid_history = list(vfea.resid_history)
            vfea.close()

        active_vals = vm[grid.active]
        max_stress_pa = float(active_vals.max()) if active_vals.size else 0.0
        scale_vmax = max(
            float(np.percentile(active_vals, 99.0)) if active_vals.size else 0.0,
            1e-6)

        safety_cap = 10.0
        safety3d = np.zeros(nelem)
        min_safety_factor = None
        if yield_pa > 0 and active_vals.size:
            eps = max(1e-9, 1e-9 * yield_pa)
            safety3d[grid.active] = np.clip(
                yield_pa / np.maximum(active_vals, eps), 0.0, safety_cap)
            min_safety_factor = float(
                (yield_pa / np.maximum(active_vals, eps)).min())

        # Element-averaged nodal displacement (see core/solve_worker.py).
        u_nodal = u_full.reshape(-1, 3)
        disp_mag = np.sqrt((u_nodal ** 2).sum(axis=1))
        max_disp_m = float(disp_mag.max()) if disp_mag.size else 0.0
        edof_full = build_edof(nx, ny, nz)
        node_ids = edof_full[:, 0::3] // 3
        disp3d = disp_mag[node_ids].mean(axis=1).reshape(nx, ny, nz)
        disp_vmax = float(disp3d[active3d].max()) if active3d.any() else 0.0
        disp_vmax = max(disp_vmax, 1e-9)

        # A deflection comparable to the part's own size means the result is
        # numerical noise regardless of CG's own convergence flag (matches
        # core/solve_worker.py's guard).
        bbox_diag_m = float(np.sqrt(((np.array([nx, ny, nz]) * grid.vsize) ** 2).sum()))
        plausible_disp = max_disp_m <= 0.5 * bbox_diag_m
        reliable = bool(vfea.last_converged) and plausible_disp

        payload = {
            "dims": (nx, ny, nz), "origin": grid.origin,
            "vsize": grid.vsize, "active3d": active3d,
            "stress3d": vm.reshape(nx, ny, nz).astype(np.float32),
            "stress_vmax": scale_vmax, "max_stress_pa": max_stress_pa,
            "safety3d": safety3d.reshape(nx, ny, nz).astype(np.float32),
            "safety_cap": safety_cap, "min_safety_factor": min_safety_factor,
            "yield_strength_pa": yield_pa,
            "displacement3d": disp3d.astype(np.float32),
            "displacement_vmax_m": disp_vmax,
            "max_displacement_m": max_disp_m,
            "cg_iters": int(vfea.last_iters), "cg_converged": bool(vfea.last_converged),
            "cg_resid_history": resid_history, "cg_tol": 1e-6,
            "reliable": reliable, "n_islands_removed": int(n_islands_removed),
            "compute_label": vfea.plan.label,
        }
        self._apply_result(context, payload)

    # -- shared result application --------------------------------------------

    def _apply_result(self, context, payload):
        s = context.scene.blendfea
        if s.part is None:
            return
        _rebuild_result_objects(context, payload, self._coll)

        max_mpa = payload["max_stress_pa"] / _MPA_TO_PA
        disp_mm = payload["max_displacement_m"] * 1000.0
        sf = payload.get("min_safety_factor")
        sf_txt = f", min safety factor {sf:.2f}" if sf is not None else ""
        conv_txt = "" if payload.get("cg_converged", True) else " (NOT CONVERGED)"
        n_isl = payload.get("n_islands_removed", 0)
        isl_txt = (f", ignored {n_isl} unsupported voxel(s)" if n_isl else "")
        # Reliability combines CG's own convergence flag with the
        # displacement-plausibility check above, so a result that
        # "converged" but produced an absurd deflection is still flagged.
        warn_txt = "" if payload.get("reliable", True) else "  ⚠ result may be unreliable"
        s.status = (f"done: max von Mises {max_mpa:.3g} MPa, "
                   f"max displacement {disp_mm:.3g} mm{sf_txt}{isl_txt} "
                   f"[{payload.get('compute_label', '?')}, "
                   f"{payload.get('cg_iters', 0)} CG it{conv_txt}]{warn_txt}")
        self.report(
            {'WARNING'} if not payload.get("reliable", True) else {'INFO'},
            s.status)

    def _finish(self, context):
        s = context.scene.blendfea
        wm = context.window_manager
        for attr in ("_voxizer", "_solver"):
            obj_ = getattr(self, attr, None)
            if obj_ is not None:
                try:
                    obj_.cancel()
                except Exception:
                    pass
                setattr(self, attr, None)
        if getattr(self, "_timer", None) is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        s.running = False
        if self._error:
            s.status = "Stopped due to error: " + self._error
        elif self._interrupted:
            s.status = "cancelled"
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}


_CLASSES = (
    FEA_OT_set_part,
    FEA_OT_add_bearing, FEA_OT_add_load,
    FEA_OT_remove_item, FEA_OT_apply_material_preset, FEA_OT_run,
)


def register():
    # NOTE: hasattr(cls, "bl_rna") is NOT a reliable "is this class
    # currently registered" check -- see properties.py for the full
    # rationale. try/except RuntimeError is the only reliable guard.
    registered = []
    try:
        for cls in _CLASSES:
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
            bpy.utils.register_class(cls)
            registered.append(cls)
    except Exception:
        print("blendfea.operators.register() failed -- see traceback below "
              "for the REAL cause (best-effort rollback follows, its own "
              "errors are swallowed so they don't hide this one):")
        traceback.print_exc()
        for cls in reversed(registered):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
        raise


def unregister():
    global _LAST_PAYLOAD
    _LAST_PAYLOAD = None
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
