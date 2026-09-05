# SPDX-License-Identifier: GPL-3.0-or-later
"""
Compute-backend selection: CPU / single GPU / multi-GPU / multi-process CPU.

This module is the single place that decides, for a given problem size (the
FEA's reduced DOF count), which array module and which parallelism to use. It
is identical in both editions (CPU-only and GPU); it only *asks* `backend`
what is available (``is_gpu_build``, ``gpu_usable``, ``gpu_device_count``) and
never imports Cu-Py itself, so it stays import-safe in the hosted CPU
edition.

Four tiers, gated by DOF count:

  ndof <  GPU_DOF                -> plain CPU (numpy), all available cores.
  GPU_DOF <= ndof < MULTI_GPU_DOF -> single GPU if the GPU edition has one
                                     that passed its self-test, else CPU.
  ndof >= MULTI_GPU_DOF           -> split the matrix-free matvec across every
                                     visible GPU (GPU edition, >1 device),
                                     else across CPU worker processes if that
                                     is available and beneficial at this size,
                                     else falls back one rung at a time down
                                     to plain CPU.

GPU_DOF (50,000) is the measured single-GPU-vs-CPU crossover: below it,
kernel-launch and host<->device transfer overhead dominates the actual
compute, so numpy on the CPU is faster in practice for the small grids
typical of coarse levels.

MULTI_GPU_DOF is set far above any size where multi-GPU has been measured to
win: splitting the matvec across devices only pools the outer CG loop (not
the multigrid V-cycle's local smoothing passes), so its cost is single-GPU's
cost plus a roughly fixed per-call dispatch/sync overhead that does not
shrink as the problem grows -- so the relative benefit gets worse with size,
not better, at every size tested so far. AUTO therefore does not pick
multi_gpu at any measured size; compute_mode="MULTI_GPU" remains available as
an explicit, forced choice (see multigrid.py's _requested_mode / _init_pool),
and this threshold should only be lowered after a fresh measurement shows a
real win.

MULTI_CPU_DOF (500,000) follows the same reasoning: multi-process CPU work
only pays for itself once there is enough element work per worker to
amortize the shared-memory hand-off/IPC cost every CG iteration; below that
scale a single process (with the BLAS thread pool sized to the machine) is
both simpler and faster.
"""

import os

from . import backend

GPU_DOF = 50_000
# Set far above any tested size so AUTO never picks multi_gpu until a real
# measured win justifies lowering this (see module docstring).
# compute_mode="MULTI_GPU" remains available as an explicit, forced choice
# regardless of this threshold (see multigrid.py's _init_pool).
MULTI_GPU_DOF = 50_000_000
MULTI_CPU_DOF = 500_000

_THREADPOOL_LIMITER = None   # kept alive for the process lifetime


class ComputePlan:
    """Resolved decision for one FEA/multigrid instance."""

    __slots__ = ("kind", "xp", "n_workers", "label")

    def __init__(self, kind, xp, n_workers, label):
        self.kind = kind            # 'cpu' | 'gpu' | 'multi_gpu' | 'multi_cpu'
        self.xp = xp                # array module to use for the (single-
                                    # device) parts of the computation
        self.n_workers = n_workers  # devices (multi_gpu) or processes
                                    # (multi_cpu); 1 otherwise
        self.label = label          # human-readable, for the debug log

    @property
    def parallel(self):
        return self.kind in ("multi_gpu", "multi_cpu")


def cpu_worker_count(requested=0):
    """How many CPU worker processes/threads to use.

    requested <= 0 means "auto": all logical cores minus one, so the UI
    thread (Blender, or on this path the solver subprocess's own main
    thread) is never starved. Always at least 1.
    """
    total = os.cpu_count() or 1
    if requested and requested > 0:
        return max(1, min(int(requested), total))
    return max(1, total - 1)


def configure_cpu_threads(n_threads, verbose=False):
    """Limit the BLAS thread pool (openblas/mkl/...) numpy is linked
    against to ``n_threads``. Uses threadpoolctl (bundled wheel), which
    patches the already-loaded BLAS library at runtime -- this works even
    though numpy was imported long before this call (by Blender itself).

    Safe no-op if threadpoolctl is unavailable for any reason: numpy just
    keeps whatever thread count the BLAS library defaulted to.
    """
    global _THREADPOOL_LIMITER
    try:
        import threadpoolctl
    except Exception as exc:
        if verbose:
            print(f"[VoxelSim FEA] threadpoolctl unavailable ({exc}); "
                  f"leaving BLAS thread count at its default")
        return None
    try:
        # Replaces any previous limiter (only one is meant to be active).
        if _THREADPOOL_LIMITER is not None:
            _THREADPOOL_LIMITER.unregister()
        limiter = threadpoolctl.threadpool_limits(limits=int(n_threads))
        _THREADPOOL_LIMITER = limiter
        if verbose:
            info = threadpoolctl.threadpool_info()
            libs = ", ".join(sorted({d.get("internal_api", "?") for d in info})) or "none detected"
            print(f"[VoxelSim FEA] BLAS thread pool set to {n_threads} "
                  f"(libraries: {libs})")
        return limiter
    except Exception as exc:
        if verbose:
            print(f"[VoxelSim FEA] could not set BLAS thread count ({exc})")
        return None


def choose(ndof, mode="AUTO", cpu_threads=0, verbose=False):
    """Resolve a ComputePlan for a problem with ``ndof`` (reduced) DOFs.

    mode: 'AUTO' (thresholds above), or a forced 'CPU' / 'GPU' / 'MULTI_GPU'
    / 'MULTI_CPU' (falls back gracefully -- towards CPU -- if the forced mode
    is not actually available, it is never allowed to silently do nothing).
    """
    mode = (mode or "AUTO").upper()
    n_workers_cpu = cpu_worker_count(cpu_threads)

    gpu_build = backend.is_gpu_build()
    gpu_ok = gpu_build and backend.gpu_usable()
    n_gpus = backend.gpu_device_count() if gpu_ok else 0

    def _cpu():
        return ComputePlan("cpu", backend.get_xp(False), 1,
                            f"CPU (numpy, {n_workers_cpu} BLAS threads)")

    def _gpu():
        return ComputePlan("gpu", backend.get_xp(True), 1, "single GPU (Cu-Py)")

    def _multi_gpu():
        return ComputePlan("multi_gpu", backend.get_xp(True), n_gpus,
                            f"multi-GPU ({n_gpus} devices, Cu-Py)")

    def _multi_cpu():
        return ComputePlan("multi_cpu", backend.get_xp(False), n_workers_cpu,
                            f"multi-process CPU ({n_workers_cpu} workers)")

    if mode == "CPU":
        plan = _cpu()
    elif mode == "GPU":
        plan = _gpu() if gpu_ok else _cpu()
    elif mode == "MULTI_GPU":
        plan = _multi_gpu() if (gpu_ok and n_gpus > 1) else (_gpu() if gpu_ok else _cpu())
    elif mode == "MULTI_CPU":
        plan = _multi_cpu() if n_workers_cpu > 1 else _cpu()
    else:  # AUTO
        if ndof >= MULTI_GPU_DOF and gpu_ok and n_gpus > 1:
            plan = _multi_gpu()
        elif ndof >= MULTI_CPU_DOF and n_workers_cpu > 1 and not gpu_ok:
            # Only reach for multi-process CPU at large scale, and only when
            # there is no usable GPU -- a single GPU is very likely faster
            # than several CPU processes at this size anyway.
            plan = _multi_cpu()
        elif ndof >= GPU_DOF and gpu_ok:
            plan = _gpu()
        else:
            plan = _cpu()

    # Main-process BLAS thread count is set after the plan is known: for
    # "cpu"/"gpu" plans this process does all the matvec work itself, so it
    # gets the full n_workers_cpu BLAS threads. For "multi_cpu", the matvec
    # is already split across n_workers_cpu OS processes (each pinned to 1
    # BLAS thread, see parallel_cpu.py's _limit_worker_blas_threads), so the
    # calling process is capped to 1 thread to avoid oversubscribing the
    # same cores twice. "multi_gpu" similarly leaves the heavy lifting to
    # the pooled devices, not host BLAS.
    main_process_threads = (1 if plan.kind in ("multi_cpu", "multi_gpu")
                             else n_workers_cpu)
    configure_cpu_threads(main_process_threads, verbose=verbose)

    if verbose:
        print(f"[VoxelSim FEA] compute plan: ndof={ndof} mode={mode} -> "
              f"{plan.label} (gpu_build={gpu_build}, gpu_ok={gpu_ok}, "
              f"n_gpus={n_gpus}, cpu_workers={n_workers_cpu})")
    return plan
