# Multigrid preconditioner — development notes

Detailed, dated engineering history for `multigrid.py` (`MGSolver`). This is
background/rationale for *why* the code looks the way it does — not required
reading to use or modify the solver day to day. The docstrings in
`multigrid.py` and `fea.py` carry short pointers here; look here when you
need the full reasoning behind a design decision, not just its conclusion.

## VoxelSim FEA-specific findings

### Small/thin bearing patches can stall or diverge MG-PCG (2026-08-23)

Ported from Blendtopo (topology optimization), where boundary conditions are
typically spread over a whole face of a SIMP design-domain box. VoxelSim FEA's
parts are arbitrary real-world geometry, and a "bearing" mesh commonly
touches only a small or thin contact patch (a bolt hole, a small mounting
foot) after voxelization.

Synthetic testing during the port found that geometric multigrid degrades
badly for exactly this case: a **fully solid** grid with a single-node point
bearing already fails to converge within 200 iterations (residual ratio
stuck around 0.89), and a thin member with the same point BC actively
**diverges** (residual ratio growing past 30x). A small (3x3-node) but
otherwise adequately-constrained patch showed the same stall pattern.

Root cause: the coarse-grid correction under-represents an essential BC that
only constrains a tiny contact patch. Node coarsening keeps only every other
node per level, so the less of the patch survives at each coarser level, and
the V-cycle's low-frequency error correction — the part multigrid depends on
for its whole speed advantage — degrades for exactly the near-rigid-body
modes it's supposed to handle well. This is a real limitation of plain
geometric MG with small-patch essential BCs, not a one-line bug.

Fix (`MGSolver.solve()`): a stall/divergence guard tracks the best (lowest)
residual ratio seen so far; if the current ratio exceeds 5x that best after
at least 10 iterations, the run is not just slow, it's losing ground — raise
immediately instead of burning the rest of the iteration budget. The caller
(`fea.VoxelFEA._solve_multigrid`) already has a try/except around
`MGSolver.solve()` (originally there only to catch construction failures)
that falls back to plain Jacobi-PCG on any exception, so this reuses that
existing fallback path. Net effect: MG runs fast (~10-11 iterations) on the
well-conditioned cases it was built for, and falls back to the slower but
robust Jacobi path — automatically, per solve, with a log line — on the
cases where it can't help. Verified both directions: Table-1-style full-face
cantilevers still hit the same flat 11-iteration MG performance with no
false-positive fallback, and the small-patch-BC cases now fall back cleanly
instead of returning a garbage (diverged) displacement.

### max_cg default: 200 -> 1000 (2026-08-23, user decision)

`solve()`'s default iteration cap (used when the caller passes
`max_cg=None`) was raised from 200 to 1000. Rationale (Philip's call, not a
measurement): most users of this add-on are not FEA experts and would much
rather wait a bit longer for a solve than see a "did not converge" warning
and a possibly-unreliable result. The stall guard above already catches the
one case that actually needs a *low* cap (genuine divergence) well before
1000 — typically within 10-20 iterations — so raising this default only
gives more headroom to legitimately-slow-but-converging problems, it does
not change how divergence is handled.

## Blendtopo multi_gpu `self.xp` history

Real-hardware benchmark timeline (2x GTX 970, see Blendtopo's
`paper/experiments/benchmark_multigpu.py`), inherited unchanged from
Blendtopo's `core/multigrid.py` since the same V-cycle/pooling code is
reused here. Kept because the reasoning that turned out to be WRONG is
exactly the kind of thing worth not re-discovering the hard way a third
time. VoxelSim FEA's own `_init_pool` additionally has a `multi_cpu` branch
(via `parallel_cpu.CPUMatVecPool`) that Blendtopo's copy no longer has —
Blendtopo removed multi_cpu outright after finding an Amdahl-ceiling
argument against it (see its own `compute_plan.py`); VoxelSim FEA kept
multi_cpu, so `multigrid.py` here still needs to support it.

  Round 1: self.xp was numpy for every "parallel" plan (multi_gpu/multi_cpu),
  copying fea.VoxelFEA's rule, AND the V-cycle smoother was routed through
  the pool along with the outer CG matvec. Multi-GPU was 1.5x-3.5x SLOWER
  than single-GPU. Root cause: routing smoothing through the pool (~5 pooled
  calls/iteration instead of ~2), each paying a full host round-trip per
  device for work too small to amortize. Fix: split MGSolver._apply (local,
  never pooled) from _apply_pooled (outer CG only, pooled).

  Round 2 (after the Round 1 fix + streams/pinned-memory in
  parallel_gpu_domain.py): self.xp still numpy. Small problems got much
  faster (~8x vs single-GPU), but large problems got WORSE (~6.5x slower,
  1,101,411 DOF, GPU utilization measured <30%). Hypothesis at the time:
  the now-un-pooled local smoothing, forced onto single-threaded host numpy,
  had become the bottleneck -- so self.xp was changed to
  backend.get_xp(True) (Cu-Py, single device) for multi_gpu, keeping only
  _apply_pooled's host round-trip to reach the multi-device pool.

  Round 3 (testing the Round 2 hypothesis): multi-GPU got WORSE AGAIN
  (45.6s vs 19.6s at the same 1.1M-DOF case) -- disproving the hypothesis.
  Two likely compounding causes: (1) Cu-Py's bincount is a scatter-add with
  colliding indices (shared FEA nodes between elements), which leans on
  atomics that apparently do not parallelize well on this hardware -- host
  numpy's bincount was actually faster per call, not slower; (2) forcing
  xp=Cu-Py added two extra full-vector D2H/H2D copies per _apply_pooled call
  (backend.asnumpy(u) / xp.asarray(...) around the pool -- a no-op when xp
  is already numpy, real transfers when it isn't) that Round 2 did not pay.
  Reverted self.xp back to numpy for every parallel plan (Round 2's rule).

  Status (as of the Round 3 revert): 19.6s vs 3.0s single-GPU (~6.5x slower)
  was the then-current best-known state for large multi-GPU problems, root
  cause unidentified. MGSolver.solve() gained wall-clock instrumentation
  (self._t_pooled / self._t_local, printed when verbose) instead of a third
  guess.

  Round 4 (2026-07-09, after fixing a real bug in parallel_gpu_domain.py --
  see that file's apply()/diagonal() docstrings): the domain-decomposed
  pool's compute kernels (indexing/matmul/bincount) were being dispatched to
  cupy's DEFAULT stream while only the explicit .set()/.get() calls used the
  per-device ``stream`` -- a genuine data race between the async H2D copy
  and the kernels reading its target, and between the kernels writing
  ``part`` and the async D2H copy reading it. This was already present
  during Round 2's test ("after streams/pinned-memory in
  parallel_gpu_domain.py" -- i.e. Round 2 ran on the buggy pool). A race
  that corrupts the level-0 matvec doesn't fail loudly at large problem
  sizes the way it does at small ones (see multigrid.py's solve(): small
  grids hit ZeroDivisionError outright, large grids just silently need many
  more CG iterations, sometimes hitting the iteration cap) -- exactly
  the "GPU utilization <30%, unexplained 6.5x slowdown" symptom Round 2
  reported. So Round 2's numbers, and by extension Round 3's "fix" of them,
  cannot be trusted as evidence about self.xp: they were measured against a
  correctness bug, not against the real cost of Cu-Py vs numpy for the
  local V-cycle work.

  With the race fixed and a fresh real-hardware run (benchmark_multigpu.py,
  2x GTX 970, verbose breakdown), Round 3's own claim is directly refuted by
  clean data: at 80x80x80 (1,594,323 DOF), single-GPU mode's "local" time
  (self.xp=Cu-Py there, since plan.parallel is False) is 1032.4ms across
  ~30 _apply calls (~34ms/call), while multi-GPU mode's "local" time
  (self.xp=numpy, the Round-3 rule) is 13,592.8ms across ~61 calls
  (~223ms/call) for the IDENTICAL per-call computation on the SAME
  hardware -- Cu-Py is ~6.5x FASTER than numpy here, the opposite of what
  Round 3 concluded. The numpy-is-faster hypothesis was never wrong because
  of atomics; it was measured against a solver that was quietly computing
  garbage.

  Fix: self.xp is Cu-Py (single device, never pooled) for every plan again --
  ``self.xp = self.plan.xp`` unconditionally, restoring Round 2's rule --
  while the outer CG matvec (_apply_pooled) still goes through the
  domain-decomposed pool when one exists. Do NOT also route V-cycle
  smoothing (_apply at any level) through the pool: that specific change IS
  independently confirmed bad by Round 1's real-hardware measurement
  (1.5-3.5x slower, well before the race bug existed, so not contaminated
  by it) -- each smoothing pass is too little compute to amortize even a
  correctly-working pool's per-call dispatch/sync overhead, which is most
  likely (not yet isolated on real hardware) the fixed per-call cost of
  `with cp.cuda.Device(dev_id):` context switches and `stream.synchronize()`
  under Windows' WDDM driver model for consumer GPUs, not raw PCIe transfer
  time (a few MB should move in <1ms, not the ~100ms/call implied by the
  pooled timing at large sizes). This also means multi-GPU can never reach
  100% aggregate GPU utilization by construction: only ONE device is active
  during the (numerous) local V-cycle calls, and the pool's brief
  synchronize-then-continue pattern in _apply_pooled means both devices are
  never simultaneously fed for the whole solve -- expected, not a further
  bug, given the current single-pooled-call-per-CG-iteration design.

  Round 4b (same day, follow-up): Round 4 made multi-GPU match single-GPU
  timing, but only ONE GPU was ever doing anything outside the single
  pooled outer-CG matvec per iteration -- the ~5 local V-cycle smoothing
  calls at level 0 all ran on device 0 alone, self.xp being single-device
  Cu-Py. Pooling each of those calls individually (i.e. routing every
  _apply(0, ...) through _apply_pooled) was tried and confirmed a net loss
  at every tested size (still true even with the race-fixed, streamed
  domain pool -- Round 1's finding holds regardless of pool quality,
  because the bottleneck is FIXED per-call dispatch overhead, not transfer
  volume: observed pooled cost scaled with call COUNT, not with problem
  size, at ~20-75ms/call even for a few-MB domain slice, which is far more
  than PCIe transfer of that size should cost -- consistent with Cu-Py
  device-context-switch + stream-synchronize overhead under Windows' WDDM
  driver model for consumer GPUs, though this specific cause is still
  unconfirmed on real hardware). Fix: added
  DomainGPUMatVecPool.smooth(), which runs ALL `iters` sweeps for a whole
  _smooth() call inside ONE dispatch per device (one context switch, one
  H2D, one D2H, one synchronize for the entire sweep sequence, not per
  sweep) -- a restricted-additive-Schwarz-style block-Jacobi smoother:
  each device iterates using its OWN copy of the shared boundary plane,
  which goes slightly stale across the `iters` local sweeps instead of
  being re-synced every sweep. This is NOT exact global Jacobi, but per
  this module's own "Correctness note" that was never required -- the
  outer CG still applies the TRUE global operator (apply()/_apply_pooled)
  for its own matvec, so an approximate preconditioner only changes
  iteration count, never correctness. MGSolver._smooth routes level 0
  through this when available (falls back to the untouched per-sweep
  _apply loop otherwise, e.g. multi_cpu or the plain broadcast pool).
  ** Like the rest of parallel_gpu_domain.py, NOT yet exercised on real
  multi-GPU hardware -- re-run benchmark_multigpu.py and check both wall-
  clock AND actual solution accuracy (the SIMP objective / compliance,
  not just CG iteration count) before trusting this for production runs,
  since the staleness tradeoff above is reasoned through, not measured. **

  Round 4c (2026-07-09, same-day real-hardware follow-up): Round 4b's first
  cut of smooth() AVERAGED the two devices' independent results at each
  shared boundary plane. On real hardware this produced a ZeroDivisionError
  at the SMALLEST tested grid (6x6x6, MULTI_GPU only) at ``p = z +
  (rz_new / rz) * p`` -- ``rz`` had gone to exact 0.0, i.e. some earlier
  ``z = vcycle(0, r)`` came back as (numerically) zero despite a nonzero
  residual, something that never happened with the single-device smoother.
  Averaging two independently-converged local Jacobi results isn't a
  standard construction and doesn't obviously preserve whatever
  symmetry/consistency plain CG's convergence relies on from iteration to
  iteration -- switched to the standard Restricted Additive Schwarz (RAS)
  convention instead: each shared plane is owned by exactly ONE neighbour
  (see parallel_gpu_domain.DomainGPUMatVecPool's ``_own_lo``/``_own_hi``),
  so the merge is a plain non-overlapping slice write, no averaging, no
  division. Also added defensive zero-guards around both CG divisions
  (``p @ Ap`` and ``rz``) in solve() regardless of root cause -- an
  iterative solver hitting a vanishing denominator on a well-posed SPD
  system usually means "nothing left to improve here", not a real error,
  and should stop/restart cleanly rather than crash and disable multigrid
  for the rest of the run. Re-verify on real hardware before trusting this
  for anything beyond benchmarking: RAS ownership is more standard than
  averaging but this specific implementation is still unexercised on real
  multi-GPU hardware.

  Round 4d (2026-07-09, REVERTED Round 4b/4c): the RAS ownership fix
  stopped the crash but real hardware then showed CG residual ratio
  GROWING across iterations (1.5x-6.2x, 3/5 solves hitting the iteration
  cap) -- not just slow, actually diverging. Root cause: every
  device's local bincount is only a PARTIAL matvec at BOTH of its boundary
  node planes (missing the neighbour's adjacent element contribution),
  regardless of which device's copy is kept -- a genuinely wrong local
  operator, not a preconditioner-quality tradeoff. _smooth's routing to
  DomainGPUMatVecPool.smooth() has been reverted; level-0 smoothing is
  back to Round 4's single-device Cu-Py loop (proven correct and already a
  massive win over the pre-Round-4 numpy state). smooth()/set_bc() remain
  in parallel_gpu_domain.py, unused, for whoever implements this properly
  with a real ghost-element ring (ADD one layer of the neighbour's
  elements, read-only, plus that layer's density each SIMP iteration, so
  each device's local matvec is EXACT at every node it owns) rather than
  patching the merge convention again.

  Round 4e (2026-07-09, user asked for the real fix rather than stopping at
  Round 4's parity-with-single-GPU win): implemented the ghost-element ring
  Round 4d called for. DomainGPUMatVecPool now builds a SECOND, wider
  element/dof range per device for smooth() specifically (apply()'s plain
  z-slab partition is untouched, still correct for its own '+='-merge use)
  -- each device's local element set gains a read-only ghost copy of the
  ONE adjacent element row from each neighbour (contiguous in iz, so it's
  just a wider slab: [a_k-1, b_k+1) clipped to grid bounds), and that ghost
  row's density is refreshed by set_density() every SIMP iteration same as
  its own elements. With the ghost row present, a device's local bincount
  at BOTH its own boundary node planes becomes numerically EXACT (every
  element touching those nodes is now included, own or ghost) -- this is
  standard overlapping Schwarz domain decomposition, not the non-
  overlapping partition Round 4c used. Ghost DOF *values* (not just the
  stiffness/density inputs) are frozen for the whole `iters` sweep sequence
  via set_bc() leaving minv/free at 0/False outside the truly-owned
  sub-range, so the Jacobi update never touches them -- classic lagged
  block-Jacobi/Schwarz boundary data, an approximation of which residual to
  smooth against, never of the operator itself, so per the module's
  "Correctness note" this can only affect iteration count, never
  correctness. _smooth routes level 0 through this again.

  Round 4f (2026-07-09, real-hardware verification of Round 4e): 16x16x16
  and 26x26x26 MULTI_GPU matched single-GPU's CG iteration count EXACTLY
  (5.2 and 9.8 avg) -- strong evidence the Round 4e ghost-ring smoother is
  now CORRECT in general, unlike Round 4b/4c. The smallest tested size
  (6x6x6) still needed ~20x more iterations than single-GPU (122 vs 6.4
  avg) and didn't converge within cap -- but the residual ratio SHRANK
  across the three consecutive warm-started solves (1.3 -> 0.18 -> 0.01),
  the opposite of Round 4b/4c's growing ratios, i.e. "slow", not "wrong".
  Root cause: smooth()'s ghost DOFs are frozen for a whole sweep sequence,
  a good approximation only when the frozen boundary is a small fraction
  of the subdomain; at nz=6 split across 2 devices the 1-element ghost row
  is 33% of each 3-element-deep subdomain (vs 7.7% at 26x26x26, 12.5% at
  16x16x16) -- an expected domain-decomposition-preconditioner weakness at
  that scale, not a bug. Since this size (1,029 DOF) is far below
  compute_plan.py's MULTI_GPU_DOF anyway (never auto-selected in real
  usage, only reachable by the benchmark script's explicit mode=MULTI_GPU),
  DomainGPUMatVecPool.__init__ now refuses to build itself when the
  smallest z-slab has fewer than 4 elements, which routes through
  MGSolver's existing exception-triggered fallback chain (domain pool ->
  plain broadcast pool -> single GPU).

  Round 4g (2026-07-09, DISABLED by default, real-hardware measurement):
  Round 4e/4f made the ghost-ring smoother CORRECT, but real hardware then
  showed it's a clear PERFORMANCE regression vs. Round 4's simpler state --
  16x16x16 went from 243.0ms/iter (parity with single-GPU's 245.4ms) to
  324.2ms (+32%); 26x26x26 went from 462.5ms (faster than single-GPU's
  484.9ms) to 744.6ms (+54%). Root cause: this hardware's fixed
  per-dispatch overhead (device-context-switch + stream-sync, ~50-100ms/
  call) applies PER POOLED CALL, not per unit of work moved -- batching all
  `iters` sweeps into one smooth() dispatch already amortizes it across the
  sweep loop, but going from 1 pooled dispatch/CG-iteration (the outer
  matvec alone) to 3 (pre-smooth, matvec, post-smooth) still roughly
  triples that fixed cost -- enough to outweigh the benefit of genuinely
  splitting the smoothing compute across 2 GPUs, at least on this specific
  hardware (2x GTX 970, Windows/WDDM). `MGSolver._TRY_POOLED_SMOOTH` is the
  on/off switch for Round 4e/4f's otherwise-correct code -- flip it to True
  to re-measure on hardware with lower per-dispatch overhead (Linux+TCC,
  NVLink, newer GPUs); left False here since Round 4 (matvec-only pooling)
  is both correct AND faster on the hardware this was actually measured on.
