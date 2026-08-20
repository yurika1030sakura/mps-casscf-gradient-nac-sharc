# Changelog

## v1.1.0 (2026-08-19)

Correctness release: three implementation defects in the response path are fixed. All three
caused hard, bond-dimension-independent stagnation of the response solve (true relative
residual ~1e-2..1e-1 against the 1e-6 certificate).

**Correction (2026-08-20) to the containment statement first published with this entry.**
The original wording said all three defects were "exactly invisible in the SA-2
equal-weight regime — the regime of every v1.0 validation benchmark." That is true of F1
only. The correct scope is:

- **F1** (state-average weight) is an exact no-op at SA-2 equal weights (`2 w = 1`) and
  affects only the sweep-Schur path.
- **F2** (zero-initialized addition fixed point) is **independent of the state-average
  weights, the penalty, `m_compress`, and inner-solve effort**. Its blind spot is
  **two-site** active spaces: a zero-norm MPS is an ALS fixed point only for `n_sites > 2`,
  so CAS(2,2) cannot exhibit it. The fix is inside `_combine_mps`, so the defect could
  reach **every** call site, including the global MPS-Krylov solver's arithmetic — not the
  Schur path alone.
- **F3** (inner solve reporting convergence at true residual 1e-2..5e-2) is likewise
  weight-independent, and affects the sweep-Schur path.

Consequently F2 and F3 were **active in the beyond-FCI runs of v1.0**, whose active spaces
have 16–24 sites, on both solver paths. What contained them is not the SA regime: it is
that (i) `certify_response` recomputes the true residual through the audited global matvec
and stayed honest — the affected large-CAS solves were published *with* their failed
certificates (`true_residual_relative` 0.09–0.18, `converged=False`) rather than as
converged results — and (ii) every v1.0 accuracy claim is a discrepancy measured against a
reference outside the response path (analytic FCI, or a response-free finite difference),
which a corrupted solve can inflate but cannot fake.

Readers should therefore treat v1.0's residual floors and wall times as measurements of the
v1.0 solver, not as properties of the method, and should not cite v1.0's Section 3.3
attribution of those floors to response-vector entanglement: at small scale the equivalent
floors collapse from ~5e-2 to ~1e-11 under these fixes at unchanged bond dimension.

### Fixed

- **F1 — missing state-average weight** (`sweep_coupled_response._ci_block_inverse`):
  the CI-block inverse applied `(H - E_i)^-1` where the CI Hessian block is
  `2 w_i P (H - E_i) P`; the missing `1/(2 w_i)` corrupted the Schur RHS, the Schur
  matvec, and the final CI multiplier whenever `2 w_i != 1`. Exact no-op for SA-2
  equal weights.
- **F2 — zero-initialized addition trap** (`cp_dmrg_response_mps_krylov._combine_mps`):
  block2 addition fits initialize from the first term, and a zero-norm MPS is an ALS
  fixed point for more than two sites, so `combine([(1, zero), (1, a)])` silently
  returned ~0 and annihilated the non-target CI response slots. Terms are now ordered
  by `|coeff| * norm` and negligible terms dropped, at every call site.
- **F3 — unreliable inner solve acceptance** (`sweep_coupled_response`):
  `DMRGDriver.multiply(left_mpo=...)` can report convergence at a true residual of
  1e-2..5e-2 under default local thresholds. The CI-block inverse now uses tight
  thresholds plus a root-projected conjugate-residual acceptor
  (`ci_inner_solver="multiply_cr"`, the new default) that accepts only on a measured
  per-slot true residual; `"cr"` and legacy `"multiply"` remain selectable.

### Hardening

- The `hcc-inverse` initial guess projects and penalizes against **all** reference
  roots (previously own-root only) and carries the `1/(2 w_i)` scaling.
- `solve_state_sweep_schur` records every resolved inner-solve setting and effort
  counter in its metadata, and `auto_response` attaches the accepted solver's metadata
  to every certificate (`extra["solver_meta"]`).

### Validation added

- `src/dmrg_analytic_dev/test_sa_response_reproducer.py`: SA-3 equal-weight
  (H4/3-21G/CAS(4,4)) and SA-2 unequal-weight (LiH/3-21G/CAS(2,2)) responses against
  kets-based dense references. Pre-fix floors are pinned as legacy-flag regressions
  (6.1e-2 / 4.1e-2); the fixed defaults certify at 5.3e-11 / 4.9e-15 with
  `|z - z_dense|` at 1e-11..1e-15. `.old_code_floors.json` preserves the pre-fix record.
- `verification/sa_response_fix_20260818/`: the full evidence chain — independent dense
  emulations of the Schur algebra, an adversarial SA-3 unequal-weight run
  (0.5/0.3/0.2), and mutation coverage showing each fix is individually load-bearing
  (disabling F1/F2/F3 one at a time floors at 2.6e-2 / 2.4e-2 / 7.5e-3 and is caught
  by the tests).
- All six v1.0 validation regressions reproduce reference-identical numbers on the
  fixed code (the fixes are no-ops in the published validation regime).

### Notes

- A cost caveat: the CR acceptor slows for the top averaged state at small
  positive-definite margin (observed 6x wall time, certifying at 6.3e-8 vs the
  1e-8-grade small-system results; still well inside the 1e-6 production certificate).
- Production hardening from 2026-08-08 (certificate binding, fail-closed release,
  exact requested M, dense-bridge audit, det=-1 parity-MPO gauge, non-singlet sectors,
  scale-consistent response stopping, raw overlap boundary) is included in this
  release.

## v1.0.0 (2026)

Public release: MPS-native SA-DMRG-CASSCF analytic gradients and nonadiabatic
couplings.
