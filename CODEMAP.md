# CODEMAP

How a molecule flows through the package, and every module kept for release,
grouped by layer.

## Data flow

```
user data (atoms, coords, basis, ncas/nelecas | ao_targets, max_bond_dim)
        │
        ▼
certified_engine.compute_certified_derivatives          [single entry point]
        │
        ├─ active_space.select_active_space_by_ao_targets     (optional AO-pop CAS)
        ├─ build_robust
        │     ├─ fci_free_guard: det vs FCI_FREE_THRESHOLD (5e7) → arm sentinel,
        │     │                  pick RootTracking mode, assert_fci_free_if_needed
        │     ├─ progressive_schedule  (auto bond-dim ladder)
        │     ├─ dmrg_fcisolver.MPSAsFCISolver  (block2 DMRGDriver as PySCF solver)
        │     └─ casscf_convergence.escalating_casscf  (AH level-shift → Newton, on failure)
        │
        ├─ response  (backend chosen by size)
        │     ├─ det < 5e7 : cp_casscf_response.CPCASSCFResponseFCI  (dense reference)
        │     └─ det >=5e7 : auto_response.solve_response_auto
        │           ├─ sweep_coupled_response  (Schur / block-elimination)  ── or ──
        │           └─ cp_dmrg_response_mps_krylov  (global MPS-Krylov)
        │
        ├─ certified_response.certify_response   (true-residual certificate per solve)
        ├─ mps_lagrange_assembly + single_site_sigma + site_replacement_density
        │        (Lagrange gradient/NAC assembly; cross_geometry_overlap for NAC gauge)
        └─ system_diagnostics.assess_point   →  PASS / WARN / FAIL   +   certificate
```

## Production (importable core)

- `certified_engine.py` — single system-general entry point; robust build +
  certified grad/NAC + PASS/WARN/FAIL. Owns `progressive_schedule`.
- `fci_free_guard.py` — `FCI_FREE_THRESHOLD=5e7`, `determinant_dimension`,
  `RootTracking` string modes, `assert_fci_free_if_needed`, `DenseBridgeSentinel`.
- `system_diagnostics.py` — `assess_point()` health protocol (all thresholds as
  documented kwargs).
- `certified_response.py` — `ResponseCertificate` + `certify_response()`
  a-posteriori true-residual/leakage certificate.
- `casscf_convergence.py` — `escalating_casscf()` AH level-shift ladder + Newton
  fallback.
- `auto_response.py` — `solve_response_auto` / `compute_all_responses_certified`;
  Schur-then-global backend picker, cert-gated accept, cache recycling.
- `sweep_coupled_response.py` — `solve_state_sweep_schur` Schur/block-elimination
  CP response (dense-orbital GMRES + per-site block2 sweeps).
- `cp_dmrg_response_mps_krylov.py` — global MPS-Krylov coupled response
  (GMRES/BiCGStab/CR), CI vectors kept as MPS.
- `cp_dmrg_response_mps.py` — MPS-native base response class (Hessian-vector).
- `cp_casscf_response.py` — FCI-side CP-CASSCF response (small-CAS reference).
- `dmrg_fcisolver.py` — `MPSAsFCISolver`: block2 DMRGDriver → PySCF FCI-solver
  interface; FCI-free fast-path flags.
- `active_space.py` — `select_active_space_by_ao_targets` AO-population CAS
  selector.
- `mps_lagrange_assembly.py` — MPS-aware Lagrange nuclear-gradient/NAC assembly.
- `single_site_sigma.py` — single-site sigma kernel for the FR CP-DMRG response.
- `site_replacement_density.py` — block-replacement transition densities +
  FCI↔MPS converters (the only sanctioned dense bridge; marks the sentinel).
- `cross_geometry_overlap.py` — MPS-native FCI-free cross-geometry wavefunction
  overlap; `ROTATION_CONVENTION`, `sigma_min >= 0.9` continuity gate.
- `response_timing_log.py` — `TimingLog` wall-time/memory instrumentation.
- `__init__.py` — package exports.

## Interface (SHARC-facing)

- `sharc_interface/SHARC_PYSCF_ext.py` — main SHARC QM interface; parses
  `PYSCF.template`/`QM.in`, builds SA-/DMRG-CASSCF solver, dispatches
  E/dipole/grad/NAC incl. analytic-CP path.
- `sharc_interface/analytic_cp_sharc.py` — per-step analytic CP-CASSCF grad+NAC
  backend (`compute_grad_nac_analytic_cp`).
- `sharc_interface/dmrg_sharc_bridge.py` — bridge: multi-root DMRG CI solvers +
  DMRG-over-SA-CASSCF overlay (experimental solvers labeled non-certified).
- `sharc_interface/make_init_chk.py` — build initial RHF+SA-CASSCF checkpoint
  from XYZ (CLI defaults).
- `sharc_interface/make_geom_veloc.py` — SHARC geom/veloc generator
  (Maxwell-Boltzmann); needs full periodic-table masses.
- `sharc_interface/variants/` — example SHARC input sets (templates + QM.in).

## Drivers (benchmarks — paper figures/tables)

- `benchmarks/engine_general/run_engine_stress_test.py` — the "no source edit"
  exemplar; `compute_certified_derivatives` over 7 singlet/doublet/triplet systems.
- `benchmarks/bvoe_convergence_study/run_bvoe_phase2.py` (+ `plot_bvoe_phase2.py`,
  `run_fci_free_endpoint_check.py`, `audit_bvoe_references.py`,
  `merge_bvoe_parallel_results.py`) — main convergence figure + FCI-free endpoint.
- `benchmarks/large_active_space/` — beyond-FCI drivers:
  `run_polyene_beyond_fci.py`, `beyond_fci_validation_clean.py`,
  `acene_beyond_fci.py`, `acene_nac.py`, `run_beyond_fci_{analytic,nac,schur}.py`,
  `run_cas_directional_fd.py`, `build_reference_lowest.py`,
  `combine_partial_derivatives.py`, `run_anthracene_pi14*.py`,
  `report_anthracene_strict_response.py`, `run_lif_avoided_crossing.py`,
  `run_mps_only_hchain_response.py`, `run_seed_independence.py`, `acene_systems.py`.
- `benchmarks/root_tracking/run_lif_cas66_root_tracking.py` +
  `lif_subspace_tracking.py` — subspace-continuity (sigma_min) scan.
- `benchmarks/response_timing/` — `run_schur_vs_global_scaling.py`,
  `run_anthracene_headtohead.py`, `run_precond_benchmark.py`,
  `run_recycle_same_geometry.py`, `run_mps_krylov_timing.py`,
  `run_repeated_call_trajectory.py`.
- `benchmarks/fd_validation/run_fd_validation_suite.py` — analytic-vs-FD
  regression with PASS/WARN/FAIL.
- `benchmarks/*/submit_*.sh` — SLURM job files used on the authors' cluster (run records; adapt paths for your machine).

## Diagnostics / validation / reference (kept, non-user-facing)

These live flat in `src/dmrg_analytic_dev/` alongside the core:
- `fd_validation.py`, `nac_validation.py`, `overlap_fci_reference.py` — FD/FCI
  validation backends used by the engine and tests.
- `test_*.py` + `*.json` fixtures + `supplementary/` — the test suite.
- `scripts/dmrg_sharc_preflight.py` — large-CAS DMRG-SHARC preflight checks.

## Docs

- `docs/ARCHITECTURE.md` — one-page module map.
- `docs/FCI_REFERENCE_PROTOCOL.md`, `docs/ROOT_MATCHING_PROTOCOL.md`,
  `docs/PRODUCTION_ROOT_TRACKING.md` — numerical protocols.
- `docs/figures/` — architecture / method figures.

The manuscript and Supporting Information are published separately (see the
citation in the README); this repository is the code and its benchmarks only.
