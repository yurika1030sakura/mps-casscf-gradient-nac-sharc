# Public Methods Status For Review

Generated 2026-04-27.

## Manuscript Scope

This is a methods/software manuscript about analytic
SA-DMRG-CASSCF gradients and nonadiabatic couplings in PySCF/pyblock2, with
SHARC-format output.

Only public benchmark systems are included here.

## Current Readiness

The project is ready to enter manuscript writing.

Supported public claims:

- Two validated implementation paths:
  `CPDMRGCASSCFResponseMPS` and `MPSAsFCISolver`.
- Test status reported as 46/46 passing across 14 test files.
- Real block2 SU2 DMRG BVOE phase-2 figure is generated:
  `bvoe_convergence_study/figures/bvoe_phase2.png` and `.pdf`.
- Public BVOE systems:
  H4/STO-3G CAS(4,4), H2O/STO-3G CAS(4,4), N2/STO-3G CAS(6,6),
  H2O/6-31G CAS(6,6), C2/STO-3G CAS(8,8), and LiF/STO-3G CAS(4,4).
- Public SHARC-interface smoke:
  ethylene/STO-3G CAS(2,2) with H, DM, gradients, and NACDR written to
  SHARC format with master error code 0.

## Main Figure Status

`bvoe_phase2` is now strong enough for a draft:

- H4, H2O/STO-3G, N2, and H2O/6-31G show the cleanest convergence story.
- C2 supplies a public CAS(8,8) stress test.
- LiF supplies a nonzero-NAC stress test, but with a NAC gauge caveat.

Do not overclaim C2 as chemical-accuracy-converged at the current M values.
Do not overclaim LiF NAC convergence.

## SHARC Status

The public ethylene smoke validates the interface plumbing:

- `method dmrg-casscf`
- SA(2)-DMRG-CASSCF(2,2)/STO-3G
- H/DM/GRAD/NACDR requested
- SHARC `QM.out` produced successfully

This is an interface demonstration, not a production trajectory claim.

## Critical Limitation To State In The Paper

The current SHARC path still uses FCI-projected response vectors. It is valid
for the benchmark sizes used here, but CAS(24,18)-scale production
trajectories require an MPS-Krylov response backend.

## Recommended Next Writing Steps

1. Draft the manuscript from `OUTLINE.md`.
2. Use `PUBLIC_BENCHMARK_SCOPE.md` as the scope boundary.
3. Use `METHODS_READINESS.md` for supported claims and caveats.
4. Put C2 and LiF caveats in the figure caption or SI.
5. Add code availability, test reproducibility, and version details.
