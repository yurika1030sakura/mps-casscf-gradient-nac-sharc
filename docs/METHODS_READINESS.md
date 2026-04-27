# Methods Paper Readiness Assessment

Generated 2026-04-27.

## Short Answer

The paper is valid as a methods/software manuscript, and writing can
start now.

Frame it as an open-source PySCF/pyblock2 implementation plus validation of
analytic SA-DMRG-CASSCF gradients/NACs, BVOE behavior, and SHARC-format
interface output on public benchmark systems.

## Claims That Are Currently Supported

- Two production code paths are validated:
  `CPDMRGCASSCFResponseMPS` and `MPSAsFCISolver`.
- Unit/integration validation is strong: 46/46 tests reported passing.
- BVOE phase-2 uses real block2 SU2 DMRG at FCI-converged orbitals, not the
  phase-1 SVD proxy.
- Public BVOE benchmarks now include H4, H2O/STO-3G, N2, H2O/6-31G, C2, and
  LiF.
- Public full DMRG-SHARC interface smoke is available with ethylene:
  H, DM, gradients, and NACDR are written in SHARC format with error code 0.

## Claims To Avoid

- Do not use unpublished application data, molecule names, trajectories,
  MECIs, or mechanistic conclusions in this manuscript.
- Do not overstate LiF NAC convergence. Energy and gradient converge, but
  the derivative coupling has a gauge/degeneracy caveat.

## Remaining Before Submission

- Replace placeholder author list, affiliations, acknowledgments, conflict
  of interest statement, and repository DOI/URL.
- Decide whether LiF stays in the main figure or moves to SI with the NAC
  caveat.
- Confirm journal template formatting with the corresponding author.  The
  current draft compiles as a neutral LaTeX article because the local TeX
  installation does not provide `achemso.cls`.
- Attach Supporting Information containing the full benchmark tables,
  representative input files, and regression-test logs.
