# JCTC Readiness Assessment

Generated 2026-04-27.

## Short Answer

The paper is valid as a JCTC methods/software manuscript, and writing can
start now.

Do not frame it as a finished large-active-space production dynamics paper.
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

- Do not claim CAS(24,18)-scale SHARC trajectories are production-ready.
  The current `method dmrg-casscf` SHARC path still uses FCI-projected
  response vectors and is guarded by `dmrg-max-fci-dets`.
- Do not use private chemistry data, molecule names, trajectories, MECIs, or
  mechanistic conclusions in this manuscript.
- Do not overstate LiF NAC convergence. Energy and gradient converge, but
  the derivative coupling has a gauge/degeneracy caveat.

## Remaining Before Submission

- Write LaTeX manuscript from `OUTLINE.md`.
- Convert validation results into a concise methods table.
- Add figure captions that explicitly state the current response-vector
  limitation.
- Decide whether LiF stays in the main figure or moves to SI with the NAC
  caveat.
- Add code availability and test-reproducibility notes.
