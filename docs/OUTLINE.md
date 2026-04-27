# JCTC Manuscript Outline

## 1. Introduction

- Nonadiabatic dynamics needs gradients and derivative couplings.
- CASSCF is robust near conical intersections, but FCI active spaces limit
  practical size.
- DMRG expands active-space reach, but open-source analytic response and
  SHARC plumbing are missing in the PySCF ecosystem.
- This work contributes implementation, validation, and public benchmark
  data, not a new response theory.

## 2. Theory

- State-averaged CASSCF Lagrangian and response equations.
- Freitag-Reiher single-site DMRG response approximation.
- MPS site-replacement transition density and sigma-vector operations.
- Exact pieces vs approximate pieces.

## 3. Implementation

- PySCF integration.
- pyblock2 MPS operations.
- `MPSAsFCISolver` path.
- `CPDMRGCASSCFResponseMPS` path.
- SHARC output path through `method dmrg-casscf`.
- Guardrail: current SHARC path uses FCI-projected response vectors for
  medium active spaces; full CAS(24,18)-scale response requires MPS-Krylov.

## 4. Validation

- Unit/integration tests.
- FCI agreement for small CAS.
- MPS/FCI converter validation.
- Gradient and NAC agreement with PySCF references.

## 5. Bond-Dimension Response Error

- H4, H2O, N2 existing phase-2 results.
- H2O/6-31G CAS(6,6), C2 CAS(8,8), and LiF avoided crossing extensions.
- Note that LiF is useful as a nonzero-NAC stress test but has a derivative
  coupling gauge caveat; use H2O/6-31G and ethylene as cleaner public
  validation anchors.
- Explain phase-1 catastrophic spikes as truncated-CI orbital reoptimization
  artifacts, removed in phase 2 by fixing FCI-optimized orbitals.

## 6. SHARC Interface Demonstration

- Use public ethylene or LiF only.
- Show that DMRG-CASSCF H/DM/GRAD/NACDR are emitted in SHARC format.
- Current public ethylene CAS(2,2) smoke passes locally with SHARC error
  code 0.
- Keep this as an interface demonstration, not a mechanistic chemistry
  claim.

## 7. Limitations

- Current `dmrg-casscf` SHARC path is medium-CAS because response vectors
  are still FCI-projected.
- CAS(24,18)-scale trajectories require MPS-Krylov response.
- Root tracking and production dynamics need additional stress tests.

## 8. Conclusions

- Open-source PySCF/pyblock2 analytic SA-DMRG-CASSCF response backend.
- SHARC-compatible output.
- Public benchmark validation and BVOE characterization.
