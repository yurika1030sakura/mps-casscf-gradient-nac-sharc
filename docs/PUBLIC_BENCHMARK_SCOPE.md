# JCTC Manuscript Scope - Public Benchmarks Only

Generated 2026-04-27.

## Hard Scope Boundary

The JCTC manuscript is a methods/software paper. It must use only public,
non-confidential benchmark systems and must not mention, cite, copy, or
summarize any molecule, mechanism, geometry, trajectory, or product-path
evidence from the separate private JACS chemistry project.

## JCTC Story

Suggested framing:

> Open-source PySCF/pyblock2 implementation of analytic state-averaged
> DMRG-CASSCF gradients and nonadiabatic couplings, with SHARC integration
> and bond-dimension orbital-response-error validation.

The manuscript contributes implementation, validation, and benchmark
characterization, not a private mechanistic chemistry application.

## Allowed Systems

Use these in the JCTC manuscript:

- H4 chain / STO-3G / CAS(4,4)
- H2O / STO-3G / CAS(4,4)
- H2O / 6-31G / CAS(6,6)
- N2 / STO-3G / CAS(6,6)
- C2 / STO-3G / CAS(8,8)
- LiF avoided crossing / STO-3G / CAS(4,4), with NAC-gauge caveat
- Ethylene / STO-3G / CAS(2,2) or CAS(4,4) as a public SHARC-interface
  smoke test; the CAS(2,2) H/DM/GRAD/NACDR smoke passed locally

## Disallowed Material For Manuscript Text

Do not use any private chemistry-project files, private trajectory files,
private MECI searches, or private enlarged-active-space application runs for
the JCTC manuscript.

These can stay in the workspace for the separate chemistry project, but they
are not part of the JCTC paper.

## Figures

Recommended JCTC figure set:

1. Method schematic: PySCF SA-CASSCF + pyblock2 MPS + analytic CP response
   + SHARC output.
2. BVOE convergence: H4, H2O, N2, H2O/6-31G, C2, LiF.
3. SHARC-interface smoke: ethylene or LiF, showing DMRG-CASSCF energies,
   gradients, and NACDR are written in SHARC format.
4. Test/validation table: PySCF FCI agreement, MPS path agreement,
   SHARC smoke, CI/MPS converter tests.

## Manuscript Title Direction

Avoid any title that mentions a chemical application. Suggested direction:

> Analytic State-Averaged DMRG-CASSCF Gradients and Nonadiabatic Couplings
> in PySCF with SHARC Integration
