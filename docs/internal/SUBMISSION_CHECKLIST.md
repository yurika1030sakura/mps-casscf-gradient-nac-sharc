# Methods Manuscript Submission Checklist

Generated 2026-04-27.

## Scientific Scope

- Main manuscript uses only public benchmark systems: H4, H2O, N2, C2, LiF,
  and ethylene.
- Private application chemistry, trajectories, mechanistic conclusions, and
  molecule names are intentionally excluded.
- The paper is framed as implementation, validation, and reproducibility of a
  PySCF/pyblock2 SA-DMRG-CASSCF response workflow, not as a new response
  theory derivation.

## Completed in the Current Draft

- Added comparison table against SA-CASSCF NAC, SA-DMRG-SCF response,
  SA-DMRG-CASSCF gradients, PySCF/block2, and SHARC.
- Added explicit public benchmark workflow.
- Added computational tolerances for FCI references, DMRG scans, and ethylene
  SHARC-interface regression.
- Added validation matrix covering solver, MPS/FCI mapping, response linear
  algebra, gradients/NACs, and SHARC output.
- Added quantitative largest-M response-error table.
- Added Supporting Information and Data/Code Availability sections.

## Must Be Filled Before Journal Submission

- Final title and running title, if the target journal template requests one.
- Author list, affiliations, corresponding author email, and ORCID handling.
- Acknowledgments and funding.
- Conflict of interest statement.
- Permanent repository URL or DOI for the public code/data bundle.
- Final Supporting Information PDF/ZIP with full benchmark tables and input
  files.
- Cover letter with article type, significance paragraph, and suggested
  reviewers.

## Reviewer-Sensitive Points

- Do not overstate LiF NAC convergence.  Keep the gauge/degeneracy caveat.
- Do not claim trajectory-scale dynamics or unpublished application results
  in this manuscript.
- Be explicit that public derivative benchmarks use active spaces where FCI
  references are available.
- Keep the ethylene example framed as an interface regression test, not a
  photochemical application.
