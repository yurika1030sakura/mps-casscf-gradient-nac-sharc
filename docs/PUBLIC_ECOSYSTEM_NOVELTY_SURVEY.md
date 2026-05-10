# Public Ecosystem And Novelty Check

Date checked: 2026-04-27

## Bottom Line

The public ecosystem already contains important adjacent capabilities:

- PySCF has FCI-based SA-CASSCF analytic gradients and nonadiabatic couplings.
- PySCF/Block and block2 document DMRG-CASCI/DMRG-CASSCF workflows, including state-averaged DMRG-CASSCF.
- block2 documents DMRGSCF nuclear gradients and geometry optimization examples.
- SHARC 4.0 has a PySCF interface for CASSCF/PDFT gradients and nonadiabatic couplings.
- QCMaquis/OpenMolcas publicly supports DMRG-CI/DMRG-SCF, state-specific and state-averaged DMRG-SCF, and state-specific DMRG-SCF analytic gradients.
- Freitag et al. 2019 clearly reports an implementation of approximate SA-DMRG-SCF gradients and NACs.

I did not find public documentation showing a user-facing PySCF/pyblock2 workflow that combines SA-DMRG-CASSCF analytic gradients, analytic NACs, response/root tracking, FCI-reference validation, and SHARC-compatible output. That is the defensible novelty boundary.

Do not claim first SA-DMRG-SCF gradient/NAC theory. The safe framing is:

> We implement and validate a PySCF/pyblock2 workflow that connects DMRG active-space roots to PySCF's SA-CASSCF derivative infrastructure and exposes the result through a SHARC-compatible interface.

## Evidence

### PySCF

PySCF documents FCI-based SA-CASSCF NACs:

- `pyscf.nac.sacasscf.NonAdiabaticCouplings` is documented as "SA-CASSCF non-adiabatic couplings (NACs) between states".
- The module source comments say the NAC extension needs the active-space solver path from `make_rdm12` to `trans_rdm12`.
- PySCF feature pages list non-relativistic CASCI, CASSCF, and state-average CASSCF nuclear gradients.

Relevant sources:

- https://pyscf.org/pyscf_api_docs/pyscf.nac.html
- https://pyscf.org/_modules/pyscf/nac/sacasscf.html
- https://pyscf.org/_modules/pyscf/grad/sacasscf.html
- https://pyscf.org/features.html

Local environment check:

- `pyscf.grad.sacasscf` and `pyscf.nac.sacasscf` are installed.
- `pyblock2.dmrgscf.py` in the current environment exposes a DMRGCI-style solver with RDM routines, but no local `trans_rdm12`, `nuc_grad_method`, `NonAdiabaticCouplings`, or `Gradients` implementation was found in that file.

### PySCF/Block and block2

Public docs show DMRG-CASCI/DMRG-CASSCF and state-average DMRG-CASSCF workflows.

- Old Block/PySCF docs describe replacing the CASSCF FCI solver with `dmrgscf.DMRGCI`, state-average DMRG-CASSCF, and `trans_rdm12`.
- block2 docs show DMRGSCF, state-average examples, and a DMRGSCF nuclear gradient/geometry optimization example.

Relevant sources:

- https://pyscf.org/interface.html
- https://pyscf.org/Block/with-pyscf.html
- https://block2.readthedocs.io/en/latest/user/dmrg-scf.html

Interpretation:

These pages establish public DMRG-CASSCF infrastructure. They do not, from the docs checked, provide the full SA-DMRG-CASSCF analytic NAC/response/SHARC workflow that this manuscript presents.

### SHARC

SHARC 4.0 documents a PySCF interface:

- `SHARC_PYSCF.py` provides dipoles, gradients, and nonadiabatic couplings at CASSCF and PDFT levels.
- The manual says this PySCF interface is restricted to singlet states.

Relevant source:

- https://sharc-md.org/?page_id=1454

Interpretation:

SHARC already has a PySCF CASSCF/PDFT interface. The contribution here should be described as adding/testing a DMRG-CASSCF backend path, not inventing SHARC-PySCF.

### QCMaquis/OpenMolcas

QCMaquis/OpenMolcas is the closest public neighboring ecosystem.

Public QCMaquis GitHub lists:

- state-specific and state-averaged DMRG-SCF calculations,
- analytic gradients for state-specific DMRG-SCF calculations,
- MPS state interaction,
- state-specific and quasi-degenerate multi-state DMRG-NEVPT2.

The QCMaquis manual text cloned from the public repository says the OpenMolcas interface implements state-specific and state-average DMRG-SCF algorithms and analytic gradients for state-specific calculations.

Relevant sources:

- https://github.com/qcscine/qcmaquis
- https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/dmrgscf.html
- https://arxiv.org/abs/2505.01405

Interpretation:

This is strong related software. Public QCMaquis materials I checked advertise state-specific analytic gradients, not a public user-facing SA-DMRG-SCF NAC interface. Freitag 2019 reports an implementation, but the public QCMaquis/OpenMolcas feature lists do not make that capability look like a standard documented public workflow.

### Freitag 2019

Freitag, Ma, Baiardi, Knecht, and Reiher 2019 is directly relevant and must be cited prominently.

The abstract states that they present approximate analytical gradients and NACs for SA-DMRG-SCF and apply their implementation to conical-intersection optimization.

Relevant source:

- https://arxiv.org/abs/1905.01558
- DOI: 10.1021/acs.jctc.9b00969

Interpretation:

This paper lowers novelty if the manuscript claims new theory or first implementation in general. It does not eliminate novelty for an open PySCF/pyblock2 implementation, benchmark suite, and SHARC-compatible workflow.

## Recommended Manuscript Framing

Use:

- "implementation, validation, and reproducibility paper"
- "PySCF/pyblock2 workflow"
- "connects DMRG active-space roots to PySCF's SA-CASSCF derivative infrastructure"
- "SHARC-compatible output path"
- "validated against fixed-orbital spin-adapted singlet FCI references"

Avoid:

- "first SA-DMRG-SCF analytic gradient/NAC theory"
- "no existing DMRG derivative implementation"
- "only public implementation" unless we add a more exhaustive software/version survey and are willing to defend it
- claims that depend on private cluster settings

## Practical Novelty Statement

Suggested wording:

> Prior work established SA-DMRG-SCF derivative theory and related implementations, while public PySCF and block2 ecosystems provide complementary CASSCF derivative and DMRG-CASSCF capabilities. The present work fills the practical gap between these components by providing a tested PySCF/pyblock2 response workflow for DMRG-backed SA-CASSCF gradients and derivative couplings, together with FCI-reference benchmarks and SHARC-compatible output.
