# Methods Paper Story And Figure Plan

## Core Message

This is a software-methods and validation paper:

1. PySCF already has mature SA-CASSCF derivative machinery.
2. block2 already provides DMRG active-space roots.
3. This work supplies the missing bridge: a PySCF-compatible DMRG active-space
   solver layer with root tracking, density/transition-density plumbing,
   response operations, benchmarks, and SHARC output.

State the positive contribution directly: the DMRG-facing layer satisfies the
solver and response contracts used by PySCF.

## Main-Text Figures

Figure 1: software workflow.

- PySCF SA-CASSCF reference and derivative infrastructure.
- block2 DMRG roots.
- MPSAsFCISolver bridge with root tracking and response operations.
- Validation branch against FCI fixed-orbital references.
- SHARC `QM.out` branch.

Figure 2: representative BVOE convergence.

- Keep readable: H4, H2O/STO-3G, H2O/6-31G, N2, C2, LiF, ethylene.
- Use log-log gradient and NAC errors.
- Emphasize high-M convergence and root/gauge diagnostics.

Figure 3 or Supporting Figure: basis and molecule matrix.

- STO-3G / 3-21G / 6-31G for each public system.
- Include well-known systems: ethylene, butadiene, formaldehyde, benzene.
- This figure supports generality but should not crowd the main result.

## Tables

Table 1: positioning relative to Freitag, Iino, PySCF/block2, SHARC.

Table 2: public systems and roles.

Table 3: validation layers.

Table 4: largest-M errors after job `8878801` completes.

## Language Rules

- Say "public benchmark systems", not private application chemistry.
- Use "interface regression test" in the manuscript.
- Avoid cluster-specific details in manuscript.
- Treat FCI references as validation only; production root following uses
  previous-step overlap.
- Mention PySCF derivative modules as library infrastructure, not as our code.

## Acknowledgments Draft

Y.L. thanks Prof. Christina L. Woo for mentorship and the research
environment, Prof. Richard Liu for helpful discussions, and Harvard FAS
Research Computing for computational resources. Funding information should be
confirmed before submission.
