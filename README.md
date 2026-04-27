# Analytic SA-DMRG-CASSCF Response Code

This is the sanitized public code bundle for the methods manuscript.
It contains only public benchmark systems and methods/software material.

## Contents

- `src/dmrg_analytic_dev/`: analytic SA-DMRG-CASSCF response code and tests.
- `sharc_interface/`: SHARC-PySCF interface with `method dmrg-casscf`.
- `sharc_interface/variants/full_dmrg_ethylene_regression/`: public ethylene
  SHARC-interface regression test.
- `benchmarks/bvoe_convergence_study/`: public BVOE convergence driver,
  summary, and figure.
- `docs/`: manuscript scope, readiness notes, and LaTeX skeleton.
- `UPLOAD_TUTORIAL.md`: step-by-step upload procedure.

## Environment

The code expects a Python environment with PySCF, pyblock2/block2, NumPy,
SciPy, and Matplotlib. SHARC itself is external; the included interface file
is for the SHARC-PySCF workflow.

## Quick Checks

Compile the Python files:

```bash
find src sharc_interface benchmarks -name '*.py' -print0 | xargs -0 python -m py_compile
```

Run the public SHARC regression after activating the correct PySCF/pyblock2
environment:

```bash
cd sharc_interface/variants/full_dmrg_ethylene_regression
PYSCF_PYTHON=python ./submit_regression.sh
```

Plot the BVOE summary:

```bash
cd benchmarks/bvoe_convergence_study
python plot_bvoe_phase2.py
```

## Scope

Do not add unpublished application systems, trajectory folders, mechanistic
conclusions, or local cluster output files to this repository.
