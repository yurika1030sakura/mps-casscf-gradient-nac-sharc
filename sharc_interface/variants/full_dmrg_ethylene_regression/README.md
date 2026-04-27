# Full DMRG-CASSCF Ethylene SHARC Regression

Public, non-confidential SHARC-interface regression for the methods
manuscript.

This replaces any unpublished application/demo in the methods manuscript.
It uses `method dmrg-casscf` to request H, DM, gradients, and S0/S1 NACDR
for ethylene at SA(2)-DMRG-CASSCF(2,2)/STO-3G.

Validated on 2026-04-27:

- `QM/QM.log` reports `method dmrg-casscf` and master error code 0.
- `QM/QM.out` contains Hamiltonian, dipoles, gradients, NACDR, and runtime.

Run directly, or submit through SLURM after adding site-specific account and
partition settings:

```bash
./submit_regression.sh
# or
sbatch submit_regression.sh
```
