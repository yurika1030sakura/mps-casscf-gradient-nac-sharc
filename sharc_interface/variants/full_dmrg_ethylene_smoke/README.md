# Full DMRG-CASSCF Ethylene SHARC Smoke

Public, non-confidential SHARC-interface smoke for the methods paper.

This replaces any private chemistry application/demo in the methods manuscript.
It uses `method dmrg-casscf` to request H, DM, gradients, and S0/S1 NACDR
for ethylene at SA(2)-DMRG-CASSCF(2,2)/STO-3G.

Validated locally on 2026-04-27:

- `QM/QM.log` reports `method dmrg-casscf` and master error code 0.
- `QM/QM.out` contains Hamiltonian, dipoles, gradients, NACDR, and runtime.

Run locally or through SLURM:

```bash
./submit_smoke.sh
# or
sbatch submit_smoke.sh
```
