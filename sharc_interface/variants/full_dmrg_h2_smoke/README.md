# Full DMRG-CASSCF SHARC H2 Smoke

This is the smallest non-hybrid test of the new `method dmrg-casscf` path.
It asks the SHARC PySCF interface for:

- DMRG-CASSCF SA(2) energies
- dipoles
- both gradients
- S0/S1 analytic NAC

The current implementation is intentionally guarded: DMRG roots are
projected to PySCF FCI ndarrays for the response equations, so this path is
valid only while the FCI vector dimension is below `dmrg-max-fci-dets`.
That is enough to validate the SHARC wiring before implementing the
MPS-Krylov response needed for CAS(24,18).

Run:

```bash
sbatch submit_smoke.sh
```
