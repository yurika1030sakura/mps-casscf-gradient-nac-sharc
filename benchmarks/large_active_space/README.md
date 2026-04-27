# Large Active-Space Benchmarks

This directory contains fixed-orbital DMRG convergence benchmarks beyond the
small FCI-validation systems used in the main BVOE scan.

## Anthracene CAS(14,14)

`run_anthracene_pi14.py` builds an idealized planar anthracene geometry,
selects the full pi space with AVAS (`C 2pz`), and confirms `CAS(14,14)` in
STO-3G.  The default benchmark then:

1. runs a fixed-orbital CASCI/FCI reference for the two lowest singlet roots,
2. runs SU2 DMRG-CASCI at `M = 64, 128, 256, 512, 1024`, and
3. reports `E(M)-E_FCI` in mEh for each root.

The FCI determinant dimension is about `1.18e7`, so this is a heavy but
tractable CAS(14,14) reference calculation on a large-memory node.

Example:

```bash
PYSCF_PYTHON=/path/to/python sbatch submit_anthracene_pi14.sh
```

For a quick active-space check:

```bash
python run_anthracene_pi14.py --preview-only
```
