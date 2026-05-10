# Large Active-Space Benchmarks

This directory contains large-active-space DMRG benchmarks beyond the small
FCI-validation systems used in the main BVOE scan.

## Anthracene CAS(14,14)

`run_anthracene_pi14.py` builds an idealized planar anthracene geometry,
selects the full pi space with AVAS (`C 2pz`), and confirms `CAS(14,14)` in
STO-3G.  The energy-only benchmark then:

1. runs a fixed-orbital PySCF `direct_spin0` singlet CASCI/FCI reference and
   records `S^2` plus residual diagnostics,
2. runs SU2 DMRG-CASCI at `M = 64, 128, 256, 512, 1024, 2048`, and
3. reports `E(M)-E_FCI` in mEh for the target roots.

The FCI determinant dimension is about `1.18e7`, so this is a heavy but
tractable CAS(14,14) reference calculation on a large-memory node.

`run_anthracene_pi14_gradnac.py` is the methods-relevant derivative
benchmark. It:

1. optimizes SA(2)-DMRG-CASSCF orbitals with the MPS-native RDM solver,
2. reruns fixed-orbital SU2 DMRG at selected bond dimensions,
3. evaluates analytic state gradients and the S0/S1 derivative coupling either
   through the historical projected-CI validation path or the MPS-Krylov
   response backend, and
4. reports response residuals, root-selection diagnostics, timings, and
   post hoc errors when an FCI reference cache is supplied.

The MPS-Krylov path does not require FCI at runtime.  For the manuscript
benchmark, a fixed-orbital spin-adapted singlet FCI cache is used only for post
hoc error scoring.

Example:

```bash
PYSCF_PYTHON=/path/to/python sbatch submit_anthracene_pi14.sh
PYSCF_PYTHON=/path/to/python sbatch submit_anthracene_pi14_gradnac.sh
```

The strict response scan used for the large-active-space derivative figure is:

```bash
PYSCF_PYTHON=/path/to/python \
ANTH_REFERENCE_NPZ=/path/to/anthracene_pi14_reference_cache.npz \
sbatch submit_anthracene_pi14_strict_m_scan.sh
```

The strict scan submits separate gradient and NAC right-hand-side jobs for each
bond dimension and combines the partial JSON files with
`combine_partial_derivatives.py`.

For a quick active-space check:

```bash
python run_anthracene_pi14.py --preview-only
python run_anthracene_pi14_gradnac.py --preview-only
```
