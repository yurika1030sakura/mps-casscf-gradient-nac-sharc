# SA-DMRG-CASSCF Analytic Gradients and NACs with SHARC Integration

Analytic state-averaged DMRG-CASSCF energies, nuclear gradients, and
non-adiabatic couplings on top of `pyblock2` / `block2`, with a SHARC-PySCF
interface for direct surface-hopping dynamics. The coupled-perturbed
response solver runs in MPS space (MPS-Krylov), so no dense FCI response
vector is ever stored — the method reaches CAS active spaces well beyond
the conventional FCI ceiling while keeping fully analytic gradients and
non-adiabatic couplings.

The default code path reproduces the submitted-manuscript numerics. The
fast-path flags described below reduce per-call wall time for production
runs without changing the converged result.

## What's here

| Path | Purpose |
|---|---|
| `src/dmrg_analytic_dev/dmrg_fcisolver.py` | `MPSAsFCISolver` — block2-backed FCI solver wrapper consumed by PySCF SA-CASSCF. End-user entry point. |
| `src/dmrg_analytic_dev/cp_dmrg_response_mps_krylov.py` | MPS-Krylov coupled-perturbed response solver. Core technical contribution. |
| `src/dmrg_analytic_dev/cp_dmrg_response_mps.py` | Full-MPS reference response solver (slower; used for validation). |
| `src/dmrg_analytic_dev/cp_casscf_response.py` | Conventional CP-CASSCF baseline. |
| `src/dmrg_analytic_dev/mps_lagrange_assembly.py` | Z-vector / Lagrangian assembly for analytic gradients. |
| `src/dmrg_analytic_dev/casci_derivative_coupling.py` | Analytic derivative-coupling assembly. |
| `src/dmrg_analytic_dev/fci_sacasscf_nac_baseline.py` | FCI baseline NAC reference. |
| `sharc_interface/SHARC_PYSCF_ext.py` | SHARC-PySCF method bridge — reads SHARC templates and dispatches `method dmrg-casscf`. |
| `sharc_interface/variants/full_dmrg_ethylene_regression/` | Runnable ethylene SHARC-interface regression test. |
| `benchmarks/large_active_space/` | Fixed-orbital MPS-only response benchmarks. |
| `benchmarks/bvoe_convergence_study/` | Convergence study driver + summary figure. |
| `benchmarks/response_timing/` | Per-component wall-time benchmarks. |
| `docs/manuscript_skeleton.tex`, `docs/references.bib` | Submitted-manuscript skeleton + bibliography. |
| `docs/FCI_REFERENCE_PROTOCOL.md`, `docs/ROOT_MATCHING_PROTOCOL.md`, `docs/PRODUCTION_ROOT_TRACKING.md` | Numerical protocols used by the benchmarks. |
| `docs/ARCHITECTURE.md` | One-page map of the modules and how they call each other. |

Every core module has a sibling `test_*.py` + `test_*.json` golden-value
regression. Run any of them stand-alone to check an install.

## Install

```bash
git clone git@github.com:yurika1030sakura/mps-casscf-gradient-nac-sharc.git
cd mps-casscf-gradient-nac-sharc
# Conda (recommended; block2 wheels are non-trivial via pip)
conda install -c conda-forge "pyscf>=2.4" "block2>=0.5" numpy scipy matplotlib
# pip alternative
pip install pyscf numpy scipy matplotlib block2
```

`block2` ships `pyblock2`. SHARC itself is external and not bundled here.

Tested with PySCF 2.4–2.6, block2 0.5.x, Python 3.10–3.12.

## Quick start

Smoke-compile everything:

```bash
find src sharc_interface benchmarks -name '*.py' -print0 | xargs -0 python -m py_compile
```

Run the focused MPS-Krylov response test (≈30 s, no SHARC required):

```bash
cd src/dmrg_analytic_dev
PYTHONPATH=../..:../../sharc_interface python test_mps_krylov_response.py
```

Expected at the end of output:

```
[mps-krylov] max |response - reference| = O(1e-8)
PASSED
```

Run the SHARC-interface regression on ethylene (requires SHARC available
on `$PATH`):

```bash
cd sharc_interface/variants/full_dmrg_ethylene_regression
PYSCF_PYTHON=python ./submit_regression.sh
```

Reproduce the BVOE convergence figure:

```bash
cd benchmarks/bvoe_convergence_study
python plot_bvoe_phase2.py
```

## Fast-path flags (production mode)

All flags default off. With every flag off the code path is the submitted
version. Enabling them changes performance, not the converged energy.

| `MPSAsFCISolver` kwarg | SHARC template key | What it does |
|---|---|---|
| `stack_mem_mb=8000` | `dmrg-stack-mem-mb 8000` | Override the 200 MB block2 stack allocation. Required on large-memory nodes. |
| `mps_native_rdms=True` | `dmrg-mps-native-rdms 1` | Compute `make_rdm12` / `trans_rdm12` / `contract_2e` via block2 NPDM kernels instead of the FCI-projected path. |
| `warm_start=True` | `dmrg-warm-start 1` | Cache the converged MPS from the previous macro iteration and reuse it as the initial guess. Cuts later-iter DMRG sweeps from ~30 to ~4–8. |
| `first_iter_warmup=True` | `dmrg-first-iter-warmup 1` | HF-occupation-biased initial MPS + bond-dim ramp 50→100→M with 12 sweeps on macro iter 1 (whose orbitals are still poor). |
| `skip_kernel_fci_conversion=True` | `dmrg-skip-fci-conversion 1` | Skip the O(n_dets) MPS→FCI ndarray conversion at the end of every kernel call — the dominant cost at CAS(14,12) and above. Requires `mps_native_rdms=1`. |
| `dmrg_symm_su2=True` | `dmrg-symm-su2 1` | Use `SymmetryTypes.SU2` (spin-adapted) instead of `SZ`. Targets the requested 2S sector by construction so the energy-sort root selection cannot accidentally pick a contaminating sector. Recommended for any spin-pure ground state. |
| `timing_log=True` | `dmrg-timing-log 1` | Per-section wall-time print at every kernel / RDM call. Diagnostic only. |

Production preset (CAS(14,12) and above, both spin sectors safe):

```python
mc.fcisolver = MPSAsFCISolver(
    mol,
    bond_dim=512, n_sweeps=30, n_threads=4,
    mps_native_rdms=True,
    warm_start=True,
    first_iter_warmup=True,
    skip_kernel_fci_conversion=True,
    dmrg_symm_su2=True,
    stack_mem_mb=8000,
)
```

SHARC template equivalent:

```
method                       dmrg-casscf
dmrg-mps-native-rdms         1
dmrg-warm-start              1
dmrg-first-iter-warmup       1
dmrg-skip-fci-conversion     1
dmrg-symm-su2                1
dmrg-stack-mem-mb            8000
```

## Validation against PySCF FCI

Spin-pure small-system checks (every fast-path flag on, FCI reference is
PySCF `fcisolver`):

| System | Active space | Test | max |dE| vs FCI |
|---|---|---|---|
| H4 chain, sto-3g | (4,4), SA(2 singlets) | `src/dmrg_analytic_dev/test_v9_skip_fci_vs_fci.py` | 2.5e-13 Ha |
| H4 chain, sto-3g | (4,4), triplet GS via SU2 | `src/dmrg_analytic_dev/test_v10_su2_triplet.py` | 8.9e-16 Ha |

Matching transition 1-RDMs and 2-RDMs agree at the 1e-7 level.

Large-active-space convergence: planar anthracene CAS(14,14)/STO-3G,
SA(2 singlets), fixed-orbital DMRG-CASCI vs the manuscript's cached FCI
reference E₀ = −529.7030437 Ha, E₁ = −529.5556316 Ha. Produced by
`benchmarks/large_active_space/run_anthracene_pi14_fastpath_mscan.py`
on 4 CPU cores; raw output in `results_anthracene_pi14_fastpath_mscan.txt`.

| M | E₀ (Ha) | E₁ (Ha) | \|dE₀\| (mHa) | \|dE₁\| (mHa) | wall (s) |
|---|---|---|---|---|---|
|   64 | −529.6581689 | −529.4823392 | 44.87 | 73.29 |   8.4 |
|  128 | −529.6832931 | −529.5167555 | 19.75 | 38.88 |  15.5 |
|  256 | −529.6951117 | −529.5377081 |  7.93 | 17.92 |  31.0 |
|  512 | −529.7006909 | −529.5508136 |  2.35 |  4.82 | 136.0 |
| 1024 | −529.7025280 | −529.5543807 |  0.52 |  1.25 | 397.9 |

Monotonic convergence in both states, sub-mHa accuracy on the ground
state by M = 1024, ≈1.3 mHa on the first excited state — well inside
chemical accuracy on this active space.

## Citation

```bibtex
@article{mps_casscf_response_sharc_2026,
  title  = {Analytic State-Averaged DMRG-CASSCF Gradients and
            Non-Adiabatic Couplings with MPS-Krylov Response
            for Surface-Hopping Dynamics},
  author = {<authors>},
  year   = {2026},
  note   = {Manuscript submitted to J. Chem. Theory Comput.}
}
```

The published DOI / journal record will be added to this block once the
manuscript is accepted.

## Scope

Public methods bundle only. Application systems, trajectory outputs, and
local cluster files belong in private project repositories.

## License

MIT. See [LICENSE](LICENSE).
