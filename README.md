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

## Numerical checks

### Manuscript benchmark, small-to-medium CAS — fixed-orbital response at M = 200

The methods manuscript validates the analytic SA-DMRG-CASSCF gradient
and derivative-coupling pipeline against fixed-orbital FCI references on
nine public benchmark systems. The cached single-point JSONs in
`benchmarks/bvoe_convergence_study/data_phase2/` reproduce the
manuscript's largest-M table directly:

| System | Active space | Δg (mE_h / Bohr) | Δd (a.u.) |
|---|---|---|---|
| H₄ / STO-3G          | (4,4) | 6.93e−11 | 1.47e−07 |
| H₂O / 6-31G          | (6,6) | 5.68e−06 | 4.29e−07 |
| N₂ / 6-31G           | (6,6) | 3.89e−06 | 7.10e−07 |
| C₂ / 6-31G           | (8,8) | 2.66e−04 | 4.08e−05 |
| LiF / 6-31G          | (4,4) | 1.32e−06 | 4.21e−06 |
| Ethylene / 6-31G     | (2,2) | 3.97e−06 | 7.06e−07 |
| Butadiene / 6-31G    | (4,4) | 1.79e−04 | 6.53e−05 |
| Formaldehyde / 6-31G | (4,4) | 1.73e−06 | 9.76e−07 |
| Benzene / 6-31G      | (6,6) | 7.53e−06 | 4.08e−07 |

`Δg` is the max absolute analytic-gradient error of state 0 vs the
fixed-orbital FCI reference; `Δd` is the analytic derivative-coupling
vector error between roots (0, 1). H₄ is an exact-rank check (M ≥ full
bipartite rank ⇒ DMRG = FCI to machine precision). Across the rest the
error sits at 10⁻⁶ to 10⁻⁴ E_h/Bohr — well below the 0.1 mE_h/Bohr
reference line used in the convergence-plot figure.

### Manuscript benchmark, large CAS — anthracene CAS(14,14) strict MPS-Krylov response

The same response pipeline is exercised at large active space on planar
anthracene with the AVAS-selected π active space: CAS(14,14) / STO-3G,
FCI dimension 11,778,624 determinants. The runtime DMRG calculation
selects spin-adapted MPS roots without using FCI CI vectors; the cached
FCI reference is loaded only after the response solve, for post hoc
scoring. The production rows use Boys-localized active orbitals ordered
by the principal molecular axis (the orbital ordering matters for
entanglement on this aromatic π system), response tolerance 10⁻⁸, and
120 MPS-Krylov iterations per right-hand side.

Reproduce from the cached strict-response JSONs:

```bash
cd benchmarks/large_active_space
python report_anthracene_strict_response.py
```

| M | \|dE\| (Ha) | Δg (E_h / Bohr) | Δd (a.u.) |
|---|---|---|---|
|  64 | 1.15e−03 | 1.11e−03 | 9.52e−03 |
| 128 | 8.83e−05 | 1.13e−04 | 1.13e−03 |
| 256 | 3.20e−06 | 1.07e−05 | 7.39e−05 |
| 512 | **6.41e−08** | **4.85e−07** | **3.81e−06** |

All three error channels are monotonic over the M ∈ {64, 128, 256, 512}
scan. At M = 512 the strict-response gradient error sits at
5×10⁻⁷ E_h/Bohr and the derivative-coupling error at 4×10⁻⁶ a.u. — the
same accuracy regime as the small-CAS benchmark table above, on a FCI
space three orders of magnitude larger. This is the headline
large-active-space response result of the manuscript.

### Spin-pure DMRG-vs-FCI equivalence on H₄

When the active-space FCI vector fits in memory, DMRG at sufficient M is
numerically equivalent to FCI. The H₄ CAS(4,4) tests pin every fast-path
flag against PySCF's `fcisolver` reference:

| System | Active space | Test | max \|dE\| vs FCI |
|---|---|---|---|
| H₄ chain, sto-3g | (4,4), SA(2 singlets) | `src/dmrg_analytic_dev/test_v9_skip_fci_vs_fci.py` | 2.5e−13 Ha |
| H₄ chain, sto-3g | (4,4), triplet GS via SU2 | `src/dmrg_analytic_dev/test_v10_su2_triplet.py` | 8.9e−16 Ha |

Matching 1- / 2-RDMs and transition RDMs agree at the 1e−7 level. These
tests verify that the placeholder-`ci` machinery, the SU2 spin sector,
the NPDM bypass, and the warm-start path produce bitwise the same
physics as the legacy FCI-projection path.

### Fast-path knobs at CAS(14,14): wall-time demonstration

The opt-in flags listed earlier let the same active space be reached
without Boys localization or the MPS-Krylov response solver, by running
DMRG-CASCI directly on AVAS-canonical orbitals. This path is faster per
M but converges more slowly in M because AVAS-canonical orbitals carry
more entanglement than Boys-localized orbitals — useful as a
fast-path-knob smoke test, not a substitute for the strict-response
benchmark above. Produced by
`benchmarks/large_active_space/run_anthracene_pi14_fastpath_mscan.py`
on 4–8 CPU cores; raw output in `results_anthracene_pi14_fastpath_mscan.txt`.

| M | \|dE₀\| (Ha) | \|dE₁\| (Ha) | wall (s) |
|---|---|---|---|
|   64 | 4.49e−02 | 7.33e−02 |    8.4 |
|  128 | 1.97e−02 | 3.89e−02 |   15.5 |
|  256 | 7.93e−03 | 1.79e−02 |   31.0 |
|  512 | 2.35e−03 | 4.82e−03 |  136.0 |
| 1024 | 5.16e−04 | 1.25e−03 |  397.9 |
| 2048 | 3.95e−04 | 8.49e−04 |  391.5 |
| 4096 | 1.49e−04 | 3.69e−04 |  677.3 |

The state-energy error halves per doubling of M on AVAS orbitals — the
1000-fold reduction that Boys localization delivers in the strict-response
table is the entanglement-structure win, not a code win.

## Citation

```bibtex
@article{mps_casscf_response_sharc_2026,
  title  = {Analytic State-Averaged DMRG-CASSCF Gradients and
            Non-Adiabatic Couplings with MPS-Krylov Response
            for Surface-Hopping Dynamics},
  author = {<authors>},
  year   = {2026},
  note   = {Manuscript in review}
}
```

The published DOI / journal record will be added once the manuscript is
accepted.

## Scope

Public methods bundle only. Application systems, trajectory outputs, and
local cluster files belong in private project repositories.

## License

MIT. See [LICENSE](LICENSE).
