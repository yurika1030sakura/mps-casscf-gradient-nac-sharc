# MPS-Native SA-DMRG-CASSCF Gradients and Nonadiabatic Couplings

This repository contains a PySCF/pyblock2 implementation of analytic state-averaged
DMRG-CASSCF nuclear gradients and nonadiabatic couplings. The electronic roots and
the active-space response vectors can remain matrix product states (MPSs), avoiding
a dense determinant-space response vector in the MPS-native path. A SHARC-compatible
interface provides the electronic quantities required by repeated-call
nonadiabatic-dynamics drivers.

## Scientific scope

The implementation is evaluated with a validation hierarchy rather than a single
benchmark:

1. **Exact controls:** fixed-orbital comparisons with analytic FCI derivatives where
   dense active-space vectors are available.
2. **MPS-native bridge:** anthracene CAS(14,14), evaluated without runtime FCI root
   selection and scored against an FCI reference only after the MPS calculation.
3. **Beyond-FCI checks:** selected gradient components are compared with finite
   differences of the same DMRG energy surface, and selected coupling components are
   compared with determinant-free cross-geometry MPS-overlap finite differences.
4. **Large-space diagnostics:** calculations through CAS(24,24) report bond-dimension
   stability, response residuals, continuity checks, timing, and memory without
   treating a residual as a universal property-error bound.
5. **Repeated-call integration:** SHARC interface regressions and a 715-frame
   ethylene trajectory test repeated derivative evaluation and energy consistency
   for one initial condition.

The excited-state derivative-coupling benchmarks reported in the manuscript use
singlet state averaging. The analytic gradient is additionally validated in two
non-singlet spin sectors against exact FCI (triplet CH2 CAS(6,6) to
`1.7e-5 Eh/Bohr`; doublet CH3 CAS(7,7) to `1.7e-4 Eh/Bohr`, maximum Cartesian
component), and an SU(2) triplet H4 energy/RDM regression is included. Non-singlet
excited-state derivative couplings and spin-orbit coupling are not validated here.

## Main numerical results

- **Anthracene CAS(14,14), strict MPS-native response, M = 512:** maximum state-wise
  gradient error `1.57e-3 mEh/Bohr`; phase-aligned derivative-coupling error
  `9.11e-6 a0^-1` against the post hoc fixed-orbital FCI reference.
- **C16 CAS(16,16):** three clean directional-gradient comparisons agree with
  same-surface DMRG finite differences to `1.0e-3 Eh/Bohr` or better. A three-state
  coupling-component comparison differs by `5.1e-3 a0^-1`.
- **Aza-C20 CAS(20,20):** three directional-gradient discrepancies are `4.4e-4`,
  `1.7e-3`, and `3.5e-3 Eh/Bohr`; one coupling-component discrepancy is
  `4.5e-3 a0^-1`.
- **Ethylene repeated-call run:** 715 sampled frames over 0-178.5 fs, `M = 200`,
  response tolerance `5e-4`, and `dt = 0.25 fs`; total-energy drift `-9.6 meV`,
  peak-to-peak fluctuation `66 meV`, and RMS fluctuation `13 meV` over the reported
  continuous segment.

These values use the error definitions and acceptance gates stated in the manuscript
and Supporting Information. Do not mix them with older values based on a different
norm, orbital set, or response setting.

## Repository map

- `src/dmrg_analytic_dev/` — DMRG solver wrapper, MPS response solvers, Lagrangian
  assembly, and regression tests.
- `benchmarks/bvoe_convergence_study/` — exact-control convergence data and plots.
- `benchmarks/large_active_space/` — anthracene and beyond-FCI benchmark drivers,
  accepted/excluded finite-difference ledgers, non-singlet gradient regressions,
  timing records, and machine-readable summaries under `data/`.
- `benchmarks/response_timing/` — component-resolved timing and memory records.
- `benchmarks/root_tracking/` — the LiF avoided-crossing scan records and figure
  script.
- `sharc_interface/` — SHARC-PySCF bridge and interface regressions.
- `sharc_interface/variants/ethylene_photochem_tight/` — the archived 178 fs
  ethylene run: SHARC input, initial geometry/velocities, `PYSCF.template`,
  per-step energies (`output.dat`, `output.lis`), sampled geometries
  (`output.xyz`), and a representative `QM.in`/`QM.out` pair. The per-step energy
  track and self-describing metadata are also mirrored at
  `benchmarks/large_active_space/data/eth_trackA_trajectory.dat` and
  `eth_trackA_trajectory_meta.json`.
- `methods_manuscript/figures/`, `methods_manuscript/figure_scripts/` — manuscript
  figures and the scripts that generate them.
- `docs/` — numerical protocols (FCI reference, root matching, production root
  tracking) and the architecture notes.

## Installation

The manuscript calculations used Python 3.11.14, PySCF 2.12.1, block2 0.5.3 through
pyblock2, and SHARC 3.0. Use an isolated environment and record exact package builds
for reproduced benchmarks.

```bash
git clone https://github.com/yurika1030sakura/mps-casscf-gradient-nac-sharc.git
cd mps-casscf-gradient-nac-sharc
conda create -n mps-deriv python=3.11
conda activate mps-deriv
conda install -c conda-forge pyscf=2.12.1 block2=0.5.3 numpy scipy matplotlib
```

SHARC is external software and is not bundled with this repository.

## Quickstart — running a new system

The single entry point is `compute_certified_derivatives` in
`src/dmrg_analytic_dev/certified_engine.py`. A system is specified by data-level
inputs rather than source edits, within the validated scope stated above; the
engine selects the dense or MPS-native response automatically at `det = 5e7`.

```python
import sys, numpy as np
sys.path.insert(0, "src/dmrg_analytic_dev")
sys.path.insert(0, "sharc_interface")
import certified_engine as ce

ANG = 1.8897261246257702  # Angstrom -> Bohr; coords are passed in Bohr

# Example: LiF, S0/S1 gradient + NAC. Change only these data lines.
atoms  = ["Li", "F"]
coords = np.array([(0, 0, 0.0), (0, 0, 1.564)]) * ANG
out = ce.compute_certified_derivatives(
    atoms, coords,
    basis="6-31G", charge=0, spin=0,
    ncas=6, nelecas=6,          # or: ao_targets=["F 2p", "Li 2s"]
    nroots=2, weights=[0.5, 0.5],
    gradient_states=(0, 1),     # which state gradients
    nac_pairs=[(0, 1)],         # which NAC vectors
    max_bond_dim=800,           # raise for larger CAS
    threads=8, stack_mem_mb=8000,
)

print(out["overall_health"])                    # PASS / WARN / FAIL
print(out["build"]["det_dim"], out["build"]["beyond_fci"])
for s, g in out["gradients"].items():
    print("grad", s, g["health"], g["certificate"]["true_residual_relative"])
for k, d in out["nacs"].items():
    print("nac", k, d["certificate"]["true_residual_relative"])
```

What happens automatically:

- **Active space** comes from `ncas`/`nelecas`, from `ao_targets` (AO-population
  selection, e.g. `["C 2pz"]` for a pi space), or defaults to the HOMO-LUMO window.
- **Backend selection** compares `det` to `FCI_FREE_THRESHOLD (5e7)`. Below it, an
  exact-FCI seed accelerates the SA-CASSCF and provides an energy-match convergence
  check; at or above it, the FCI-free MPS path is forced and the dense-bridge
  sentinel is armed.
- **Convergence** follows the progressive bond-dimension schedule; the escalation
  ladder (extra macros, AH level shift, Newton) fires only if the plain build does
  not converge.
- **Every derivative** is accepted only after its true-residual diagnostic confirms
  the response solve, and the point is scored by `assess_point`. Large-active-space
  results should be read with the stated active space, orbitals, bond dimension,
  tolerances, and validation category.

To exercise the same data-driven call across spin sectors and sizes:

```bash
cd benchmarks/engine_general
python run_engine_stress_test.py --out data/engine_stress.json
```

### SHARC (repeated-call) use

All physics settings live in `PYSCF.template`:

```
method                   dmrg-casscf
basis                    6-31G
dmrg-ncas                6
dmrg-nelecas             6
roots                    2
dmrg-response-mode       mps-krylov     # or projected-ci for small CAS
dmrg-symm-su2            1
dmrg-stack-mem-mb        8000
```

The archived 178 fs run's exact template is
`sharc_interface/variants/ethylene_photochem_tight/PYSCF.template`
(`dmrg-maxm 200`, `dmrg-response-tol 5e-4`, `dt = 0.25 fs`). See
`docs/PRODUCTION_ROOT_TRACKING.md` for state following without FCI.

## Reproduce the paper

Figures and tables regenerate from cached JSONs without a cluster. See
[`benchmarks/README.md`](benchmarks/README.md) for the figure-to-driver map and
[`CODEMAP.md`](CODEMAP.md) for how a calculation flows through the package.

```bash
# Exact-control fixed-orbital convergence (9 systems x 3 bases)
cd benchmarks/bvoe_convergence_study
python plot_bvoe_phase2.py          # figure from cached summary

# Small/medium-CAS gradient+NAC validation vs fixed-orbital FCI
python run_fci_free_endpoint_check.py

# Anthracene CAS(14,14) strict MPS-Krylov response table
cd ../large_active_space
python report_anthracene_strict_response.py

# Beyond-FCI analytic gradient vs same-surface finite difference
python beyond_fci_validation_clean.py --ncarbon 20

# LiF avoided-crossing root-tracking figure
cd ../root_tracking
python make_root_tracking_figure.py

# 178 fs trajectory figure + self-describing metadata
cd ../large_active_space
python make_trajectory_figure.py
```

Generated summaries record method, basis, active space, state average, spin sector,
orbital preparation, bond dimensions, sweep schedules, tolerances, error definitions
and units, continuity metrics with acceptance status, and (for timing rows) hardware
and thread counts.

## Interpretation and limitations

- A low response residual indicates response-vector convergence under the stated
  representation; it is not by itself a rigorous bound on a contracted gradient or
  coupling.
- The LiF calculation is a one-dimensional avoided-crossing/root-subspace tracking
  stress test, not a multidimensional conical-intersection benchmark.
- The 178 fs ethylene run is a repeated-call and energy-consistency test for one
  initial condition. It is not an ensemble photodynamics calculation and no
  mechanistic conclusion is inferred.
- Timings from different active spaces, orbitals, bond dimensions, hardware, or
  workloads must not be divided to claim a controlled speedup.

## Data release

- Release tag: `v1.0`
- Archived data DOI: pending Zenodo deposit of the tagged release.

## Citation

```bibtex
@article{Li2026MPSNativeDMRGCASSCF,
  author = {Li, Yuli},
  title = {MPS-Native Analytic State-Averaged DMRG-CASSCF Gradients and Nonadiabatic Couplings beyond the Dense-FCI Limit},
  journal = {Journal of Chemical Theory and Computation},
  year = {2026},
  note = {Manuscript under review}
}
```

## License

MIT License. See `LICENSE`.
