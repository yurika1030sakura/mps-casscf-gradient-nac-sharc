# Architecture map

One-page guide to the modules in `src/dmrg_analytic_dev/` and how they
fit together. Read this before opening a PR or extending the response
solver.

## User-facing entry points

| Module | Role |
|---|---|
| `dmrg_fcisolver.py` (`MPSAsFCISolver`) | block2-backed FCI solver wrapper consumed by `pyscf.mcscf.CASSCF` / `CASCI`. This is the object a user constructs. |
| `sharc_interface/SHARC_PYSCF_ext.py` | Parses SHARC templates with `method dmrg-casscf` and dispatches to `MPSAsFCISolver` with the right kwargs. |
| `sharc_interface/dmrg_sharc_bridge.py` | Glue between SHARC's per-step request format and the response solver. |
| `sharc_interface/analytic_cp_sharc.py` | Driver that turns a SA-CASSCF result into the gradient / NAC tensors SHARC expects. |

## Core response path

The analytic-gradient / NAC pipeline runs the coupled-perturbed (CP)
equations in MPS space rather than FCI space. The main solver is the
MPS-Krylov variant.

| Module | Role |
|---|---|
| `cp_dmrg_response_mps_krylov.py` | MPS-Krylov CP solver — the main contribution. Iteratively builds the response without ever forming a dense FCI vector. |
| `cp_dmrg_response_mps.py` | Full-MPS reference solver (slower; used to validate the Krylov solver). |
| `cp_casscf_response.py` | Conventional FCI-CP-CASSCF baseline used as a reference at small CAS sizes. |
| `single_site_sigma.py` | Builds single-site σ tensors used inside the CP equations. |
| `site_replacement_density.py` | Site-replacement densities consumed by the response and by NPDM kernels. |

## Gradient and derivative-coupling assembly

| Module | Role |
|---|---|
| `mps_lagrange_assembly.py` | Z-vector / Lagrangian assembly that turns response amplitudes into the analytic nuclear gradient. |
| `casci_derivative_coupling.py` | Analytic derivative-coupling assembly between two CASSCF states. |
| `fci_sacasscf_nac_baseline.py` | FCI baseline for NAC validation. |
| `overlap_fci_reference.py` | FCI overlap utilities used by the root-tracking protocol. |

## Tests

Every core module has a sibling `test_<module>.py` plus a `test_<module>.json`
golden-value file. The tests are the canonical specification of input /
output shapes.

To run a single test:

```bash
cd src/dmrg_analytic_dev
PYTHONPATH=../..:../../sharc_interface python test_<module>.py
```

To run them all:

```bash
cd src/dmrg_analytic_dev
for t in test_*.py; do
  PYTHONPATH=../..:../../sharc_interface python "$t" || break
done
```

## Numerical protocols

`docs/FCI_REFERENCE_PROTOCOL.md`, `docs/ROOT_MATCHING_PROTOCOL.md`, and
`docs/PRODUCTION_ROOT_TRACKING.md` describe the conventions used to
generate the cached benchmark references and to track states across
geometries. Read these before adding a new benchmark molecule.
