# Supplementary regression tests (not manuscript-cited)

The tests in this directory pin the production fast-path knobs (the
seven opt-in flags listed under the **Production fast path** section of
the top-level README) against PySCF FCI references on a small system
that fits in memory.

They are **not** part of the manuscript's benchmark suite. The
manuscript-cited validation lives under

- `benchmarks/bvoe_convergence_study/` (nine small CAS systems, paper
  Figures 2 & 3),
- `benchmarks/large_active_space/` (anthracene CAS(14,14) strict
  MPS-Krylov response, paper Table tab:mps-only-response).

The tests below exist so that anyone modifying the fast-path code can
keep the knobs honest against a known-good reference.

## What each test checks

| Test | Knob exercised | Reference |
|---|---|---|
| `test_skip_fci_conversion_vs_fci.py` | `skip_kernel_fci_conversion=True` + `mps_native_rdms=True` placeholder-`ci` path | PySCF FCI on H4 chain CAS(4,4) singlet pair |
| `test_su2_mode_triplet.py` | `dmrg_symm_su2=True` (spin-adapted SU2 sector) on a triplet ground state | PySCF FCI on H4 chain CAS(4,4) triplet GS |

Each test prints `PASS` / `FAIL` per quantity (state energies, 1-RDMs,
2-RDMs, transition RDMs) at the bottom of stdout.

## Run

```bash
cd src/dmrg_analytic_dev/supplementary
PYTHONPATH=.. python test_skip_fci_conversion_vs_fci.py
PYTHONPATH=.. python test_su2_mode_triplet.py
```
