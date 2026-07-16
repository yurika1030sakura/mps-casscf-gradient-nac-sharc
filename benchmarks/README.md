# Benchmarks — figure/table → driver map

Every paper figure and table regenerates from a driver here. The heavy runs
write machine-readable JSON to each study's `data/`; the plotting/report steps
read those cached JSONs and need no cluster. Drivers take only data-level
arguments — no source edits.

Directory | What it produces
--- | ---
`engine_general/` | The "no source edit" exemplar: one call over 7 singlet/doublet/triplet systems at a uniform bond dimension.
`bvoe_convergence_study/` | Fixed-orbital bond-dimension convergence across 9 systems × 3 basis sets; the FCI-free endpoint check.
`large_active_space/` | The beyond-FCI headline: polyene/acene/aza gradients and NACs from CAS(14,14) to CAS(24,24), the anthracene strict-response m-ladder, and the ethylene SHARC trajectory.
`root_tracking/` | The LiF avoided-crossing scan and its subspace-continuity (σ_min, assignment-margin) figure.
`response_timing/` | Response cost: global MPS-Krylov vs sweep-Schur, recycling, repeated-call.
`fd_validation/` | Analytic-vs-finite-difference regression with PASS/WARN/FAIL.

## Reproduce the main results

```bash
# 1. "No source edit" across spin sectors and sizes
cd engine_general && python run_engine_stress_test.py --out data/engine_stress.json

# 2. Bond-dimension convergence figure (cached data -> figure, no cluster)
cd ../bvoe_convergence_study && python plot_bvoe_phase2.py
python run_fci_free_endpoint_check.py           # small/medium-CAS grad+NAC vs fixed-orbital FCI

# 3. Anthracene CAS(14,14) strict MPS-Krylov response (FCI-scored bridge)
cd ../large_active_space && python report_anthracene_strict_response.py

# 4. Beyond-FCI analytic gradient / NAC = finite difference (where a clean FD exists)
python run_beyond_fci_nac.py --ncarbon 16 --nroots 3          # C16 SA(3) NAC, clean
python run_beyond_fci_nac.py --ncarbon 20 --aza --nroots 5    # aza-C20, both grad + NAC clean
#   small-gap / high-amplitude coupling: use the conjugate-residual solver
python acene_nac.py --nrings 5 --nroots 5 --nac-solver cr --nac-m-compress 600   # pentacene, clean

# 5. Bond-dimension invariance at the largest active space (cert-only regime)
python polyene_pinned_fd.py --ncarbon 24 --aza --directions central_cc --bond-dim 1600

# 6. Root / subspace tracking figure through the LiF avoided crossing
cd ../root_tracking && python make_root_tracking_figure.py

# 7. Ethylene S1 trajectory energy-conservation figure (from cached per-step data)
cd ../large_active_space && python make_trajectory_figure.py
```

## Notes

- `submit_*.sh` scripts are the exact SLURM job files used on the authors'
  cluster; treat them as run records and adapt the account, partition, scratch
  paths, and Python environment to your machine. The physics comes from the
  `.py` drivers and `PYSCF.template`, not from these.
- The cross-geometry finite-difference NAC additionally requires, before it is
  accepted, that the R / +h / −h state-averaged reference energies agree to
  ≲10⁻⁵ Eh (a displaced multireference build can be gauge-continuous yet on a
  different SA-CASSCF solution; the orbital overlap cannot see this, the energy
  can). The drivers print this consistency check.
