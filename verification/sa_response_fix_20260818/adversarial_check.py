"""Independent adversarial verification of the response-exact-projection fix.

Design (not in any committed test):
  System: H4 rectangle / 3-21G / CAS(4,4) / SA-3 UNEQUAL weights (0.5,0.3,0.2)
  -- exercises per-state 1/(2*w_i) scaling with three DIFFERENT scale factors
  (1.0, 5/3, 5/2), the D5 trap (4 sites), and the D2 inner-solve floor, for
  ALL THREE states including state 2 (never solved in the committed test).

  A. Default solver must certify <= 1e-8 for states 0,1,2 AND match the
     kets-based dense projected reference (|z_mps - z_dense| < 1e-6), AND the
     certificate value must agree with the independently computed dense
     relative residual of the MPS solution.
  B. Mutation sensitivity (state 1, where 2w=0.6 != 1): disabling each fix
     individually must push the certificate visibly above 1e-8 -- proving each
     fix is load-bearing and the certificate catches a partial regression.
       M1: hcc_weight_scaling=False        (F1 off)
       M2: obj._mps_combine_zero_safe=False (F2 off)
       M3: ci_inner_solver="multiply"       (F3 off, block2 default thrds)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WT = Path("/n/home04/yulili/daisuan/verify_new_wt")
for p in (WT / "src/dmrg_analytic_dev", WT / "sharc_interface"):
    sys.path.insert(0, str(p))

import fd_validation as fdv  # noqa: E402
import sweep_coupled_response as scr  # noqa: E402
import test_sa_response_reproducer as tsr  # noqa: E402
from analytic_cp_sharc import _make_mps_krylov_response  # noqa: E402
from certified_response import certify_response  # noqa: E402
from cp_dmrg_response_mps_krylov import MPSKrylovVector  # noqa: E402

ANG = 1.8897261246257702
T0 = time.perf_counter()


def log(m):
    print(f"[{time.perf_counter() - T0:8.1f}s] {m}", flush=True)


def build_h4_unequal():
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=200, n_sweeps=30, sweep_tol=1.0e-10, n_threads=1)
    sr = os.environ.get("DMRG_TEST_SCRATCH")
    if sr:
        Path(sr).mkdir(parents=True, exist_ok=True)
        cfg["scratch_root"] = sr
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.00],
        [0.0, 1.40, 0.0],
        [0.0, 1.40, 1.00],
    ]) * ANG
    _mol, _mf, mc, _s = fdv.build_sa_dmrg_casscf(
        ["H", "H", "H", "H"], coords, basis="3-21G", charge=0, spin=0,
        ncas=4, nelecas=4, nroots=3, weights=[0.5, 0.3, 0.2],
        solver_cfg=cfg,
    )
    assert mc.converged, "SA-CASSCF did not converge"
    log(f"built H4 SA-3 (0.5,0.3,0.2); E = {np.asarray(mc.e_states)}")
    return mc


def solve(mc, state, label, *, zero_safe=True, **kw):
    obj = _make_mps_krylov_response(mc)
    obj._mps_combine_zero_safe = zero_safe
    t0 = time.perf_counter()
    kappa, zc, info, meta = scr.solve_state_sweep_schur(
        obj, state, orb_tol=1.0e-9, ci_tol=1.0e-11, **kw)
    wall = time.perf_counter() - t0
    z = MPSKrylovVector(obj, kappa, zc, label=f"ADV-{label}-{state}")
    cert = certify_response(obj, z, state=state, tol=1.0e-8,
                            solver=f"adv_{label}", wall_s=wall,
                            extra={"solver_meta": meta})
    return obj, kappa, zc, cert


results = {}
fails = []

mc = build_h4_unequal()

# --- A: default solver, all three states, vs dense --------------------------
for s in (0, 1, 2):
    obj, k, zc, cert = solve(mc, s, "default")
    rel = cert.true_residual_relative
    dz, nz, drel = tsr.compare_mps_to_dense(mc, obj, s, k, zc)
    ok = (rel <= 1.0e-8) and (dz < 1.0e-6) and (drel < 1.0e-6)
    results[f"default_state{s}"] = dict(
        cert=rel, z_err_vs_dense=dz, z_dense_norm=nz, dense_rel_resid=drel,
        weight=float(np.asarray(mc.weights).ravel()[s]), ok=ok)
    log(f"A default s{s} (w={np.asarray(mc.weights).ravel()[s]}): "
        f"cert={rel:.3e} |z-z_dense|={dz:.3e} dense_rel={drel:.3e} "
        f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"default_state{s}")

# --- B: mutation sensitivity on state 1 (2w = 0.6) ---------------------------
mutations = {
    "M1_no_weight_scaling": dict(kw=dict(hcc_weight_scaling=False),
                                 zero_safe=True),
    "M2_no_zero_safe": dict(kw=dict(), zero_safe=False),
    "M3_multiply_only": dict(kw=dict(ci_inner_solver="multiply"),
                             zero_safe=True),
}
for name, m in mutations.items():
    obj, k, zc, cert = solve(mc, 1, name, zero_safe=m["zero_safe"], **m["kw"])
    rel = cert.true_residual_relative
    caught = rel > 1.0e-8   # the certificate must flag the regression
    results[name] = dict(cert=rel, caught=caught)
    log(f"B {name}: cert={rel:.3e} -> {'CAUGHT' if caught else 'MISSED'}")
    if not caught:
        fails.append(name)

status = "pass" if not fails else "fail"
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(dict(status=status, fails=fails, results=results),
                          indent=2, default=float) + "\n")
log(f"ADVERSARIAL {status.upper()} ({len(fails)} failures) -> {out}")
sys.exit(0 if status == "pass" else 1)
