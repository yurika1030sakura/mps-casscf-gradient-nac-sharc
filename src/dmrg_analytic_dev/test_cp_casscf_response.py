"""Validate CP-CASSCF response building blocks (Freitag-Reiher Step 3).

Tests each Hessian-vector block and the combined matvec for:
  C1. H_CC linearity in v
  C2. H_CC Hermiticity: <v|H_CC w> = <w|H_CC v>
  C3. H_OC linearity in v
  C4. H_CO linearity in κ
  C5. Combined _matvec linearity
  C6. Combined _matvec Hermiticity (the key consistency check that orbital
      and CI block factors are mutually consistent — if this passes, the
      coupled CP-CASSCF Hessian is symmetric as it must be)
  C7. GMRES converges on the assembled system for state 0 of an SA(2)
      problem with ncore>0/nvirt>0 so the orbital block is nontrivial.

Test system: HeH+/sto-3g CAS(2,2) for C1-C2 (no orbital rotations exist, so
H_OO/H_OC/H_CO trivially vanish and we exercise only H_CC).
For C3-C7 we use H2/3-21G CAS(2,2) which has ncore=0, ncas=2, nvirt=2 →
4 independent (active-virtual) orbital rotations.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from pyscf import gto, mcscf, scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cp_casscf_response import CPCASSCFResponseFCI, _project_orthogonal


def setup_heh_sto3g():
    mol = gto.M(atom="He 0 0 0; H 0 0 1.4", basis="sto-3g",
                charge=1, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = 2
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-10
    mc.max_cycle_macro = 200
    mc = mc.state_average_([0.5, 0.5])
    mc.kernel()
    return mc


def setup_heh_321g():
    """HeH+/3-21G CAS(2,2) SA(2): nrot = 4 (active-virtual). Asymmetric atoms
    so per-state orbital gradients are nonzero (good for non-trivial RHS in C7)."""
    mol = gto.M(atom="He 0 0 0; H 0 0 1.4", basis="3-21G",
                charge=1, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = 2
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-10
    mc.max_cycle_macro = 200
    mc = mc.state_average_([0.5, 0.5])
    mc.kernel()
    return mc


def test_C1_HCC_linearity():
    mc = setup_heh_sto3g()
    cp = CPCASSCFResponseFCI(mc)
    rng = np.random.default_rng(0)
    v = [rng.standard_normal(ci.shape) for ci in cp.ci_list]
    w = [rng.standard_normal(ci.shape) for ci in cp.ci_list]
    a, b = 0.7, -1.3
    Hv = cp.H_CC_apply(v)
    Hw = cp.H_CC_apply(w)
    Hcombo = cp.H_CC_apply([a*vi + b*wi for vi, wi in zip(v, w)])
    diff = max(float(np.linalg.norm(Hcombo[i] - (a*Hv[i] + b*Hw[i])))
               for i in range(cp.nstates))
    return {"name": "C1_HCC_linearity",
            "abs_diff": diff,
            "tol": 1e-10,
            "status": "pass" if diff < 1e-10 else "fail"}


def test_C2_HCC_hermiticity():
    mc = setup_heh_sto3g()
    cp = CPCASSCFResponseFCI(mc)
    rng = np.random.default_rng(1)
    v = [rng.standard_normal(ci.shape) for ci in cp.ci_list]
    w = [rng.standard_normal(ci.shape) for ci in cp.ci_list]
    Hv = cp.H_CC_apply(v)
    Hw = cp.H_CC_apply(w)
    lhs = sum(float(np.tensordot(wi, Hv_i, axes=([0,1],[0,1])))
              for wi, Hv_i in zip(w, Hv))
    rhs = sum(float(np.tensordot(vi, Hw_i, axes=([0,1],[0,1])))
              for vi, Hw_i in zip(v, Hw))
    diff = abs(lhs - rhs)
    return {"name": "C2_HCC_hermiticity",
            "lhs": lhs, "rhs": rhs, "abs_diff": diff,
            "tol": 1e-10,
            "status": "pass" if diff < 1e-10 else "fail"}


def test_C3_HOC_linearity():
    mc = setup_heh_321g()
    cp = CPCASSCFResponseFCI(mc)
    rng = np.random.default_rng(2)
    v = [rng.standard_normal(ci.shape) for ci in cp.ci_list]
    w = [rng.standard_normal(ci.shape) for ci in cp.ci_list]
    a, b = 0.4, -0.9
    Tv = cp.H_OC_apply(v)
    Tw = cp.H_OC_apply(w)
    Tcombo = cp.H_OC_apply([a*vi + b*wi for vi, wi in zip(v, w)])
    diff = float(np.linalg.norm(Tcombo - (a*Tv + b*Tw)))
    return {"name": "C3_HOC_linearity",
            "abs_diff": diff,
            "tol": 1e-10,
            "status": "pass" if diff < 1e-10 else "fail"}


def test_C4_HCO_linearity():
    mc = setup_heh_321g()
    cp = CPCASSCFResponseFCI(mc)
    rng = np.random.default_rng(3)
    nmo = cp.nmo
    K1 = rng.standard_normal((nmo, nmo)); K1 = K1 - K1.T
    K2 = rng.standard_normal((nmo, nmo)); K2 = K2 - K2.T
    a, b = 1.1, -0.3
    out1 = cp.H_CO_apply(K1)
    out2 = cp.H_CO_apply(K2)
    outc = cp.H_CO_apply(a*K1 + b*K2)
    diff = max(float(np.linalg.norm(outc[i] - (a*out1[i] + b*out2[i])))
               for i in range(cp.nstates))
    return {"name": "C4_HCO_linearity",
            "abs_diff": diff,
            "tol": 1e-10,
            "status": "pass" if diff < 1e-10 else "fail"}


def test_C5_matvec_linearity():
    mc = setup_heh_321g()
    cp = CPCASSCFResponseFCI(mc)
    op = cp.get_linear_operator()
    n = op.shape[0]
    rng = np.random.default_rng(4)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    a, b = 0.6, -0.8
    Ax = op @ x
    Ay = op @ y
    Ac = op @ (a*x + b*y)
    diff = float(np.linalg.norm(Ac - (a*Ax + b*Ay)))
    return {"name": "C5_matvec_linearity",
            "n": int(n),
            "abs_diff": diff,
            "tol": 1e-10,
            "status": "pass" if diff < 1e-10 else "fail"}


def test_C6_matvec_hermiticity():
    """Hermiticity on the orthogonal complement of the SA-redundant nullspace.

    The assembled matvec uses the same one-sided SA projection as
    `pyscf.grad.sacasscf.project_Aop` (output-only): it removes the SA
    cross-state CI directions from A·x but does not pre-project x. So
    <y | A x> = <x | A y> holds on the projected subspace, not the full
    flattened vector space. Project x and y first to match the operator's
    actual domain.
    """
    mc = setup_heh_321g()
    cp = CPCASSCFResponseFCI(mc)
    op = cp.get_linear_operator()
    n = op.shape[0]
    rng = np.random.default_rng(5)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    # Project x and y orthogonal to the SA-redundant CI directions.
    def project(z):
        kappa, v_list = cp._unflatten(z)
        v_proj = list(v_list)
        for i in range(cp.nstates):
            for j in range(cp.nstates):
                ovlp = float(np.tensordot(v_proj[i], cp.ci_list[j],
                                          axes=([0, 1], [0, 1])))
                v_proj[i] = v_proj[i] - ovlp * cp.ci_list[j]
        return cp._flatten(kappa, v_proj)
    x_p, y_p = project(x), project(y)
    Ax = op @ x_p
    Ay = op @ y_p
    lhs = float(y_p @ Ax)
    rhs = float(x_p @ Ay)
    diff = abs(lhs - rhs)
    rel = diff / max(abs(lhs), abs(rhs), 1e-30)
    return {"name": "C6_matvec_hermiticity",
            "lhs_yAx": lhs, "rhs_xAy": rhs,
            "abs_diff": diff, "rel_diff": rel,
            "tol_rel": 1e-8,
            "status": "pass" if rel < 1e-8 else "fail"}


def test_C7_gmres_converges():
    """GMRES on the assembled CP-CASSCF system converges for state 0."""
    mc = setup_heh_321g()
    cp = CPCASSCFResponseFCI(mc)
    kappa, v_list, info = cp.solve(state=0, tol=1e-8, max_iter=500, verbose=False)
    op = cp.get_linear_operator()
    rhs_O, rhs_C = cp.build_rhs(0)
    rhs_flat = cp._flatten(rhs_O, rhs_C)
    x_flat = cp._flatten(kappa, v_list)
    residual = float(np.linalg.norm(op @ x_flat - rhs_flat))
    rhs_norm = float(np.linalg.norm(rhs_flat))
    return {"name": "C7_gmres_converges",
            "info": int(info),
            "residual_norm": residual,
            "rhs_norm": rhs_norm,
            "rel_residual": residual / max(rhs_norm, 1e-30),
            "kappa_norm": float(np.linalg.norm(kappa)),
            "tol_rel": 1e-6,
            "status": "pass" if (info == 0 and residual / max(rhs_norm, 1e-30) < 1e-6)
                              else "fail"}


def main():
    cases = [test_C1_HCC_linearity, test_C2_HCC_hermiticity,
             test_C3_HOC_linearity, test_C4_HCO_linearity,
             test_C5_matvec_linearity, test_C6_matvec_hermiticity,
             test_C7_gmres_converges]
    results = []
    for c in cases:
        try:
            r = c()
        except Exception as exc:
            r = {"name": c.__name__, "status": "fail",
                 "exception": type(exc).__name__,
                 "message": str(exc),
                 "traceback_tail": traceback.format_exc()[-2000:]}
        results.append(r)
        print(f"  {r['name']}: {r['status']}")

    out_path = Path(__file__).with_suffix(".json")
    out = {"milestone": "FreitagReiher_Step3_cp_casscf_response_blocks",
           "results": results}
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
