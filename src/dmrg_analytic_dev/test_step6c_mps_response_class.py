"""Step 6.3b/d validation: CPDMRGCASSCFResponseMPS.

Reproducibility test on CAS(2,2) HeH+/3-21G SA(2): the new MPS-routed class
must produce H_*_apply outputs that match the validated FCI parent
(``CPCASSCFResponseFCI`` with backend="freitag_reiher") to ~ 1e-8 element-wise.

Tests:

  M1: H_CC element-wise equivalence (FCI parent vs MPS class) on a random
      v_list trial.
  M2: H_OC element-wise equivalence on the same v_list.
  M3: H_CO element-wise equivalence on a random kappa.
  M4: end-to-end ``solve(state)`` produces matching kappa, v_list to 1e-8
      between the two classes.

Note: at CAS(2,2) DMRG = FCI, so we expect machine-precision agreement up
to fitting tolerance (1e-12 from the multiply call) and the SZ-mode
det-routing overhead (no error, exact mapping).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dmrg_fcisolver import MPSAsFCISolver
from cp_casscf_response import CPCASSCFResponseFCI
from cp_dmrg_response_mps import CPDMRGCASSCFResponseMPS


def _setup_heh(d_bohr: float = 1.4, basis: str = "3-21G", bond_dim: int = 64,
               nroots: int = 2):
    mol = gto.M(atom=f"He 0 0 0; H 0 0 {d_bohr}", basis=basis,
                charge=1, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 2, 2)
    solver = MPSAsFCISolver(mol, bond_dim=bond_dim, n_sweeps=20)
    solver.nroots = nroots
    mc.fcisolver = solver
    if nroots > 1:
        mc.state_average_([1.0 / nroots] * nroots)
    mc.kernel()
    # state_average_ wraps fcisolver as a view; the underlying object
    # holds the live driver. Return mc.fcisolver (post-wrap) for downstream
    # code so we use the live driver / kets.
    return mc, mc.fcisolver


def _build_active_mpo(driver, mol, mf, mo_coeff, ncore, ncas):
    mo_act = mo_coeff[:, ncore:ncore + ncas]
    h_core = mf.get_hcore()
    if ncore > 0:
        mo_core = mo_coeff[:, :ncore]
        dm_core = 2 * mo_core @ mo_core.T
        j_core, k_core = mf.get_jk(mol, dm_core)
        h_eff = h_core + j_core - 0.5 * k_core
        h_act = mo_act.T @ h_eff @ mo_act
    else:
        h_act = mo_act.T @ h_core @ mo_act
    eri_act = ao2mo.kernel(mol, mo_act, compact=False).reshape((ncas,) * 4)
    mpo = driver.get_qc_mpo(np.asarray(h_act), np.asarray(eri_act),
                             ecore=0.0, iprint=0)
    return mpo


def test_m1_HCC_elementwise():
    mc, solver = _setup_heh()
    mpo_act = _build_active_mpo(solver._driver, mc.mol, mc._scf,
                                  mc.mo_coeff, mc.ncore, mc.ncas)
    fci_resp = CPCASSCFResponseFCI(mc, backend="freitag_reiher")
    mps_resp = CPDMRGCASSCFResponseMPS(mc, solver._driver, mpo_act,
                                        m_compress=64)

    rng = np.random.default_rng(42)
    v_list = []
    for c in mc.ci:
        v = rng.standard_normal(c.shape)
        # symmetrize for singlet
        v = 0.5 * (v + v.T)
        v_list.append(v)

    out_fci = fci_resp.H_CC_apply(v_list)
    out_mps = mps_resp.H_CC_apply(v_list)
    diffs = [float(np.linalg.norm(a - b)) for a, b in zip(out_fci, out_mps)]
    return {"name": "M1_HCC_elementwise",
            "diffs_per_root": diffs, "tol": 1e-8,
            "status": "pass" if max(diffs) < 1e-8 else "fail"}


def test_m2_HOC_elementwise():
    mc, solver = _setup_heh()
    mpo_act = _build_active_mpo(solver._driver, mc.mol, mc._scf,
                                  mc.mo_coeff, mc.ncore, mc.ncas)
    fci_resp = CPCASSCFResponseFCI(mc, backend="freitag_reiher")
    mps_resp = CPDMRGCASSCFResponseMPS(mc, solver._driver, mpo_act,
                                        m_compress=64)

    rng = np.random.default_rng(43)
    v_list = []
    for c in mc.ci:
        v = rng.standard_normal(c.shape)
        v = 0.5 * (v + v.T)
        v_list.append(v)

    out_fci = fci_resp.H_OC_apply(v_list)
    out_mps = mps_resp.H_OC_apply(v_list)
    diff = float(np.linalg.norm(out_fci - out_mps))
    return {"name": "M2_HOC_elementwise",
            "diff": diff, "tol": 1e-8,
            "status": "pass" if diff < 1e-8 else "fail"}


def test_m3_HCO_elementwise():
    mc, solver = _setup_heh()
    mpo_act = _build_active_mpo(solver._driver, mc.mol, mc._scf,
                                  mc.mo_coeff, mc.ncore, mc.ncas)
    fci_resp = CPCASSCFResponseFCI(mc, backend="freitag_reiher")
    mps_resp = CPDMRGCASSCFResponseMPS(mc, solver._driver, mpo_act,
                                        m_compress=64)

    rng = np.random.default_rng(44)
    nmo = mc.mo_coeff.shape[1]
    kappa = rng.standard_normal((nmo, nmo))
    kappa = kappa - kappa.T  # antisymmetrize

    out_fci = fci_resp.H_CO_apply(kappa)
    out_mps = mps_resp.H_CO_apply(kappa)
    diffs = [float(np.linalg.norm(a - b)) for a, b in zip(out_fci, out_mps)]
    return {"name": "M3_HCO_elementwise",
            "diffs_per_root": diffs, "tol": 1e-8,
            "status": "pass" if max(diffs) < 1e-8 else "fail"}


def test_m4_solve_state0():
    """End-to-end solve(state=0): kappa, v_list match between FCI and MPS class."""
    mc, solver = _setup_heh()
    mpo_act = _build_active_mpo(solver._driver, mc.mol, mc._scf,
                                  mc.mo_coeff, mc.ncore, mc.ncas)
    fci_resp = CPCASSCFResponseFCI(mc, backend="freitag_reiher")
    mps_resp = CPDMRGCASSCFResponseMPS(mc, solver._driver, mpo_act,
                                        m_compress=64)

    k_f, v_f, info_f = fci_resp.solve(state=0, tol=1e-9, max_iter=200)
    k_m, v_m, info_m = mps_resp.solve(state=0, tol=1e-9, max_iter=200)

    diff_k = float(np.linalg.norm(k_f - k_m))
    diff_v = max(float(np.linalg.norm(a - b)) for a, b in zip(v_f, v_m))
    ok = (info_f == 0 and info_m == 0 and diff_k < 1e-7 and diff_v < 1e-7)
    return {"name": "M4_solve_state0",
            "info_fci": int(info_f), "info_mps": int(info_m),
            "diff_kappa": diff_k, "diff_v_max": diff_v, "tol": 1e-7,
            "status": "pass" if ok else "fail"}


def main():
    cases = [test_m1_HCC_elementwise, test_m2_HOC_elementwise,
             test_m3_HCO_elementwise, test_m4_solve_state0]
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
    out = {
        "milestone": "Step6.3b_CPDMRGCASSCFResponseMPS_validation",
        "purpose": "CPDMRGCASSCFResponseMPS reproduces CPCASSCFResponseFCI on CAS(2,2) HeH+ where DMRG = FCI.",
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2,
        default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
