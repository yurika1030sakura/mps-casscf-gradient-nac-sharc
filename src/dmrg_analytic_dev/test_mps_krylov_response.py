"""Validation tests for the MPS-Krylov response backend.

The response RHS and CI Krylov vectors are stored as block2 MPS objects.  FCI
arrays are used only to construct random small-CAS validation vectors and to
compare the final outputs against the established FCI response backend.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cp_casscf_response import CPCASSCFResponseFCI
from cp_dmrg_response_mps_krylov import (  # noqa: E402
    CPDMRGCASSCFResponseMPSKrylov,
    MPSKrylovVector,
)
from test_step6c_mps_response_class import _build_active_mpo, _setup_heh  # noqa: E402


def _make_pair():
    mc, solver = _setup_heh()
    mpo_act = _build_active_mpo(
        solver._driver, mc.mol, mc._scf, mc.mo_coeff, mc.ncore, mc.ncas,
    )
    fci_resp = CPCASSCFResponseFCI(mc, backend="freitag_reiher")
    mps_resp = CPDMRGCASSCFResponseMPSKrylov(
        mc, solver._driver, mpo_act, m_compress=64,
        mps_fit_sweeps=12, mps_fit_tol=1.0e-12,
    )
    return mc, fci_resp, mps_resp


def test_m1_matvec_matches_fci():
    mc, fci_resp, mps_resp = _make_pair()
    rng = np.random.default_rng(123)
    nmo = mc.mo_coeff.shape[1]
    kappa = rng.standard_normal((nmo, nmo))
    kappa = kappa - kappa.T
    v_list = []
    for ci in mc.ci:
        v = rng.standard_normal(ci.shape)
        v = 0.5 * (v + v.T)
        v_list.append(v)

    x_fci = fci_resp._flatten(kappa, v_list)
    ax_fci = fci_resp._matvec_fr(x_fci)

    x_mps = mps_resp.vector_from_fci(kappa, v_list, label="M1")
    ax_mps = mps_resp.matvec_mps(x_mps)
    ax_mps_fci = mps_resp.flatten_for_validation(ax_mps)

    diff = float(np.linalg.norm(ax_mps_fci - ax_fci))
    return {
        "name": "M1_mps_krylov_matvec_matches_fci",
        "diff": diff,
        "norm_fci": float(np.linalg.norm(ax_fci)),
        "norm_mps": float(np.linalg.norm(ax_mps_fci)),
        "tol": 1.0e-8,
        "status": "pass" if diff < 1.0e-8 else "fail",
    }


def test_m2_gradient_rhs_matches_fci_without_fci_builder():
    mc, fci_resp, mps_resp = _make_pair()
    rhs_o, rhs_c = fci_resp.build_rhs(0)
    rhs_fci = fci_resp._flatten(rhs_o, rhs_c)
    rhs_mps = mps_resp.flatten_for_validation(mps_resp.build_rhs_mps(0))
    diff = float(np.linalg.norm(rhs_mps - rhs_fci))
    return {
        "name": "M2_mps_gradient_rhs_matches_fci",
        "diff": diff,
        "norm_fci": float(np.linalg.norm(rhs_fci)),
        "norm_mps": float(np.linalg.norm(rhs_mps)),
        "tol": 1.0e-8,
        "status": "pass" if diff < 1.0e-8 else "fail",
    }


def test_m3_nac_rhs_matches_fci_without_fci_builder():
    mc, fci_resp, mps_resp = _make_pair()
    rhs_o, rhs_c = fci_resp.build_rhs_nac((0, 1))
    rhs_fci = fci_resp._flatten(rhs_o, rhs_c)
    rhs_mps = mps_resp.flatten_for_validation(mps_resp.build_rhs_nac_mps((0, 1)))
    diff = float(np.linalg.norm(rhs_mps - rhs_fci))
    return {
        "name": "M3_mps_nac_rhs_matches_fci",
        "diff": diff,
        "norm_fci": float(np.linalg.norm(rhs_fci)),
        "norm_mps": float(np.linalg.norm(rhs_mps)),
        "tol": 1.0e-8,
        "status": "pass" if diff < 1.0e-8 else "fail",
    }


def test_m4_solve_state0_matches_fci():
    mc, fci_resp, mps_resp = _make_pair()
    k_fci, v_fci, info_fci = fci_resp.solve(0, tol=1.0e-8, max_iter=100)
    k_mps, v_mps, info_mps, meta = mps_resp.solve_mps(
        0, tol=1.0e-8, max_iter=20, verbose=False,
    )
    v_mps_fci = mps_resp.vector_to_fci_list(
        MPSKrylovVector(
            mps_resp, np.zeros((mc.mo_coeff.shape[1], mc.mo_coeff.shape[1])),
            v_mps,
        )
    )
    diff_kappa = float(np.linalg.norm(k_mps - k_fci))
    diff_v = float(max(np.linalg.norm(a - b) for a, b in zip(v_mps_fci, v_fci)))
    ok = info_fci == 0 and info_mps == 0 and diff_kappa < 1.0e-8 and diff_v < 1.0e-8
    return {
        "name": "M4_mps_krylov_solve_state0_matches_fci",
        "info_fci": int(info_fci),
        "info_mps": int(info_mps),
        "mps_residual": float(meta["residual"]),
        "mps_niter": int(meta["niter"]),
        "diff_kappa": diff_kappa,
        "diff_v_max": diff_v,
        "tol": 1.0e-8,
        "status": "pass" if ok else "fail",
    }


def test_m5_mps_only_initializer_solves_without_dense_ci_roots():
    mc, fci_resp, mps_resp = _make_pair()
    mps_only = CPDMRGCASSCFResponseMPSKrylov(
        mc,
        mps_resp._driver_su2,
        mps_resp._mpo_active,
        mps_states=mps_resp._mps_states,
        weights=mps_resp.weights,
        m_compress=64,
        mps_fit_sweeps=12,
        mps_fit_tol=1.0e-12,
        mps_only=True,
    )
    k_fci, _v_fci, info_fci = fci_resp.solve(0, tol=1.0e-8, max_iter=100)
    k_mps, _v_mps, info_mps, meta = mps_only.solve_mps(
        0, tol=1.0e-8, max_iter=20, verbose=False,
    )
    d_cache = float(np.linalg.norm(
        mps_resp._build_eris_cache()["casdm1_avg"]
        - mps_only._build_eris_cache()["casdm1_avg"]
    ))
    d_kappa = float(np.linalg.norm(k_mps - k_fci))
    ok = info_fci == 0 and info_mps == 0 and d_cache < 1.0e-10 and d_kappa < 1.0e-8
    return {
        "name": "M5_mps_only_initializer_solves_without_dense_ci_roots",
        "info_fci": int(info_fci),
        "info_mps": int(info_mps),
        "mps_residual": float(meta["residual"]),
        "density_cache_diff": d_cache,
        "diff_kappa": d_kappa,
        "tol": 1.0e-8,
        "status": "pass" if ok else "fail",
    }


def test_m6_bicgstab_solver_matches_fci():
    mc, fci_resp, mps_resp = _make_pair()
    bicg = CPDMRGCASSCFResponseMPSKrylov(
        mc,
        mps_resp._driver_su2,
        mps_resp._mpo_active,
        m_compress=64,
        mps_fit_sweeps=12,
        mps_fit_tol=1.0e-12,
        linear_solver="bicgstab",
    )
    k_fci, _v_fci, info_fci = fci_resp.solve(0, tol=1.0e-8, max_iter=100)
    k_mps, _v_mps, info_mps, meta = bicg.solve_mps(
        0, tol=1.0e-8, max_iter=30, verbose=False,
    )
    d_kappa = float(np.linalg.norm(k_mps - k_fci))
    ok = info_fci == 0 and info_mps == 0 and d_kappa < 1.0e-8
    return {
        "name": "M6_mps_bicgstab_solve_state0_matches_fci",
        "info_fci": int(info_fci),
        "info_mps": int(info_mps),
        "mps_residual": float(meta["residual"]),
        "mps_niter": int(meta["niter"]),
        "diff_kappa": d_kappa,
        "tol": 1.0e-8,
        "status": "pass" if ok else "fail",
    }


def test_m7_hcc_inverse_initial_guess_matches_fci():
    mc, fci_resp, mps_resp = _make_pair()
    hcc_guess = CPDMRGCASSCFResponseMPSKrylov(
        mc,
        mps_resp._driver_su2,
        mps_resp._mpo_active,
        m_compress=64,
        mps_fit_sweeps=12,
        mps_fit_tol=1.0e-12,
        initial_guess="hcc-inverse",
        initial_guess_sweeps=6,
        initial_guess_tol=1.0e-10,
        initial_guess_proj_weight=20.0,
    )
    k_fci, _v_fci, info_fci = fci_resp.solve(0, tol=1.0e-8, max_iter=100)
    k_mps, _v_mps, info_mps, meta = hcc_guess.solve_mps(
        0, tol=1.0e-8, max_iter=20, verbose=False,
    )
    d_kappa = float(np.linalg.norm(k_mps - k_fci))
    ok = info_fci == 0 and info_mps == 0 and d_kappa < 1.0e-8
    timings = dict(meta.get("timings_s", {}))
    return {
        "name": "M7_mps_hcc_inverse_initial_guess_matches_fci",
        "info_fci": int(info_fci),
        "info_mps": int(info_mps),
        "mps_residual": float(meta["residual"]),
        "mps_relative_residual": float(meta.get("relative_residual", 0.0)),
        "mps_niter": int(meta["niter"]),
        "initial_guess": str(meta.get("initial_guess")),
        "initial_guess_hcc_inverse_s": float(
            timings.get("initial_guess_hcc_inverse", 0.0)
        ),
        "diff_kappa": d_kappa,
        "tol": 1.0e-8,
        "status": "pass" if ok else "fail",
    }


def test_m8_cr_solver_matches_fci():
    mc, fci_resp, mps_resp = _make_pair()
    cr = CPDMRGCASSCFResponseMPSKrylov(
        mc,
        mps_resp._driver_su2,
        mps_resp._mpo_active,
        m_compress=64,
        mps_fit_sweeps=12,
        mps_fit_tol=1.0e-12,
        linear_solver="cr",
    )
    k_fci, _v_fci, info_fci = fci_resp.solve(0, tol=1.0e-8, max_iter=100)
    k_mps, _v_mps, info_mps, meta = cr.solve_mps(
        0, tol=1.0e-8, max_iter=60, verbose=False,
    )
    d_kappa = float(np.linalg.norm(k_mps - k_fci))
    ok = info_fci == 0 and info_mps == 0 and d_kappa < 1.0e-8
    return {
        "name": "M8_mps_cr_solve_state0_matches_fci",
        "info_fci": int(info_fci),
        "info_mps": int(info_mps),
        "mps_residual": float(meta["residual"]),
        "mps_niter": int(meta["niter"]),
        "diff_kappa": d_kappa,
        "tol": 1.0e-8,
        "status": "pass" if ok else "fail",
    }


def test_m9_gmres_recycle_initial_guess_matches_fci():
    mc, fci_resp, mps_resp = _make_pair()
    recycle = CPDMRGCASSCFResponseMPSKrylov(
        mc,
        mps_resp._driver_su2,
        mps_resp._mpo_active,
        m_compress=64,
        mps_fit_sweeps=12,
        mps_fit_tol=1.0e-12,
        initial_guess="gmres-recycle",
    )
    recycle.solve_mps(0, tol=1.0e-8, max_iter=20, verbose=False)
    k_fci, _v_fci, info_fci = fci_resp.solve(1, tol=1.0e-8, max_iter=100)
    k_mps, _v_mps, info_mps, meta = recycle.solve_mps(
        1, tol=1.0e-8, max_iter=20, verbose=False,
    )
    d_kappa = float(np.linalg.norm(k_mps - k_fci))
    timings = dict(meta.get("timings_s", {}))
    ok = info_fci == 0 and info_mps == 0 and d_kappa < 1.0e-8
    return {
        "name": "M9_mps_gmres_recycle_solve_state1_matches_fci",
        "info_fci": int(info_fci),
        "info_mps": int(info_mps),
        "mps_residual": float(meta["residual"]),
        "mps_niter": int(meta["niter"]),
        "diff_kappa": d_kappa,
        "recycle_project_s": float(
            timings.get("initial_guess_gmres_recycle_project", 0.0)
        ),
        "recycle_build_s": float(
            timings.get("initial_guess_gmres_recycle_build", 0.0)
        ),
        "tol": 1.0e-8,
        "status": "pass" if ok else "fail",
    }


def main():
    cases = [
        test_m1_matvec_matches_fci,
        test_m2_gradient_rhs_matches_fci_without_fci_builder,
        test_m3_nac_rhs_matches_fci_without_fci_builder,
        test_m4_solve_state0_matches_fci,
        test_m5_mps_only_initializer_solves_without_dense_ci_roots,
        test_m6_bicgstab_solver_matches_fci,
        test_m7_hcc_inverse_initial_guess_matches_fci,
        test_m8_cr_solver_matches_fci,
        test_m9_gmres_recycle_initial_guess_matches_fci,
    ]
    results = []
    for case in cases:
        try:
            result = case()
        except Exception as exc:
            result = {
                "name": case.__name__,
                "status": "fail",
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc()[-2000:],
            }
        results.append(result)
        print(f"  {result['name']}: {result['status']}")

    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps({
        "milestone": "MPS_Krylov_response_backend",
        "purpose": (
            "Validate MPS-valued CI Krylov vectors against the FCI response "
            "backend on HeH+ CAS(2,2)."
        ),
        "results": results,
    }, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
