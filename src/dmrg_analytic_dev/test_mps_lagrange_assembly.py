"""Validate MPS transition-RDM CI-Lagrange nuclear assembly.

This test checks the next piece needed after an MPS-valued Krylov solve: the
final CI-Lagrange contribution to the nuclear derivative can be assembled from
MPS transition RDMs without storing the Lagrange CI vector as an ndarray.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from pyscf.grad.sacasscf import Lci_dot_dgci_dx, Lorb_dot_dgorb_dx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cp_dmrg_response_mps_krylov import CPDMRGCASSCFResponseMPSKrylov  # noqa: E402
from cp_casscf_response import _project_orthogonal  # noqa: E402
from test_step6c_mps_response_class import _build_active_mpo, _setup_heh  # noqa: E402


def _setup_response():
    mc, solver = _setup_heh()
    mpo_act = _build_active_mpo(
        solver._driver, mc.mol, mc._scf, mc.mo_coeff, mc.ncore, mc.ncas,
    )
    response = CPDMRGCASSCFResponseMPSKrylov(
        mc, solver._driver, mpo_act, m_compress=64,
        mps_fit_sweeps=12, mps_fit_tol=1.0e-12,
    )
    return mc, response


def test_l1_lci_dot_from_mps_transition_rdm_matches_pyscf():
    mc, response = _setup_response()
    rng = np.random.default_rng(321)
    lci = []
    for ci in mc.ci:
        v = rng.standard_normal(ci.shape)
        v = 0.5 * (v + v.T)
        lci.append(v)

    eris = mc.ao2mo(mc.mo_coeff)
    mf_grad = mc._scf.nuc_grad_method()
    ref = Lci_dot_dgci_dx(
        lci, response.weights, mc,
        mo_coeff=mc.mo_coeff, ci=mc.ci, atmlst=None,
        mf_grad=mf_grad, eris=eris, verbose=0,
    )

    lci_mps = response.ci_mps_from_fci_list(lci, label="L1")
    got = response.Lci_dot_dgci_dx_mps(
        lci_mps, mo_coeff=mc.mo_coeff, atmlst=None,
        mf_grad=mf_grad, eris=eris, verbose=0,
    )

    diff = float(np.linalg.norm(got - ref))
    return {
        "name": "L1_lci_dot_from_mps_transition_rdm_matches_pyscf",
        "diff": diff,
        "norm_ref": float(np.linalg.norm(ref)),
        "norm_mps": float(np.linalg.norm(got)),
        "tol": 1.0e-8,
        "status": "pass" if diff < 1.0e-8 else "fail",
    }


def test_l2_full_lagrange_nuclear_term_matches_pyscf():
    mc, response = _setup_response()
    rng = np.random.default_rng(654)
    lci = []
    for ci in mc.ci:
        v = rng.standard_normal(ci.shape)
        v = 0.5 * (v + v.T)
        lci.append(v)

    kraw = rng.standard_normal((response.nmo, response.nmo))
    kappa = response._canonical_kappa(kraw - kraw.T)
    eris = mc.ao2mo(mc.mo_coeff)
    mf_grad = mc._scf.nuc_grad_method()

    ref_ci = Lci_dot_dgci_dx(
        lci, response.weights, mc,
        mo_coeff=mc.mo_coeff, ci=mc.ci, atmlst=None,
        mf_grad=mf_grad, eris=eris, verbose=0,
    )
    ref_orb = Lorb_dot_dgorb_dx(
        kappa, mc,
        mo_coeff=mc.mo_coeff, ci=mc.ci, atmlst=None,
        mf_grad=mf_grad, eris=eris, verbose=0,
    )

    lci_mps = response.ci_mps_from_fci_list(lci, label="L2")
    got = response.LdotJnuc_mps(
        kappa, lci_mps, mo_coeff=mc.mo_coeff, ci=mc.ci,
        atmlst=None, mf_grad=mf_grad, eris=eris, verbose=0,
    )

    ref = ref_ci + ref_orb
    diff = float(np.linalg.norm(got - ref))
    return {
        "name": "L2_full_lagrange_nuclear_term_matches_pyscf",
        "diff": diff,
        "norm_ref": float(np.linalg.norm(ref)),
        "norm_mps": float(np.linalg.norm(got)),
        "tol": 1.0e-8,
        "status": "pass" if diff < 1.0e-8 else "fail",
    }


def test_l3_final_mps_ci_projection_matches_dense_gauge():
    mc, response = _setup_response()
    rng = np.random.default_rng(987)
    lci_raw = []
    lci_projected = []
    for i, ci in enumerate(mc.ci):
        v = rng.standard_normal(ci.shape)
        v = 0.5 * (v + v.T)
        contaminated = v + (0.25 + 0.1 * i) * ci
        lci_raw.append(contaminated)
        lci_projected.append(_project_orthogonal(contaminated, ci))

    eris = mc.ao2mo(mc.mo_coeff)
    mf_grad = mc._scf.nuc_grad_method()
    ref = Lci_dot_dgci_dx(
        lci_projected, response.weights, mc,
        mo_coeff=mc.mo_coeff, ci=mc.ci, atmlst=None,
        mf_grad=mf_grad, eris=eris, verbose=0,
    )

    lci_mps_raw = response.ci_mps_from_fci_list(lci_raw, label="L3RAW")
    lci_mps_projected, proj_meta = response.project_ci_mps_list(
        lci_mps_raw, label="L3PROJ",
    )
    got = response.Lci_dot_dgci_dx_mps(
        lci_mps_projected, mo_coeff=mc.mo_coeff, atmlst=None,
        mf_grad=mf_grad, eris=eris, verbose=0,
    )

    diff = float(np.linalg.norm(got - ref))
    return {
        "name": "L3_final_mps_ci_projection_matches_dense_gauge",
        "diff": diff,
        "norm_ref": float(np.linalg.norm(ref)),
        "norm_mps": float(np.linalg.norm(got)),
        "max_overlap_before": float(
            proj_meta["max_root_overlap_before_projection"]
        ),
        "max_overlap_after": float(
            proj_meta["max_root_overlap_after_projection"]
        ),
        "tol": 1.0e-8,
        "status": "pass" if diff < 1.0e-8 else "fail",
    }


def main():
    cases = [
        test_l1_lci_dot_from_mps_transition_rdm_matches_pyscf,
        test_l2_full_lagrange_nuclear_term_matches_pyscf,
        test_l3_final_mps_ci_projection_matches_dense_gauge,
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
        "milestone": "MPS_Lci_nuclear_assembly",
        "purpose": (
            "Validate CI-Lagrange nuclear derivative assembly from MPS "
            "transition RDMs against PySCF's dense Lci path."
        ),
        "results": results,
    }, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
