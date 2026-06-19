"""Corrected PoC: solve a NONZERO CI-block linear system with block2's sweep
solver, the engine the coupled Schur response needs.

The earlier PoC fed the state-gradient CI residual, which is ~0 for a converged
state, so it solved a trivial system.  In the coupled response the CI-block RHS
is the orbital->CI coupling H_CO(z_kappa), which is genuinely nonzero.  Here we
build such a nonzero RHS, solve  P (H_CC - E_i) P v = P w  by a block2 sweep,
and check the reference-free projected residual.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "sharc_interface"))

from analytic_cp_sharc import _make_mps_krylov_response
from test_step6c_mps_response_class import _setup_heh


def _norm(obj, v):
    return float(np.sqrt(max(obj._mps_overlap(v, v), 0.0)))


def _proj_out_roots(obj, v, tag):
    """Remove every state-root component from v (the matvec projector P)."""
    out = v
    for j, sm in enumerate(obj._state_mps):
        ov = obj._mps_overlap(sm, out)
        if abs(ov) > 1.0e-14:
            out = obj._combine_mps(
                [(1.0, out), (-ov, sm)], tag=obj._new_tag(f"{tag}-P{j}"),
            )
    return out


def ci_block_inverse(obj, w_ci, state, *, n_sweeps=50, tol=1.0e-11,
                     solver_type="MinRes", proj_weight=1.0e3):
    """Solve (H_CC - E_state) v = w_ci with v ⟂ Psi_state via block2 sweeps."""
    sm = obj._state_mps[state]
    hmE = obj._hcc_shifted_mpo(state)
    bra = obj._copy_mps(w_ci, tag=obj._new_tag("CIINV-BRA"))
    with obj._use_su2_frame():
        obj._driver_su2.multiply(
            bra, obj._identity(), w_ci,
            left_mpo=hmE, n_sweeps=int(n_sweeps), tol=float(tol),
            bra_bond_dims=[obj._m_compress],
            proj_mpss=[sm], proj_weights=[float(proj_weight)],
            linear_max_iter=4000, solver_type=solver_type, iprint=0,
        )
    ov = obj._mps_overlap(sm, bra)
    if abs(ov) > 1.0e-15:
        bra = obj._combine_mps([(1.0, bra), (-ov, sm)],
                               tag=obj._new_tag("CIINV-PROJ"))
    return bra


def run_poc(state=0):
    mc, _solver = _setup_heh()
    obj = _make_mps_krylov_response(mc)
    obj._build_eris_cache(); obj._build_hcc_state_cache()

    # nonzero CI-block RHS: orbital->CI coupling of a nonzero orbital vector
    rng = np.random.default_rng(0)
    kappa = rng.standard_normal((obj.nmo, obj.nmo))
    kappa = obj._canonical_kappa(kappa - kappa.T)
    w_list = obj.H_CO_apply_mps(kappa)        # list of MPS, one per state slot
    w = _proj_out_roots(obj, w_list[state], "W")   # project to the solve space
    w_norm = _norm(obj, w)

    # solve (H_CC - E) v = w
    v = ci_block_inverse(obj, w, state)

    # reference-free projected residual ||P[(H-E)v - w]|| / ||P w||
    sig = obj._sigma_mps(obj._hcc_shifted_mpo(state), v, tag=obj._new_tag("SIG"))
    res = obj._combine_mps([(1.0, sig), (-1.0, w)], tag=obj._new_tag("RES"))
    res = _proj_out_roots(obj, res, "RES")
    rel_res = _norm(obj, res) / max(w_norm, 1.0e-30)

    return {
        "state": state,
        "rhs_norm": w_norm,
        "solution_norm": _norm(obj, v),
        "projected_residual_rel": rel_res,
        "pass_criterion": "projected_residual_rel < 1e-6 on a NONZERO rhs",
        "status": "pass" if (w_norm > 1e-8 and rel_res < 1.0e-6) else "fail",
    }


def main():
    try:
        result = run_poc()
    except Exception as exc:
        result = {"status": "fail", "exception": type(exc).__name__,
                  "message": str(exc),
                  "traceback_tail": traceback.format_exc()[-3000:]}
    print(json.dumps(result, indent=2))
    Path(__file__).with_suffix(".json").write_text(json.dumps({
        "milestone": "PoC_ci_block_inverse_nonzero_rhs",
        "system": "HeH+ / 3-21G / CAS(2,2) / SA(2)",
        "result": result,
    }, indent=2) + "\n")
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
