"""Step 5a: validate `backend="freitag_reiher"` against `backend="newton_casscf"`.

For CAS small enough that DMRG = FCI, the two backends must be element-wise
equivalent — newton_casscf is the validation gate, FR is the MPS-replaceable
generalization. This test set verifies:

  E1. Random-x matvec equivalence: cp_n._matvec(x) ≈ cp_fr._matvec(x) at 1e-10
  E2. solve(state=0) Lvec equivalence at 1e-8
  E3. solve_nac((0,1)) Lvec equivalence at 1e-6

System: HeH+/3-21G CAS(2,2) SA(2). Same setup as the existing baseline tests.
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

from cp_casscf_response import CPCASSCFResponseFCI


def setup_heh_321g():
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


def test_E1_matvec_equivalence():
    """For random x, cp_fr._matvec(x) must equal cp_n._matvec(x) to 1e-10."""
    mc = setup_heh_321g()
    cp_n = CPCASSCFResponseFCI(mc, backend="newton_casscf")
    cp_fr = CPCASSCFResponseFCI(mc, backend="freitag_reiher")
    op_n = cp_n.get_linear_operator()
    op_fr = cp_fr.get_linear_operator()
    n = op_n.shape[0]
    rng = np.random.default_rng(42)
    diffs = []
    for trial in range(5):
        x = rng.standard_normal(n)
        y_n = op_n @ x
        y_fr = op_fr @ x
        diffs.append(float(np.linalg.norm(y_n - y_fr)))
    max_diff = max(diffs)
    return {"name": "E1_matvec_equivalence",
            "trials": 5,
            "max_diff": max_diff,
            "all_diffs": diffs,
            "tol": 1e-10,
            "status": "pass" if max_diff < 1e-10 else "fail"}


def test_E2_solve_state0_equivalence():
    """Lvec from solve(state=0) must agree between backends to 1e-8."""
    mc = setup_heh_321g()
    cp_n = CPCASSCFResponseFCI(mc, backend="newton_casscf")
    cp_fr = CPCASSCFResponseFCI(mc, backend="freitag_reiher")
    kappa_n, vlist_n, info_n = cp_n.solve(state=0, tol=1e-10, max_iter=500)
    kappa_fr, vlist_fr, info_fr = cp_fr.solve(state=0, tol=1e-10, max_iter=500)
    diff_kappa = float(np.linalg.norm(kappa_n - kappa_fr))
    diff_v = max(float(np.linalg.norm(v_n - v_fr))
                 for v_n, v_fr in zip(vlist_n, vlist_fr))
    return {"name": "E2_solve_state0_equivalence",
            "info_n": int(info_n),
            "info_fr": int(info_fr),
            "kappa_diff": diff_kappa,
            "v_diff": diff_v,
            "tol": 1e-8,
            "status": "pass" if (info_n == 0 and info_fr == 0
                                 and diff_kappa < 1e-8
                                 and diff_v < 1e-8) else "fail"}


def test_E3_solve_nac_equivalence():
    """Lvec from solve_nac((0,1)) must agree between backends to 1e-6."""
    mc = setup_heh_321g()
    cp_n = CPCASSCFResponseFCI(mc, backend="newton_casscf")
    cp_fr = CPCASSCFResponseFCI(mc, backend="freitag_reiher")
    kappa_n, vlist_n, info_n = cp_n.solve_nac((0, 1), tol=1e-10, max_iter=500)
    kappa_fr, vlist_fr, info_fr = cp_fr.solve_nac((0, 1), tol=1e-10, max_iter=500)
    diff_kappa = float(np.linalg.norm(kappa_n - kappa_fr))
    diff_v = max(float(np.linalg.norm(v_n - v_fr))
                 for v_n, v_fr in zip(vlist_n, vlist_fr))
    return {"name": "E3_solve_nac_equivalence",
            "info_n": int(info_n),
            "info_fr": int(info_fr),
            "kappa_diff": diff_kappa,
            "v_diff": diff_v,
            "tol": 1e-6,
            "status": "pass" if (info_n == 0 and info_fr == 0
                                 and diff_kappa < 1e-6
                                 and diff_v < 1e-6) else "fail"}


def main():
    cases = [test_E1_matvec_equivalence,
             test_E2_solve_state0_equivalence,
             test_E3_solve_nac_equivalence]
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
    out = {"milestone": "Step5a_FR_backend_equivalence",
           "results": results}
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
