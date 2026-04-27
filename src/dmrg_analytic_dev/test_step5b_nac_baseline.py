"""Step 5b: validate NAC computation against pyscf.nac.sacasscf.

Workflow:
  N1. Set up HeH+/3-21G CAS(2,2) SA(2). State pair (0, 1).
  N2. Run pyscf.nac.sacasscf.NonAdiabaticCouplings(mc).kernel(state=(0,1))
      for the reference NAC vector AND extract internal Lvec_nac.
  N3. Run our CPCASSCFResponseFCI.solve_nac((0,1)) → (κ̄, ṽ̄).
  N4. Compare flattened my-Lvec against pyscf's Lvec_nac.
  N5. Plug my Lvec into pyscf's NAC kernel via monkey-patched solve_lagrange,
      compare full nuclear NAC vector to the reference.

If N4/N5 pass at machine precision, our solver is validated for both
gradient (Step 4) and NAC. Step 5b complete; ready to wire to SHARC.
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


def run_pyscf_nac_reference(mc, state_pair=(0, 1)):
    """Run pyscf.nac.sacasscf and capture both the final NAC vector and
    the internal Lagrange multiplier."""
    from pyscf.nac import sacasscf as nac_sacasscf
    nac = nac_sacasscf.NonAdiabaticCouplings(mc)
    captured = {}
    orig_get_callback = nac.get_lagrange_callback

    def my_get_callback(Lvec_last, itvec, geff_op):
        cb = orig_get_callback(Lvec_last, itvec, geff_op)
        def wrapped(x):
            cb(x)
            captured["Lvec_last"] = np.asarray(x).copy()
        return wrapped
    nac.get_lagrange_callback = my_get_callback

    de = nac.kernel(state=state_pair)
    return de, captured.get("Lvec_last"), nac


def test_N1_nac_rhs_nontrivial():
    mc = setup_heh_321g()
    cp = CPCASSCFResponseFCI(mc)
    rhs_O, rhs_C = cp.build_rhs_nac((0, 1))
    rhs_flat = cp._flatten(rhs_O, rhs_C)
    return {"name": "N1_nac_rhs_nontrivial",
            "rhs_O_norm": float(np.linalg.norm(rhs_O)),
            "rhs_flat_norm": float(np.linalg.norm(rhs_flat)),
            "tol_min": 1e-3,
            "status": "pass" if np.linalg.norm(rhs_flat) > 1e-3 else "fail"}


def test_N2_my_nac_lvec_matches_pyscf():
    mc = setup_heh_321g()
    de_ref, Lvec_ref, nac_obj = run_pyscf_nac_reference(mc, state_pair=(0, 1))

    cp = CPCASSCFResponseFCI(mc)
    kappa, v_list, info = cp.solve_nac((0, 1), tol=1e-10, max_iter=500)
    Lvec_mine = nac_obj.pack_uniq_var(kappa, v_list)

    diff_pos = float(np.linalg.norm(Lvec_mine - Lvec_ref))
    diff_neg = float(np.linalg.norm(Lvec_mine + Lvec_ref))
    diff_min = min(diff_pos, diff_neg)
    return {"name": "N2_nac_lvec_matches_pyscf",
            "info": int(info),
            "Lvec_pyscf_norm": float(np.linalg.norm(Lvec_ref)),
            "Lvec_mine_norm": float(np.linalg.norm(Lvec_mine)),
            "diff_pos": diff_pos, "diff_neg": diff_neg,
            "diff_min": diff_min,
            "tol": 1e-6,
            "status": "pass" if diff_min < 1e-6 else "fail"}


def test_N3_full_nac_matches():
    mc = setup_heh_321g()
    de_ref, Lvec_ref, nac_obj = run_pyscf_nac_reference(mc, state_pair=(0, 1))

    cp = CPCASSCFResponseFCI(mc)
    kappa, v_list, info = cp.solve_nac((0, 1), tol=1e-10, max_iter=500)
    Lvec_mine = nac_obj.pack_uniq_var(kappa, v_list)

    def fake_solve(*args, **kwargs):
        bvec = nac_obj.get_wfn_response(state=(0, 1))
        Aop, Adiag = nac_obj.get_Aop_Adiag(state=(0, 1))
        return True, Lvec_mine, bvec, Aop, Adiag
    nac_obj.solve_lagrange = fake_solve
    de_mine = nac_obj.kernel(state=(0, 1))

    max_abs_diff = float(np.max(np.abs(de_mine - de_ref)))
    rel_diff = max_abs_diff / max(np.max(np.abs(de_ref)), 1e-30)
    return {"name": "N3_full_nac_matches",
            "de_ref": de_ref.tolist(),
            "de_mine": de_mine.tolist(),
            "Lvec_diff": float(np.linalg.norm(Lvec_mine - Lvec_ref)),
            "max_abs_diff": max_abs_diff,
            "rel_diff": rel_diff,
            "tol_rel": 1e-6,
            "status": "pass" if rel_diff < 1e-6 else "fail"}


def main():
    cases = [test_N1_nac_rhs_nontrivial,
             test_N2_my_nac_lvec_matches_pyscf,
             test_N3_full_nac_matches]
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
    out = {"milestone": "FreitagReiher_Step5b_NAC_baseline_validation",
           "results": results}
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
