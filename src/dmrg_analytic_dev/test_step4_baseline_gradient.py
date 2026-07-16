"""Step 4: validate full GMRES-solved gradient against pyscf.grad.sacasscf.

Workflow:
  G1. Set up HeH+/3-21G CAS(2,2) SA(2). RHS is non-trivial (state-specific
      orbital gradient is nonzero by atomic asymmetry).
  G2. Run pyscf.grad.sacasscf.Gradients(mc).kernel(state=0) for the reference
      nuclear gradient AND extract the internal Lagrange solution Lvec.
  G3. Run our CPCASSCFResponseFCI.solve(state=0) to get (κ̄, ṽ̄).
  G4. Compare flattened Lvec from (3) to pyscf's solution from (2). Should
      match to GMRES tolerance.
  G5. Use pyscf.grad.sacasscf's get_LdotJnuc + get_ham_response to assemble
      a nuclear gradient using OUR (κ̄, ṽ̄). Compare to the reference from (2).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.grad import sacasscf as sacasscf_grad

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


def run_pyscf_reference(mc, state=0):
    """Run pyscf.grad.sacasscf and capture both the final gradient and
    the internal Lagrange multiplier vector via monkey-patching."""
    grad = sacasscf_grad.Gradients(mc)
    captured = {}
    # Hook into Lagrange callback to capture the converged Lvec
    orig_get_callback = grad.get_lagrange_callback

    def my_get_callback(Lvec_last, itvec, geff_op):
        cb = orig_get_callback(Lvec_last, itvec, geff_op)
        def wrapped(x):
            cb(x)
            captured["Lvec_last"] = np.asarray(x).copy()
        return wrapped
    grad.get_lagrange_callback = my_get_callback

    de = grad.kernel(state=state)
    return de, captured.get("Lvec_last"), grad


def test_G1_rhs_is_nontrivial():
    mc = setup_heh_321g()
    cp = CPCASSCFResponseFCI(mc)
    rhs_O, rhs_C = cp.build_rhs(state=0)
    rhs_flat = cp._flatten(rhs_O, rhs_C)
    return {"name": "G1_rhs_is_nontrivial",
            "rhs_O_norm": float(np.linalg.norm(rhs_O)),
            "rhs_flat_norm": float(np.linalg.norm(rhs_flat)),
            "tol_min": 1e-3,
            "status": "pass" if np.linalg.norm(rhs_flat) > 1e-3 else "fail"}


def test_G2_my_lvec_matches_pyscf_lvec():
    """Both my solver and pyscf use newton_casscf for the matvec, so up to
    GMRES tolerance and RHS sign convention they should produce the same Lvec."""
    mc = setup_heh_321g()
    de_ref, Lvec_ref, grad_obj = run_pyscf_reference(mc, state=0)

    cp = CPCASSCFResponseFCI(mc)
    kappa, v_list, info = cp.solve(state=0, tol=1e-10, max_iter=500)
    Lvec_mine = grad_obj.pack_uniq_var(kappa, v_list)

    # Test both signs (the pyscf RHS sign convention may differ)
    diff_pos = float(np.linalg.norm(Lvec_mine - Lvec_ref))
    diff_neg = float(np.linalg.norm(Lvec_mine + Lvec_ref))
    diff_min = min(diff_pos, diff_neg)
    sign = "+" if diff_pos < diff_neg else "-"
    return {"name": "G2_lvec_matches_pyscf",
            "info": int(info),
            "Lvec_pyscf_norm": float(np.linalg.norm(Lvec_ref)),
            "Lvec_mine_norm": float(np.linalg.norm(Lvec_mine)),
            "diff_pos": diff_pos, "diff_neg": diff_neg,
            "matched_sign": sign,
            "diff_min": diff_min,
            "tol": 1e-6,
            "status": "pass" if diff_min < 1e-6 else "fail"}


def test_G3_full_gradient_matches():
    """Compare full nuclear gradient: drive pyscf.grad.sacasscf with my Lvec
    by replacing grad_obj.Lvec, then re-run the assembly path that's already
    been exercised internally during kernel().
    """
    mc = setup_heh_321g()
    de_ref, Lvec_ref, grad_obj = run_pyscf_reference(mc, state=0)

    cp = CPCASSCFResponseFCI(mc)
    kappa, v_list, info = cp.solve(state=0, tol=1e-10, max_iter=500)
    Lvec_mine = grad_obj.pack_uniq_var(kappa, v_list)

    # Lvec match means de_mine = de_ref exactly. Verify by re-running
    # pyscf's kernel after monkey-patching solve_lagrange to return Lvec_mine.
    def fake_solve(*args, **kwargs):
        # replicate signature: (converged, Lvec, bvec, Aop, Adiag)
        bvec = grad_obj.get_wfn_response(state=0)
        Aop, Adiag = grad_obj.get_Aop_Adiag(state=0)
        return True, Lvec_mine, bvec, Aop, Adiag
    grad_obj.solve_lagrange = fake_solve
    de_mine = grad_obj.kernel(state=0)

    max_abs_diff = float(np.max(np.abs(de_mine - de_ref)))
    rel_diff = max_abs_diff / max(np.max(np.abs(de_ref)), 1e-30)
    return {"name": "G3_full_gradient_matches",
            "de_ref": de_ref.tolist(),
            "de_mine": de_mine.tolist(),
            "Lvec_diff": float(np.linalg.norm(Lvec_mine - Lvec_ref)),
            "max_abs_diff": max_abs_diff,
            "rel_diff": rel_diff,
            "tol_rel": 1e-6,
            "status": "pass" if rel_diff < 1e-6 else "fail"}


def main():
    cases = [test_G1_rhs_is_nontrivial,
             test_G2_my_lvec_matches_pyscf_lvec,
             test_G3_full_gradient_matches]
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
    out = {"milestone": "FreitagReiher_Step4_baseline_gradient_validation",
           "results": results}
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
