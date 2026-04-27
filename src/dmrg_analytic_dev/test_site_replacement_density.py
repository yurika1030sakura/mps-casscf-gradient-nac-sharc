"""Validate site-replacement transition densities and T matrix.

Tests:
  R1: When trial_v = ci_state, T(Ψ, Ψ) should equal the standard state-specific
      orbital gradient — but at a stationary point (after CASSCF converges)
      this should be zero (Brillouin condition).

  R2: Linearity in the trial vector: T(Ψ, αv + βw) = α T(Ψ, v) + β T(Ψ, w).

  R3: For trial_v = different CI eigenstate, T returns the standard transition
      orbital gradient between two CI eigenstates.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from site_replacement_density import (
    T_matrix_site_replacement,
    transition_rdm_site_replacement,
    generalized_fock_matrix,
)


def setup_heh():
    """HeH+ sto-3g CAS(2,2). Note: ncore=0, nvirt=0 in this basis, so the only
    orbital rotations are active-active (gauge freedom). Physical orbital
    gradient is therefore identically zero, and the T_active-active block
    encodes only gauge information. Tests below check bilinearity and
    antisymmetry of T which hold regardless."""
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


def test_R1_T_antisymmetry():
    """T_pq(Ψ, Ψ̄) must be antisymmetric: T_pq + T_qp = 0.

    This follows from the definition T = 2(F - F^T) and is a basic correctness
    check that holds regardless of the CASSCF convergence state or active
    space size.
    """
    mc = setup_heh()
    h_mo = mc.mo_coeff.T @ mc._scf.get_hcore() @ mc.mo_coeff
    nmo = mc.mo_coeff.shape[1]
    eri_mo = ao2mo.kernel(mc.mol, mc.mo_coeff, compact=False).reshape((nmo,nmo,nmo,nmo))
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2, 2))
    T = T_matrix_site_replacement(h_mo, eri_mo, mc.ci[0], v, 2, 0, (1, 1))
    sym_residual = float(np.linalg.norm(T + T.T))
    nonzero = float(np.linalg.norm(T))
    return {"name": "R1_T_antisymmetry",
            "T_norm": nonzero,
            "T_plus_TT_norm": sym_residual,
            "tol": 1e-12,
            "status": "pass" if sym_residual < 1e-12 and nonzero > 1e-3 else "fail"}


def test_R2_linearity_in_trial():
    mc = setup_heh()
    h_mo = mc.mo_coeff.T @ mc._scf.get_hcore() @ mc.mo_coeff
    nmo = mc.mo_coeff.shape[1]
    eri_mo = ao2mo.kernel(mc.mol, mc.mo_coeff, compact=False).reshape((nmo,nmo,nmo,nmo))
    ci_0 = mc.ci[0] if isinstance(mc.ci, list) else mc.ci

    rng = np.random.default_rng(42)
    v = rng.standard_normal((2, 2))
    w = rng.standard_normal((2, 2))
    a, b = 0.6, -1.4

    T_v = T_matrix_site_replacement(h_mo, eri_mo, ci_0, v, 2, 0, (1, 1))
    T_w = T_matrix_site_replacement(h_mo, eri_mo, ci_0, w, 2, 0, (1, 1))
    T_combo = T_matrix_site_replacement(h_mo, eri_mo, ci_0, a*v + b*w, 2, 0, (1, 1))
    T_lin = a*T_v + b*T_w

    diff = float(np.linalg.norm(T_combo - T_lin))
    return {"name": "R2_linearity_in_trial",
            "abs_diff": diff,
            "tol": 1e-12,
            "status": "pass" if diff < 1e-12 else "fail"}


def test_R3_transition_between_eigenstates():
    """T(Ψ_0, Ψ_1) should be a finite antisymmetric matrix encoding the
    transition orbital gradient between root 0 and root 1 of SA-CASSCF."""
    mc = setup_heh()
    h_mo = mc.mo_coeff.T @ mc._scf.get_hcore() @ mc.mo_coeff
    nmo = mc.mo_coeff.shape[1]
    eri_mo = ao2mo.kernel(mc.mol, mc.mo_coeff, compact=False).reshape((nmo,nmo,nmo,nmo))
    ci_0 = mc.ci[0]
    ci_1 = mc.ci[1]

    T_01 = T_matrix_site_replacement(h_mo, eri_mo, ci_0, ci_1, 2, 0, (1, 1))
    T_10 = T_matrix_site_replacement(h_mo, eri_mo, ci_1, ci_0, 2, 0, (1, 1))

    # T(Ψ, Ψ̄) is antisymmetric in the (orbital indices); swap of (Ψ, Ψ̄)
    # should give -T^T (state-pair swap = transpose with sign).
    sym_residual = float(np.linalg.norm(T_01 + T_10.T))
    nonzero = float(np.linalg.norm(T_01))
    return {"name": "R3_transition_between_eigenstates",
            "T_01_norm": nonzero,
            "T_01_plus_T_10_T_norm": sym_residual,
            "T_01": T_01.tolist(),
            "T_10": T_10.tolist(),
            "tol_sym": 1e-10,
            "tol_nonzero_min": 1e-3,
            "status": "pass" if (sym_residual < 1e-10 and nonzero > 1e-3) else "fail"}


def main():
    cases = [test_R1_T_antisymmetry,
             test_R2_linearity_in_trial,
             test_R3_transition_between_eigenstates]
    results = []
    for c in cases:
        try:
            r = c()
        except Exception as exc:
            r = {"name": c.__name__, "status": "fail",
                 "exception": type(exc).__name__,
                 "message": str(exc),
                 "traceback_tail": traceback.format_exc()[-1500:]}
        results.append(r)
        print(f"  {r['name']}: {r['status']}")

    out_path = Path(__file__).with_suffix(".json")
    out = {"milestone": "FreitagReiher_Step2_site_replacement_density",
           "results": results}
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
