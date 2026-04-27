"""Validate the single-site sigma vector via FCI fallback.

The Freitag-Reiher CP-DMRG-CASSCF algorithm needs σ = H · v on the chosen
linear-response site. For validation against the FCI baseline (CAS(2,2)/(4,4)),
the single site IS the entire FCI tensor — DMRG hasn't compressed anything
because the active space is small enough.

This test verifies:

  S1: For a converged FCI eigenstate |Ψ⟩ with energy E,
      σ(M_Ψ) = E · M_Ψ  (eigenvalue equation)

  S2: σ is a linear operator: σ(αv + βw) = α σ(v) + β σ(w).

  S3: σ is symmetric: <v | σ(w)> = <w | σ(v)>  (Hermitian H).

These checks confirm that the FCI sigma vector path is correct and ready to
plug into the CP-CASSCF PCG solver.
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

from single_site_sigma import single_site_sigma_fci_fallback


def setup_h2_fci(d_bohr=1.4):
    """HeH+ for nontrivial transitions; same setup as M3-CASCI tests."""
    mol = gto.M(atom=f"He 0 0 0; H 0 0 {d_bohr}", basis="sto-3g",
                charge=1, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cas = mcscf.CASCI(mf, 2, 2)
    cas.fix_spin_(ss=0)
    cas.fcisolver.nroots = 2
    cas.kernel()

    mo = mf.mo_coeff
    mo_act = mo[:, :2]
    h1_act = mo_act.T @ mf.get_hcore() @ mo_act
    h2_act = ao2mo.kernel(mol, mo_act, compact=False).reshape(2, 2, 2, 2)

    e_states = list(map(float, cas.e_tot))
    return cas, h1_act, h2_act, e_states


def test_eigenvector_identity():
    """For a CASCI eigenstate ci with energy E (incl. ecore + nuc),
    σ(ci) = (E - E_core_total) · ci where E_core_total = ecore_active + nuc_rep.

    For our setup ecore=0, nuc_rep is in mol, and absorb_h1e/contract_2e operate
    on active integrals so we expect σ(ci) = E_active · ci where
    E_active = E_total - nuc_repulsion."""
    cas, h1, h2, e_tot = setup_h2_fci()
    nuc = cas.mol.energy_nuc()
    ci_0 = cas.ci[0]
    ci_1 = cas.ci[1]
    sigma_0 = single_site_sigma_fci_fallback(h1, h2, ci_0, 2, (1, 1))
    sigma_1 = single_site_sigma_fci_fallback(h1, h2, ci_1, 2, (1, 1))

    # Active-space energy (no nuclear repulsion since ecore=0 in our integrals)
    E0_act = e_tot[0] - nuc
    E1_act = e_tot[1] - nuc

    diff_0 = float(np.linalg.norm(sigma_0 - E0_act * ci_0))
    diff_1 = float(np.linalg.norm(sigma_1 - E1_act * ci_1))

    return {"name": "S1_eigenvector_identity",
            "E0_active": E0_act, "E1_active": E1_act,
            "diff_root0": diff_0, "diff_root1": diff_1,
            "tol": 1e-10,
            "status": "pass" if max(diff_0, diff_1) < 1e-10 else "fail"}


def test_linearity():
    cas, h1, h2, _ = setup_h2_fci()
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2, 2))
    w = rng.standard_normal((2, 2))
    a, b = 0.7, -1.3
    sv = single_site_sigma_fci_fallback(h1, h2, v, 2, (1, 1))
    sw = single_site_sigma_fci_fallback(h1, h2, w, 2, (1, 1))
    s_combo = single_site_sigma_fci_fallback(h1, h2, a*v + b*w, 2, (1, 1))
    diff = float(np.linalg.norm(s_combo - (a*sv + b*sw)))
    return {"name": "S2_linearity",
            "abs_diff": diff,
            "tol": 1e-12,
            "status": "pass" if diff < 1e-12 else "fail"}


def test_hermiticity():
    cas, h1, h2, _ = setup_h2_fci()
    rng = np.random.default_rng(1)
    v = rng.standard_normal((2, 2))
    w = rng.standard_normal((2, 2))
    sv = single_site_sigma_fci_fallback(h1, h2, v, 2, (1, 1))
    sw = single_site_sigma_fci_fallback(h1, h2, w, 2, (1, 1))
    vsw = float(np.tensordot(v, sw, axes=([0, 1], [0, 1])))
    wsv = float(np.tensordot(w, sv, axes=([0, 1], [0, 1])))
    diff = abs(vsw - wsv)
    return {"name": "S3_hermiticity",
            "<v|sw>": vsw, "<w|sv>": wsv, "abs_diff": diff,
            "tol": 1e-12,
            "status": "pass" if diff < 1e-12 else "fail"}


def main():
    cases = [test_eigenvector_identity, test_linearity, test_hermiticity]
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
    out = {"milestone": "FreitagReiher_Step1_single_site_sigma_FCI_fallback",
           "purpose": "Validate FCI fallback used for CAS(2,2)/(4,4) regime in CP-DMRG-CASSCF.",
           "results": results}
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
