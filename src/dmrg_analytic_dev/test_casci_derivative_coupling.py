"""Validation tests for casci_derivative_coupling.py (Milestone 3-CASCI).

Honest scope: this milestone delivers
  - analytic 1e (`<I|dh/dR|J>`) part validated against FD,
  - FD 2e (`<I|dg/dR|J>`) part as the working implementation,
  - hybrid total `casci_dh_dR_numerator` = analytic 1e + FD 2e,
  - pure-FD reference `casci_dh_dR_numerator_full_fd`,
  - overlap-based time-derivative coupling `tau` (includes wavefunction response).

The fully analytic 2e contraction is documented as a follow-up.

Tests on HeH+ / sto-3g CAS(2,2) (avoids g/u and S=0/S=1 selection issues that
make the H2 transitions trivially zero).

Run with:
    python test_casci_derivative_coupling.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from casci_derivative_coupling import (
    casci_dh_dR_numerator,
    casci_dh_dR_numerator_full_fd,
    overlap_based_time_derivative_coupling,
)


def make_mol(d_bohr: float = 1.4):
    return gto.M(atom=f"He 0 0 0; H 0 0 {d_bohr}", basis="sto-3g",
                 charge=1, spin=0, unit="Bohr",
                 symmetry=False, verbose=0)


def run_cas22(mol, nroots: int = 3):
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cas = mcscf.CASCI(mf, 2, 2)
    cas.fix_spin_(ss=0)
    cas.fcisolver.nroots = nroots
    cas.kernel()
    ci_list = cas.ci if isinstance(cas.ci, list) else [cas.ci]
    return cas, ci_list


def _trans_rdm12(ci_bra, ci_ket, ncas, nelec):
    return fci.direct_spin1.trans_rdm12(ci_bra, ci_ket, ncas, nelec)


def test_hybrid_vs_full_fd():
    """Hybrid (analytic 1e + FD 2e) must match pure-FD numerator to FD precision."""
    mol = make_mol()
    cas, ci_list = run_cas22(mol)
    bra, ket = 0, 1
    t1, t2 = _trans_rdm12(ci_list[bra], ci_list[ket], cas.ncas, cas.nelecas)

    nac_hyb = casci_dh_dR_numerator(
        mol, cas.mo_coeff, t1, t2, cas.ncore, cas.ncas, fd_step_bohr=1e-3,
    )
    nac_fd = casci_dh_dR_numerator_full_fd(
        mol, cas.mo_coeff, t1, t2, cas.ncore, cas.ncas, fd_step_bohr=1e-3,
    )
    diff = float(np.linalg.norm(nac_hyb - nac_fd))
    return {
        "name": "C1_hybrid_matches_full_FD",
        "nac_hybrid": nac_hyb.tolist(),
        "nac_full_fd": nac_fd.tolist(),
        "abs_diff": diff,
        "tol": 1e-6,
        "status": "pass" if diff < 1e-6 else "fail",
    }


def test_hermiticity_symmetry():
    """N^x_{IJ} = N^x_{JI} for real CI vectors (Hermiticity of H).

    Note: the **derivative coupling** d_IJ = N_IJ / (E_J - E_I) is antisymmetric,
    but the numerator N is symmetric for real wavefunctions.
    """
    mol = make_mol()
    cas, ci_list = run_cas22(mol)
    t1_01, t2_01 = _trans_rdm12(ci_list[0], ci_list[1], cas.ncas, cas.nelecas)
    t1_10, t2_10 = _trans_rdm12(ci_list[1], ci_list[0], cas.ncas, cas.nelecas)
    nac_01 = casci_dh_dR_numerator(mol, cas.mo_coeff, t1_01, t2_01,
                                   cas.ncore, cas.ncas, fd_step_bohr=1e-3)
    nac_10 = casci_dh_dR_numerator(mol, cas.mo_coeff, t1_10, t2_10,
                                   cas.ncore, cas.ncas, fd_step_bohr=1e-3)
    sym = float(np.linalg.norm(nac_01 - nac_10))

    # Also check that d_IJ = N/(E_J-E_I) is antisymmetric.
    dE = float(cas.e_tot[1] - cas.e_tot[0])
    d_01 = nac_01 / dE
    d_10 = nac_10 / (-dE)
    asym_d = float(np.linalg.norm(d_01 + d_10))

    return {
        "name": "C2_hermiticity_symmetry_and_d_IJ_antisymmetry",
        "nac_01": nac_01.tolist(),
        "nac_10": nac_10.tolist(),
        "sym_residual_of_N": sym,
        "asym_residual_of_d_IJ": asym_d,
        "tol_N_sym": 1e-6,
        "tol_d_asym": 1e-6,
        "status": "pass" if sym < 1e-6 and asym_d < 1e-6 else "fail",
    }


def test_translational_invariance():
    """Sum over atoms of N^x_{IJ} ~ 0 (Hellmann-Feynman trans inv)."""
    mol = make_mol()
    cas, ci_list = run_cas22(mol)
    t1, t2 = _trans_rdm12(ci_list[0], ci_list[1], cas.ncas, cas.nelecas)
    nac = casci_dh_dR_numerator(mol, cas.mo_coeff, t1, t2,
                                cas.ncore, cas.ncas, fd_step_bohr=1e-3)
    trans = float(np.linalg.norm(nac.sum(axis=0)))
    return {
        "name": "C3_translational_invariance",
        "nac": nac.tolist(),
        "trans_residual": trans,
        "norm": float(np.linalg.norm(nac)),
        "tol": 1e-3,
        "status": "pass" if trans < 1e-3 else "fail",
    }


def test_overlap_nac_consistency():
    """Sanity check: overlap-NAC tau ~ N^x / dE in magnitude.

    Loose order-of-magnitude check. Exact equality requires full CASSCF
    response (orbital + CI), which is M3 full / M4 territory.
    """
    mol = make_mol()
    cas, ci_list = run_cas22(mol)
    bra, ket = 0, 1
    t1, t2 = _trans_rdm12(ci_list[bra], ci_list[ket], cas.ncas, cas.nelecas)
    nac_an = casci_dh_dR_numerator(mol, cas.mo_coeff, t1, t2,
                                   cas.ncore, cas.ncas, fd_step_bohr=1e-3)
    dE = float(cas.e_tot[ket] - cas.e_tot[bra])

    def mol_factory(coords_bohr):
        if coords_bohr is None:
            return mol
        return make_mol(d_bohr=float(coords_bohr[1, 2] - coords_bohr[0, 2])) \
            if False else gto.M(
                atom=[("He", tuple(coords_bohr[0])), ("H", tuple(coords_bohr[1]))],
                basis="sto-3g", charge=1, spin=0, unit="Bohr",
                symmetry=False, verbose=0,
            )

    def cas_factory(m):
        return run_cas22(m)

    tau = overlap_based_time_derivative_coupling(
        mol_factory, cas_factory, bra, ket, dx_bohr=1e-3,
    )
    norm_tau_dE = float(np.linalg.norm(tau * dE))
    norm_nac = float(np.linalg.norm(nac_an))
    ratio = norm_nac / max(norm_tau_dE, 1e-12)

    return {
        "name": "C4_overlap_nac_order_of_magnitude",
        "dE_hartree": dE,
        "tau_overlap": tau.tolist(),
        "tau_times_dE": (tau * dE).tolist(),
        "nac_analytic_hybrid": nac_an.tolist(),
        "norm_tau_times_dE": norm_tau_dE,
        "norm_nac": norm_nac,
        "ratio": ratio,
        "tol": "0.01 < ratio < 100",
        "note": "Loose check; exact match requires full CASSCF response.",
        "status": "pass" if 0.01 < ratio < 100 else "fail",
    }


def main():
    cases = [
        test_hybrid_vs_full_fd,
        test_hermiticity_symmetry,
        test_translational_invariance,
        test_overlap_nac_consistency,
    ]
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
    out = {"milestone": "M3_CASCI_derivative_coupling",
           "scope": "analytic 1e + FD 2e (hybrid). Fully analytic 2e is M3-followup.",
           "results": results}
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x)
                                   if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
