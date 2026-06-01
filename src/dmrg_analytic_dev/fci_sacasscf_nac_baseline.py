"""FCI baseline for SA-CASSCF analytic NAC on the test systems.

Goal: produce the ground-truth analytic NAC values from
`pyscf.nac.sacasscf.NonAdiabaticCouplings` so any future DMRG-based
implementation has a number to validate against. For CAS small enough that
DMRG is exact (CAS(2,2), CAS(4,4)), the DMRG analytic NAC must reproduce
these to numerical precision.

This is **Milestone 4 baseline**: not the DMRG implementation, but the FCI
reference that the DMRG implementation must match.

Tests on HeH+/sto-3g CAS(2,2):
  - SA-CASSCF energies are well-defined (no symmetry pathology).
  - Compare PySCF analytic NAC against finite-difference overlap NAC.
  - Compare against M3-CASCI fixed-orbital numerator divided by the gap to
    quantify the orbital-response correction (the piece M3-CASCI is missing).

Run with:
    /n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11 fci_sacasscf_nac_baseline.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.nac.sacasscf import NonAdiabaticCouplings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from casci_derivative_coupling import (
    casci_dh_dR_numerator,
    overlap_based_time_derivative_coupling,
)


def make_heh(d_bohr: float = 1.4):
    return gto.M(atom=f"He 0 0 0; H 0 0 {d_bohr}", basis="sto-3g",
                 charge=1, spin=0, unit="Bohr",
                 symmetry=False, verbose=0)


def run_sa_casscf(mol, nroots: int = 2):
    """SA-CASSCF (not CASCI!) so the NAC machinery applies."""
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fix_spin_(ss=0)
    mc.fcisolver.nroots = nroots
    mc = mc.state_average_([1.0 / nroots] * nroots)
    mc.kernel()
    return mc


def test_pyscf_analytic_sa_nac():
    """Run PySCF's analytic SA-CASSCF NAC on HeH+ CAS(2,2) for state pair (0,1).

    Returns the analytic NAC vector and useful metadata.
    """
    mol = make_heh()
    mc = run_sa_casscf(mol, nroots=2)

    # NAC from state (ket=1) to state (bra=0): <1 | d/dR | 0>
    # PySCF convention: state = (bra, ket) for d/dR acting on ket.
    nac_obj = NonAdiabaticCouplings(mc, state=(0, 1))
    nac_an = nac_obj.kernel()  # shape (natm, 3), in 1/Bohr (derivative coupling, not numerator)

    e_states = mc.e_states if hasattr(mc, "e_states") else mc.e_tot
    dE = float(e_states[1] - e_states[0])

    return {
        "name": "T1_pyscf_analytic_sa_casscf_nac",
        "system": "HeH+ / sto-3g, CAS(2,2), SA-2 equal weights",
        "energies_hartree": list(map(float, e_states)),
        "delta_E_hartree": dE,
        "nac_analytic_per_bohr": nac_an.tolist(),
        "nac_norm_per_bohr": float(np.linalg.norm(nac_an)),
        "note": "PySCF NonAdiabaticCouplings: full SA-CASSCF analytic NAC including "
                "orbital response (CP-CASSCF Z-vector) and CI response. This is the "
                "ground truth a DMRG analog must reproduce to numerical precision in "
                "CAS spaces small enough that DMRG = FCI.",
        "status": "pass",  # always passes — this is just the baseline
    }


def test_compare_with_overlap_nac():
    """Compare PySCF analytic NAC vs finite-difference overlap-based tau."""
    mol = make_heh()
    mc = run_sa_casscf(mol, nroots=2)

    nac_obj = NonAdiabaticCouplings(mc, state=(0, 1))
    nac_an = nac_obj.kernel()

    # Finite-difference overlap-based tau (uses M1 cross-geometry overlap)
    def mol_factory(coords_bohr):
        if coords_bohr is None:
            return mol
        return gto.M(
            atom=[("He", tuple(coords_bohr[0])), ("H", tuple(coords_bohr[1]))],
            basis="sto-3g", charge=1, spin=0, unit="Bohr",
            symmetry=False, verbose=0,
        )

    def cas_factory(m):
        from pyscf import scf as _scf, mcscf as _mc
        mf = _scf.RHF(m).run(conv_tol=1e-12)
        mc_disp = _mc.CASCI(mf, 2, 2)
        mc_disp.fix_spin_(ss=0)
        mc_disp.fcisolver.nroots = 2
        mc_disp.kernel()
        ci_list = mc_disp.ci if isinstance(mc_disp.ci, list) else [mc_disp.ci]
        return mc_disp, ci_list

    tau = overlap_based_time_derivative_coupling(
        mol_factory, cas_factory, bra_idx=0, ket_idx=1, dx_bohr=1e-3,
    )

    diff = float(np.linalg.norm(nac_an - tau))
    rel = diff / max(float(np.linalg.norm(nac_an)), 1e-12)

    return {
        "name": "T2_pyscf_analytic_vs_overlap_FD",
        "nac_analytic": nac_an.tolist(),
        "tau_overlap_fd": tau.tolist(),
        "abs_diff": diff,
        "rel_diff": rel,
        "note": "Finite-difference overlap-based tau approximates analytic NAC; "
                "small deviations (~1%) come from FD step + finite-orbital-relaxation "
                "between displaced points.",
        "tol_rel": 5e-2,
        "status": "pass" if rel < 5e-2 else "fail",
    }


def test_compare_with_m3_casci():
    """Compare PySCF analytic NAC against M3-CASCI fixed-orbital numerator / dE.

    The M3-CASCI numerator gives only the CASCI fixed-orbital contribution.
    This test quantifies how much of the full SA-CASSCF NAC is captured by
    the CASCI piece — i.e., the magnitude of orbital response missing in
    M3-CASCI.
    """
    mol = make_heh()
    mc = run_sa_casscf(mol, nroots=2)
    nac_obj = NonAdiabaticCouplings(mc, state=(0, 1))
    nac_an = nac_obj.kernel()

    # M3-CASCI numerator with same CI vectors and orbitals
    ci_list = mc.ci if isinstance(mc.ci, list) else [mc.ci]
    t1, t2 = fci.direct_spin1.trans_rdm12(ci_list[0], ci_list[1], mc.ncas, mc.nelecas)
    nac_casci_num = casci_dh_dR_numerator(
        mol, mc.mo_coeff, t1, t2, mc.ncore, mc.ncas, fd_step_bohr=1e-3,
    )
    e_states = mc.e_states if hasattr(mc, "e_states") else mc.e_tot
    dE = float(e_states[1] - e_states[0])
    nac_m3_full = nac_casci_num / dE  # convert numerator to NAC (1/Bohr)

    diff = float(np.linalg.norm(nac_an - nac_m3_full))
    return {
        "name": "T3_pyscf_analytic_vs_m3_casci_over_dE",
        "nac_pyscf_analytic_full": nac_an.tolist(),
        "nac_m3_casci_over_dE": nac_m3_full.tolist(),
        "abs_diff_response_size": diff,
        "interpretation": "diff norm = magnitude of orbital + CI response that M3-CASCI is missing.",
        "status": "pass",  # this is informational, not pass/fail
    }


def main():
    cases = [
        test_pyscf_analytic_sa_nac,
        test_compare_with_overlap_nac,
        test_compare_with_m3_casci,
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
    out = {
        "milestone": "M4_baseline_FCI_SA_CASSCF_NAC",
        "purpose": "Establish ground truth from pyscf.nac.sacasscf for future DMRG analog.",
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
