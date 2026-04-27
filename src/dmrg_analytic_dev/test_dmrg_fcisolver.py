"""Validate `MPSAsFCISolver` end-to-end against the FCI baseline.

Test structure:
  E1: Energies from DMRG-as-FCI match PySCF FCI for HeH+ CAS(2,2).
  E2: 1- and 2-RDMs from pyblock2 match PySCF FCI RDMs.
  E3: Transition RDMs from pyblock2 match PySCF FCI transition RDMs.
  E4: Plug `MPSAsFCISolver` into PySCF SA-CASSCF + nac.sacasscf and check the
      analytic NAC equals the all-FCI baseline to numerical precision.

Test E4 is the meaningful CP-DMRG-CASSCF analytic NAC validation: PySCF's
existing CP-CASSCF response solver runs end-to-end with DMRG providing the
underlying eigensolver and RDMs.

Run with:
    /n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11 test_dmrg_fcisolver.py
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

from dmrg_fcisolver import MPSAsFCISolver


def make_heh(d_bohr: float = 1.4):
    return gto.M(atom=f"He 0 0 0; H 0 0 {d_bohr}", basis="sto-3g",
                 charge=1, spin=0, unit="Bohr",
                 symmetry=False, verbose=0)


def test_e1_dmrg_energies_match_fci():
    mol = make_heh()
    mf = scf.RHF(mol).run(conv_tol=1e-12)

    # FCI reference
    cas_fci = mcscf.CASCI(mf, 2, 2)
    cas_fci.fix_spin_(ss=0)
    cas_fci.fcisolver.nroots = 2
    cas_fci.kernel()
    e_fci = list(map(float, cas_fci.e_tot))

    # DMRG
    cas_dmrg = mcscf.CASCI(mf, 2, 2)
    cas_dmrg.fcisolver = MPSAsFCISolver(mol)
    cas_dmrg.fcisolver.nroots = 2
    cas_dmrg.kernel()
    e_dmrg = list(map(float, cas_dmrg.e_tot))

    diff = float(max(abs(e_fci[i] - e_dmrg[i]) for i in range(2)))
    return {
        "name": "E1_energies_match",
        "fci": e_fci, "dmrg": e_dmrg,
        "max_abs_diff": diff,
        "tol": 1e-7,
        "status": "pass" if diff < 1e-7 else "fail",
    }


def test_e2_state_rdms_match():
    mol = make_heh()
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cas_fci = mcscf.CASCI(mf, 2, 2)
    cas_fci.fix_spin_(ss=0)
    cas_fci.fcisolver.nroots = 2
    cas_fci.kernel()
    ci_fci_0 = cas_fci.ci[0]
    dm1_fci, dm2_fci = fci.direct_spin1.make_rdm12(ci_fci_0, 2, (1, 1))

    cas_dmrg = mcscf.CASCI(mf, 2, 2)
    cas_dmrg.fcisolver = MPSAsFCISolver(mol)
    cas_dmrg.fcisolver.nroots = 2
    cas_dmrg.kernel()
    ci_dmrg_0 = cas_dmrg.ci[0]
    dm1_dmrg, dm2_dmrg = cas_dmrg.fcisolver.make_rdm12(ci_dmrg_0, 2, (1, 1))

    d1 = float(np.linalg.norm(np.asarray(dm1_fci) - np.asarray(dm1_dmrg)))
    d2 = float(np.linalg.norm(np.asarray(dm2_fci) - np.asarray(dm2_dmrg)))
    return {
        "name": "E2_state_rdms_match",
        "dm1_diff": d1, "dm2_diff": d2,
        "tol": 1e-7,
        "status": "pass" if max(d1, d2) < 1e-7 else "fail",
    }


def test_e3_trans_rdms_match():
    mol = make_heh()
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cas_fci = mcscf.CASCI(mf, 2, 2)
    cas_fci.fix_spin_(ss=0)
    cas_fci.fcisolver.nroots = 2
    cas_fci.kernel()
    t1_fci, t2_fci = fci.direct_spin1.trans_rdm12(cas_fci.ci[0], cas_fci.ci[1], 2, (1, 1))

    cas_dmrg = mcscf.CASCI(mf, 2, 2)
    cas_dmrg.fcisolver = MPSAsFCISolver(mol)
    cas_dmrg.fcisolver.nroots = 2
    cas_dmrg.kernel()
    t1_dmrg, t2_dmrg = cas_dmrg.fcisolver.trans_rdm12(
        cas_dmrg.ci[0], cas_dmrg.ci[1], 2, (1, 1),
    )

    d1 = float(np.linalg.norm(np.asarray(t1_fci) - np.asarray(t1_dmrg)))
    d2 = float(np.linalg.norm(np.asarray(t2_fci) - np.asarray(t2_dmrg)))
    # Allow sign difference (trans rdm has gauge freedom for degenerate-state pairs)
    d1_alt = float(np.linalg.norm(np.asarray(t1_fci) + np.asarray(t1_dmrg)))
    d2_alt = float(np.linalg.norm(np.asarray(t2_fci) + np.asarray(t2_dmrg)))
    sign_consistent = (d1 < d1_alt) == (d2 < d2_alt)
    use_d1 = min(d1, d1_alt); use_d2 = min(d2, d2_alt)
    return {
        "name": "E3_trans_rdms_match",
        "t1_diff": d1, "t1_diff_neg": d1_alt,
        "t2_diff": d2, "t2_diff_neg": d2_alt,
        "sign_consistent": sign_consistent,
        "max_after_phase": max(use_d1, use_d2),
        "tol": 1e-6,
        "status": "pass" if max(use_d1, use_d2) < 1e-6 and sign_consistent else "fail",
    }


def test_e4_analytic_nac_pipeline():
    """End-to-end analytic SA-CASSCF NAC using DMRG fcisolver.

    This is the headline test: PySCF's CP-CASSCF response solver runs with
    DMRG providing the eigensolver and RDMs. The result must match the
    all-FCI analytic NAC baseline to numerical precision.
    """
    mol = make_heh()
    mf_fci = scf.RHF(mol).run(conv_tol=1e-12)

    # FCI baseline
    mc_fci = mcscf.CASSCF(mf_fci, 2, 2)
    mc_fci.fix_spin_(ss=0)
    mc_fci.fcisolver.nroots = 2
    mc_fci = mc_fci.state_average_([0.5, 0.5])
    mc_fci.kernel()
    nac_fci = NonAdiabaticCouplings(mc_fci, state=(0, 1)).kernel()

    # DMRG version. We deliberately do NOT call fix_spin_ here: pyblock2 SU2
    # mode already targets the singlet sector, so adding a spin penalty on top
    # creates two-state-averaged solver layering issues during CASSCF macro
    # iterations.
    mf_dmrg = scf.RHF(mol).run(conv_tol=1e-12)
    mc_dmrg = mcscf.CASSCF(mf_dmrg, 2, 2)
    mc_dmrg.fcisolver = MPSAsFCISolver(mol)
    mc_dmrg.fcisolver.nroots = 2
    mc_dmrg = mc_dmrg.state_average_([0.5, 0.5])
    mc_dmrg.kernel()
    nac_dmrg = NonAdiabaticCouplings(mc_dmrg, state=(0, 1)).kernel()

    # Allow overall sign flip (NAC has gauge phase from CI vector sign)
    diff_pos = float(np.linalg.norm(nac_fci - nac_dmrg))
    diff_neg = float(np.linalg.norm(nac_fci + nac_dmrg))
    diff = min(diff_pos, diff_neg)
    return {
        "name": "E4_analytic_nac_pipeline",
        "nac_fci": nac_fci.tolist(),
        "nac_dmrg": nac_dmrg.tolist(),
        "diff_pos": diff_pos,
        "diff_neg": diff_neg,
        "min_diff_after_phase": diff,
        "tol": 1e-5,
        "status": "pass" if diff < 1e-5 else "fail",
    }


def main():
    cases = [
        test_e1_dmrg_energies_match_fci,
        test_e2_state_rdms_match,
        test_e3_trans_rdms_match,
        test_e4_analytic_nac_pipeline,
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
        "milestone": "M4_dmrg_fcisolver_validation",
        "purpose": "Validate MPSAsFCISolver wrapper end-to-end with PySCF SA-CASSCF NAC.",
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
