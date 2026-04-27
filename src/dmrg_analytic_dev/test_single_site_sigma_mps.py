"""Validate the MPS-native single-site sigma vector.

For CAS(2,2) singlet (where DMRG = FCI exactly):

  M1: For a converged DMRG eigenstate |Ψ⟩ of energy E (active-space MPO,
      ecore=0), σ = MPO·|Ψ⟩ should satisfy ‖σ‖ = |E_active|, and the
      expectation ⟨Ψ|σ⟩ should equal E_active.

  M2: Cross-check the FCI representation of σ_mps against the FCI fallback
      σ_fci. Both should produce the same FCI ndarray to bond-truncation
      precision.

The MPS-native path uses ``driver.multiply``: |bra⟩ = MPO|ket⟩ via fitting
sweeps. This is the *production* primitive that scales as O(M^3 ncas) instead
of O(FCI dim).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dmrg_fcisolver import MPSAsFCISolver, _csf_to_fci22_singlet
from single_site_sigma import (
    single_site_sigma_fci_fallback,
    single_site_sigma_mps_native,
    site_tensor_to_fci,
)


def setup_heh_dmrg(d_bohr: float = 1.4, bond_dim: int = 64):
    """Build HeH+ CAS(2,2) and run DMRG via MPSAsFCISolver. Return the solver
    + active-space integrals (with ecore=0, nuc included implicitly via mol)."""
    mol = gto.M(atom=f"He 0 0 0; H 0 0 {d_bohr}", basis="sto-3g",
                charge=1, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)

    cas = mcscf.CASCI(mf, 2, 2)
    cas.fcisolver = MPSAsFCISolver(mol, bond_dim=bond_dim, n_sweeps=20)
    cas.fcisolver.nroots = 2
    cas.kernel()

    mo_act = mf.mo_coeff[:, :2]
    h1_act = mo_act.T @ mf.get_hcore() @ mo_act
    eri_full = ao2mo.kernel(mol, mo_act, compact=False).reshape(2, 2, 2, 2)

    nuc = mol.energy_nuc()
    return cas, h1_act, eri_full, nuc


def test_m1_eigenstate_norm_and_expectation():
    """For a DMRG eigenstate |Ψ⟩ (singlet) we have

        σ = MPO|Ψ⟩
        ‖σ‖ = |E_full|     where E_full = E_total = E_active + nuc (ecore=0)
                            since DMRGDriver builds MPO with ecore included.
        ⟨Ψ|σ⟩ = E_full

    We use the driver's own MPO (which carries ecore from kernel()), so the
    MPS expectation includes the nuclear repulsion.
    """
    cas, h1, h2, nuc = setup_heh_dmrg()
    solver: MPSAsFCISolver = cas.fcisolver
    driver = solver._driver
    kets = solver._kets
    e_states = solver.e_states

    # Build a fresh MPO matching the integrals stored in MPSAsFCISolver.kernel().
    # MPSAsFCISolver uses `ecore=float(ecore)` where ecore came in from
    # CASCI.kernel via self._scf.energy_nuc(). PySCF passes ecore=ecore_ext
    # through; the driver MPO in the solver was built with the *same* h1/h2.
    # So the MPO eigenvalue is E_total = e_states[i].

    # We need access to the same MPO. The solver doesn't keep it, so we
    # rebuild it consistent with kernel(): h1 = MO h1, eri = MO eri (compact=False),
    # ecore = nuc.
    mpo = driver.get_qc_mpo(h1, h2, ecore=float(nuc), iprint=0)

    results = []
    for i, (mps_i, E_i) in enumerate(zip(kets, e_states)):
        sigma_mps, sigma_norm = single_site_sigma_mps_native(
            driver, mpo, mps_i, out_tag=f"SIGMA-{i}",
            n_sweeps=8, tol=1e-12, iprint=0,
        )
        # Expectation <psi|sigma>: equivalent to <psi|MPO|psi> = E.
        expect = float(driver.expectation(mps_i, mpo, mps_i, iprint=0))
        # ‖sigma‖ from the driver.
        norm_check = float(driver.expectation(sigma_mps, driver.get_identity_mpo(), sigma_mps, iprint=0)) ** 0.5

        diff_norm = abs(sigma_norm - abs(E_i))
        diff_expect = abs(expect - E_i)

        results.append({
            "root": i,
            "E_dmrg": float(E_i),
            "sigma_mps_norm": sigma_norm,
            "norm_via_overlap": norm_check,
            "psi_MPO_psi_expectation": expect,
            "abs(E_dmrg - sigma_norm)": diff_norm,
            "abs(E_dmrg - <psi|MPO|psi>)": diff_expect,
        })

    max_diff = max(max(r["abs(E_dmrg - sigma_norm)"], r["abs(E_dmrg - <psi|MPO|psi>)"])
                   for r in results)
    return {
        "name": "M1_eigenstate_norm_and_expectation",
        "results": results,
        "max_abs_diff": max_diff,
        "tol": 1e-6,
        "status": "pass" if max_diff < 1e-6 else "fail",
    }


def test_m2_mps_sigma_matches_fci_fallback():
    """Compare σ_mps converted to FCI ndarray against σ_fci.

    For HeH+ CAS(2,2) DMRG = FCI, so σ_mps mapped to FCI should equal σ_fci
    up to (a) bond-fit tolerance and (b) overall sign/phase from CSF → FCI
    conversion.
    """
    cas, h1, h2, nuc = setup_heh_dmrg()
    solver: MPSAsFCISolver = cas.fcisolver
    driver = solver._driver

    # Build the *active-only* MPO with ecore=0 so σ_mps in FCI form is the
    # same operator as single_site_sigma_fci_fallback (which never sees nuc).
    mpo_act = driver.get_qc_mpo(h1, h2, ecore=0.0, iprint=0)

    results = []
    for i, mps_i in enumerate(solver._kets):
        # σ_mps from MPS multiply
        sigma_mps, sigma_norm = single_site_sigma_mps_native(
            driver, mpo_act, mps_i, out_tag=f"SIGMA-ACT-{i}",
            n_sweeps=12, tol=1e-12, iprint=0,
        )
        # Convert to FCI
        sigma_fci_from_mps = site_tensor_to_fci(driver, sigma_mps, 2, (1, 1))

        # Reference σ_fci from the FCI fallback acting on the same |Ψ⟩
        ci_i = cas.ci[i]
        sigma_fci_direct = single_site_sigma_fci_fallback(h1, h2, ci_i, 2, (1, 1))

        # Resolve overall-sign ambiguity from CSF→FCI mapping
        # (block2 returns coefs with arbitrary global phase).
        # We compare two normalized vectors up to ±1.
        a = sigma_fci_from_mps.reshape(-1)
        b = sigma_fci_direct.reshape(-1)
        if np.linalg.norm(a) > 1e-12 and np.linalg.norm(b) > 1e-12:
            sign = np.sign(np.dot(a, b))
            if sign == 0:
                sign = 1.0
        else:
            sign = 1.0
        diff = float(np.linalg.norm(sigma_fci_from_mps - sign * sigma_fci_direct))

        results.append({
            "root": i,
            "sign": float(sign),
            "‖sigma_fci_direct‖": float(np.linalg.norm(b)),
            "‖sigma_fci_from_mps‖": float(np.linalg.norm(a)),
            "diff_after_sign_align": diff,
        })

    max_diff = max(r["diff_after_sign_align"] for r in results)
    return {
        "name": "M2_mps_sigma_matches_fci_fallback",
        "results": results,
        "max_abs_diff": max_diff,
        "tol": 1e-5,
        "status": "pass" if max_diff < 1e-5 else "fail",
    }


def main():
    cases = [test_m1_eigenstate_norm_and_expectation,
             test_m2_mps_sigma_matches_fci_fallback]
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
    out = {
        "milestone": "Step6.1_single_site_sigma_MPS_native",
        "purpose": "Validate that driver.multiply(bra, MPO, ket) gives σ = MPO·|ψ⟩ in MPS form, matching FCI fallback.",
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
