"""Step 6.3 (minimal): integration test that exercises BOTH MPS primitives
(`single_site_sigma_mps_native` and `T_matrix_site_replacement_mps`) within
the same workflow, against the FCI-fallback FR primitives that the validated
freitag_reiher backend in `cp_casscf_response.py` uses.

Setup: HeH+/sto-3g CAS(2,2) SA(2). DMRG = FCI in this regime, so the MPS
primitives should reproduce the FCI primitives' outputs to bond-truncation
precision (machine ε on this CAS).

Tests:

  K1: σ_MPS(state, MPS) — converted to FCI ndarray — matches σ_FCI(state, ci)
      element-wise on each SA root, for the active-only Hamiltonian used in
      the FR backend's CC block (`H_CC_apply` line 263 in cp_casscf_response).

  K2: T_MPS(Ψ, ṽ) matches T_FCI(Ψ, ṽ) for ṽ drawn as a random CI tensor on
      the same active space — this exercises the same call path that the FR
      backend uses in `H_OC_apply` and `H_CO_apply` for the orbital-CI
      coupling.

  K3 [optional]: build the Hessian-vector products κ → H_OC κ and ṽ → H_OC ṽ
      using the MPS primitives where the FR backend would use the FCI
      primitives, and verify element-wise equivalence with the FR backend's
      output on a random vector.

Note: We do NOT modify the FR backend. We rebuild the H_OC/H_CO contraction
locally using both primitives and check they agree on the same inputs.
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

from dmrg_fcisolver import MPSAsFCISolver
from single_site_sigma import (
    single_site_sigma_fci_fallback,
    single_site_sigma_mps_native,
    site_tensor_to_fci,
)
from site_replacement_density import (
    T_matrix_site_replacement,
    T_matrix_site_replacement_mps,
)


def setup_heh_dmrg(d_bohr: float = 1.4, bond_dim: int = 64):
    mol = gto.M(atom=f"He 0 0 0; H 0 0 {d_bohr}", basis="sto-3g",
                charge=1, spin=0, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    cas = mcscf.CASCI(mf, 2, 2)
    cas.fcisolver = MPSAsFCISolver(mol, bond_dim=bond_dim, n_sweeps=20)
    cas.fcisolver.nroots = 2
    cas.kernel()

    nmo = mf.mo_coeff.shape[1]
    h_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)
    h1_act = h_mo[:2, :2]
    eri_act = eri_mo[:2, :2, :2, :2]
    return cas, mf, h_mo, eri_mo, h1_act, eri_act


def _resolve_sign(a, b):
    fa, fb = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if np.linalg.norm(fa) < 1e-14 or np.linalg.norm(fb) < 1e-14:
        return 1.0
    return float(np.sign(np.dot(fa, fb)) or 1.0)


def test_k1_sigma_primitive_integration():
    """For each SA root i: σ_FCI(ci_i) should equal σ_MPS(mps_i) (FCI form)
    to machine precision. This is the call signature used by H_CC_apply line
    263.
    """
    cas, mf, h_mo, eri_mo, h1_act, eri_act = setup_heh_dmrg()
    solver: MPSAsFCISolver = cas.fcisolver
    driver = solver._driver

    # Active-only MPO (ecore=0) — same operator the FR backend's H_CC uses.
    mpo_act = driver.get_qc_mpo(h1_act, eri_act, ecore=0.0, iprint=0)

    diffs = []
    for i in range(2):
        # FCI side
        sig_fci = single_site_sigma_fci_fallback(
            h1_act, eri_act, cas.ci[i], 2, (1, 1),
        )
        # MPS side
        sig_mps_obj, _ = single_site_sigma_mps_native(
            driver, mpo_act, solver._kets[i],
            out_tag=f"K1-SIG-{i}", n_sweeps=12, tol=1e-12, iprint=0,
        )
        sig_mps_fci = site_tensor_to_fci(driver, sig_mps_obj, 2, (1, 1))
        sign = _resolve_sign(sig_mps_fci, sig_fci)
        diffs.append(float(np.linalg.norm(sig_mps_fci - sign * sig_fci)))

    max_diff = max(diffs)
    return {"name": "K1_sigma_primitive_integration",
            "diffs_per_root": diffs,
            "tol": 1e-9,
            "status": "pass" if max_diff < 1e-9 else "fail"}


def test_k2_T_matrix_integration_random_trial():
    """For a random spin-symmetrized trial CI tensor, T_FCI and T_MPS must
    agree element-wise. This is the call shape used by H_OC_apply.

    The trial is drawn from random ndarray + symmetrized for closed-shell
    singlet (matching the singlet projection that the FR backend's V vectors
    obey).
    """
    cas, mf, h_mo, eri_mo, h1_act, eri_act = setup_heh_dmrg()
    solver: MPSAsFCISolver = cas.fcisolver
    driver = solver._driver

    rng = np.random.default_rng(7)
    diffs = []
    for state_i in range(2):
        # Random singlet trial: closed-shell + symmetric open-shell pieces
        v = rng.standard_normal((2, 2))
        # Symmetrize off-diagonal (singlet condition: v[0,1]=v[1,0])
        v_sym = v.copy()
        v_sym[0, 1] = v_sym[1, 0] = 0.5 * (v[0, 1] + v[1, 0])

        # FCI reference T
        T_fci = T_matrix_site_replacement(
            h_mo, eri_mo, cas.ci[state_i], v_sym, 2, 0, (1, 1),
            symmetrize_density=True,
        )

        # MPS T
        T_mps = T_matrix_site_replacement_mps(
            driver, h_mo, eri_mo, solver._kets[state_i], v_sym, 2, 0, (1, 1),
            trial_tag=f"K2-TR-{state_i}", symmetrize_density=True,
        )

        sign = _resolve_sign(T_mps, T_fci)
        diffs.append(float(np.linalg.norm(T_mps - sign * T_fci)))

    max_diff = max(diffs)
    return {"name": "K2_T_matrix_integration_random_trial",
            "diffs_per_root": diffs,
            "tol": 1e-9,
            "status": "pass" if max_diff < 1e-9 else "fail"}


def main():
    cases = [test_k1_sigma_primitive_integration,
             test_k2_T_matrix_integration_random_trial]
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
        "milestone": "Step6.3_MPS_primitives_integration",
        "purpose": "Validate that single_site_sigma_mps_native + T_matrix_site_replacement_mps reproduce their FCI counterparts on the call shapes used by the freitag_reiher backend (HeH+ CAS(2,2) DMRG=FCI).",
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
