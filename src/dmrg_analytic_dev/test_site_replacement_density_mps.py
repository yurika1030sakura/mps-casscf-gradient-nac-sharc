"""Validate MPS-native site-replacement transition density and T matrix.

For HeH+ CAS(2,2) singlet (DMRG = FCI):

  R1: γ_AB, Γ_AB from `transition_rdm_site_replacement_mps` (with trial round-
      tripped through CSF→MPS) match the FCI-fallback
      `transition_rdm_site_replacement` to bond-truncation tolerance.

  R2: The full T matrix from `T_matrix_site_replacement_mps` matches the FCI
      version `T_matrix_site_replacement` for the same (Ψ, trial) pair.

The numerical comparison is element-wise after resolving any global sign that
arises from CSF→MPS round-trip (block2 returns coefficients with arbitrary
overall phase). For DMRG = FCI this should match to ~1e-10.
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
from site_replacement_density import (
    transition_rdm_site_replacement,
    transition_rdm_site_replacement_mps,
    T_matrix_site_replacement,
    T_matrix_site_replacement_mps,
)


def setup_heh(d_bohr: float = 1.4, bond_dim: int = 64):
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
    """Find the overall ±1 phase that minimizes ‖a - sign·b‖."""
    fa = np.asarray(a).reshape(-1)
    fb = np.asarray(b).reshape(-1)
    if np.linalg.norm(fa) < 1e-14 or np.linalg.norm(fb) < 1e-14:
        return 1.0
    return float(np.sign(np.dot(fa, fb)) or 1.0)


def test_r1_trans_rdm_matches_fci():
    cas, mf, h_mo, eri_mo, h1_act, eri_act = setup_heh()
    solver: MPSAsFCISolver = cas.fcisolver
    driver = solver._driver

    ci0 = cas.ci[0]
    ci1 = cas.ci[1]

    # FCI-fallback reference: state = root 0, trial = root 1
    gamma_fci, Gamma_fci = transition_rdm_site_replacement(ci0, ci1, 2, (1, 1))

    # MPS-native: state MPS = solver._kets[0], trial as FCI array (auto-converted)
    gamma_mps, Gamma_mps = transition_rdm_site_replacement_mps(
        driver, solver._kets[0], ci1, 2, (1, 1), trial_tag="TR-R1",
    )

    # Resolve overall phase from CSF round-trip on trial
    sign_g = _resolve_sign(gamma_mps, gamma_fci)
    diff_gamma = float(np.linalg.norm(gamma_mps - sign_g * gamma_fci))
    sign_G = _resolve_sign(Gamma_mps, Gamma_fci)
    diff_Gamma = float(np.linalg.norm(Gamma_mps - sign_G * Gamma_fci))

    return {
        "name": "R1_trans_rdm_matches_fci",
        "‖gamma_fci‖": float(np.linalg.norm(gamma_fci)),
        "‖Gamma_fci‖": float(np.linalg.norm(Gamma_fci)),
        "sign_g": sign_g, "sign_G": sign_G,
        "diff_gamma": diff_gamma, "diff_Gamma": diff_Gamma,
        "tol": 1e-8,
        "status": "pass" if max(diff_gamma, diff_Gamma) < 1e-8 else "fail",
    }


def test_r2_T_matrix_matches_fci():
    cas, mf, h_mo, eri_mo, h1_act, eri_act = setup_heh()
    solver: MPSAsFCISolver = cas.fcisolver
    driver = solver._driver

    ci0 = cas.ci[0]
    ci1 = cas.ci[1]

    nmo = h_mo.shape[0]
    ncas = 2
    ncore = 0

    # FCI reference
    T_fci = T_matrix_site_replacement(
        h_mo, eri_mo, ci0, ci1, ncas, ncore, (1, 1), symmetrize_density=True,
    )

    # MPS path
    T_mps = T_matrix_site_replacement_mps(
        driver, h_mo, eri_mo, solver._kets[0], ci1, ncas, ncore, (1, 1),
        trial_tag="TR-R2", symmetrize_density=True,
    )

    # T matrix is built from a symmetrized density, so the global phase from
    # CSF round-trip cancels. (Symmetrize: sign² = 1 on both rdm contributions.)
    # But to be robust, resolve sign here too.
    sign = _resolve_sign(T_mps, T_fci)
    diff = float(np.linalg.norm(T_mps - sign * T_fci))

    return {
        "name": "R2_T_matrix_matches_fci",
        "‖T_fci‖": float(np.linalg.norm(T_fci)),
        "‖T_mps‖": float(np.linalg.norm(T_mps)),
        "sign": sign,
        "diff": diff,
        "tol": 1e-8,
        "status": "pass" if diff < 1e-8 else "fail",
    }


def main():
    cases = [test_r1_trans_rdm_matches_fci, test_r2_T_matrix_matches_fci]
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
        "milestone": "Step6.2_site_replacement_density_MPS_native",
        "purpose": "Validate transition_rdm_site_replacement_mps + T_matrix_site_replacement_mps on HeH+ CAS(2,2) (DMRG=FCI).",
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
