"""End-to-end finite-difference validation on HeH+ CAS(2,2).

HeH+/3-21G CAS(2,2) is the regime where FCI = DMRG and the MPS->FCI bridge is
trivially exact, so every piece can be cross-checked against ground truth and
against the analytic response simultaneously:

  C1  self-overlap <I|I> = 1 and orthogonality <0|1> = 0 at one geometry,
      proving the MPS->FCI bridge + non-orthogonal overlap are sign-correct;
  C2  central-difference gradient of E_state vs the analytic SA-DMRG-CASSCF
      gradient (Reviewer 2 core check);
  C3  overlap finite-difference derivative coupling vs the analytic NAC
      (Reviewer 3 check).

Emits a golden JSON next to this file in the repo's test convention.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import fd_validation as fdv
from overlap_fci_reference import cross_geometry_S_act, overlap_fci


HEH = dict(
    atoms=["He", "H"],
    coords_bohr=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
    basis="3-21G",
    charge=1,
    spin=0,
    ncas=2,
    nelecas=2,
    nroots=2,
    weights=[0.5, 0.5],
)
# small, exact CAS(2,2): DMRG = FCI; keep bond dim modest, sweeps tight
SOLVER_CFG = dict(fdv.DEFAULT_SOLVER_CFG)
SOLVER_CFG.update(bond_dim=64, n_sweeps=20, sweep_tol=1.0e-12)


def _build_ref():
    return fdv.build_sa_dmrg_casscf(
        HEH["atoms"], HEH["coords_bohr"],
        basis=HEH["basis"], charge=HEH["charge"], spin=HEH["spin"],
        ncas=HEH["ncas"], nelecas=HEH["nelecas"], nroots=HEH["nroots"],
        weights=HEH["weights"], solver_cfg=SOLVER_CFG,
    )


def test_c1_overlap_canary():
    mol0, _mf0, mc0, solver0 = _build_ref()
    ci = fdv.mps_ci_list(solver0, HEH["ncas"], mc0.nelecas, HEH["nroots"])
    S_id = cross_geometry_S_act(
        mol0, mol0, mc0.mo_coeff, mc0.mo_coeff, HEH["ncas"], mc0.ncore,
    )
    o00 = overlap_fci(ci[0], ci[0], S_id, HEH["ncas"], mc0.nelecas)
    o11 = overlap_fci(ci[1], ci[1], S_id, HEH["ncas"], mc0.nelecas)
    o01 = overlap_fci(ci[0], ci[1], S_id, HEH["ncas"], mc0.nelecas)
    err_norm = max(abs(o00 - 1.0), abs(o11 - 1.0))
    err_orth = abs(o01)
    ok = err_norm < 1.0e-8 and err_orth < 1.0e-8
    return {
        "name": "C1_overlap_canary",
        "o00": o00, "o11": o11, "o01": o01,
        "err_norm": err_norm, "err_orth": err_orth,
        "tol": 1.0e-8, "status": "pass" if ok else "fail",
    }


def test_c2_fd_gradient_vs_analytic():
    # H atom (index 1), bond axis (z, component 2)
    _mol0, _mf0, mc0, _s0 = _build_ref()
    results = {}
    statuses = []
    for state in (0, 1):
        g_an = fdv.analytic_gradient(mc0, state, backend="mps-krylov",
                                     tol=1.0e-8, max_iter=100)
        g_fd_arr = fdv.fd_gradient(
            HEH["atoms"], HEH["coords_bohr"], state=state,
            basis=HEH["basis"], charge=HEH["charge"], spin=HEH["spin"],
            ncas=HEH["ncas"], nelecas=HEH["nelecas"], nroots=HEH["nroots"],
            weights=HEH["weights"], solver_cfg=SOLVER_CFG,
            h_bohr=1.0e-3, atmlst=[1], components=[2], track_roots=True,
        )
        g_an_z = float(g_an[1, 2])
        g_fd_z = float(g_fd_arr[1, 2])
        err = abs(g_an_z - g_fd_z)
        results[f"state{state}"] = {
            "g_analytic_z": g_an_z, "g_fd_z": g_fd_z, "abs_err": err,
        }
        # central-difference truncation at h=1e-3 scales with the gradient
        # curvature, so the steep excited-state component sits near 1e-5;
        # sub-1e-4 agreement validates the analytic gradient against the
        # solver's own energy (full step-size scan in the benchmark runner).
        statuses.append(err < 5.0e-5)
    ok = all(statuses)
    return {
        "name": "C2_fd_gradient_vs_analytic",
        "components": results, "tol": 5.0e-5,
        "status": "pass" if ok else "fail",
    }


def test_c3_fd_nac_vs_analytic():
    from analytic_cp_sharc import compute_grad_nac_analytic_cp
    _mol0, _mf0, mc0, _s0 = _build_ref()
    res = compute_grad_nac_analytic_cp(
        mc0, gradient_states=None, nac_pairs=[(0, 1)],
        backend="mps-krylov", tol=1.0e-8, max_iter=100,
    )
    d_an = np.asarray(res["nac"][(0, 1)], dtype=float)
    tau_fd = fdv.fd_nac(
        HEH["atoms"], HEH["coords_bohr"], bra=0, ket=1,
        basis=HEH["basis"], charge=HEH["charge"], spin=HEH["spin"],
        ncas=HEH["ncas"], nelecas=HEH["nelecas"], nroots=HEH["nroots"],
        weights=HEH["weights"], solver_cfg=SOLVER_CFG,
        h_bohr=1.0e-3, atmlst=[1], components=[2],
    )
    d_an_z = float(d_an[1, 2])
    tau_z = float(tau_fd[1, 2])
    # the overlap-FD tau is the CSF/derivative-coupling component; compare
    # phase-aware against the analytic NAC z-component
    err = min(abs(d_an_z - tau_z), abs(d_an_z + tau_z))
    # report ratio too: the analytic NAC may carry an energy-gap factor
    ratio = (d_an_z / tau_z) if abs(tau_z) > 1e-12 else float("nan")
    ok = err < 1.0e-4
    return {
        "name": "C3_fd_nac_vs_analytic",
        "nac_analytic_z": d_an_z, "tau_fd_z": tau_z,
        "abs_err_phaseaware": err, "ratio_analytic_over_fd": ratio,
        "tol": 1.0e-4, "status": "pass" if ok else "fail",
    }


def main():
    cases = [
        test_c1_overlap_canary,
        test_c2_fd_gradient_vs_analytic,
        test_c3_fd_nac_vs_analytic,
    ]
    results = []
    for case in cases:
        try:
            result = case()
        except Exception as exc:
            result = {
                "name": case.__name__,
                "status": "fail",
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc()[-2500:],
            }
        results.append(result)
        print(f"  {result['name']}: {result['status']}", flush=True)

    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps({
        "milestone": "FD_validation_HeH_CAS22",
        "purpose": (
            "Finite-difference validation of analytic SA-DMRG-CASSCF gradients "
            "(vs central difference of DMRG state energies) and derivative "
            "couplings (vs cross-geometry wavefunction overlap finite "
            "differences) on HeH+ CAS(2,2)."
        ),
        "system": "HeH+ / 3-21G / CAS(2,2) / SA(2)",
        "results": results,
    }, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
