"""Finite-difference validation on N2 and LiF (the artifacts the revision needs).

The original HeH+ suite (test_fd_validation.py) is the only committed FD-validation
artifact.  The revision text referenced N2 gradient and LiF NAC accuracies that had
no backing file; this module generates real, committed golden JSON for both, using
the same three checks as HeH+:

  C1  overlap canary (<I|I>=1, <0|1>=0)            -- MPS->FCI bridge sign-correct
  C2  central-difference gradient vs analytic       -- Reviewer 2 core check
  C3  cross-geometry overlap FD NAC vs analytic     -- Reviewer 3 check

Both systems are FCI-feasible small CAS so DMRG == FCI and every number is ground
truth.  N2/6-31G CAS(6,6) is the gradient case; LiF/6-31G CAS(6,6) is the NAC case
(the ionic/covalent pair carries a real derivative coupling).  Numbers reported
here are whatever the code computes -- no previously-quoted value is assumed.
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

BOHR = 1.0 / 0.52917721067

SOLVER_CFG = dict(fdv.DEFAULT_SOLVER_CFG)
# n_threads=8 (DEFAULT is 1 -> the DMRG-CASSCF build ran single-threaded and
# stalled the job); skip the per-macro dense FCI conversion (the readout happens
# once in mps_ci_list for these FCI-feasible CAS).
SOLVER_CFG.update(bond_dim=128, n_sweeps=24, sweep_tol=1.0e-12, n_threads=8,
                  skip_kernel_fci_conversion=True, mps_native_rdms=True)

# Test systems.  test_atom/test_comp pick a coupling/force component along the bond.
N2 = dict(
    label="N2_CAS66_631G", atoms=["N", "N"],
    coords_bohr=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.098 * BOHR]]),
    basis="6-31G", charge=0, spin=0, ncas=6, nelecas=6, nroots=2,
    weights=[0.5, 0.5], test_atom=1, test_comp=2,
)
LIF = dict(
    label="LiF_CAS66_631G", atoms=["Li", "F"],
    coords_bohr=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.564 * BOHR]]),
    basis="6-31G", charge=0, spin=0, ncas=6, nelecas=6, nroots=2,
    weights=[0.5, 0.5], test_atom=1, test_comp=2,
)


def _build(sys_):
    out = fdv.build_sa_dmrg_casscf(
        sys_["atoms"], sys_["coords_bohr"], basis=sys_["basis"],
        charge=sys_["charge"], spin=sys_["spin"], ncas=sys_["ncas"],
        nelecas=sys_["nelecas"], nroots=sys_["nroots"], weights=sys_["weights"],
        solver_cfg=SOLVER_CFG)
    # Use the short-recurrence cr solver + fewer fit sweeps for the analytic
    # response: the default gmres is the no-preconditioner O(iter^2) path that
    # ground N2/LiF to a halt; cr avoids that and is reliable for these small CAS.
    mc = out[2]
    mc.fcisolver.response_linear_solver = "cr"
    mc.fcisolver.mps_fit_sweeps = 4
    return out


def c1_canary(sys_):
    mol0, _mf0, mc0, solver0 = _build(sys_)
    ci = fdv.mps_ci_list(solver0, sys_["ncas"], mc0.nelecas, sys_["nroots"])
    S_id = cross_geometry_S_act(mol0, mol0, mc0.mo_coeff, mc0.mo_coeff,
                                sys_["ncas"], mc0.ncore)
    o00 = overlap_fci(ci[0], ci[0], S_id, sys_["ncas"], mc0.nelecas)
    o11 = overlap_fci(ci[1], ci[1], S_id, sys_["ncas"], mc0.nelecas)
    o01 = overlap_fci(ci[0], ci[1], S_id, sys_["ncas"], mc0.nelecas)
    err = max(abs(o00 - 1.0), abs(o11 - 1.0), abs(o01))
    return {"name": "C1_overlap_canary", "o00": o00, "o11": o11, "o01": o01,
            "err": err, "tol": 1.0e-8, "status": "pass" if err < 1.0e-8 else "fail"}


def c2_gradient(sys_):
    _m, _mf, mc0, _s = _build(sys_)
    a, x = sys_["test_atom"], sys_["test_comp"]
    out = {}
    statuses = []
    for state in (0, 1):
        g_an = fdv.analytic_gradient(mc0, state, backend="mps-krylov",
                                     tol=1.0e-8, max_iter=200)
        g_fd = fdv.fd_gradient(
            sys_["atoms"], sys_["coords_bohr"], state=state, basis=sys_["basis"],
            charge=sys_["charge"], spin=sys_["spin"], ncas=sys_["ncas"],
            nelecas=sys_["nelecas"], nroots=sys_["nroots"], weights=sys_["weights"],
            solver_cfg=SOLVER_CFG, h_bohr=1.0e-3, atmlst=[a], components=[x],
            track_roots=True)
        ga, gf = float(g_an[a, x]), float(g_fd[a, x])
        err = abs(ga - gf)
        out[f"state{state}"] = {"g_analytic": ga, "g_fd": gf, "abs_err": err}
        statuses.append(err < 5.0e-5)
    return {"name": "C2_fd_gradient_vs_analytic", "components": out,
            "tol": 5.0e-5, "status": "pass" if all(statuses) else "fail"}


def c3_nac(sys_):
    from analytic_cp_sharc import compute_grad_nac_analytic_cp
    _m, _mf, mc0, _s = _build(sys_)
    a, x = sys_["test_atom"], sys_["test_comp"]
    res = compute_grad_nac_analytic_cp(mc0, gradient_states=None,
                                       nac_pairs=[(0, 1)], backend="mps-krylov",
                                       tol=1.0e-8, max_iter=200)
    d_an = float(np.asarray(res["nac"][(0, 1)], dtype=float)[a, x])
    tau = fdv.fd_nac(sys_["atoms"], sys_["coords_bohr"], bra=0, ket=1,
                     basis=sys_["basis"], charge=sys_["charge"], spin=sys_["spin"],
                     ncas=sys_["ncas"], nelecas=sys_["nelecas"],
                     nroots=sys_["nroots"], weights=sys_["weights"],
                     solver_cfg=SOLVER_CFG, h_bohr=1.0e-3, atmlst=[a],
                     components=[x])
    tau_z = float(tau[a, x])
    err = min(abs(d_an - tau_z), abs(d_an + tau_z))
    return {"name": "C3_fd_nac_vs_analytic", "nac_analytic": d_an,
            "tau_fd": tau_z, "abs_err_phaseaware": err, "tol": 1.0e-4,
            "status": "pass" if err < 1.0e-4 else "fail"}


def run_system(sys_):
    results = []
    for fn in (c1_canary, c2_gradient, c3_nac):
        try:
            r = fn(sys_)
        except Exception as exc:  # noqa: BLE001
            r = {"name": fn.__name__, "status": "fail",
                 "exception": type(exc).__name__, "message": str(exc)[:300],
                 "traceback_tail": traceback.format_exc()[-2000:]}
        results.append(r)
        print(f"  [{sys_['label']}] {r['name']}: {r['status']}", flush=True)
    return {"system": sys_["label"], "results": results}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="N2 or LiF")
    args = ap.parse_args()
    systems = {"N2": N2, "LiF": LIF}
    if args.only:
        systems = {k: v for k, v in systems.items() if k == args.only}
    out = {"milestone": "FD_validation_N2_LiF",
           "purpose": ("Committed finite-difference validation artifacts for N2 "
                       "(gradient) and LiF (NAC), CAS(6,6)/6-31G, FCI-feasible."),
           "systems": []}
    for s in systems.values():
        out["systems"].append(run_system(s))
    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {out_path}", flush=True)
    allpass = all(r["status"] == "pass" for sysr in out["systems"]
                  for r in sysr["results"])
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
