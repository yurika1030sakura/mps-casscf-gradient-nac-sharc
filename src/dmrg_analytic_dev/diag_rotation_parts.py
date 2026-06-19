"""Isolate the unitary (U) and symmetric (P) parts of the orbital rotation.

The combined polar rotation left a ~3e-3 residual at finite-difference
magnitude.  The unitary convention (apply exp(-X-hat)) is already consistent
across calibrations, so this pins the symmetric-stretch part separately: it
applies each generator alone and sweeps sign x hermitian-flag against the exact
overlap_fci reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SHARC = _HERE.parents[1] / "sharc_interface"
if str(_SHARC) not in sys.path:
    sys.path.insert(0, str(_SHARC))

import fd_validation as fdv
from analytic_cp_sharc import _make_mps_krylov_response
from overlap_fci_reference import overlap_fci
from cross_geometry_overlap import _apply_onebody_exp, _real_log

ANG = 1.8897261246257702


def main():
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=100, n_sweeps=24, sweep_tol=1.0e-10, n_threads=1)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.90]]) * ANG
    _mol, _mf, mc, solver = fdv.build_sa_dmrg_casscf(
        ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
        ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5], solver_cfg=cfg,
    )
    ncas = mc.ncas
    nelec = (mc.nelecas[0], mc.nelecas[1]) if isinstance(mc.nelecas, (tuple, list)) \
        else (mc.nelecas // 2 + mc.nelecas % 2, mc.nelecas // 2)
    obj = _make_mps_krylov_response(mc)
    states = obj._state_mps
    nst = len(states)
    ci = fdv.mps_ci_list(solver, ncas, nelec, nst)

    rng = np.random.default_rng(1)
    G = rng.standard_normal((ncas, ncas))
    M_U = expm(5.0e-3 * (G - G.T))          # pure unitary
    M_P = expm(5.0e-3 * (G + G.T))          # pure symmetric positive

    tag = {"n": 0}

    def overlap_matrix(M, Gmat, sign, hermitian):
        rotated = []
        for j in range(nst):
            tag["n"] += 1
            r = _apply_onebody_exp(obj._driver_su2, obj._driver_su2.copy_mps(
                states[j], f"DIAG{tag['n']}"), Gmat, ncas=ncas,
                tag=f"DIAGr{tag['n']}", sign=sign, hermitian=hermitian,
                n_steps=24, bond_dim=None)
            rotated.append(r)
        return np.array([[obj._mps_overlap(states[i], rotated[j])
                          for j in range(nst)] for i in range(nst)])

    results = {}
    for name, M in (("U_only", M_U), ("P_only", M_P)):
        Gmat = _real_log(M)
        O_gt = np.array([[overlap_fci(ci[i], ci[j], M, ncas, nelec)
                          for j in range(nst)] for i in range(nst)])
        rows = []
        for sign in (-1.0, 1.0):
            for hermitian in (False, True):
                try:
                    O = overlap_matrix(M, Gmat, sign, hermitian)
                    err = float(np.max(np.abs(O - O_gt)))
                    rows.append({"sign": sign, "hermitian": hermitian, "err": err})
                except Exception as exc:  # noqa: BLE001
                    rows.append({"sign": sign, "hermitian": hermitian,
                                 "error": repr(exc)[:90]})
        best = min((r for r in rows if "err" in r), key=lambda r: r["err"], default=None)
        results[name] = {"rows": rows, "best": best}

    (_HERE / "diag_rotation_parts.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, r in results.items():
        print(f"=== {name} ===")
        for row in r["rows"]:
            if "err" in row:
                print(f"  sign={row['sign']:+.0f} herm={row['hermitian']}  err={row['err']:.3e}")
            else:
                print(f"  sign={row['sign']:+.0f} herm={row['hermitian']}  ERR {row['error']}")
        if r["best"]:
            print(f"  BEST sign={r['best']['sign']:+.0f} herm={r['best']['hermitian']} err={r['best']['err']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
