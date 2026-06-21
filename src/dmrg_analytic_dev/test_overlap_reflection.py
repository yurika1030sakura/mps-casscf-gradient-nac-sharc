"""Regression test: the discrete-gauge factorization that fixes the cross-geometry
overlap reflection crash.

At a near-degenerate active space (e.g. polyene C20) the displaced CASSCF returns
active orbitals in a different DISCRETE gauge -- a sign flip, a column swap, or a
reflection -- so the active cross overlap ``s`` is not near-identity and
``logm(polar(s))`` is complex (the observed 'orbital-rotation log has imaginary
part ~pi' crash).  ``discrete_gauge(s)`` factors ``s = G @ s_res`` with ``G`` the
nearest signed permutation and ``s_res`` near-identity.

The load-bearing identity (verified here to machine precision) is

    overlap_fci(ci_i, ci_j, s) = overlap_fci(ci_i, transform_ci(ci_j, G.T), s @ G.T)

i.e. the discrete gauge MUST be applied to the ket state (a naive "just rotate by
s_res" is WRONG -- it leaves an O(1) error, also checked here).  This pins the MPS
fix: apply the signed permutation G.T to the displaced MPS, then rotate by the
near-identity s @ G.T.
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.linalg import expm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from pyscf.fci import addons
from overlap_fci_reference import overlap_fci
from cross_geometry_overlap import discrete_gauge

NCAS, NELEC = 4, (2, 2)
GAUGES = {
    "identity": np.eye(4),
    "signflip": np.diag([1.0, -1.0, 1.0, 1.0]),          # det -1
    "swap_1_2": np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1.0]]),  # det -1
    "signflip_plus_swap": np.array([[0, -1, 0, 0], [1, 0, 0, 0],
                                    [0, 0, 1, 0], [0, 0, 0, -1.0]]),
}


def main():
    d = comb(NCAS, NELEC[0])
    rng = np.random.default_rng(1)
    ci_i = rng.standard_normal((d, d)); ci_i /= np.linalg.norm(ci_i)
    ci_j = rng.standard_normal((d, d)); ci_j /= np.linalg.norm(ci_j)
    A = rng.standard_normal((NCAS, NCAS))
    U = expm(0.02 * (A - A.T))            # near-identity proper rotation

    results = []
    ok = True
    for name, Gt in GAUGES.items():
        s = Gt @ U
        o_gt = overlap_fci(ci_i, ci_j, s, NCAS, NELEC)
        G, s_res = discrete_gauge(s)
        recovered = bool(np.allclose(np.abs(G), np.abs(Gt)))
        # correct (same-side) decomposition: transform ket by G.T, pair with s@G.T
        cj = addons.transform_ci_for_orbital_rotation(ci_j, NCAS, NELEC, G.T)
        err_fix = abs(overlap_fci(ci_i, cj, s @ G.T, NCAS, NELEC) - o_gt)
        near_id = float(np.max(np.abs(s @ G.T - np.eye(NCAS))))
        # naive "no surgery" (rotate by s_res, no ket transform) -- must be WRONG
        err_naive = abs(overlap_fci(ci_i, ci_j, s_res, NCAS, NELEC) - o_gt)
        passed = (err_fix < 1.0e-10) and (near_id < 0.2) and (
            name == "identity" or err_naive > 1.0e-3)
        ok &= passed
        results.append({"gauge": name, "detG": float(round(np.linalg.det(G))),
                        "G_recovered": recovered, "err_discrete_gauge_fix": err_fix,
                        "max_abs_sGT_minus_I": near_id,
                        "err_naive_no_surgery": err_naive,
                        "status": "pass" if passed else "fail"})
        print(f"  {name:20s} detG={round(np.linalg.det(G)):+d} "
              f"fix_err={err_fix:.1e} naive_err={err_naive:.1e} "
              f"{'pass' if passed else 'FAIL'}", flush=True)

    out = {"milestone": "overlap_discrete_gauge_decomposition",
           "identity": "overlap_fci(i,j,s) == overlap_fci(i, transform(j,G.T), s@G.T)",
           "results": results}
    (_HERE / "test_overlap_reflection.json").write_text(json.dumps(out, indent=2) + "\n")
    print("OVERLAP DISCRETE-GAUGE TEST:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
