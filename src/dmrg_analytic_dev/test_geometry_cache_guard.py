"""Geometry-cache safety guard for the MPS-Krylov response object.

When a response object is reused across geometries (geometry scans, trajectory
steps), the integral/RDM caches it holds are only valid at the geometry where
they were built.  These tests check that:

  G1  prepare_geometry_cache() assembles the cache and stamps a fingerprint,
      and the cached gradient matches a freshly built object (cache reuse does
      not change the operator);
  G2  invalidate_geometry_cache() clears every geometry-dependent cache;
  G3  reusing the object after the orbitals/geometry change WITHOUT calling
      invalidate raises, rather than silently returning a stale-cache result.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "sharc_interface"))

from analytic_cp_sharc import _make_mps_krylov_response
from test_step6c_mps_response_class import _setup_heh


def _grad0(resp):
    kappa, ci_mps, info, meta = resp.solve_mps(0, tol=1.0e-8, max_iter=30)
    return kappa, info, meta


def test_g1_prepare_cache_matches_fresh():
    mc, _solver = _setup_heh()
    r1 = _make_mps_krylov_response(mc)
    r1.prepare_geometry_cache()
    k_prepared, info1, _ = _grad0(r1)
    # fingerprint stamped, cache populated
    fp_ok = (r1._geom_fingerprint is not None
             and getattr(r1, "_eris_cache", None) is not None)
    r2 = _make_mps_krylov_response(mc)  # fresh object, lazy cache
    k_fresh, info2, _ = _grad0(r2)
    diff = float(np.linalg.norm(np.asarray(k_prepared) - np.asarray(k_fresh)))
    ok = fp_ok and info1 == 0 and info2 == 0 and diff < 1.0e-8
    return {
        "name": "G1_prepare_cache_matches_fresh",
        "fingerprint_stamped": bool(fp_ok),
        "kappa_diff_prepared_vs_fresh": diff,
        "tol": 1.0e-8, "status": "pass" if ok else "fail",
    }


def test_g2_invalidate_clears_caches():
    mc, _solver = _setup_heh()
    r = _make_mps_krylov_response(mc)
    r.prepare_geometry_cache()
    before = getattr(r, "_eris_cache", None) is not None
    r.invalidate_geometry_cache()
    cleared = (
        getattr(r, "_eris_cache", None) is None
        and r._hci0_mps_cache is None
        and r._corr_mps_cache is None
        and r._eci0_mps_cache is None
        and r._hcc_shifted_mpo_cache == {}
        and r._state_transition_rdm_cache == {}
        and r._gmres_recycle_cache is None
        and r._geom_fingerprint is None
    )
    ok = before and cleared
    return {
        "name": "G2_invalidate_clears_caches",
        "cache_built_before": bool(before),
        "all_cleared_after": bool(cleared),
        "status": "pass" if ok else "fail",
    }


def test_g3_stale_cache_raises():
    mc, _solver = _setup_heh()
    r = _make_mps_krylov_response(mc)
    r.prepare_geometry_cache()
    # Simulate a geometry/orbital change on a reused object WITHOUT invalidating:
    # mutate the captured mo_coeff in place so the fingerprint no longer matches
    # while the stale cache is still present.
    r.mo_coeff = r.mo_coeff + 1.0e-3
    raised = False
    try:
        r._build_eris_cache()
    except RuntimeError as exc:
        raised = "stale" in str(exc).lower()
    # after a proper invalidate + restore, it must work again
    r.mo_coeff = r.mo_coeff - 1.0e-3
    r.invalidate_geometry_cache()
    recovered = False
    try:
        r._build_eris_cache()
        recovered = getattr(r, "_eris_cache", None) is not None
    except Exception:
        recovered = False
    ok = raised and recovered
    return {
        "name": "G3_stale_cache_raises",
        "raised_on_stale": bool(raised),
        "recovered_after_invalidate": bool(recovered),
        "status": "pass" if ok else "fail",
    }


def main():
    cases = [
        test_g1_prepare_cache_matches_fresh,
        test_g2_invalidate_clears_caches,
        test_g3_stale_cache_raises,
    ]
    results = []
    for case in cases:
        try:
            result = case()
        except Exception as exc:
            result = {
                "name": case.__name__, "status": "fail",
                "exception": type(exc).__name__, "message": str(exc),
                "traceback_tail": traceback.format_exc()[-2000:],
            }
        results.append(result)
        print(f"  {result['name']}: {result['status']}", flush=True)
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps({
        "milestone": "geometry_cache_guard",
        "purpose": "Stale-cache safety for reused MPS-Krylov response objects.",
        "results": results,
    }, indent=2) + "\n")
    print(f"Wrote {out}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
