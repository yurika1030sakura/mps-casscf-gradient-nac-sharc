"""Reproducer + regression pin: SA!=2 stagnation of the sweep-Schur response.

THE DISEASE (production, 2026-08): solve_state_sweep_schur stagnates at true
relative residual ~1e-1..1e-2 on SA-3 equal-weight systems (6m dithiocarbonate
CAS(22,18)/def2-SVP: S0 9.1e-2, S1 4.4e-2 after ~13 h; aza-acene SA-5: 0.023)
while every repo validation -- ALL of which is SA-2 equal-weight -- certifies
to 1e-6..1e-16.  The certificate demands 1e-6.

CONFIRMED DEFECTS (MPS-level dissection, scratch repro stages 1-16,
2026-08-18; every number below was measured through the repo's actual block2
path):

D1  MISSING SA WEIGHT in the CI-block inverse.  The coupled CI-CI block is
    A_CC,I = 2*w_I*P(H-E_I)P (pyscf newton_casscf; cp_casscf_response.py;
    H_CC_apply_mps scale=2.0*w), but _ci_block_inverse solved the BARE
    (H-E_I).  SA-2 equal weights have 2w = 1 exactly -- every validated test
    was blind.  Isolated at CAS(2,2) where nothing else can bite:
    LiH/3-21G/CAS(2,2) SA-2 (0.7,0.3) floors at 4.06e-2 on old code; with
    ONLY the 1/(2w) fix it certifies at ~5e-15 and matches the dense
    reference to ~3e-15.

D5  ZERO-INITIALIZED ADDITION TRAP (dominant at ncas>2; weight- and
    solver-independent).  block2 addition fits initialize the bra from the
    FIRST _combine_mps term; a zero-norm MPS is an ALS fixed point for
    n_sites > 2 (zero site tensors give zero environments), so
    combine([(1, zero), (1, a)]) silently returns ~0 while
    combine([(1, a), (1, ~0)]) is exact to 1e-18.  The Schur final CI RHS
    combine([(1, b_ci[i]), (-1, hco_zk[i])]) has b_ci[i] == the cached zero
    MPS for every non-target slot, so the non-target CI response was
    ANNIHILATED (H_CC z_C[j!=state] ~ 1e-12) and the certificate floor
    equals |P H_CO z_kappa| per slot -- independent of weights, penalty,
    m_compress, or inner-solver effort.  The in-function residual had the
    same trap and under-reported (meta 0.0089 vs certificate 0.049).
    CAS(2,2) has 2 sites (no environment), so the whole SA-2/CAS(2,2) suite
    was structurally blind.

D2  SILENTLY UNDER-CONVERGED INNER SOLVES (bites after D5 is fixed).
    DMRGDriver.multiply(left_mpo=H-E_i) reports convergence (DF ~ 1e-19,
    per-site Error = 0) at a stationary point whose TRUE residual is
    1e-2..5e-2 with block2's DEFAULT local thresholds thrds=[1e-6]*4+[1e-7]
    (never overridden); thrds=1e-12 improves it only to ~4e-5.  A
    root-projected conjugate-residual loop over the repo's own exact
    primitives reaches 1e-11 in <= 20 iterations: on the all-roots
    complement (H-E_i) is positive definite for every slot i.

    Fix ladder measured on H4/3-21G/CAS(4,4) SA-3 equal (state 0 / state 1
    certificates): none 6.07e-2/4.44e-2 -> +F1 5.05e-2/4.26e-2 ->
    +F2 7.35e-3/6.51e-3 -> +F3 2.45e-11/2.55e-11, with |z - z_dense| ~ 7e-12
    and dense residual of the MPS solution ~ 5e-11.

D4  (why this file carries its own dense reference) mc.ci from the kernel's
    MPS->FCI conversion is NOT an eigenvector of the dense active
    Hamiltonian for ncas>2 (|H c - <H> c| ~ 0.1-0.2; the MPS kets convert to
    exact eigenvectors at 1e-11; overlap signature 1/sqrt(2) = missing
    spin-partner determinant), so any mc.ci-based dense cross-check at
    ncas>2 is invalid.  And CPCASSCFResponseFCI.solve() does not project its
    RHS, so for equal SA weights the singular+inconsistent full system makes
    its GMRES diverge (info=400, |z| ~ 1e11).  The dense reference below
    therefore rebuilds ci_list from the MPS kets and solves the projected
    system by min-norm lstsq.

WHAT THIS FILE ASSERTS
  * H4/3-21G/CAS(4,4) SA-3 EQUAL (production-shaped): the default sweep-Schur
    solve certifies to <= 1e-8 for states 0 and 1 and matches the kets-based
    dense response to <= 1e-6 (on pre-fix code these cases FAIL at the
    5e-2..6e-2 floor -- that failure is the recorded disease).
  * legacy pins (run only when the fix flags exist): re-enabling the historic
    behavior (hcc_weight_scaling=False, ci_inner_solver="multiply",
    ci_proj_all_roots=False, obj._mps_combine_zero_safe=False) must still
    show the >1e-3 floor, so the disease stays demonstrable forever.
  * LiH/3-21G/CAS(2,2) SA-2 UNEQUAL (0.7,0.3): default solve certifies to
    <= 1e-8 and matches dense (isolates D1; old code floors at 4.06e-2).
  * HeH+/3-21G/CAS(2,2) SA-2 EQUAL control: certifies on old AND new code
    (2w = 1: the weight fix is a no-op by construction in the historic
    validation regime).

ORDERING CONSTRAINT: block2's Global.frame is process-global, so each
system's cases run contiguously and lazily -- a system is never used after
the next one is built.  main() enforces this.

Run:  python test_sa_response_reproducer.py [--case h4|lih|heh|all]
Writes test_sa_response_reproducer.json next to this file.  Optional env
DMRG_TEST_SCRATCH overrides the block2 scratch root (default: solver default).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[1] / "sharc_interface"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fd_validation as fdv  # noqa: E402
import sweep_coupled_response as scr  # noqa: E402
from analytic_cp_sharc import _make_mps_krylov_response, _make_response  # noqa: E402
from certified_response import certify_response  # noqa: E402
from cp_dmrg_response_mps_krylov import MPSKrylovVector  # noqa: E402

ANG = 1.8897261246257702

CERT_TOL = 1.0e-6    # what the production certificate demands
FLOOR = 1.0e-3       # "visibly stagnated": >= 3 orders above CERT_TOL
FIXED_TOL = 1.0e-8   # what the fixed solver must reach (standard of proof)
DENSE_MATCH = 1.0e-6  # |z_mps - z_dense| (absolute; |z_dense| is O(0.1))

_T0 = time.perf_counter()


def _log(msg):
    print(f"[{time.perf_counter() - _T0:8.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# System builders (one at a time -- see the Global.frame ordering constraint)
# ---------------------------------------------------------------------------

def build_case(case: str):
    """Converge SA-DMRG-CASSCF for one reproducer case; returns mc."""
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=200, n_sweeps=30, sweep_tol=1.0e-10, n_threads=1)
    scratch_root = os.environ.get("DMRG_TEST_SCRATCH")
    if scratch_root:
        Path(scratch_root).mkdir(parents=True, exist_ok=True)
        cfg["scratch_root"] = scratch_root

    if case == "h4":
        # H4 rectangle, 3-21G, CAS(4,4), SA-3 EQUAL weights: the smallest
        # production-shaped case.  The CI space is much larger than 3 roots
        # (nontrivial projected complement) and 4 sites arm the D5 trap.
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.00],
            [0.0, 1.40, 0.0],
            [0.0, 1.40, 1.00],
        ]) * ANG
        _mol, _mf, mc, _s = fdv.build_sa_dmrg_casscf(
            ["H", "H", "H", "H"], coords, basis="3-21G", charge=0, spin=0,
            ncas=4, nelecas=4, nroots=3, weights=[1 / 3, 1 / 3, 1 / 3],
            solver_cfg=cfg,
        )
        assert mc.converged
    elif case == "lih":
        # LiH, 3-21G, CAS(2,2), SA-2 (0.7, 0.3).  CAS(2,2) = 2 sites makes
        # the D5 zero-init trap inert and the inner solve exact, isolating
        # the D1 weight bug cleanly.
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.60]]) * ANG
        _mol, _mf, mc, _s = fdv.build_sa_dmrg_casscf(
            ["Li", "H"], coords, basis="3-21G", charge=0, spin=0,
            ncas=2, nelecas=2, nroots=2, weights=[0.7, 0.3],
            solver_cfg=cfg,
        )
        assert mc.converged
    elif case == "heh":
        # HeH+, 3-21G, CAS(2,2), SA-2 EQUAL -- the repo's historically
        # certified system, the blind-spot control.  No mc.converged assert:
        # at tight conv_tol PySCF's flag can stay False on HeH+ at an
        # excellent solution; the certificate below is the arbiter.
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]) * ANG
        _mol, _mf, mc, _s = fdv.build_sa_dmrg_casscf(
            ["He", "H"], coords, basis="3-21G", charge=1, spin=0,
            ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5],
            solver_cfg=cfg,
        )
    else:
        raise ValueError(case)
    return mc


# ---------------------------------------------------------------------------
# Kets-based dense projected reference (valid for any small CAS; see D4)
# ---------------------------------------------------------------------------

def _zc_to_fci(obj, mc, zc_list, tag="ZC2FCI"):
    """Read general SU2 response MPS slots out as UNNORMALIZED PySCF FCI
    arrays (mirrors fd_validation.mps_ci_list without the normalization: a
    response vector's length carries physics)."""
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    solver = mc.fcisolver
    nelec = obj.nelec
    sz_driver = DMRGDriver(
        scratch=solver._scratch, clean_scratch=False,
        stack_mem=int(solver.stack_mem_mb) * 1024 * 1024,
        n_threads=int(solver.n_threads),
        symm_type=SymmetryTypes.SZ,
    )
    sz_driver.initialize_system(
        n_sites=int(obj.ncas), n_elec=int(nelec[0] + nelec[1]), spin=0,
        orb_sym=[0] * int(obj.ncas),
    )
    out = []
    for i, m in enumerate(zc_list):
        norm2 = float(obj._mps_overlap(m, m))
        arr = np.asarray(fdv._su2_ket_to_fci(
            obj._driver_su2, sz_driver, m, int(obj.ncas), nelec,
            tag=f"{tag}-{i}",
        ))
        # _su2_ket_to_fci normalizes; restore the true length.
        n = float(np.linalg.norm(arr))
        if n > 1.0e-30:
            arr = arr / n * np.sqrt(max(norm2, 0.0))
        out.append(arr)
    return out


def dense_projected_solution(mc, obj, state):
    """Exact dense solution of the root-projected coupled response system,
    built on the MPS kets' FCI readouts (NOT mc.ci -- see D4).  The dense
    solve() is not used because it does not project its RHS and diverges for
    equal SA weights; the projected system is solved by min-norm lstsq."""
    resp = _make_response(mc, backend="newton_casscf")
    kets_fci = _zc_to_fci(obj, mc, obj._state_mps, tag=f"DREF{state}")
    resp.ci_list = [np.asarray(c) for c in kets_fci]
    resp._eris_cache = None
    resp._h_op_cache = None
    nmo = mc.mo_coeff.shape[1]
    nrot = mc.pack_uniq_var(np.zeros((nmo, nmo))).size
    ci_size = resp.ci_list[0].size
    nst = resp.nstates
    n = nrot + nst * ci_size
    roots = [c.ravel() for c in resp.ci_list]

    def P(x):
        x = np.asarray(x, float).copy()
        for i in range(nst):
            s = slice(nrot + i * ci_size, nrot + (i + 1) * ci_size)
            for r in roots:
                x[s] -= np.dot(x[s], r) * r
        return x

    M = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        M[:, k] = resp._matvec(P(e))
    rhs_O, rhs_C = resp.build_rhs(state)
    b = P(resp._flatten(rhs_O, rhs_C))
    z, _res, _rank, _sv = np.linalg.lstsq(M, b, rcond=1e-10)
    return P(z), resp, P, M, b


def compare_mps_to_dense(mc, obj, state, kappa, zc_list):
    """Return (|z_mps - z_dense|, |z_dense|, dense rel resid of z_mps)."""
    z_dense, resp, P, M, b = dense_projected_solution(mc, obj, state)
    zc_fci = _zc_to_fci(obj, mc, zc_list, tag=f"DCMP{state}")
    kp = mc.pack_uniq_var(obj._canonical_kappa(kappa))
    z_mps = P(resp._flatten(mc.unpack_uniq_var(kp), zc_fci))
    dz = float(np.linalg.norm(z_mps - z_dense))
    rel = float(np.linalg.norm(b - M @ z_mps) / max(np.linalg.norm(b), 1e-300))
    return dz, float(np.linalg.norm(z_dense)), rel


# ---------------------------------------------------------------------------
# Solve + certify
# ---------------------------------------------------------------------------

def _legacy_mode_available() -> bool:
    """The legacy pins need the fix-era flags on solve_state_sweep_schur."""
    params = set(inspect.signature(scr.solve_state_sweep_schur).parameters)
    return {"hcc_weight_scaling", "ci_inner_solver", "ci_proj_all_roots"} <= params


def run_schur_and_certify(mc, state, *, mode="default", tol=CERT_TOL):
    """One sweep-Schur solve on a fresh response object + true-residual
    certificate against the global operator.

    mode="default": the shipped solver configuration (post-fix: F1+F2+F3).
    mode="legacy":  re-enable the historic behavior (bare (H-E_i) inverse,
                    block2-multiply-only inner solve at default thrds,
                    own-root-only penalty, zero-unsafe combines) to pin the
                    disease.  Requires the fix-era flags.
    """
    obj = _make_mps_krylov_response(mc)
    kwargs = {}
    if mode == "legacy":
        if not _legacy_mode_available():
            raise RuntimeError("legacy mode flags not available")
        obj._mps_combine_zero_safe = False
        kwargs = dict(
            hcc_weight_scaling=False,
            ci_inner_solver="multiply",
            ci_proj_all_roots=False,
        )
    t0 = time.perf_counter()
    kappa, zc, info, meta = scr.solve_state_sweep_schur(
        obj, state, orb_tol=1.0e-9, ci_tol=1.0e-11, **kwargs,
    )
    wall = time.perf_counter() - t0
    z = MPSKrylovVector(obj, kappa, zc, label=f"REPRO-{mode.upper()}{state}")
    cert = certify_response(
        obj, z, state=state, tol=tol, solver=f"sweep_schur_{mode}",
        wall_s=wall, extra={"solver_meta": meta},
    )
    return obj, kappa, zc, cert, meta


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_h4_default(mc, state):
    obj, k, zc, cert, meta = run_schur_and_certify(
        mc, state, mode="default", tol=FIXED_TOL)
    rel = cert.true_residual_relative
    dz, nz, drel = compare_mps_to_dense(mc, obj, state, k, zc)
    rec = {
        "true_residual_relative": rel,
        "meta_residual": float(meta["true_residual_rel"]),
        "z_err_abs_vs_dense": dz,
        "z_dense_norm": nz,
        "dense_rel_residual": drel,
        "wall_s": cert.wall_s,
    }
    _log(f"[H4 SA-3 equal, default] state {state}: cert={rel:.3e} "
         f"meta={rec['meta_residual']:.3e} |z-z_dense|={dz:.3e} "
         f"dense_rel={drel:.3e}")
    assert rel <= FIXED_TOL, f"certificate {rel:.3e} > {FIXED_TOL}"
    assert cert.converged
    # the in-function residual must be honest (D5 under-reported it)
    assert rec["meta_residual"] <= FIXED_TOL
    assert dz < DENSE_MATCH, f"|z-z_dense| {dz:.3e} >= {DENSE_MATCH}"
    assert drel < 1.0e-6
    return rec


def case_h4_legacy_floor(mc, state=0):
    obj, k, zc, cert, meta = run_schur_and_certify(
        mc, state, mode="legacy", tol=CERT_TOL)
    rel = cert.true_residual_relative
    _log(f"[H4 SA-3 equal, LEGACY pin] state {state}: cert={rel:.3e}")
    rec = {"true_residual_relative": rel, "wall_s": cert.wall_s}
    assert rel > FLOOR, (
        f"legacy floor vanished ({rel:.3e} <= {FLOOR}); the disease pin no "
        f"longer demonstrates the historic stagnation")
    assert not cert.converged
    return rec


def case_lih_default(mc, state):
    obj, k, zc, cert, meta = run_schur_and_certify(
        mc, state, mode="default", tol=FIXED_TOL)
    rel = cert.true_residual_relative
    dz, nz, _drel = compare_mps_to_dense(mc, obj, state, k, zc)
    rec = {
        "true_residual_relative": rel,
        "z_err_abs_vs_dense": dz,
        "z_dense_norm": nz,
        "wall_s": cert.wall_s,
    }
    _log(f"[LiH SA-2 0.7/0.3, default] state {state}: cert={rel:.3e} "
         f"|z-z_dense|={dz:.3e}")
    assert rel <= FIXED_TOL, f"certificate {rel:.3e} > {FIXED_TOL}"
    assert cert.converged
    assert dz < DENSE_MATCH
    return rec


def case_lih_legacy_floor(mc, state=0):
    obj, k, zc, cert, meta = run_schur_and_certify(
        mc, state, mode="legacy", tol=CERT_TOL)
    rel = cert.true_residual_relative
    _log(f"[LiH SA-2 0.7/0.3, LEGACY pin] state {state}: cert={rel:.3e}")
    rec = {"true_residual_relative": rel, "wall_s": cert.wall_s}
    assert rel > FLOOR, (
        f"legacy floor vanished ({rel:.3e} <= {FLOOR})")
    return rec


def case_heh_control(mc):
    obj, k, zc, cert, meta = run_schur_and_certify(
        mc, 0, mode="default", tol=CERT_TOL)
    rel = cert.true_residual_relative
    _log(f"[HeH+ SA-2 equal control] state 0: cert={rel:.3e}")
    rec = {"true_residual_relative": rel, "wall_s": cert.wall_s}
    assert cert.converged
    assert rel < CERT_TOL
    return rec


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["h4", "lih", "heh", "all"])
    args = ap.parse_args(argv)

    results = {}
    counts = {"pass": 0, "fail": 0, "skip": 0}

    def run(name, fn, *fargs, needs_legacy=False):
        if needs_legacy and not _legacy_mode_available():
            _log(f"SKIP {name} (fix-era flags not present yet)")
            results[name] = {"status": "skip",
                             "reason": "legacy-mode flags not available"}
            counts["skip"] += 1
            return
        try:
            rec = fn(*fargs)
            rec["status"] = "pass"
            counts["pass"] += 1
            _log(f"PASS {name}")
        except AssertionError as exc:
            rec = {"status": "fail", "assertion": str(exc)}
            counts["fail"] += 1
            _log(f"FAIL {name}: {exc}")
        except Exception:
            rec = {"status": "fail",
                   "traceback_tail": traceback.format_exc()[-2000:]}
            counts["fail"] += 1
            _log(f"ERROR {name}")
            traceback.print_exc()
        results[name] = rec

    # NOTE: one system at a time (block2 Global.frame is process-global).
    if args.case in ("h4", "all"):
        _log("building H4/3-21G/CAS(4,4) SA-3 equal ...")
        mc = build_case("h4")
        for s in (0, 1):
            run(f"h4_sa3_equal_default_certifies[{s}]", case_h4_default, mc, s)
        run("h4_sa3_equal_legacy_floor_pin", case_h4_legacy_floor, mc,
            needs_legacy=True)
        del mc

    if args.case in ("lih", "all"):
        _log("building LiH/3-21G/CAS(2,2) SA-2 (0.7,0.3) ...")
        mc = build_case("lih")
        for s in (0, 1):
            run(f"lih_sa2_unequal_default_certifies[{s}]", case_lih_default,
                mc, s)
        run("lih_sa2_unequal_legacy_floor_pin", case_lih_legacy_floor, mc,
            needs_legacy=True)
        del mc

    if args.case in ("heh", "all"):
        _log("building HeH+/3-21G/CAS(2,2) SA-2 equal control ...")
        mc = build_case("heh")
        run("heh_sa2_equal_control_certifies", case_heh_control, mc)
        del mc

    status = "pass" if counts["fail"] == 0 else "fail"
    payload = {
        "milestone": "sa_response_reproducer",
        "systems": {
            "h4": "H4 rectangle / 3-21G / CAS(4,4) / SA-3 equal (1/3,1/3,1/3)",
            "lih": "LiH / 3-21G / CAS(2,2) / SA-2 (0.7,0.3)",
            "heh": "HeH+ / 3-21G / CAS(2,2) / SA-2 equal (control)",
        },
        "thresholds": {"fixed_tol": FIXED_TOL, "cert_tol": CERT_TOL,
                       "floor": FLOOR, "dense_match": DENSE_MATCH},
        "legacy_mode_available": _legacy_mode_available(),
        "counts": counts,
        "results": results,
        "status": status,
    }
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    _log(f"{counts['pass']} passed, {counts['fail']} failed, "
         f"{counts['skip']} skipped -> {out.name}: {status}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
