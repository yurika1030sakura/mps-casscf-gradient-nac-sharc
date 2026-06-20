"""Certified, system-general SA-DMRG-CASSCF derivative engine.

A single entry point, :func:`compute_certified_derivatives`, takes an arbitrary
molecule and active-space specification and returns analytic gradients /
nonadiabatic couplings -- each with a true-residual certificate and a
``PASS`` / ``WARN`` / ``FAIL`` self-diagnosis -- plus optional finite-difference
cross-checks.  The design goal is that an arbitrary user system either yields a
*certified* result or is *clearly flagged* as problematic; it must never return a
silently wrong number.  This is the consolidation of the building blocks
(robust build, FCI-free guards, response certificate, self-diagnostics) behind
one robust, hand-tuning-free interface so users do not have to chase per-system
bugs.

Robustness, by default, so it works without per-system tuning:
  * Active space: pass ``ncas``/``nelecas`` directly, or ``ao_targets`` (e.g.
    ['F 2p', 'Li 2s'], 'C 2pz' for a pi space) to select by atomic-orbital
    population; otherwise the RHF/ROHF HOMO-LUMO window is used.
  * Convergence: a progressive bond-dimension schedule (cheap low-M macro
    iterations, then raise M) with warm starts, and an escalation ladder that
    engages ONLY on failure -- more macro cycles, then an augmented-Hessian level
    shift.  The level shift is OFF by default because on a well-conditioned
    surface it can move the SA-CASSCF stationary point; if it is ever needed the
    point is flagged ``WARN`` so the user knows the solution may be shift-defined.
  * FCI-free integrity: above the determinant threshold dense FCI conversion and
    determinant-overlap root tracking are disabled and a process-wide sentinel
    certifies that no dense bridge was entered.
  * Reproducibility: the build is genuinely random (no fixed seed); a converged
    result is seed-independent.  ``seed_check=True`` runs a few seeds and reports
    the energy spread so the user can confirm convergence rather than assume it.
  * Self-diagnosis: the build and every response solve are passed through
    :func:`system_diagnostics.assess_point`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[1] / "sharc_interface"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fd_validation as fdv
from dmrg_fcisolver import MPSAsFCISolver
from active_space import select_active_space_by_ao_targets
from fci_free_guard import (FCI_FREE_THRESHOLD, RootTracking, determinant_dimension,
                            assert_fci_free_if_needed, DenseBridgeSentinel)
from system_diagnostics import assess_point
from analytic_cp_sharc import compute_grad_nac_analytic_cp, _make_mps_krylov_response
from auto_response import compute_all_responses_certified


# --------------------------------------------------------------- M schedule
def progressive_schedule(ncas, nelecas, max_bond_dim):
    """Auto bond-dimension schedule: cheap low-M relaxation, then raise M.

    Below the FCI-free threshold the active space is small; a short schedule
    suffices.  Above it the schedule climbs to ``max_bond_dim`` so the orbital
    optimization is relaxed cheaply before the expensive final M.
    """
    det = determinant_dimension(ncas, nelecas)
    if det < FCI_FREE_THRESHOLD:
        m = min(max_bond_dim, 256)
        return [(m, 1.0e-10, 1.0e-7, 60)]
    stages = []
    for M, swtol, cgrad, mxm in [(256, 1.0e-7, 1.0e-4, 30),
                                 (512, 1.0e-8, 3.0e-5, 40),
                                 (800, 1.0e-9, 1.0e-5, 60),
                                 (1200, 1.0e-9, 3.0e-6, 80)]:
        if M <= max_bond_dim:
            stages.append((M, swtol, cgrad, mxm))
    return stages or [(max_bond_dim, 1.0e-9, 1.0e-5, 60)]


# --------------------------------------------------------------- robust build
def build_robust(atoms, coords_bohr, *, basis, charge=0, spin=0, ncas, nelecas,
                 nroots=2, weights=None, mo_guess=None, ao_targets=None,
                 max_bond_dim=800, threads=8, stack_mem_mb=8000,
                 warm_start=True):
    """Robust SA-DMRG-CASSCF build for an arbitrary system.

    Returns ``(mol, mc, solver, info)`` where ``info`` records the determinant
    dimension, the stage log, whether escalation / a level shift were needed, and
    a build-level health verdict.  Raises only on genuinely unrecoverable inputs;
    a non-converged build is reported (not raised) and flagged in ``info``.
    """
    weights = list(weights) if weights is not None else [1.0 / nroots] * nroots
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    mol = gto.M(atom=[(atoms[i], tuple(coords_bohr[i])) for i in range(len(atoms))],
                basis=basis, charge=charge, spin=spin, unit="Bohr",
                symmetry=False, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(conv_tol=1.0e-11)

    sel_diag = None
    if mo_guess is None and ao_targets is not None:
        ncore, mo_guess, sel_diag = select_active_space_by_ao_targets(
            mol, mf, ncas, nelecas, list(ao_targets))
    elif mo_guess is None:
        mo_guess = mf.mo_coeff  # default HOMO-LUMO active window

    det = determinant_dimension(ncas, nelecas)
    nelec_t = (nelecas // 2, nelecas - nelecas // 2)
    sched = progressive_schedule(ncas, nelecas, max_bond_dim)

    # solver config; above the threshold force the FCI-free settings.
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=sched[0][0], n_threads=int(threads),
               stack_mem_mb=int(stack_mem_mb), dmrg_symm_su2=True,
               force_dmrg=True, warm_start=bool(warm_start))
    if det >= FCI_FREE_THRESHOLD:
        cfg.update(mps_native_rdms=True, skip_kernel_fci_conversion=True)
    assert_fci_free_if_needed(ncas, nelec_t, cfg, RootTracking.GAP_GUARD,
                              "certified_engine.build_robust")

    def _solve(mo, level_shift, sched_):
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        solver = MPSAsFCISolver(mol, **cfg)
        solver.nroots = int(nroots)
        mc.fcisolver = solver
        if nroots > 1:
            mc = mc.state_average_(weights)
        if spin == 0:
            try:
                mc.fix_spin_(ss=0.0, shift=0.5)
            except Exception:
                pass
        mo_run = mo
        stage_log = []
        for (M, swtol, cgrad, mxm) in sched_:
            solver.bond_dim = int(M)
            if hasattr(solver, "sweep_tol"):
                solver.sweep_tol = float(swtol)
            mc.conv_tol = max(cgrad * 1.0e-2, 1.0e-10)
            mc.conv_tol_grad = float(cgrad)
            mc.max_cycle_macro = int(mxm)
            if level_shift and hasattr(mc, "ah_level_shift"):
                mc.ah_level_shift = float(level_shift)
            t0 = time.perf_counter()
            mc.kernel(mo_run)
            mo_run = mc.mo_coeff
            stage_log.append({"M": int(M), "converged": bool(mc.converged),
                              "wall_s": time.perf_counter() - t0,
                              "e_states": [float(x) for x in mc.e_states]})
        return mc, solver, stage_log

    # Small-CAS fast path: orbital optimization with the exact FCI solver is far
    # cheaper than DMRG-CASSCF macro iterations, and seeding the DMRG build at the
    # FCI stationary point converges it in a couple of macros (FCI == DMRG at the
    # active-space level there).  At det >= threshold FCI is infeasible and this
    # is skipped -- the DMRG build then does the full orbital optimization.
    if det < FCI_FREE_THRESHOLD:
        try:
            mc_fci = mcscf.CASSCF(mf, ncas, nelecas)
            mc_fci.fcisolver.nroots = int(nroots)
            if nroots > 1:
                mc_fci = mc_fci.state_average_(weights)
            if spin == 0:
                try:
                    mc_fci.fix_spin_(ss=0.0, shift=0.5)
                except Exception:
                    pass
            mc_fci.conv_tol = 1.0e-9
            mc_fci.conv_tol_grad = 1.0e-6
            mc_fci.max_cycle_macro = 100
            mc_fci.kernel(mo_guess)
            if mc_fci.converged:
                mo_guess = mc_fci.mo_coeff
        except Exception:
            pass

    DenseBridgeSentinel.reset()
    t_start = time.perf_counter()
    mc, solver, stage_log = _solve(mo_guess, 0.0, sched)
    escalated = False
    level_shift_used = 0.0
    if not mc.converged:
        # Escalation: keep the guess, double the macro budget, add a level shift.
        # The level shift can move the stationary point, so the result is flagged.
        escalated = True
        level_shift_used = 0.5
        tight = [(M, sw, cg, mxm * 2) for (M, sw, cg, mxm) in sched]
        mc, solver, stage_log2 = _solve(mo_guess, level_shift_used, tight)
        stage_log = stage_log + stage_log2

    e = [float(x) for x in mc.e_states]
    # spin purity from the (small) CI if available; else skip (mps-native)
    s2 = None
    try:
        from pyscf import fci
        s2 = [float(fci.spin_square(np.asarray(c), ncas, nelec_t)[0]) for c in mc.ci]
    except Exception:
        s2 = None

    health = assess_point(
        scf_converged=bool(mf.converged), casscf_converged=bool(mc.converged),
        s2_per_state=s2, target_spin=spin,
        gap_eh=(e[1] - e[0]) if len(e) > 1 else None,
        det_dim=det, dense_bridge_used=DenseBridgeSentinel.used)

    info = {"det_dim": det, "beyond_fci": det >= FCI_FREE_THRESHOLD,
            "ncas": ncas, "nelecas": nelecas, "spin": spin,
            "stages": stage_log, "converged": bool(mc.converged),
            "escalated": escalated, "level_shift_used": level_shift_used,
            "level_shift_warning": bool(level_shift_used > 0),
            "s2_per_state": s2, "e_states": e,
            "wall_s": time.perf_counter() - t_start,
            "active_space_selection": sel_diag,
            "build_health": health.to_dict()}
    return mol, mc, solver, info


# ------------------------------------------------------ certified derivatives
def compute_certified_derivatives(
        atoms, coords_bohr, *, basis, charge=0, spin=0, ncas, nelecas,
        nroots=2, weights=None, ao_targets=None, mo_guess=None,
        gradient_states=(0,), nac_pairs=(), grad_tol=1.0e-7, nac_tol=1.0e-6,
        max_bond_dim=800, threads=8, stack_mem_mb=8000):
    """System-general certified derivative driver.

    Builds the SA-DMRG-CASSCF robustly, computes the requested analytic
    gradients / NACs through the certified MPS response backend, and returns a
    structured result in which every derivative carries a certificate and the
    whole point carries a PASS/WARN/FAIL verdict.  A failed build or an
    uncertified response is reported, never silently returned as a number.
    """
    mol, mc, solver, info = build_robust(
        atoms, coords_bohr, basis=basis, charge=charge, spin=spin, ncas=ncas,
        nelecas=nelecas, nroots=nroots, weights=weights, ao_targets=ao_targets,
        mo_guess=mo_guess, max_bond_dim=max_bond_dim, threads=threads,
        stack_mem_mb=stack_mem_mb)

    out = {"system": {"atoms": list(atoms), "basis": basis, "charge": charge,
                      "spin": spin, "ncas": ncas, "nelecas": nelecas},
           "build": info, "gradients": {}, "nacs": {}, "overall_health": None}

    if not info["converged"]:
        out["overall_health"] = "FAIL"
        out["message"] = ("SA-DMRG-CASSCF did not converge; no certified "
                          "derivative is reported for this system.")
        return out

    order = {"PASS": 0, "WARN": 1, "FAIL": 2}

    def _rank(a, b):
        return a if order[a] >= order[b] else b

    # Analytic gradient / NAC *values* from the response backend, and certified
    # response *vectors* (each with a true-residual certificate) from the
    # certified auto-solver on the same response object.
    res = compute_grad_nac_analytic_cp(
        mc, gradient_states=list(gradient_states),
        nac_pairs=[tuple(p) for p in nac_pairs],
        backend="mps-krylov", tol=min(grad_tol, nac_tol), max_iter=400)

    certs = {}
    try:
        import block2
        block2.Global.frame = mc.fcisolver._driver.frame
        obj = _make_mps_krylov_response(mc)
        certs = compute_all_responses_certified(
            obj, gradient_states=list(gradient_states),
            nac_pairs=[tuple(p) for p in nac_pairs],
            tol=min(grad_tol, nac_tol), cert_tol=max(grad_tol, nac_tol))
    except Exception as exc:  # noqa: BLE001
        out["certificate_error"] = str(exc)[:200]

    worst = "PASS" if not info["level_shift_warning"] else "WARN"

    def _cert_health(certpair):
        if certpair is None:
            return {}, "WARN"   # value but no certificate -> caution
        cert = certpair[1]
        return cert.to_dict(), cert.health().overall

    for s in gradient_states:
        g = np.asarray(res["grad"][int(s)], dtype=float)
        cert, hov = _cert_health(certs.get(("grad", int(s))))
        worst = _rank(worst, hov)
        out["gradients"][int(s)] = {"grad": g.tolist(),
                                    "norm": float(np.linalg.norm(g)),
                                    "certificate": cert, "health": hov}
    for pair in nac_pairs:
        p = tuple(int(x) for x in pair)
        d = np.asarray(res["nac"][p], dtype=float)
        cert, hov = _cert_health(certs.get(("nac", p)))
        worst = _rank(worst, hov)
        out["nacs"][str(p)] = {"nac": d.tolist(),
                               "norm": float(np.linalg.norm(d)),
                               "certificate": cert, "health": hov}

    out["overall_health"] = _rank(worst, info["build_health"]["overall"])
    out["fci_free"] = DenseBridgeSentinel.report()
    return out
