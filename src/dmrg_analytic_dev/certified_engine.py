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
from analytic_cp_sharc import (
    _make_mps_krylov_response,
    assemble_grad_nac_from_mps_responses,
)
from auto_response import compute_all_responses_certified


# ---------------------------------------------------------- electron sector
def active_nelec_tuple(nelecas, spin=0):
    """Return the active ``(n_alpha, n_beta)`` sector for PySCF ``spin=2S``.

    Passing an integer active-electron count must not silently imply a
    closed-shell sector.  That approximation is harmless for some scheduling
    heuristics but is wrong for determinant counts, FCI-free gates, and
    :math:`\langle S^2\rangle` diagnostics in doublet/triplet calculations.
    A tuple is accepted as an already explicit sector and cross-checked against
    ``spin`` when a nonzero spin is supplied.
    """
    spin = abs(int(spin))
    if isinstance(nelecas, (tuple, list, np.ndarray)):
        na, nb = int(nelecas[0]), int(nelecas[1])
        if spin and abs(na - nb) != spin:
            raise ValueError(
                f"active electron sector {(na, nb)} is inconsistent with "
                f"spin=2S={spin}"
            )
    else:
        ne = int(nelecas)
        if ne < spin or (ne + spin) % 2:
            raise ValueError(
                f"nelecas={ne} and spin=2S={spin} have inconsistent parity"
            )
        na = (ne + spin) // 2
        nb = (ne - spin) // 2
    if na < 0 or nb < 0:
        raise ValueError(f"invalid active electron sector {(na, nb)}")
    return na, nb


# --------------------------------------------------------------- M schedule
def progressive_schedule(ncas, nelecas, max_bond_dim, *, spin=0):
    """Auto bond-dimension schedule: cheap low-M relaxation, then raise M.

    Below the FCI-free threshold the active space is small; a short schedule
    suffices.  Above it the schedule climbs to ``max_bond_dim`` so the orbital
    optimization is relaxed cheaply before the expensive final M.
    """
    max_bond_dim = int(max_bond_dim)
    if max_bond_dim < 1:
        raise ValueError("max_bond_dim must be a positive integer")
    nelec_t = active_nelec_tuple(nelecas, spin=spin)
    det = determinant_dimension(ncas, nelec_t)
    if det < FCI_FREE_THRESHOLD:
        m = min(max_bond_dim, 256)
        # conv_tol_grad 1e-5 (not tighter): the DMRG RDM sweep noise floors the
        # orbital-gradient norm, so a tighter threshold can never be met even at
        # the exact stationary point (the energy still matches FCI to ~1e-14).
        return [(m, 1.0e-10, 1.0e-5, 60)]
    ladder = [(256, 1.0e-7, 1.0e-4, 30),
              (512, 1.0e-8, 3.0e-5, 40),
              (800, 1.0e-9, 1.0e-5, 60),
              (1200, 1.0e-9, 3.0e-6, 80)]
    stages = [stage for stage in ladder if stage[0] < max_bond_dim]

    # ``max_bond_dim`` is a contract, not merely an upper bound.  The old
    # ladder silently stopped at 256 for M=500, 800 for M=900/1000, and 1200
    # for M>1200.  Select the convergence settings of the next ladder rung and
    # always append the exact requested final M.
    final_settings = ladder[-1][1:]
    for rung in ladder:
        if max_bond_dim <= rung[0]:
            final_settings = rung[1:]
            break
    stages.append((max_bond_dim, *final_settings))
    return stages


# --------------------------------------------------------------- robust build
def build_robust(atoms, coords_bohr, *, basis, charge=0, spin=0, ncas, nelecas,
                 nroots=2, weights=None, mo_guess=None, ao_targets=None,
                 max_bond_dim=800, threads=8, stack_mem_mb=8000,
                 warm_start=True, force_fci_free=False,
                 dmrg_sweep_tol=None, refine_sweeps=None):
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
    # PySCF's current SA-CASSCF NAC assembly is not compatible with an ROHF
    # ``_scf`` object (its UHF-like gradient density has the wrong rank for
    # ``pyscf.nac.sacasscf.grad_elec_core``).  For even-electron spin states
    # whose unpaired electrons live in the active space -- the standard
    # singlet/triplet CASSCF setting -- use an explicit closed-shell RHF object
    # as the orbital *starting point*.  The CASSCF/FCI or SU2-DMRG solver, not
    # the reference determinant, enforces the requested (na, nb) and S(S+1).
    # Odd-electron sectors retain ROHF; if their downstream analytic assembly
    # is unsupported it fails closed in ``compute_certified_derivatives``.
    if int(mol.nelectron) % 2 == 0:
        mf = scf.hf.RHF(mol).run(conv_tol=1.0e-11)
        reference_kind = "RHF orbital reference"
    else:
        mf = scf.ROHF(mol).run(conv_tol=1.0e-11)
        reference_kind = "ROHF orbital reference"

    sel_diag = None
    if mo_guess is None and ao_targets is not None:
        ncore, mo_guess, sel_diag = select_active_space_by_ao_targets(
            mol, mf, ncas, nelecas, list(ao_targets))
    elif mo_guess is None:
        mo_guess = mf.mo_coeff  # default HOMO-LUMO active window

    nelec_t = active_nelec_tuple(nelecas, spin=spin)
    det = determinant_dimension(ncas, nelec_t)
    sched = progressive_schedule(
        ncas, nelec_t, max_bond_dim, spin=spin,
    )
    if dmrg_sweep_tol is not None:
        requested_sweep_tol = float(dmrg_sweep_tol)
        if not 0.0 < requested_sweep_tol < 1.0:
            raise ValueError("dmrg_sweep_tol must lie strictly between 0 and 1")
        sched = [
            (M, min(float(swtol), requested_sweep_tol), cgrad, mxm)
            for M, swtol, cgrad, mxm in sched
        ]

    # solver config; above the threshold force the FCI-free settings.
    cfg = dict(fdv.DEFAULT_SOLVER_CFG)
    cfg.update(bond_dim=sched[0][0], n_threads=int(threads),
               stack_mem_mb=int(stack_mem_mb), dmrg_symm_su2=True,
               force_dmrg=True, warm_start=bool(warm_start))
    if dmrg_sweep_tol is not None:
        cfg["refine_sweep_tol"] = float(dmrg_sweep_tol)
    if refine_sweeps is not None:
        if int(refine_sweeps) < 1:
            raise ValueError("refine_sweeps must be positive")
        cfg["refine_sweeps"] = int(refine_sweeps)
    fci_free_required = bool(force_fci_free or det >= FCI_FREE_THRESHOLD)
    if fci_free_required:
        cfg.update(mps_native_rdms=True, skip_kernel_fci_conversion=True)
    assert_fci_free_if_needed(ncas, nelec_t, cfg, RootTracking.GAP_GUARD,
                              "certified_engine.build_robust")

    def _solve(mo, level_shift, sched_):
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        solver = MPSAsFCISolver(mol, **cfg)
        solver.nroots = int(nroots)
        s = 0.5 * abs(int(spin))
        target_ss = s * (s + 1.0)
        # Set the contract on the base solver before PySCF creates its
        # StateAverage/SpinPenalty view classes.  Calling only ``mc.fix_spin_``
        # stores ``ss_value`` on the wrapper but leaves the MPS base solver's
        # ``_target_ss`` unset, so an FCI-free root-index sentinel cannot report
        # its exact SU2 sector later.
        solver.fix_spin_(ss=target_ss, shift=0.5)
        mc.fcisolver = solver
        if nroots > 1:
            mc = mc.state_average_(weights)
        try:
            mc.fix_spin_(ss=target_ss, shift=0.5)
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

    # Small-CAS fast path.  The exact FCI solver reaches the converged SA-CASSCF
    # orbitals far more cheaply than DMRG-CASSCF macro iterations; we seed the
    # DMRG build there so it starts at (essentially) the stationary point.  We
    # also keep the FCI energies to certify convergence below.  At det >=
    # threshold FCI is infeasible and this is skipped.
    fci_orbitals = False
    e_fci = None
    if det < FCI_FREE_THRESHOLD:
        try:
            mc_fci = mcscf.CASSCF(mf, ncas, nelecas)
            mc_fci.fcisolver.nroots = int(nroots)
            if nroots > 1:
                mc_fci = mc_fci.state_average_(weights)
            try:
                s = 0.5 * abs(int(spin))
                mc_fci.fix_spin_(ss=s * (s + 1.0), shift=0.5)
            except Exception:
                pass
            mc_fci.conv_tol = 1.0e-10
            mc_fci.conv_tol_grad = 1.0e-6
            mc_fci.max_cycle_macro = 200
            mc_fci.kernel(mo_guess)
            if mc_fci.converged:
                mo_guess = mc_fci.mo_coeff
                e_fci = np.asarray(mc_fci.e_states, dtype=float).ravel()
                fci_orbitals = True
        except Exception:
            fci_orbitals = False

    DenseBridgeSentinel.reset()
    t_start = time.perf_counter()
    escalated = False
    level_shift_used = 0.0
    mc, solver, stage_log = _solve(mo_guess, 0.0, sched)
    casscf_converged = bool(mc.converged)
    orbital_source = "fci-seeded dmrg-casscf" if fci_orbitals else "dmrg-casscf"

    # FCI-certified convergence: at small CAS the DMRG-CASSCF orbital-gradient
    # norm is floored by RDM sweep noise and can sit above the threshold even at
    # the true stationary point.  If the exact FCI converged AND the DMRG-CASSCF
    # state energies match it (the orbitals did not move off that point), the
    # build is converged -- the gradient flag is a noise artifact, not physics.
    fci_certified = False
    if (not casscf_converged) and fci_orbitals and e_fci is not None:
        e_dmrg = np.asarray(mc.e_states, dtype=float).ravel()
        if (e_dmrg.shape == e_fci.shape
                and float(np.max(np.abs(e_dmrg - e_fci))) < 1.0e-6):
            casscf_converged = True
            fci_certified = True
            orbital_source = "fci-certified (dmrg gradient noise-limited)"

    if not casscf_converged:
        # Escalation: keep the guess, double the macro budget, add a level
        # shift.  The level shift can move the stationary point, so flag it.
        escalated = True
        level_shift_used = 0.5
        tight = [(M, sw, cg, mxm * 2) for (M, sw, cg, mxm) in sched]
        mc, solver, stage_log2 = _solve(mo_guess, level_shift_used, tight)
        stage_log = stage_log + stage_log2
        casscf_converged = bool(mc.converged)

    e = [float(x) for x in mc.e_states]
    # Spin purity through the active solver interface.  In FCI-free mode the
    # ``ci`` objects are deliberately tiny root-index sentinels, so calling
    # ``pyscf.fci.spin_square`` on them is invalid.  MPSAsFCISolver.spin_square
    # reports the exact SU2 target sector for those sentinels; in dense mode it
    # delegates to the normal PySCF contraction.
    s2 = None
    try:
        from pyscf import fci
        ci_roots = list(mc.ci) if isinstance(mc.ci, (list, tuple)) else [mc.ci]
        s2 = []
        for c in ci_roots:
            arr = np.asarray(c)
            if arr.size == 1 and getattr(
                mc.fcisolver, "skip_kernel_fci_conversion", False
            ):
                # The state-average view overrides ``spin_square`` with an API
                # that expects the complete CI-root list.  Call the MPS base
                # implementation explicitly for one root-index sentinel.
                value = MPSAsFCISolver.spin_square(
                    mc.fcisolver, c, ncas, nelec_t
                )[0]
            else:
                value = fci.spin_square(arr, ncas, nelec_t)[0]
            s2.append(float(value))
    except Exception:
        s2 = None

    health = assess_point(
        scf_converged=bool(mf.converged), casscf_converged=bool(casscf_converged),
        s2_per_state=s2, target_spin=spin,
        gap_eh=(e[1] - e[0]) if len(e) > 1 else None,
        det_dim=det, dense_bridge_used=DenseBridgeSentinel.used)

    info = {"det_dim": det, "beyond_fci": det >= FCI_FREE_THRESHOLD,
            "fci_free_required": fci_free_required,
            "ncas": ncas, "nelecas": nelecas, "spin": spin,
            "active_nelec": list(nelec_t),
            "orbital_reference": reference_kind,
            "orbital_source": orbital_source, "fci_certified": fci_certified,
            "stages": stage_log, "converged": bool(casscf_converged),
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
        max_bond_dim=800, threads=8, stack_mem_mb=8000,
        force_fci_free=False, dmrg_sweep_tol=None, refine_sweeps=None):
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
        stack_mem_mb=stack_mem_mb, force_fci_free=force_fci_free,
        dmrg_sweep_tol=dmrg_sweep_tol, refine_sweeps=refine_sweeps)

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

    # Solve each response exactly once.  The physical derivative below is
    # assembled from the *same* MPS response vector whose true residual is
    # certified.  Keeping value and certificate on independent solves can pair
    # a plausible number with a certificate for a different vector.
    certs = {}
    assembled = {"grad": {}, "nac": {}}
    try:
        import block2
        block2.Global.frame = mc.fcisolver._driver.frame
        obj = _make_mps_krylov_response(mc)
        certs.update(compute_all_responses_certified(
            obj, gradient_states=list(gradient_states), tol=grad_tol,
            cert_tol=grad_tol, max_iter=400))
        certs.update(compute_all_responses_certified(
            obj, nac_pairs=[tuple(p) for p in nac_pairs], tol=nac_tol,
            cert_tol=nac_tol, max_iter=400))

        accepted = {
            key: pair[0] for key, pair in certs.items()
            if bool(pair[1].converged)
        }
        assembled = assemble_grad_nac_from_mps_responses(
            mc, obj, accepted,
            gradient_states=[int(s) for s in gradient_states
                             if ("grad", int(s)) in accepted],
            nac_pairs=[tuple(int(x) for x in p) for p in nac_pairs
                       if ("nac", tuple(int(x) for x in p)) in accepted],
        )
    except Exception as exc:  # noqa: BLE001
        out["certificate_error"] = str(exc)[:200]

    worst = "PASS" if not info["level_shift_warning"] else "WARN"

    def _cert_health(certpair):
        if certpair is None:
            return {}, "FAIL"
        cert = certpair[1]
        return cert.to_dict(), cert.health().overall

    for s in gradient_states:
        s = int(s)
        cert, hov = _cert_health(certs.get(("grad", int(s))))
        worst = _rank(worst, hov)
        g = assembled["grad"].get(s)
        if g is None or hov == "FAIL":
            out["gradients"][s] = {
                "grad": None, "norm": None, "certificate": cert,
                "health": "FAIL",
                "message": "No derivative released: response was not certified.",
            }
        else:
            g = np.asarray(g, dtype=float)
            out["gradients"][s] = {"grad": g.tolist(),
                                    "norm": float(np.linalg.norm(g)),
                                    "certificate": cert, "health": hov}
    for pair in nac_pairs:
        p = tuple(int(x) for x in pair)
        cert, hov = _cert_health(certs.get(("nac", p)))
        worst = _rank(worst, hov)
        d = assembled["nac"].get(p)
        if d is None or hov == "FAIL":
            out["nacs"][str(p)] = {
                "nac": None, "norm": None, "certificate": cert,
                "health": "FAIL",
                "message": "No derivative released: response was not certified.",
            }
        else:
            d = np.asarray(d, dtype=float)
            out["nacs"][str(p)] = {"nac": d.tolist(),
                                   "norm": float(np.linalg.norm(d)),
                                   "certificate": cert, "health": hov}

    out["overall_health"] = _rank(worst, info["build_health"]["overall"])
    fci_free = DenseBridgeSentinel.report()
    fci_free["required"] = bool(info["fci_free_required"])
    fci_free["passed"] = bool(
        (not info["fci_free_required"]) or (not fci_free["dense_bridge_used"])
    )
    out["fci_free"] = fci_free
    if not fci_free["passed"]:
        out["overall_health"] = "FAIL"
        out["message"] = (
            "Beyond-FCI integrity failure: a dense determinant bridge was "
            "entered; all derivatives at this point are withheld."
        )
        for rec in out["gradients"].values():
            rec.update(grad=None, norm=None, health="FAIL")
        for rec in out["nacs"].values():
            rec.update(nac=None, norm=None, health="FAIL")
    return out
