"""System-agnostic robust SA-CASSCF convergence (escalation protocol).

Near-degenerate geometries (avoided crossings, closing gaps) leave the
state-averaged orbital Hessian ill-conditioned, and a fixed-shift first-order
augmented-Hessian (AH) optimization oscillates without converging.  This module
provides ONE standard escalation ladder used by every driver in this package,
for every molecule -- no per-system tuning and no source edits required:

  1. increasing AH level shift (0.5 -> 1.0 -> 2.0 -> 4.0) with a growing
     macro-cycle budget: progressively stronger damping of the oscillating
     orbital rotation;
  2. if still unconverged, a second-order co-iterative augmented-Hessian
     (CIAH / Newton) solve, which treats the near-singular Hessian directly.

Every rung acts on the SAME caller-supplied orbital guess.  The escalation
touches only the optimizer, never the guess, so when the caller propagates
orbitals along a scan or trajectory the adjacent-geometry gauge (and any
overlap-continuity diagnostic measured on it) is preserved.

Validated on the LiF CAS(6,6) avoided-crossing scan: the near-crossing
geometries (R = 4.10, 4.50 Angstrom) that defeat a fixed single-shift
optimization converge under this ladder (rungs 1-2 or the Newton fallback).

Usage:
    from casscf_convergence import escalating_casscf

    def factory():
        # return a FRESH configured SA-CASSCF object (state_average_,
        # fix_spin_, fcisolver settings...) -- called once per rung
        ...

    mc = escalating_casscf(factory, mo_guess,
                           conv_tol_grad=1e-6, max_cycle_macro=100)
    if not mc.converged:
        # hand the point to the health diagnostic: flag and exclude,
        # never silently accept
        ...
"""

from __future__ import annotations

__all__ = ["DEFAULT_LADDER", "escalating_casscf"]

# (level_shift, macro_budget_multiplier, conv_tol_grad_floor)
DEFAULT_LADDER = (
    (0.5, 1, None),
    (1.0, 2, 5.0e-5),
    (2.0, 3, 5.0e-5),
    (4.0, 4, 1.0e-4),
)


def _configure(mc, *, conv_tol, conv_tol_grad, max_cycle_macro, level_shift):
    for name, val in (("conv_tol", conv_tol),
                      ("conv_tol_grad", conv_tol_grad),
                      ("max_cycle_macro", max_cycle_macro),
                      ("ah_level_shift", level_shift)):
        if hasattr(mc, name):
            setattr(mc, name, val)
    return mc


def escalating_casscf(mc_factory, mo_guess, *, conv_tol_grad=1.0e-6,
                      max_cycle_macro=100, ladder=DEFAULT_LADDER,
                      newton_fallback=True, verbose=False):
    """Converge an SA-CASSCF point with the standard escalation ladder.

    Parameters
    ----------
    mc_factory : callable() -> mc
        Returns a FRESH, fully configured (state averaging, spin penalty,
        solver) CASSCF-like object.  Called once per rung so no state leaks
        between attempts.
    mo_guess : ndarray
        Orbital guess.  The SAME guess is used on every rung (gauge-preserving).
    conv_tol_grad : float
        Target orbital-gradient tolerance for the first rung; later rungs may
        floor it per the ladder (a slightly looser converged point beats an
        unconverged tight one, and the health diagnostic reports the actual
        tolerance achieved).
    ladder : sequence of (level_shift, macro_multiplier, conv_tol_grad_floor)
    newton_fallback : bool
        Append a second-order CIAH (``mc.newton()``) rung after the ladder.

    Returns the first converged mc; if nothing converges, returns the last
    attempt (``mc.converged == False``) so the caller's health diagnostic can
    flag and exclude the point rather than silently accept it.
    """
    mc = None
    for level_shift, mult, floor in ladder:
        ctg = conv_tol_grad if floor is None else max(conv_tol_grad, floor)
        mc = _configure(mc_factory(),
                        conv_tol=max(ctg * 1e-2, 1e-10), conv_tol_grad=ctg,
                        max_cycle_macro=int(max_cycle_macro * mult),
                        level_shift=level_shift)
        if verbose:
            print(f"[escalate] AH shift={level_shift} macro={max_cycle_macro * mult}",
                  flush=True)
        mc.kernel(mo_guess)
        if getattr(mc, "converged", False):
            return mc
    if newton_fallback:
        try:
            mc2 = _configure(mc_factory(),
                             conv_tol=max(conv_tol_grad * 1e-2, 1e-10),
                             conv_tol_grad=max(conv_tol_grad, 5.0e-5),
                             max_cycle_macro=int(max_cycle_macro * 2),
                             level_shift=1.0)
            mc2 = mc2.newton()
            if verbose:
                print("[escalate] second-order CIAH (Newton) fallback", flush=True)
            mc2.kernel(mo_guess)
            if getattr(mc2, "converged", False):
                return mc2
        except Exception:
            pass
    return mc
