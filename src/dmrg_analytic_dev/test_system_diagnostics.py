"""Unit test for the system-agnostic health diagnostics.

Verifies that assess_point flags each failure mode a user could hit on their
own system: non-convergence, wrong spin sector, active-space discontinuity,
unconverged response, and an accidental dense FCI bridge in a beyond-FCI run --
and that a genuine near-degeneracy is a WARN (usable with caveat), not a FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from system_diagnostics import assess_point, PASS, WARN, FAIL


def _check(h, name):
    return next(c for c in h.checks if c.name == name)


def main():
    ok = True

    # 1) Clean point -> PASS, trustworthy.
    h = assess_point(scf_converged=True, casscf_converged=True,
                     response_converged=True, s2_per_state=[0.0, 0.0],
                     target_spin=0, gap_eh=0.05,
                     active_subspace_sigma_min=0.999,
                     response_true_residual_rel=2e-9,
                     root_projector_leakage=1e-10,
                     det_dim=3.4e10, dense_bridge_used=False)
    ok &= h.overall == PASS and h.trustworthy
    print(f"clean point: {h.overall} (expect PASS)")

    # 2) Non-converged CASSCF -> FAIL.
    h = assess_point(casscf_converged=False)
    ok &= h.overall == FAIL and not h.trustworthy
    print(f"non-converged: {h.overall} (expect FAIL)")

    # 3) Triplet leak with singlet target -> spin_purity FAIL.
    h = assess_point(s2_per_state=[0.0, 2.0], target_spin=0)
    ok &= _check(h, "spin_purity").status == FAIL
    print(f"triplet leak: spin_purity={_check(h,'spin_purity').status} (expect FAIL)")

    # 4) Doublet target with correct S^2 -> spin_purity PASS.
    h = assess_point(s2_per_state=[0.75, 0.75], target_spin=1)
    ok &= _check(h, "spin_purity").status == PASS
    print(f"doublet ok: spin_purity={_check(h,'spin_purity').status} (expect PASS)")

    # 5) Genuine near-degeneracy -> WARN, still trustworthy.
    h = assess_point(gap_eh=1.5e-4, active_subspace_sigma_min=0.999)
    ok &= _check(h, "near_degeneracy").status == WARN and h.trustworthy
    print(f"near-degeneracy: {h.overall} trustworthy={h.trustworthy} (expect WARN/True)")

    # 6) Active-space discontinuity -> FAIL.
    h = assess_point(active_subspace_sigma_min=0.3)
    ok &= _check(h, "subspace_continuity").status == FAIL
    print(f"discontinuity: {_check(h,'subspace_continuity').status} (expect FAIL)")

    # 7) Response residual over tol -> FAIL.
    h = assess_point(response_true_residual_rel=1e-4, response_residual_tol=1e-7)
    ok &= _check(h, "response_certificate").status == FAIL
    print(f"bad residual: {_check(h,'response_certificate').status} (expect FAIL)")

    # 8) Beyond-FCI + dense bridge used -> FAIL.
    h = assess_point(det_dim=3.4e10, dense_bridge_used=True)
    ok &= _check(h, "fci_free_integrity").status == FAIL
    print(f"fci bridge leak: {_check(h,'fci_free_integrity').status} (expect FAIL)")

    print("ALL PASS" if ok else "FAILURES DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
