"""A posteriori certificate for a solved CP-DMRG-CASSCF response vector.

The response solvers (global MPS-Krylov ``solve_mps`` and the sweep-localized
``solve_state_sweep_schur``) already accept a solution only after checking the
*true* residual ``||b - A z|| / ||b||`` of the projected coupled-perturbed
equation -- not the (optimistic) projected Arnoldi residual, which lossy MPS
addition makes unreliable.  This module collects that check, together with the
solution's bond dimension and its leakage out of the reference-root space, into
a single object that can be serialized next to every gradient/NAC result.

The certificate is about the linear solve: it states that the accepted vector
satisfies the projected response equation to a stated tolerance, with an a
posteriori error governed by the smallest non-zero singular value of the
(gauge/root-projected) response operator.  It is deliberately separate from the
finite-M truncation error and from the SA-DMRG-CASSCF model error, which are
reported independently.  No solver behaviour is changed by computing it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ResponseCertificate:
    """A posteriori certificate for one accepted response solution."""

    state: int
    solver: str
    converged: bool
    rhs_norm: float
    true_residual_norm: float
    true_residual_relative: float
    residual_tol: float
    orbital_dim: int
    response_bond_dim: int
    root_projector_leakage: Optional[float]
    wall_s: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _orbital_dim(obj, z) -> int:
    """Number of independent orbital-rotation parameters in the solution."""
    try:
        return int(obj.mc.pack_uniq_var(obj._canonical_kappa(z.kappa)).size)
    except Exception:
        return int(np.asarray(z.kappa).size)


def _response_bond_dim(z) -> int:
    bond = 0
    for m in getattr(z, "ci_mps", []) or []:
        try:
            bond = max(bond, int(m.info.bond_dim))
        except Exception:
            continue
    return int(bond)


def _root_projector_leakage(obj, z) -> Optional[float]:
    """Largest overlap of the response active-space block with any reference
    root.  The CP response lives in the complement of the reference roots, so a
    well-converged, well-projected solution has leakage near zero; a growing
    value flags loss of orthogonality (e.g. near a root crossing)."""
    states = getattr(obj, "_state_mps", None)
    ci = getattr(z, "ci_mps", None)
    if not states or not ci:
        return None
    worst = 0.0
    seen = False
    for comp in ci:
        if comp is None:
            continue
        for root in states:
            try:
                worst = max(worst, abs(float(obj._mps_overlap(root, comp))))
                seen = True
            except Exception:
                continue
    return worst if seen else None


def certify_response(
    obj,
    z,
    *,
    state: int,
    tol: float,
    solver: str,
    rhs=None,
    wall_s: Optional[float] = None,
    leakage_tol: float = 1.0e-6,
    check_leakage: bool = True,
    extra: Optional[dict] = None,
) -> ResponseCertificate:
    """Build the certificate for an accepted response solution ``z``.

    Recomputes the explicit residual ``b - A z`` against the global coupled
    operator (``obj.matvec_mps``); this is the same arbiter the solvers use, so
    the certificate never disagrees with the accepted solution.  ``rhs`` is
    rebuilt for ``state`` if not supplied.
    """
    state = int(state)
    if rhs is None:
        rhs = obj.build_rhs_mps(state)
    rhs_norm = float(rhs.norm())
    resid = rhs.add_scaled(obj.matvec_mps(z), -1.0, label=f"CERT-RES{state}")
    r_norm = float(resid.norm())
    rel = r_norm / max(rhs_norm, 1.0e-300)

    leakage = _root_projector_leakage(obj, z) if check_leakage else None
    converged = bool(rel < tol and (leakage is None or leakage < leakage_tol))

    return ResponseCertificate(
        state=state,
        solver=str(solver),
        converged=converged,
        rhs_norm=rhs_norm,
        true_residual_norm=r_norm,
        true_residual_relative=rel,
        residual_tol=float(tol),
        orbital_dim=_orbital_dim(obj, z),
        response_bond_dim=_response_bond_dim(z),
        root_projector_leakage=leakage,
        wall_s=wall_s,
        extra=dict(extra or {}),
    )
