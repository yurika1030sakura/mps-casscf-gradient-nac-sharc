"""Runtime guards that make the FCI-free claim verifiable.

A beyond-FCI active space cannot form a dense determinant/FCI vector at all, so
"we never built one" must be *enforced and provable*, not merely intended.  This
module provides four tools used across the beyond-FCI drivers:

* ``determinant_dimension`` / ``FCI_FREE_THRESHOLD`` -- the determinant-space
  size past which a dense FCI vector cannot be stored.
* ``RootTracking`` -- the allowed root-tracking modes (string constants, never
  booleans), with the FCI-free subset called out explicitly.
* ``assert_fci_free_if_needed`` -- a precondition check: above the threshold it
  requires ``skip_kernel_fci_conversion``, ``mps_native_rdms``, and an FCI-free
  root-tracking mode, raising otherwise.
* ``DenseBridgeSentinel`` -- a process-wide tripwire.  Every code path that
  converts an MPS to a dense determinant/FCI array calls ``mark``; a beyond-FCI
  driver calls ``assert_not_used`` at the end of a run, so a reported result can
  certify that no dense bridge was ever entered, regardless of caller.
"""
from __future__ import annotations

from math import comb
from typing import List

# Determinant-space size above which a dense FCI vector cannot be formed.
FCI_FREE_THRESHOLD = 5.0e7


class RootTracking:
    """Allowed root-tracking modes (no booleans).

    ``FCI_OVERLAP`` forms dense CI vectors to assign roots by wavefunction
    overlap and is only valid below the threshold; the others are FCI-free.
    """

    FCI_OVERLAP = "fci_overlap"
    GAP_GUARD = "gap_guard"
    MPS_SUBSPACE = "mps_subspace"
    ENERGY_ORDERED = "energy_ordered"

    ALL = (FCI_OVERLAP, GAP_GUARD, MPS_SUBSPACE, ENERGY_ORDERED)
    FCI_FREE = (GAP_GUARD, MPS_SUBSPACE, ENERGY_ORDERED)

    @classmethod
    def normalize(cls, track_roots) -> str:
        """Map legacy boolean flags onto the string modes."""
        if track_roots is True:
            return cls.FCI_OVERLAP
        if track_roots is False or track_roots is None:
            return cls.ENERGY_ORDERED
        if track_roots not in cls.ALL:
            raise ValueError(
                f"unknown track_roots={track_roots!r}; "
                f"choose one of {cls.ALL}"
            )
        return track_roots


def determinant_dimension(ncas, nelecas) -> float:
    """Half-filled determinant product dimension binom(ncas,na)*binom(ncas,nb)."""
    if isinstance(nelecas, (tuple, list)):
        na, nb = int(nelecas[0]), int(nelecas[1])
    else:
        na = int(nelecas) // 2 + int(nelecas) % 2
        nb = int(nelecas) // 2
    return float(comb(int(ncas), na) * comb(int(ncas), nb))


def assert_fci_free_if_needed(ncas, nelecas, cfg, track_roots, context) -> float:
    """Precondition check for a derivative driver.

    Returns the determinant dimension.  Above ``FCI_FREE_THRESHOLD`` it raises
    unless the configuration is genuinely FCI-free (no dense conversion,
    MPS-native RDMs, FCI-free root tracking).
    """
    det = determinant_dimension(ncas, nelecas)
    if det >= FCI_FREE_THRESHOLD:
        if not cfg.get("skip_kernel_fci_conversion", False):
            raise RuntimeError(
                f"{context}: skip_kernel_fci_conversion required for "
                f"det_dim={det:.3e} >= {FCI_FREE_THRESHOLD:.1e}"
            )
        if not cfg.get("mps_native_rdms", False):
            raise RuntimeError(
                f"{context}: mps_native_rdms required for "
                f"det_dim={det:.3e} >= {FCI_FREE_THRESHOLD:.1e}"
            )
        if RootTracking.normalize(track_roots) == RootTracking.FCI_OVERLAP:
            raise RuntimeError(
                f"{context}: FCI-overlap root tracking forbidden for "
                f"det_dim={det:.3e} >= {FCI_FREE_THRESHOLD:.1e}; use one of "
                f"{RootTracking.FCI_FREE}"
            )
    return det


class DenseBridgeSentinel:
    """Process-wide tripwire for dense MPS->FCI/determinant conversions.

    Usage::

        DenseBridgeSentinel.reset()
        ... run a beyond-FCI driver ...
        DenseBridgeSentinel.assert_not_used("CAS(20,20) gradient")

    Any conversion path calls ``DenseBridgeSentinel.mark("name")``; if one fires
    during a beyond-FCI run, ``assert_not_used`` raises and names the offender.
    """

    used: bool = False
    calls: List[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.used = False
        cls.calls = []

    @classmethod
    def mark(cls, name: str) -> None:
        cls.used = True
        cls.calls.append(str(name))

    @classmethod
    def assert_not_used(cls, context: str) -> None:
        if cls.used:
            raise RuntimeError(
                f"{context}: a dense FCI/determinant bridge was used "
                f"({cls.calls}); a beyond-FCI run must never form one."
            )

    @classmethod
    def report(cls) -> dict:
        return {"dense_bridge_used": bool(cls.used), "calls": list(cls.calls)}
