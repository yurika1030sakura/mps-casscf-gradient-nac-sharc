"""System-agnostic health diagnostics for SA-DMRG-CASSCF derivative runs.

A user running this code on an arbitrary system must be able to tell whether a
result is trustworthy *without knowing the right answer in advance*.  Every
derivative / overlap / response calculation can be passed through
``assess_point``, which returns a :class:`SystemHealth` verdict
(``PASS`` / ``WARN`` / ``FAIL``) with a per-check breakdown and the underlying
numbers, so a problematic system is flagged explicitly rather than silently
producing wrong numbers.

Design choices:
  * Nothing here is system-specific: every threshold is a keyword with a
    documented default, and every check is skipped (not failed) when its input
    is not supplied, so the same function serves a CAS(2,2) sanity run and a
    beyond-FCI CAS(20,20) production point.
  * Status semantics:
      - ``PASS``  trustworthy as reported;
      - ``WARN``  valid but only with the documented caveat (e.g. a genuine
        near-degeneracy, where adiabatic root labels are gauge-dependent and the
        subspace-continuity diagnostic must be used instead of root identity);
      - ``FAIL``  do not trust this point.
  * The overall verdict is the worst single check (FAIL > WARN > PASS).

This is the user-facing counterpart of the internal guards in
``fci_free_guard`` and the response certificate in ``certified_response``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


def target_ss_from_spin(spin: int) -> float:
    """S(S+1) for a state with ``spin`` = 2S unpaired electrons.

    spin=0 -> 0.0 (singlet); spin=1 -> 0.75 (doublet); spin=2 -> 2.0 (triplet).
    """
    s = spin / 2.0
    return s * (s + 1.0)


@dataclass
class Check:
    name: str
    status: str
    value: Any
    message: str

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "value": self.value, "message": self.message}


@dataclass
class SystemHealth:
    overall: str = PASS
    checks: List[Check] = field(default_factory=list)

    def add(self, name, status, value, message):
        self.checks.append(Check(name, status, value, message))
        if _RANK[status] > _RANK[self.overall]:
            self.overall = status

    @property
    def trustworthy(self) -> bool:
        """True unless any check FAILed (WARN points are usable with caveats)."""
        return self.overall != FAIL

    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.status == WARN]

    def to_dict(self) -> dict:
        return {"overall": self.overall, "trustworthy": self.trustworthy,
                "checks": [c.to_dict() for c in self.checks]}

    def summary(self) -> str:
        head = f"[{self.overall}] {len(self.checks)} checks"
        bad = self.failures() + self.warnings()
        if not bad:
            return head + " — all PASS"
        lines = [head] + [f"  {c.status:4s} {c.name}: {c.message}" for c in bad]
        return "\n".join(lines)


def assess_point(
    *,
    # convergence flags (None -> skip)
    scf_converged: Optional[bool] = None,
    casscf_converged: Optional[bool] = None,
    response_converged: Optional[bool] = None,
    # spin purity: measured S^2 per state vs the target sector
    s2_per_state: Optional[List[float]] = None,
    target_spin: int = 0,
    s2_tol: float = 5.0e-2,
    # energy gap / near-degeneracy (Eh)
    gap_eh: Optional[float] = None,
    gap_warn: float = 1.0e-3,
    # active-subspace continuity (sigma_min of the cross-geometry active overlap)
    active_subspace_sigma_min: Optional[float] = None,
    sigma_warn: float = 0.98,
    sigma_fail: float = 0.50,
    # response certificate
    response_true_residual_rel: Optional[float] = None,
    response_residual_tol: float = 1.0e-7,
    root_projector_leakage: Optional[float] = None,
    leakage_tol: float = 1.0e-6,
    # FCI-free integrity (beyond-FCI runs only)
    det_dim: Optional[float] = None,
    fci_free_threshold: float = 5.0e7,
    dense_bridge_used: Optional[bool] = None,
    # bond-dimension saturation
    discarded_weight: Optional[float] = None,
    discarded_weight_warn: float = 1.0e-6,
) -> SystemHealth:
    """Assess one calculated point; return a structured PASS/WARN/FAIL verdict.

    Every argument is optional: a check is only run when its inputs are given,
    so the function adapts to whatever the caller measured.
    """
    h = SystemHealth()

    for flag, name in [(scf_converged, "scf_converged"),
                       (casscf_converged, "casscf_converged"),
                       (response_converged, "response_converged")]:
        if flag is not None:
            h.add(name, PASS if flag else FAIL, bool(flag),
                  "converged" if flag else "did NOT converge — result unreliable")

    if s2_per_state is not None:
        tgt = target_ss_from_spin(target_spin)
        worst = max((abs(float(s) - tgt) for s in s2_per_state), default=0.0)
        if worst <= s2_tol:
            h.add("spin_purity", PASS, worst,
                  f"all states in the target sector (S^2~{tgt:.3g}, max dev {worst:.2e})")
        else:
            h.add("spin_purity", FAIL, worst,
                  f"a root left the target spin sector (S^2 target {tgt:.3g}, "
                  f"max dev {worst:.2e}); enforce fix_spin / SU2 twos")

    if gap_eh is not None and gap_eh < gap_warn:
        h.add("near_degeneracy", WARN, float(gap_eh),
              f"gap {gap_eh:.2e} Eh < {gap_warn:.1e}: adiabatic labels are "
              f"gauge-dependent; use the subspace-continuity diagnostic, not root identity")
    elif gap_eh is not None:
        h.add("near_degeneracy", PASS, float(gap_eh), f"gap {gap_eh:.2e} Eh")

    if active_subspace_sigma_min is not None:
        s = float(active_subspace_sigma_min)
        if s < sigma_fail:
            h.add("subspace_continuity", FAIL, s,
                  f"active-subspace sigma_min {s:.3f} < {sigma_fail}: the active "
                  f"space is discontinuous here; finite differences / overlaps invalid")
        elif s < sigma_warn:
            h.add("subspace_continuity", WARN, s,
                  f"active-subspace sigma_min {s:.3f} < {sigma_warn}: reduce the "
                  f"step or realign the active space")
        else:
            h.add("subspace_continuity", PASS, s, f"sigma_min {s:.3f}")

    if response_true_residual_rel is not None:
        r = float(response_true_residual_rel)
        h.add("response_certificate", PASS if r <= response_residual_tol else FAIL, r,
              f"true residual {r:.2e} "
              + ("within" if r <= response_residual_tol else "EXCEEDS")
              + f" tol {response_residual_tol:.1e}")
    if root_projector_leakage is not None:
        lk = float(root_projector_leakage)
        h.add("root_leakage", PASS if lk <= leakage_tol else FAIL, lk,
              f"root-projector leakage {lk:.2e} "
              + ("within" if lk <= leakage_tol else "EXCEEDS") + f" tol {leakage_tol:.1e}")

    if det_dim is not None and dense_bridge_used is not None:
        beyond = det_dim >= fci_free_threshold
        if beyond and dense_bridge_used:
            h.add("fci_free_integrity", FAIL, True,
                  f"det_dim {det_dim:.2e} is beyond FCI but a dense FCI/determinant "
                  f"bridge was used — the FCI-free claim is violated")
        elif beyond:
            h.add("fci_free_integrity", PASS, False,
                  f"det_dim {det_dim:.2e} beyond FCI; no dense bridge used")

    if discarded_weight is not None and discarded_weight > discarded_weight_warn:
        h.add("bond_dimension", WARN, float(discarded_weight),
              f"discarded weight {discarded_weight:.2e} > {discarded_weight_warn:.1e}: "
              f"increase the bond dimension for a tighter result")
    elif discarded_weight is not None:
        h.add("bond_dimension", PASS, float(discarded_weight),
              f"discarded weight {discarded_weight:.2e}")

    return h
