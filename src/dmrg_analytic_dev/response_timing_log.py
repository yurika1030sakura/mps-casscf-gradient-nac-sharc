"""Structured wall-time / memory instrumentation for the CP-DMRG-CASSCF
response backend.

The earlier development builds printed ad-hoc progress lines to stdout while
diagnosing where the MPS response spent its time.  Those lines made the cost
of each stage hard to aggregate and looked like leftover debugging output.
This module replaces them with a context-manager timer that records one row
per instrumented stage and serialises the result to JSON, so the per-component
wall-time and peak-memory breakdown can be tabulated directly from a run.

Usage::

    timing = TimingLog(enabled=True)
    with timing.section("state_rdm12", state=0, M=512):
        d1, d2 = build_state_rdm(...)
    ...
    timing.write("response_timing.json")

A disabled log (the default) has negligible overhead: ``section`` yields
immediately without reading the clock or the resident set size.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager

try:
    import resource

    def _peak_rss_kb() -> int:
        # ru_maxrss is kB on Linux, bytes on macOS; callers on the cluster are
        # Linux, so kB is reported as-is.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
except ImportError:  # pragma: no cover - resource is POSIX-only
    def _peak_rss_kb() -> int:
        return 0


class TimingLog:
    """Collect ``(name, wall_s, peak_rss_kb, metadata)`` rows for response stages.

    Parameters
    ----------
    enabled
        When ``False`` (default) the timer is inert and ``section`` adds no
        measurable overhead, so production paths can leave calls in place.
    label
        Optional run label stored in the serialised header.
    """

    def __init__(self, enabled: bool = False, label: str = ""):
        self.enabled = bool(enabled)
        self.label = str(label)
        self.events: list[dict] = []

    @contextmanager
    def section(self, name: str, **meta):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - t0
            self.events.append(
                {
                    "name": name,
                    "wall_s": wall,
                    "peak_rss_kb": _peak_rss_kb(),
                    **meta,
                }
            )

    def add(self, name: str, wall_s: float, **meta) -> None:
        """Record a stage whose wall time was measured outside a ``section``."""
        if not self.enabled:
            return
        self.events.append(
            {
                "name": name,
                "wall_s": float(wall_s),
                "peak_rss_kb": _peak_rss_kb(),
                **meta,
            }
        )

    def totals_by_name(self) -> dict[str, float]:
        """Aggregate wall time per stage name (for a breakdown table)."""
        out: dict[str, float] = {}
        for e in self.events:
            out[e["name"]] = out.get(e["name"], 0.0) + e.get("wall_s", 0.0)
        return out

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "events": self.events,
            "totals_by_name": self.totals_by_name(),
        }

    def write(self, path: str) -> None:
        if not self.enabled:
            return
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
