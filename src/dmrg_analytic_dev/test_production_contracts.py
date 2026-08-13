"""Fast regression tests for production-facing fail-closed contracts."""
from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from certified_engine import active_nelec_tuple, progressive_schedule
from cp_dmrg_response_mps_krylov import relative_residual_target
from fci_free_guard import DenseBridgeSentinel
from site_replacement_density import mps_to_fci_generic


def main():
    # Use a determinant dimension above the FCI-free threshold so the full
    # ladder is exercised.  Every arbitrary requested maximum must be present.
    for requested in (128, 500, 900, 1000, 1200, 1600):
        schedule = progressive_schedule(20, 20, requested)
        assert schedule[-1][0] == requested, (requested, schedule)
        assert all(schedule[i][0] < schedule[i + 1][0]
                   for i in range(len(schedule) - 1)), schedule

    try:
        progressive_schedule(20, 20, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive max_bond_dim was accepted")

    # Integer active-electron counts must preserve the requested spin sector.
    assert active_nelec_tuple(4, spin=2) == (3, 1)
    assert active_nelec_tuple(3, spin=1) == (2, 1)
    assert active_nelec_tuple((3, 1), spin=2) == (3, 1)
    try:
        active_nelec_tuple(4, spin=1)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent electron/spin parity was accepted")

    # Response tolerances are relative to the physical RHS.  In particular, a
    # small RHS must not silently turn tol into an absolute stopping threshold.
    assert math.isclose(
        relative_residual_target(1.0e-7, 1.0e-6), 1.0e-13,
        rel_tol=1.0e-15, abs_tol=0.0,
    )
    try:
        relative_residual_target(1.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive relative response tolerance accepted")

    # The primitive must trip the sentinel before it does any driver work.
    DenseBridgeSentinel.reset()
    try:
        mps_to_fci_generic(None, None, 2, (1, 1))
    except Exception:
        pass
    assert DenseBridgeSentinel.used
    assert DenseBridgeSentinel.calls == [
        "site_replacement_density.mps_to_fci_generic"
    ]

    print("PRODUCTION CONTRACT TESTS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
