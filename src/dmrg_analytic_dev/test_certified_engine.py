"""End-to-end test of the system-general certified derivative engine.

Runs ``certified_engine.compute_certified_derivatives`` on a small system and
checks that it returns a PASS verdict, a converged build, certificates at
machine precision, and a clean FCI-free integrity report -- i.e. the single
general entry point produces a certified, self-diagnosed result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sharc_interface"))
import certified_engine as ce

ANG = 1.8897261246257702


def main():
    ok = True
    atoms = ["He", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.90 * ANG]])
    out = ce.compute_certified_derivatives(
        atoms, coords, basis="3-21G", charge=1, spin=0, ncas=2, nelecas=2,
        nroots=2, gradient_states=[0], nac_pairs=[(0, 1)], max_bond_dim=64)

    print("HeH+ overall_health:", out["overall_health"])
    print("  build converged:", out["build"]["converged"],
          "| s2:", out["build"].get("s2_per_state"),
          "| wall %.0fs" % out["build"]["wall_s"])
    g = out["gradients"][0]
    gr = g["certificate"].get("true_residual_relative")
    print("  grad health=%s true_residual=%.2e norm=%.5f" % (g["health"], gr, g["norm"]))
    n = out["nacs"]["(0, 1)"]
    nr = n["certificate"].get("true_residual_relative")
    print("  nac  health=%s true_residual=%.2e norm=%.5f" % (n["health"], nr, n["norm"]))
    print("  fci_free:", out["fci_free"])

    ok &= out["overall_health"] == "PASS"
    ok &= bool(out["build"]["converged"])
    ok &= g["health"] == "PASS" and gr is not None and gr < 1.0e-10
    ok &= n["health"] == "PASS" and nr is not None and nr < 1.0e-9
    ok &= out["fci_free"]["dense_bridge_used"] is False

    print("ENGINE TEST: PASS" if ok else "ENGINE TEST: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
