#!/usr/bin/env python3
"""Combine parallel partial derivative JSON files from anthracene benchmark.

Each partial job evaluates the same fixed-orbital DMRG roots but only one
response RHS, for example grad0, grad1, or NAC(0,1).  This script merges the
partial derivative maps into the legacy full-result fields used by plotting and
manuscript scripts.  It refuses silently inconsistent inputs by reporting the
maximum root-energy spread across the partial jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def phase_aware_l2(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Combined JSON output path.")
    parser.add_argument("partials", nargs="+", help="Partial derivative JSON files.")
    parser.add_argument(
        "--energy-spread-warn-mEh",
        type=float,
        default=0.05,
        help="Warn if repeated DMRG root energies differ by more than this.",
    )
    return parser.parse_args()


def _result(payload: dict) -> dict:
    results = payload.get("results") or []
    if len(results) != 1:
        raise ValueError("each partial JSON must contain exactly one result")
    return results[0]


def main() -> int:
    args = parse_args()
    paths = [Path(p).resolve() for p in args.partials]
    payloads = [json.loads(path.read_text()) for path in paths]
    records = [_result(payload) for payload in payloads]

    base = copy.deepcopy(payloads[0])
    rec = copy.deepcopy(records[0])
    nstates = int(base.get("nroots", 2))

    energies = []
    for item in records:
        e = item.get("response_energies_hartree", item["projected_energies_hartree"])
        energies.append(np.asarray(e, dtype=float))
    min_len = min(len(e) for e in energies)
    spread_mEh = 1000.0 * np.ptp(np.stack([e[:min_len] for e in energies]), axis=0)

    gradients = {}
    nacs = {}
    diagnostics = {"response_mode": "mps-krylov", "response_solver": {"grad": {}, "nac": {}}}
    for item in records:
        for key, value in item.get("gradients_by_state_hartree_per_bohr", {}).items():
            gradients[int(key)] = np.asarray(value, dtype=float)
        for key, value in item.get("nac_by_pair_au", {}).items():
            left, right = key.split("-", 1)
            nacs[(int(left), int(right))] = np.asarray(value, dtype=float)
        diag = item.get("derivative_diagnostics", {}).get("response_solver", {})
        diagnostics["response_solver"]["grad"].update(diag.get("grad", {}))
        diagnostics["response_solver"]["nac"].update(diag.get("nac", {}))

    rec["stage"] = (
        "derivatives_complete"
        if sorted(gradients) == list(range(nstates)) and (0, 1) in nacs
        else "partial_derivatives_complete"
    )
    rec["combined_from_partial_derivative_json"] = [str(p) for p in paths]
    rec["partial_root_energy_spread_mEh"] = [float(x) for x in spread_mEh]
    rec["gradient_states_computed"] = sorted(int(k) for k in gradients)
    rec["nac_pairs_computed"] = [f"{i}-{j}" for i, j in sorted(nacs)]
    rec["gradients_by_state_hartree_per_bohr"] = {
        str(k): gradients[k].tolist() for k in sorted(gradients)
    }
    rec["gradient_norms_by_state_hartree_per_bohr"] = {
        str(k): float(np.linalg.norm(gradients[k])) for k in sorted(gradients)
    }
    rec["nac_by_pair_au"] = {
        f"{i}-{j}": nacs[(i, j)].tolist() for i, j in sorted(nacs)
    }
    rec["nac_norms_by_pair_au"] = {
        f"{i}-{j}": float(np.linalg.norm(nacs[(i, j)])) for i, j in sorted(nacs)
    }
    rec["derivative_diagnostics"] = diagnostics
    rec["runtime_derivatives_s"] = float(
        max(float(item.get("runtime_derivatives_s", 0.0)) for item in records)
    )
    rec["runtime_derivatives_wall_model"] = "max_partial_runtime_s"

    if sorted(gradients) == list(range(nstates)):
        rec["gradients_hartree_per_bohr"] = [
            gradients[i].tolist() for i in range(nstates)
        ]
        rec["gradient_norms_hartree_per_bohr"] = [
            float(np.linalg.norm(gradients[i])) for i in range(nstates)
        ]
    if (0, 1) in nacs:
        rec["nac_01_au"] = nacs[(0, 1)].tolist()
        rec["nac_norm_au"] = float(np.linalg.norm(nacs[(0, 1)]))

    fci_ref = base.get("fci_reference")
    if fci_ref is not None:
        e = np.asarray(rec.get("response_energies_hartree", rec["projected_energies_hartree"]))
        e_ref = np.asarray(fci_ref["energies_hartree"])
        delta = {
            "energy_mEh": [
                float(1000.0 * (e[i] - e_ref[i]))
                for i in range(min(len(e), len(e_ref)))
            ],
            "gradient_l2_by_state_mEh_per_bohr": {},
        }
        g_ref = [np.asarray(x) for x in fci_ref["gradients_hartree_per_bohr"]]
        for state, grad in gradients.items():
            if state < len(g_ref):
                delta["gradient_l2_by_state_mEh_per_bohr"][str(state)] = float(
                    np.linalg.norm(grad - g_ref[state]) * 1000.0
                )
        if sorted(gradients) == list(range(min(nstates, len(g_ref)))):
            delta["gradient_l2_per_state_mEh_per_bohr"] = [
                delta["gradient_l2_by_state_mEh_per_bohr"][str(i)]
                for i in range(min(nstates, len(g_ref)))
            ]
        if (0, 1) in nacs:
            delta["nac_l2_phase_aware_au"] = phase_aware_l2(
                nacs[(0, 1)], np.asarray(fci_ref["nac_01_au"])
            )
            delta["nac_l2_by_pair_phase_aware_au"] = {
                "0-1": delta["nac_l2_phase_aware_au"]
            }
        rec["delta_vs_fci_reference"] = delta

    base["results"] = [rec]
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(base, indent=2) + "\n")

    max_spread = float(max(spread_mEh)) if spread_mEh.size else 0.0
    print(f"wrote {out}")
    print(f"max partial root-energy spread = {max_spread:.6f} mEh")
    if max_spread > float(args.energy_spread_warn_mEh):
        print("WARNING: partial jobs did not reproduce identical root energies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
