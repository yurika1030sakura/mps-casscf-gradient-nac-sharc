#!/usr/bin/env python3
"""Summarize a short SHARC trajectory smoke test.

The output is intentionally small and manuscript-friendly: it records whether
the SHARC master run completed, how many trajectory frames were written, and
whether the latest QM.out contains the electronic-structure blocks required by
the DMRG-CASSCF interface claim.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def count_xyz_frames(path: Path) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    frames = 0
    while i < len(lines):
        try:
            natom = int(lines[i].strip())
        except Exception:
            i += 1
            continue
        if natom <= 0:
            i += 1
            continue
        frames += 1
        i += natom + 2
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    sharc_log = root / "sharc.log"
    output_log = root / "output.log"
    output_xyz = root / "output.xyz"
    output_dat = root / "output.dat"
    qm_out = root / "QM" / "QM.out"
    qm_err = root / "QM" / "QM.err"

    log_text = sharc_log.read_text(errors="replace") if sharc_log.exists() else ""
    output_log_text = (
        output_log.read_text(errors="replace") if output_log.exists() else ""
    )
    qm_text = qm_out.read_text(errors="replace") if qm_out.exists() else ""
    err_text = qm_err.read_text(errors="replace") if qm_err.exists() else ""

    completed = (
        "Program SHARC successfully terminated" in log_text
        or "Program SHARC finished" in log_text
        or "Total wallclock time" in log_text
        or "Total wallclock time" in output_log_text
    )
    failed = any(
        marker in log_text
        for marker in (
            "QM call was not successful",
            "STOP 1",
            "aborting the run",
        )
    ) or "Traceback" in err_text

    qm_blocks = {
        "H": "! 1 Hamiltonian Matrix" in qm_text,
        "DM": "! 2 Dipole Moment Matrices" in qm_text,
        "GRAD": "! 3 Gradient Vectors" in qm_text,
        "NACDR": "! 5 Nonadiabatic couplings" in qm_text,
        "PHASES": "! 7 Phases" in qm_text,
        "RUNTIME": "! 8 Runtime" in qm_text,
    }

    step_numbers = [
        int(match.group(1))
        for match in re.finditer(r"!\s*0\s+Step\s*\n\s*(\d+)", output_dat.read_text(errors="replace"))
    ] if output_dat.exists() else []

    summary = {
        "label": args.label,
        "root": str(root),
        "sharc_completed": bool(completed and not failed),
        "sharc_failed": bool(failed),
        "xyz_frames": count_xyz_frames(output_xyz),
        "output_dat_steps": step_numbers,
        "max_output_dat_step": max(step_numbers) if step_numbers else None,
        "qmout_blocks": qm_blocks,
        "qmout_all_required_blocks": all(qm_blocks.values()),
        "sharc_log_tail": "\n".join((log_text or output_log_text).splitlines()[-30:]),
        "qm_err_tail": "\n".join(err_text.splitlines()[-30:]),
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
