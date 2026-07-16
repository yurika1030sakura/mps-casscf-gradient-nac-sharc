#!/usr/bin/env python3
"""Merge isolated parallel BVOE runs into the main benchmark dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parallel-root",
        type=Path,
        default=ROOT / "parallel_runs",
    )
    parser.add_argument("--dest-data", type=Path, default=ROOT / "data_phase2")
    parser.add_argument(
        "--dest-summary",
        type=Path,
        default=ROOT / "summary_phase2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parallel_root = args.parallel_root.resolve()
    dest_data = args.dest_data.resolve()
    dest_summary = args.dest_summary.resolve()
    dest_data.mkdir(parents=True, exist_ok=True)

    summary = load_json(dest_summary) if dest_summary.exists() else {}
    merged_systems = []
    copied_files = 0
    for summary_path in sorted(parallel_root.glob("*/summary_phase2.json")):
        run_dir = summary_path.parent
        run_summary = load_json(summary_path)
        data_dir = run_dir / "data_phase2"
        if not data_dir.exists():
            continue
        for system, entry in sorted(run_summary.items()):
            summary[system] = entry
            merged_systems.append(system)
        for path in sorted(data_dir.glob("*.json")):
            shutil.copy2(path, dest_data / path.name)
            copied_files += 1

    write_json(dest_summary, summary)
    print(f"Merged systems: {sorted(set(merged_systems))}")
    print(f"Copied JSON files: {copied_files}")
    print(f"Wrote {dest_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
