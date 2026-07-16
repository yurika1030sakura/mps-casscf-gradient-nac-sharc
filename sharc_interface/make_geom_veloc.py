#!/usr/bin/env python3
"""Build SHARC geom/veloc files from an XYZ geometry."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

ANG_TO_BOHR = 1.8897261254578281
AMU_TO_ME = 1822.888486209
KB_HARTREE = 3.166811563e-6

MASS = {
    "H": 1.007825,
    "C": 12.000000,
    "N": 14.003074,
    "O": 15.994915,
}

Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2 : 2 + natom]:
        sym, x, y, z = line.split()[:4]
        atoms.append((sym, float(x), float(y), float(z)))
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz", help="Input XYZ in Angstrom.")
    parser.add_argument("--geom", default="geom", help="Output SHARC geom path.")
    parser.add_argument("--veloc", default="veloc", help="Output SHARC velocity path.")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    atoms = read_xyz(Path(args.xyz))

    with open(args.geom, "w", encoding="utf-8") as fh:
        for sym, x, y, z in atoms:
            mass = MASS[sym]
            fh.write(
                f"{sym:>4s} {float(Z[sym]):5.1f} "
                f"{x * ANG_TO_BOHR:14.8f} {y * ANG_TO_BOHR:14.8f} {z * ANG_TO_BOHR:14.8f} "
                f"{mass:14.8f}\n"
            )

    with open(args.veloc, "w", encoding="utf-8") as fh:
        for sym, _, _, _ in atoms:
            sigma = math.sqrt(KB_HARTREE * args.temperature / (MASS[sym] * AMU_TO_ME))
            vx = rng.gauss(0.0, sigma)
            vy = rng.gauss(0.0, sigma)
            vz = rng.gauss(0.0, sigma)
            fh.write(f"{vx:15.8f} {vy:15.8f} {vz:15.8f}\n")

    print(f"Wrote {args.geom} and {args.veloc} from {args.xyz}")


if __name__ == "__main__":
    main()
