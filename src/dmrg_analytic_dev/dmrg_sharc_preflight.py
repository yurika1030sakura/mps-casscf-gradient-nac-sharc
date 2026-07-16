#!/usr/bin/env python3
"""Preflight checks for large-active-space DMRG-SHARC jobs.

The methods-paper validation path can project DMRG roots to PySCF CI arrays
when the determinant space is still moderate.  Production CAS(24,18)-scale
work should not silently enter that path.  This small helper reports the
determinant dimension and recommends the currently supported workflow.
"""

from __future__ import annotations

import argparse
import math


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(int(n), int(k))


def nelec_tuple(nelecas: int, spin: int = 0) -> tuple[int, int]:
    if (nelecas + spin) % 2:
        raise ValueError("nelecas and spin parity are inconsistent")
    na = (int(nelecas) + int(spin)) // 2
    nb = int(nelecas) - na
    return na, nb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ncas", type=int, required=True)
    p.add_argument("--nelecas", type=int, required=True)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--nroots", type=int, default=2)
    p.add_argument("--max-fci-dets", type=int, default=20_000_000)
    p.add_argument(
        "--needs",
        default="h,dm,grad,nacdr",
        help="Comma-separated SHARC quantities: h,dm,grad,nacdr,overlap.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    na, nb = nelec_tuple(args.nelecas, args.spin)
    n_alpha = comb(args.ncas, na)
    n_beta = comb(args.ncas, nb)
    n_det = n_alpha * n_beta
    needs = {x.strip().lower() for x in args.needs.replace(" ", ",").split(",") if x.strip()}
    projected_ok = n_det <= int(args.max_fci_dets)

    print("DMRG-SHARC preflight")
    print(f"CAS({args.nelecas},{args.ncas}) spin={args.spin} nroots={args.nroots}")
    print(f"alpha strings={n_alpha}")
    print(f"beta strings={n_beta}")
    print(f"determinant dimension={n_det}")
    print(f"projected-CI limit={args.max_fci_dets}")

    if projected_ok:
        print("recommendation=projected-ci validation path is feasible")
        print("notes=Use method dmrg-casscf for H/DM/GRAD/NACDR benchmarks.")
        return 0

    print("recommendation=FCI-free production path required")
    if {"grad", "nacdr"} & needs:
        print(
            "notes=Do not use projected-CI analytic response at this active "
            "space. Use DMRG static validation plus finite-difference or "
            "wavefunction-overlap dynamics until the MPS-native response "
            "backend is promoted to production."
        )
        return 2
    print("notes=H/DM static DMRG overlay can be feasible if MPS-native RDMs are used.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
