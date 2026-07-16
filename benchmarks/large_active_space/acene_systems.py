"""Linear [n]acene geometries + FD directions for the beyond-FCI gradient/NAC
validation, generalizing the polyene template to fused-ring (2D) pi systems.

[n]acene = n linearly fused regular hexagons (C-C = a), planar in xy (pi axis z),
C(4n+2)H(2n+4), pi active space CAS(4n+2, 4n+2):
    n=2 naphthalene  CAS(10,10)  det 6.35e4   (FCI-checkable proof of concept)
    n=3 anthracene   CAS(14,14)  det 1.18e7
    n=4 tetracene    CAS(18,18)  det 2.36e9   (beyond FCI)
    n=5 pentacene    CAS(22,22)  det 4.97e11  (beyond FCI)

Carbon skeleton: n+1 vertical "rails" at x = (2j-1)h - ... with carbons at y=+-a/2,
and 2n apex carbons at (ring_centre, +-a).  Bridgehead (fused) carbons are the
INNER rails (3 C neighbours, no H); apexes and the two OUTER rails bear H.

FD directions are unambiguous single-bond stretches identified by geometry (no
chain-ordering assumption): central fusion bond, central peripheral bond, terminal
peripheral bond -- each via the general stretch_direction.
"""
from __future__ import annotations

import numpy as np

ANG = 1.8897261246257702


def acene_geometry(n_rings: int, a: float = 1.40, ch: float = 1.09):
    """Idealized planar [n]acene in xy, Angstrom. Returns (atoms, coords_ang)."""
    assert n_rings >= 2
    h = a * np.sqrt(3.0) / 2.0
    hy = a / 2.0
    ring_cx = [2.0 * h * k for k in range(n_rings)]          # ring centres on x
    # rails at x = -h, +h, +3h, ... (n_rings+1 of them)
    rail_x = [-h + 2.0 * h * j for j in range(n_rings + 1)]

    carbons = []
    ch_carbon_idx = []           # indices (into carbons) that bear an H
    ch_ring_cx = []              # ring centre used to place that H
    # rail carbons (top & bottom of each rail)
    for j, rx in enumerate(rail_x):
        is_outer = (j == 0 or j == n_rings)
        for sy in (+1.0, -1.0):
            carbons.append((rx, sy * hy))
            if is_outer:
                ch_carbon_idx.append(len(carbons) - 1)
                # outer rail H points away along x (use the adjacent ring centre)
                ch_ring_cx.append(ring_cx[0] if j == 0 else ring_cx[-1])
    # apex carbons (top & bottom of each ring)
    for cx in ring_cx:
        for sy in (+1.0, -1.0):
            carbons.append((cx, sy * a))
            ch_carbon_idx.append(len(carbons) - 1)
            ch_ring_cx.append(cx)

    hydrogens = []
    for idx, ring_cx_h in zip(ch_carbon_idx, ch_ring_cx):
        cx_c, cy_c = carbons[idx]
        dx, dy = cx_c - ring_cx_h, cy_c - 0.0
        nrm = np.hypot(dx, dy)
        hydrogens.append((cx_c + ch * dx / nrm, cy_c + ch * dy / nrm))

    atoms = ["C"] * len(carbons) + ["H"] * len(hydrogens)
    xy = np.array(carbons + hydrogens, dtype=float)
    coords = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])

    nC, nH = 4 * n_rings + 2, 2 * n_rings + 4
    assert atoms.count("C") == nC and atoms.count("H") == nH, \
        f"[{n_rings}]acene wrong composition: {atoms.count('C')}C {atoms.count('H')}H (want {nC}C {nH}H)"
    # bond sanity: all C-C bonds == a, all C-H == ch
    cc = [np.linalg.norm(coords[i] - coords[j])
          for i in range(nC) for j in range(i + 1, nC)
          if np.linalg.norm(coords[i] - coords[j]) < 1.6]
    assert max(abs(d - a) for d in cc) < 1e-9, "C-C bonds not all = a"
    return atoms, coords


def _carbon_xy(symbols, coords):
    return [(i, coords[i, 0], coords[i, 1]) for i in range(len(symbols)) if symbols[i] == "C"]


def _bond_between(coords, i, j):
    """Unit stretch direction on atoms i,j (opposite signs), zero elsewhere."""
    d = np.zeros_like(coords)
    u = coords[j] - coords[i]
    u = u / np.linalg.norm(u)
    d[i] = -u
    d[j] = +u
    return d / np.linalg.norm(d)


def acene_named_directions(symbols, coords_bohr):
    """Unambiguous single-bond stretch directions, identified by geometry.

    - fusion_cc:   central bridgehead-bridgehead bond (the inner-most fused C-C,
                   the bond shared by the two central rings; |y| small, x ~ centre).
    - peripheral_cc: a central peripheral C-C bond near the long-axis centre, |y|
                   large (an outer edge bond of a central ring).
    - terminal_cc: a peripheral C-C bond on the terminal ring.
    """
    C = _carbon_xy(symbols, coords_bohr)
    xs = np.array([c[1] for c in C]); ys = np.array([c[2] for c in C])
    xc = xs.mean()
    a_bohr = 1.40 * ANG
    # all C-C bonded pairs
    pairs = []
    for ii in range(len(C)):
        for jj in range(ii + 1, len(C)):
            i, j = C[ii][0], C[jj][0]
            d = np.linalg.norm(coords_bohr[i] - coords_bohr[j])
            if d < 1.6 * ANG:
                mx = 0.5 * (coords_bohr[i, 0] + coords_bohr[j, 0])
                my = 0.5 * (coords_bohr[i, 1] + coords_bohr[j, 1])
                # vertical bond (rail edge, same x) vs slanted
                vertical = abs(coords_bohr[i, 0] - coords_bohr[j, 0]) < 0.2 * ANG
                pairs.append((i, j, mx, my, vertical))
    dirs = {}
    # fusion_cc: vertical bond closest to x-centre (a shared inner rail edge)
    vert = [p for p in pairs if p[4]]
    if vert:
        i, j, *_ = min(vert, key=lambda p: abs(p[2] - xc))
        dirs["fusion_cc"] = _bond_between(coords_bohr, i, j)
    # peripheral_cc: slanted bond with large |y| nearest x-centre
    slant = [p for p in pairs if not p[4]]
    if slant:
        ymax = max(abs(p[3]) for p in slant)
        outer = [p for p in slant if abs(p[3]) > 0.6 * ymax]
        i, j, *_ = min(outer, key=lambda p: abs(p[2] - xc))
        dirs["peripheral_cc"] = _bond_between(coords_bohr, i, j)
    # terminal_cc: slanted bond with largest |x| (terminal ring), upper half
    if slant:
        i, j, *_ = max(slant, key=lambda p: abs(p[2] - xc))
        dirs["terminal_cc"] = _bond_between(coords_bohr, i, j)
    return dirs


if __name__ == "__main__":
    import sys
    for n in (2, 3, 4, 5):
        atoms, coords = acene_geometry(n)
        nC = atoms.count("C")
        from math import comb
        det = comb(nC, nC // 2) ** 2
        dirs = acene_named_directions(atoms, coords * ANG)
        name = {2: "naphthalene", 3: "anthracene", 4: "tetracene", 5: "pentacene"}[n]
        print(f"[{n}]acene {name}: {nC}C{atoms.count('H')}H  CAS({nC},{nC})  det={det:.3e}  "
              f"dirs={list(dirs.keys())}")
