"""System-agnostic active-space selection.

Picking a sensible active space is the one step that is genuinely
system-specific, so a blind default (energy-window or undirected AVAS) is a
common reason a user's run silently goes wrong (spectator orbitals enter the
active space, the optimizer stalls, or a wrong-spin root sneaks in).  This module
provides a transparent, general selector driven by *which atomic orbitals the
user expects to be chemically active*, plus a population diagnostic the caller
can hand to :func:`system_diagnostics.assess_point`.

Nothing here is specific to any molecule: the LiF benchmark just passes
``ao_targets=['F 2p', 'Li 2s', 'Li 2pz']``; an organic chromophore would pass
its pi-AO labels, a metal complex its d-AO labels, and so on.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _ao_matches(label: str, targets) -> bool:
    """True if an AO label (e.g. '0 F 2pz') matches any target token.

    A target may be an atom+shell prefix like 'F 2p' (matches 2px/2py/2pz) or a
    full 'Li 2pz'.  Matching is on the whitespace-joined 'atom orb' fields, so
    'F 2p' matches 'F 2px' but not 'F 3p'.
    """
    toks = label.split()
    if len(toks) < 3:
        return False
    atom, orb = toks[1], toks[2]
    key = f"{atom} {orb}"
    for t in targets:
        t = t.strip()
        if key == t or key.startswith(t):
            return True
    return False


def select_active_space_by_ao_targets(
    mol, mf, ncas: int, nelecas: int, ao_targets: List[str],
) -> Tuple[int, np.ndarray, dict]:
    """Choose a valence-adapted active space by target-AO population.

    The ``ncas`` non-core MOs with the largest summed population on ``ao_targets``
    become the active space; the lowest-energy ``ncore`` MOs become core.  The MO
    coefficient matrix is reordered to ``[core | active | rest]`` so it can seed a
    CASSCF/CASCI directly.

    Returns ``(ncore, mo_reordered, diagnostics)``.  ``diagnostics`` carries the
    per-active-orbital target population and energies; an active orbital whose
    target population is small (<~0.2) flags a poorly matched active space, which
    the caller can surface through the system diagnostics layer.
    """
    if (mol.nelectron - nelecas) % 2 != 0:
        raise ValueError(
            f"nelecas={nelecas} incompatible with {mol.nelectron} electrons "
            f"(ncore would be non-integer)")
    ncore = (mol.nelectron - nelecas) // 2

    mo = mf.mo_coeff
    S = mf.get_ovlp()
    mo_energy = mf.mo_energy
    labels = mol.ao_labels()

    target_ao = [i for i, lab in enumerate(labels) if _ao_matches(lab, ao_targets)]
    if not target_ao:
        raise RuntimeError(
            f"no AOs matched ao_targets={ao_targets}; available examples: "
            f"{labels[:6]}")

    Smo = S @ mo
    target_pop = np.einsum("ai,ai->i", mo[target_ao, :], Smo[target_ao, :])

    core = sorted(range(mo.shape[1]), key=lambda i: mo_energy[i])[:ncore]
    noncore = [i for i in range(mo.shape[1]) if i not in core]
    active = sorted(
        sorted(noncore, key=lambda i: target_pop[i], reverse=True)[:ncas],
        key=lambda i: mo_energy[i],
    )
    rest = sorted([i for i in range(mo.shape[1]) if i not in core and i not in active],
                  key=lambda i: mo_energy[i])
    mo_reordered = mo[:, core + active + rest]

    pops = [float(target_pop[i]) for i in active]
    diagnostics = {
        "ncore": ncore, "ncas": ncas, "nelecas": nelecas,
        "ao_targets": list(ao_targets),
        "active_target_pop": pops,
        "active_mo_energy": [float(mo_energy[i]) for i in active],
        "min_active_target_pop": float(min(pops)) if pops else 0.0,
        "active_space_well_matched": bool(pops and min(pops) > 0.2),
    }
    return ncore, mo_reordered, diagnostics
