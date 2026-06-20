"""Energy-invariant gauge tooling for the LiF CAS(6,6) subspace-tracking scan.

This module provides *gauge-only* helpers used by
``run_lif_cas66_root_tracking.py`` to push the adjacent-geometry subspace
overlap diagnostic to its physically meaningful limit.  None of these
operations change any energy: they are unitary rotations within (a) the active
orbital space and (b) the degenerate-up-to-phase state-average subspace, plus a
state-overlap polar decomposition.  The orbital rotation is applied to BOTH the
MO coefficients and the CI vectors (via PySCF's
``transform_ci_for_orbital_rotation``) so that the represented many-body
wavefunction is *bit-for-bit the same state* -- only its coordinate
representation in the active basis changes.

Why this matters
----------------
Near the LiF ionic/covalent avoided crossing the two lowest singlets swap
character, so energy-sorted adiabatic labels are gauge-dependent.  The
gauge-INVARIANT continuity diagnostic is the singular-value spectrum
(``sigma_min``) of the adjacent-geometry state-overlap matrix ``O``.  Two
distinct sources can depress ``sigma_min``:

  1. An orbital/state *gauge* mismatch between adjacent points -- a fixable
     artifact.  Active-orbital polar alignment (:func:`align_active_orbitals`)
     and state-subspace polar alignment (:func:`polar_align_overlap`) remove
     this.  Crucially, BOTH leave the singular values of ``O`` unchanged: if
     ``sigma_min`` is still low after alignment, the depression is NOT a gauge
     artifact.

  2. Genuine multi-state mixing -- the two-state subspace at R_l is not closed;
     it leaks weight into a third (or higher) state at R_r.  A k-state *buffer*
     (:func:`buffered_overlap`, SA(nroots>2)) exposes this directly: if the
     buffered (e.g. 3-state) ``sigma_min`` recovers to ~0.99 while the
     lowest-two-state value sits at ~0.94, the 0.94 is *physical* coupling to a
     third state -- a real, reportable result, not a bug.

Public API
----------
  align_active_orbitals(mol_prev, mo_prev, mol_curr, mo_curr, ci_curr,
                        ncas, ncore, nelecas)
      -> dict with rotated mo/ci and active_sigma_min before/after.
  polar_align_overlap(O) -> (O_aligned, R, singular_values)
  buffered_overlap(...)  -> raw + polar-aligned |O|, singular values, both the
                            lowest-two-state and buffered-k-state sigma_min.
  adaptive_refine(records, target=0.98, hard=0.95) -> list of midpoint R values.
"""
from __future__ import annotations

import numpy as np
from pyscf import gto
from pyscf.fci import addons as fci_addons

from overlap_fci_reference import (overlap_matrix_fci, assign_roots_by_overlap,
                                    cross_geometry_S_act)


# --------------------------------------------------------------------------- 1.
def align_active_orbitals(mol_prev, mo_prev, mol_curr, mo_curr, ci_curr,
                          ncas, ncore, nelecas):
    """Polar-align the current point's active orbitals to the previous point's.

    Builds the active-active cross-geometry overlap

        X = C_act_prev^T @ S_cross @ C_act_curr,   S_cross = <AO_prev|AO_curr>,

    takes its SVD ``X = U s Vh``, and rotates the current active orbitals by the
    nearest orthogonal matrix that maximally aligns them to the previous
    geometry's active orbitals:

        Q = (U @ Vh).T          (an ncas x ncas orthogonal rotation)
        C_act_curr  <-  C_act_curr @ Q

    Because this is a *rotation within the active space*, the energies are
    unchanged; but the CI vectors are expressed in the OLD active basis, so to
    keep the represented wavefunction identical each CI vector is transformed
    consistently via ``transform_ci_for_orbital_rotation`` with the SAME ``Q``.
    (Verified property: ``transform_ci_for_orbital_rotation`` applied with the
    orbital rotation that maps old->new orbitals reproduces the CI that the
    solver would have found in the new basis, with no energy change.)

    Returns a dict::

        {"mo_aligned": full MO matrix with rotated active block,
         "ci_aligned": list of rotated CI vectors,
         "Q": the ncas x ncas rotation applied,
         "active_sigma_min_before": min singular value of X (raw),
         "active_sigma_min_after":  min singular value of the aligned X
                                    (== same singular values; alignment makes X
                                     symmetric-PD, not larger),
         "active_singular_values": list of singular values of X}
    """
    nelec = (nelecas // 2, nelecas - nelecas // 2)
    actv = slice(ncore, ncore + ncas)

    S_cross = gto.intor_cross("int1e_ovlp", mol_prev, mol_curr)  # (nao_p, nao_c)
    C_prev = mo_prev[:, actv]
    C_curr = mo_curr[:, actv]
    X = C_prev.T @ S_cross @ C_curr                              # (ncas, ncas)

    U, s, Vh = np.linalg.svd(X)
    sigma_min_before = float(np.min(s))

    # Nearest-orthogonal rotation aligning curr -> prev.  X = U s Vh, so the
    # orthogonal polar factor of X is (U @ Vh); rotating C_curr by its transpose
    # makes the aligned cross-overlap C_prev^T S_cross (C_curr Q) = U s Vh Q^T
    # symmetric positive-(semi)definite (= U s U^T), i.e. maximally diagonal.
    Q = (U @ Vh).T

    # Rotate active orbitals (energy-invariant: rotation inside the active space).
    mo_aligned = mo_curr.copy()
    mo_aligned[:, actv] = C_curr @ Q

    # Transform CI consistently so the represented wavefunction is identical.
    # transform_ci_for_orbital_rotation(ci, norb, nelec, u) expects the rotation
    # u mapping the OLD one-particle basis to the NEW one, i.e. C_new = C_old @ u,
    # which is exactly Q here.
    ci_aligned = [fci_addons.transform_ci_for_orbital_rotation(
        np.asarray(c), ncas, nelec, Q) for c in ci_curr]

    # The singular values are invariant under the right-rotation by Q (they are
    # the principal angles' cosines); recompute the aligned X purely to confirm
    # it became symmetric / to expose the (identical) spectrum.
    X_aligned = C_prev.T @ S_cross @ (C_curr @ Q)
    s_after = np.linalg.svd(X_aligned, compute_uv=False)

    return {
        "mo_aligned": mo_aligned,
        "ci_aligned": ci_aligned,
        "Q": Q,
        "active_sigma_min_before": sigma_min_before,
        "active_sigma_min_after": float(np.min(s_after)),
        "active_singular_values": [float(x) for x in s],
        # diagnostic: how far the aligned X is from symmetric (should be ~0)
        "aligned_asymmetry": float(np.max(np.abs(X_aligned - X_aligned.T))),
    }


# --------------------------------------------------------------------------- 2.
def polar_align_overlap(O):
    """State-subspace polar alignment of a (k x k) state-overlap matrix ``O``.

    Returns ``(O_aligned, R, singular_values)`` where ``R`` is the orthogonal
    (real) factor that rotates the ket-side state basis so that the aligned
    overlap ``O @ R`` is symmetric positive-(semi)definite -- i.e. maximally
    diagonal and phase-consistent.  With ``O = U s Vh`` the right rotation is

        R = Vh.conj().T @ U.conj().T          (so O @ R = U s U^H, SPD)

    The singular values returned are those of ``O`` and are *gauge-invariant*:
    they are IDENTICAL before and after the polar alignment.  That invariance is
    the entire point -- ``min(singular_values)`` is the continuity diagnostic,
    and it cannot be inflated by any choice of state-basis gauge.
    """
    O = np.asarray(O)
    U, s, Vh = np.linalg.svd(O)
    R = Vh.conj().T @ U.conj().T
    O_aligned = O @ R
    return O_aligned, R, [float(x) for x in s]


# --------------------------------------------------------------------------- 3.
def buffered_overlap(mol_l, mo_l, ci_l, mol_r, mo_r, ci_r,
                     ncas, ncore, nelecas, n_lowest=2):
    """Adjacent-geometry overlap with a k-state buffer (k = len(ci_*)).

    Computes the full k x k determinant-level state-overlap matrix ``O`` between
    the two geometries (k = number of SA roots solved, e.g. 4), and reports:

      * the RAW ``|O_ij|`` and the polar-ALIGNED ``|O_ij|`` (alignment is gauge,
        singular values unchanged);
      * the full k-state singular values and ``sigma_min`` (the *buffered*
        diagnostic);
      * the lowest-``n_lowest``-state (default 2) ``sigma_min``, computed from
        the leading ``n_lowest x n_lowest`` block of ``O`` -- this is the value
        that a 2-state SA run would report.

    If ``buffered_sigma_min`` (k>=3) recovers toward 1 while
    ``lowest_two_sigma_min`` stays low, the low two-state value is *physical*
    multi-state mixing into the buffer states, not a gauge/algorithm artifact.
    """
    nelec = (nelecas // 2, nelecas - nelecas // 2)
    S_act = cross_geometry_S_act(mol_l, mol_r, mo_l, mo_r, ncas, ncore)
    O = np.asarray(overlap_matrix_fci(list(ci_l), list(ci_r), S_act, ncas, nelec))

    sv_full = np.linalg.svd(O, compute_uv=False)
    O_aligned, R, _ = polar_align_overlap(O)

    k = O.shape[0]
    nl = min(int(n_lowest), k)
    O_low = O[:nl, :nl]
    sv_low = np.linalg.svd(O_low, compute_uv=False)

    perm, signs = assign_roots_by_overlap(O)
    asv = np.linalg.svd(np.asarray(S_act), compute_uv=False)

    return {
        "nroots_buffer": int(k),
        "O_abs_raw": np.abs(O).tolist(),
        "O_abs_polar_aligned": np.abs(np.asarray(O_aligned)).tolist(),
        "subspace_singular_values_buffered": [float(x) for x in sv_full],
        "buffered_sigma_min": float(np.min(sv_full)),
        "n_lowest": int(nl),
        "subspace_singular_values_lowest": [float(x) for x in sv_low],
        "lowest_state_sigma_min": float(np.min(sv_low)),
        "assignment": [[int(i), int(perm[i])] for i in range(len(perm))],
        "active_orbital_sigma_min": float(np.min(asv)),
    }


# --------------------------------------------------------------------------- 4.
def adaptive_refine(records, target=0.98, hard=0.95, R_key="R_ang",
                    sigma_getter=None):
    """Propose midpoint R values to insert where adjacent continuity is weak.

    ``records`` is an ordered list of point records, each carrying an adjacent
    sigma_min relative to the *previous* record.  For every adjacent pair
    (R_left, R_right) whose sigma_min is below ``target`` one midpoint is
    proposed; if it is below the (stricter) ``hard`` floor, TWO midpoints are
    proposed (R_left + 1/3 and + 2/3 of the interval) for a denser refinement.

    Parameters
    ----------
    records : list of dict
        Must be sorted by R.  Each record (except possibly the first) exposes an
        adjacent sigma_min relative to its predecessor.
    target, hard : float
        Soft / hard continuity thresholds (hard < target).
    R_key : str
        Key holding the geometry coordinate.
    sigma_getter : callable, optional
        ``rec -> float | None`` returning the adjacent sigma_min for ``rec``
        relative to the previous record.  Defaults to reading
        ``rec["active_sigma_from_prev"]["subspace_sigma_min"]`` and, if absent,
        ``rec["buffered_from_prev"]["buffered_sigma_min"]``.

    Returns
    -------
    list of float
        Sorted, de-duplicated midpoint R values to insert and recompute.
    """
    if sigma_getter is None:
        def sigma_getter(rec):
            d = rec.get("active_sigma_from_prev")
            if isinstance(d, dict) and "subspace_sigma_min" in d:
                return float(d["subspace_sigma_min"])
            d = rec.get("buffered_from_prev")
            if isinstance(d, dict) and "buffered_sigma_min" in d:
                return float(d["buffered_sigma_min"])
            return None

    recs = [r for r in records if R_key in r]
    recs = sorted(recs, key=lambda r: float(r[R_key]))

    midpoints = []
    for i in range(1, len(recs)):
        sigma = sigma_getter(recs[i])
        if sigma is None:
            continue
        R_l = float(recs[i - 1][R_key])
        R_r = float(recs[i][R_key])
        if sigma < hard:
            midpoints.append(R_l + (R_r - R_l) / 3.0)
            midpoints.append(R_l + 2.0 * (R_r - R_l) / 3.0)
        elif sigma < target:
            midpoints.append(0.5 * (R_l + R_r))

    # de-duplicate to 6 decimals, sorted
    uniq = sorted({round(float(x), 6) for x in midpoints})
    return uniq
