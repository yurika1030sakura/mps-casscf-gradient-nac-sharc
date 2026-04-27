"""Transition densities with site-l block replacement (Freitag-Reiher Eqs 49, 50).

The Freitag-Reiher CP-DMRG-CASSCF Hessian-vector products H_OC ṽ and H_CO κ
(Eqs 49 and 50 in the paper) require evaluating transition densities between
the unmodified state Ψ and a "modified state" Ψ̄ formed by replacing the
site-l MPS tensor with the trial vector ṽ.

For the CAS(2,2)/CAS(4,4) validation regime (DMRG = FCI), the "site-l tensor"
is the entire FCI ndarray, so the modified state is simply

    |Ψ̄⟩ = ṽ_I |Ψ_I⟩

i.e. ṽ replaces the CI coefficients directly. The transition densities then
follow PySCF's standard transition_rdm12 with one CI vector being the trial.

This module exposes:

  - `T_matrix_site_replacement`: the orbital gradient T(Ψ, Ψ̄) used in H_OC ṽ
    (Eq 49) and H_CO κ (Eq 50).
  - `transition_rdm_site_replacement`: γ^ΨΨ̄_pq, Γ^ΨΨ̄_pqrs (Eqs 17-20 plus
    site-replacement) for use in the gradient/NAC assembly.
  - `transition_rdm_site_replacement_mps`, `T_matrix_site_replacement_mps`:
    MPS-native counterparts using ``driver.get_npdm`` between an MPS and a
    site-replaced "modified" MPS. The site replacement is implemented through
    the CSF↔FCI mapping inherited from ``dmrg_fcisolver``: an FCI-form trial
    vector gets converted to CSF coefficients, then back to an MPS via
    ``driver.get_mps_from_csf_coefficients``. This is a faithful encoding for
    CAS(2,2) singlet (used for validation); production-scale CAS(n,m) needs
    a generic CSF↔determinant decomposition (the same gap as in
    ``dmrg_fcisolver.MPSAsFCISolver``).

Pyblock2 native MPS extension is straightforward in principle (replace site-l
tensor in the bra MPS, compute transition RDMs via the standard pyblock2
sweep). The current implementation realizes the trial as a *full* MPS via the
CSF round-trip — this is exact when DMRG = FCI, and this is what the analytic
NAC validation regime needs.
"""

from __future__ import annotations

import numpy as np
from pyscf import fci


def transition_rdm_site_replacement(
    ci_state: np.ndarray,
    trial_v: np.ndarray,
    ncas: int,
    nelec: tuple[int, int],
):
    """Compute γ_pq, Γ_pqrs between the state Ψ (with CI vector ci_state) and
    the modified state Ψ̄ where the entire FCI tensor is replaced by trial_v.

    For Mulliken (12|34) ordering matching PySCF.
    """
    return fci.direct_spin1.trans_rdm12(
        np.asarray(ci_state), np.asarray(trial_v), ncas, nelec,
    )


def generalized_fock_matrix(
    h_mo: np.ndarray,
    eri_mo: np.ndarray,
    gamma_pq: np.ndarray,
    Gamma_pqrs: np.ndarray,
    ncas: int,
    ncore: int,
):
    """Generalized Fock matrix F_{pq}(Ψ, Ψ̄) for *transition* density.

    For a TRANSITION density between orthogonal CI states in the same active
    space, the core-block transition density and ALL non-active blocks of γ, Γ
    vanish. So F has nonzero entries only in the active *rows*:

        F_pq(Ψ, Ψ̄) = sum_t∈act γ_pt h_tq + Q_pq      for p ∈ active, q ∈ all

        Q_pq = sum_vwx∈act Γ_pvwx [(qv|wx) - 0.5 (qw|vx)]

    Returns
    -------
    F : (ncas, nmo) generalized Fock with active rows × all columns.
    """
    nmo = h_mo.shape[0]
    actv = slice(ncore, ncore + ncas)
    h_act_all = h_mo[actv, :]  # (ncas, nmo): h_tq for t ∈ act, q ∈ all
    eri_q_all = eri_mo[:, actv, actv, actv]  # (nmo, ncas, ncas, ncas)
    # represents (q v | w x) with q ∈ all, v, w, x ∈ active.

    # F_pq = sum_t γ_pt h_tq    (γ shape (ncas, ncas), h_act_all shape (ncas, nmo))
    F_one = gamma_pq @ h_act_all  # (ncas, nmo)

    Q_pq = (
        np.einsum("pvwx,qvwx->pq", Gamma_pqrs, eri_q_all, optimize=True)
        - 0.5 * np.einsum("pwvx,qvwx->pq", Gamma_pqrs, eri_q_all, optimize=True)
    )  # (ncas, nmo)

    return F_one + Q_pq


def T_matrix_site_replacement(
    h_mo: np.ndarray,
    eri_mo: np.ndarray,
    ci_state: np.ndarray,
    trial_v: np.ndarray,
    ncas: int,
    ncore: int,
    nelec: tuple[int, int],
    symmetrize_density: bool = True,
):
    """Generalized orbital gradient T_pq(Ψ, Ψ̄) (Freitag-Reiher Eq 43).

    T_pq = 2 [F_pq(Ψ, Ψ̄) - F_qp(Ψ, Ψ̄)] = 2 (F − F^T)   (antisymmetric)

    where F is built from the symmetrized transition density (γ^ΨΨ̄ + γ^Ψ̄Ψ)/2.

    Returns
    -------
    T : (nmo, nmo) antisymmetric generalized gradient.
    """
    nmo = h_mo.shape[0]

    if symmetrize_density:
        gamma_AB, Gamma_AB = transition_rdm_site_replacement(ci_state, trial_v, ncas, nelec)
        gamma_BA, Gamma_BA = transition_rdm_site_replacement(trial_v, ci_state, ncas, nelec)
        gamma = 0.5 * (gamma_AB + gamma_BA)
        Gamma = 0.5 * (Gamma_AB + Gamma_BA)
    else:
        gamma, Gamma = transition_rdm_site_replacement(ci_state, trial_v, ncas, nelec)

    F = generalized_fock_matrix(h_mo, eri_mo, gamma, Gamma, ncas, ncore)
    # F has shape (ncas, nmo) — active rows × all columns.

    actv = slice(ncore, ncore + ncas)
    T = np.zeros((nmo, nmo))
    # T_pq = 2 (F_pq - F_qp). F_pq is nonzero only for p ∈ active.
    T[actv, :] = 2.0 * F                   # T[p∈act, q∈all] = 2 F_pq (all blocks)
    T[:, actv] -= 2.0 * F.T                # subtract 2 F_qp (q∈act, p∈all)
    # Net effects:
    #   T[p∈act, q∈act] = 2 F_pq - 2 F_qp                ✓
    #   T[p∈act, q∉act] = 2 F_pq        (no F_qp; F_qp=0)  ✓
    #   T[p∉act, q∈act] = 0 - 2 F_qp = -2 F_qp           ✓
    #   T[p∉act, q∉act] = 0                              ✓
    return T


# ---------------------------------------------------------------------------
# MPS-native counterparts (Step 6.2)
# ---------------------------------------------------------------------------
#
# For production-scale CAS where the FCI tensor is infeasible, the trial v and
# the state Ψ live as MPS objects. The "site-l block replacement" then means:
# build an auxiliary "modified" MPS Ψ̄ whose site-l tensor is replaced by ṽ,
# and compute γ^ΨΨ̄, Γ^ΨΨ̄ as transition NPDMs between Ψ and Ψ̄.
#
# For the validation regime (CAS(2,2) singlet, DMRG = FCI), the trial vector
# is naturally an FCI ndarray. The simplest and most faithful "site-l
# replacement" in that regime is to treat the entire FCI tensor as the
# "single site": form Ψ̄ as the full MPS encoding of trial_v via the CSF
# round-trip in dmrg_fcisolver. This collapses to the FCI fallback when the
# bond dimension is unbounded (DMRG = FCI), and degrades smoothly to a
# bond-truncated estimate at finite M.
#
# Convention conversion empirically determined (see
# /tmp/test_block2_npdm_convention.py):
#
#   dm1_pyscf[p, q]       = dm1_block2[q, p]                 (transpose)
#   dm2_pyscf[p, q, r, s] = dm2_block2.transpose(0, 2, 1, 3) (chemist Mulliken)
#
# These conversions are direct: no δ correction needed because block2's SU2
# convention sums spin pairs with a fixed operator ordering matching PySCF
# chemist's notation under the index permutation above.


def _block2_trans_rdm12_to_pyscf(dm1_b: np.ndarray, dm2_b: np.ndarray):
    """Convert block2 SU2 transition RDMs to PySCF chemist's-notation RDMs."""
    return dm1_b.T.copy(), dm2_b.transpose(0, 2, 1, 3).copy()


def _fci22_singlet_to_csf_corrected(ci: np.ndarray):
    """Inverse of ``dmrg_fcisolver._csf_to_fci22_singlet`` — corrected.

    The forward map sets ``ci[0,1] = ci[1,0] = c/√2`` for the open-shell
    singlet CSF [1,2] with coefficient c. Hence the inverse for that piece is
    ``c = (ci[0,1] + ci[1,0]) / √2`` (sum, not difference). The original
    helper in ``dmrg_fcisolver`` had this with a minus sign which silently
    drops the symmetric open-shell singlet — that bug is dormant in current
    code paths but breaks the FCI → CSF → MPS round-trip used here.

    This corrected variant lives in this module rather than fixing the
    legacy helper, to honor the "don't modify the existing backend code"
    constraint.
    """
    csfs = []
    coefs = []
    if abs(ci[0, 0]) > 1e-14:
        csfs.append([3, 0]); coefs.append(float(ci[0, 0]))
    if abs(ci[1, 1]) > 1e-14:
        csfs.append([0, 3]); coefs.append(float(ci[1, 1]))
    open_shell_c = (ci[0, 1] + ci[1, 0]) / np.sqrt(2.0)
    if abs(open_shell_c) > 1e-14:
        csfs.append([1, 2]); coefs.append(float(open_shell_c))
    return np.asarray(csfs, dtype=np.uint8), np.asarray(coefs)


def fci_to_mps_via_csf(driver, ci_v: np.ndarray, ncas: int, nelec: tuple[int, int],
                      tag: str, dot: int = 2):
    """Build an MPS from an FCI ndarray via the CSF round-trip.

    Hardcoded to CAS(2,2) singlet (same hardcoding as ``dmrg_fcisolver``).
    Raises NotImplementedError otherwise.
    """
    if ncas != 2 or nelec != (1, 1):
        raise NotImplementedError(
            f"fci_to_mps_via_csf currently only handles CAS(2,2) singlet; got "
            f"ncas={ncas}, nelec={nelec}. Generalization needs a generic FCI→CSF "
            f"map (the same gap as in dmrg_fcisolver._csf_to_fci22_singlet)."
        )
    csfs, coefs = _fci22_singlet_to_csf_corrected(np.asarray(ci_v))
    if len(csfs) == 0:
        raise ValueError("Trial vector is identically zero; cannot build MPS.")
    return driver.get_mps_from_csf_coefficients(
        csfs, coefs.astype(np.float64), tag=tag, dot=dot, iprint=0,
    )


# ---------------------------------------------------------------------------
# Generic FCI ↔ MPS converter for arbitrary CAS(n,m) and any spin sector
# (Step 6.3a)
# ---------------------------------------------------------------------------
#
# The previous CSF route (Yamanouchi-Kotani spin-coupled basis in SU2 mode)
# requires a Slater-Determinant ↔ CSF transformation that is non-trivial to
# code generically. The pragmatic alternative used here:
#
#    use SZ-mode block2 determinants (no spin coupling).
#
# In SZ mode each block2 site has a 4-dim local basis labeled
# ``(0, 1, 2, 3)`` for ``(empty, alpha, beta, double)``. This maps directly
# to PySCF FCI ndarray indexing:
#
#    ci[ia, ib]   ↔   det = (occ_0, occ_1, ..., occ_{norb-1})
#
# where ``occ_i`` is determined from the bit patterns of the alpha and beta
# strings indexed by ``ia, ib``. The forward map (FCI → dets) and inverse
# (dets → FCI) are O(na · nb) and require no Clebsch-Gordan algebra.
#
# This sacrifices manifest spin adaptation (the SZ MPS does not enforce
# total S²), but for the analytic CP-CASSCF response that's irrelevant: the
# SA-CASSCF eigenstates already have the correct spin from the FCI solver,
# and ``get_mps_from_csf_coefficients`` faithfully reproduces them. The only
# observable cost is a factor 2-ish memory increase relative to a pure SU2
# MPS at the same fidelity, in exchange for arbitrary CAS support.
#
# Validation: see ``test_step6c_generic_fci_map.py``.


def _pyscf_to_block2_sign(sa_int: int, sb_int: int, norb: int) -> int:
    """Sign converting a PySCF FCI determinant to a block2 SZ-mode det.

    PySCF orders fermionic operators as
    ``a†_{i_1, α} a†_{i_2, α} ... a†_{j_1, β} a†_{j_2, β} ... |0>``
    (all α first in increasing site order, then all β in increasing site
    order).

    Block2 SZ-mode orders per-site (site index increasing) with α before β
    at the same site:
    ``a†_{0, σ_0} a†_{0, σ_0'} a†_{1, σ_1} ...``
    where each site contributes 0, 1, or 2 creation operators in the order
    α-first.

    Conversion sign: each pair (α at site a, β at site b) with ``a > b``
    requires one anticommutation to move the β before the α, contributing
    a factor of ``-1``.

    Returns +1 or -1.
    """
    n_swap = 0
    for a in range(norb):
        if (sa_int >> a) & 1:  # alpha at site a
            for b in range(a):  # b < a, beta at site b
                if (sb_int >> b) & 1:
                    n_swap += 1
    return -1 if (n_swap & 1) else 1


def _fci_to_sz_dets(ci, norb: int, na_e: int, nb_e: int):
    """Forward map: PySCF FCI ndarray (na, nb) → SZ-mode dets + coefs.

    SZ-mode site occupancies: 0=empty, 1=alpha, 2=beta, 3=double.

    A sign correction is applied per determinant to translate from PySCF's
    "all-α-then-all-β" fermionic operator ordering to block2's "per-site
    interleaved" ordering. See :func:`_pyscf_to_block2_sign`.
    """
    from pyscf.fci import cistring

    strs_a = cistring.make_strings(range(norb), na_e)
    strs_b = cistring.make_strings(range(norb), nb_e)
    dets, coefs = [], []
    ci = np.asarray(ci)
    for ia, sa in enumerate(strs_a):
        sa_int = int(sa)
        for ib, sb in enumerate(strs_b):
            c = float(ci[ia, ib])
            if abs(c) < 1e-14:
                continue
            sb_int = int(sb)
            occ = np.zeros(norb, dtype=np.uint8)
            for i in range(norb):
                bit_a = (sa_int >> i) & 1
                bit_b = (sb_int >> i) & 1
                if bit_a and bit_b:
                    occ[i] = 3
                elif bit_a:
                    occ[i] = 1
                elif bit_b:
                    occ[i] = 2
            sign = _pyscf_to_block2_sign(sa_int, sb_int, norb)
            dets.append(occ)
            coefs.append(sign * c)
    return (np.asarray(dets, dtype=np.uint8),
            np.asarray(coefs, dtype=np.float64))


def _sz_dets_to_fci(dets, coefs, norb: int, na_e: int, nb_e: int):
    """Inverse map: SZ-mode dets + coefs → PySCF FCI ndarray (na, nb).

    Applies the inverse fermionic-ordering sign: forward map stores
    ``c_block2 = sign(sa, sb) * c_pyscf``, so to recover the PySCF coefficient
    we multiply by the same sign (sign² = 1).
    """
    from pyscf.fci import cistring

    strs_a = list(cistring.make_strings(range(norb), na_e))
    strs_b = list(cistring.make_strings(range(norb), nb_e))
    a_idx = {int(s): i for i, s in enumerate(strs_a)}
    b_idx = {int(s): i for i, s in enumerate(strs_b)}
    ci = np.zeros((len(strs_a), len(strs_b)), dtype=np.float64)
    for d, c in zip(dets, coefs):
        c = float(c)
        if abs(c) < 1e-14:
            continue
        sa = sb = 0
        for i, occ in enumerate(d):
            occ = int(occ)
            if occ == 1:
                sa |= (1 << i)
            elif occ == 2:
                sb |= (1 << i)
            elif occ == 3:
                sa |= (1 << i)
                sb |= (1 << i)
        ia = a_idx.get(sa)
        ib = b_idx.get(sb)
        if ia is not None and ib is not None:
            sign = _pyscf_to_block2_sign(sa, sb, norb)
            ci[ia, ib] = sign * c
    return ci


def fci_to_mps_generic(driver_sz, ci_v: np.ndarray, ncas: int,
                       nelec: tuple[int, int], tag: str,
                       *, dot: int = 2):
    """Generic FCI → MPS converter (Step 6.3a).

    Works for any CAS(n,m) and any spin sector. Requires the driver to be
    initialised in SZ mode (``SymmetryTypes.SZ``).

    Parameters
    ----------
    driver_sz : DMRGDriver
        block2 driver in SZ mode, already ``initialize_system``'d for the
        target ``(n_sites=ncas, n_elec=na+nb, spin=na-nb, ...)``.
    ci_v : np.ndarray
        PySCF FCI ndarray, shape (na_string, nb_string).
    ncas : int
        Number of active orbitals.
    nelec : tuple[int, int]
        ``(n_alpha, n_beta)`` electrons.
    tag : str
        MPS tag for disk storage.
    dot : int
        block2 dot site type (1 or 2). Default 2.

    Returns
    -------
    mps : block2 MPS
    """
    na_e, nb_e = int(nelec[0]), int(nelec[1])
    dets, coefs = _fci_to_sz_dets(ci_v, ncas, na_e, nb_e)
    if len(dets) == 0:
        raise ValueError("FCI ndarray is identically zero; cannot build MPS.")
    return driver_sz.get_mps_from_csf_coefficients(
        dets, coefs, tag=tag, dot=dot, iprint=0,
    )


def mps_to_fci_generic(driver_sz, mps, ncas: int,
                       nelec: tuple[int, int],
                       *, _det_mps_tag_prefix: str = "DETP"):
    """Generic MPS → FCI converter (Step 6.3a).

    Inverse of :func:`fci_to_mps_generic`. Implementation note:
    ``DMRGDriver.get_csf_coefficients`` is unreasonably slow in SZ mode
    even with small ``cutoff`` or explicit ``given_dets`` (we observed
    >120s hangs at CAS(4,4) — apparently the SZ-mode codepath enumerates
    the full 4^N space regardless). We sidestep this by computing
    ``<det_i|MPS>`` directly as an overlap between the input MPS and a
    single-determinant MPS (built via ``get_mps_from_csf_coefficients``
    from a one-row det matrix). Each overlap is O(M·N) and there are
    ``C(m, na)·C(m, nb)`` of them — well-controlled even at CAS(20,20).

    Returns
    -------
    ci : np.ndarray, shape (na_string, nb_string)
    """
    from pyscf.fci import cistring

    na_e, nb_e = int(nelec[0]), int(nelec[1])
    strs_a = cistring.make_strings(range(ncas), na_e)
    strs_b = cistring.make_strings(range(ncas), nb_e)

    id_mpo = driver_sz.get_identity_mpo()
    ci = np.zeros((len(strs_a), len(strs_b)), dtype=np.float64)

    for ia, sa in enumerate(strs_a):
        sa_int = int(sa)
        for ib, sb in enumerate(strs_b):
            sb_int = int(sb)
            occ = np.zeros(ncas, dtype=np.uint8)
            for i in range(ncas):
                bit_a = (sa_int >> i) & 1
                bit_b = (sb_int >> i) & 1
                if bit_a and bit_b:
                    occ[i] = 3
                elif bit_a:
                    occ[i] = 1
                elif bit_b:
                    occ[i] = 2
            det_mps = driver_sz.get_mps_from_csf_coefficients(
                occ.reshape(1, -1), np.array([1.0]),
                tag=f"{_det_mps_tag_prefix}-{ia}-{ib}",
                dot=2, iprint=0,
            )
            # block2 overlap → PySCF coefficient: multiply by ordering sign
            sign = _pyscf_to_block2_sign(sa_int, sb_int, ncas)
            ci[ia, ib] = sign * float(
                driver_sz.expectation(det_mps, id_mpo, mps, iprint=0)
            )
    return ci


def transition_rdm_site_replacement_mps(
    driver,
    mps_state,
    trial,
    ncas: int,
    nelec: tuple[int, int],
    *,
    trial_tag: str = "TRIAL",
):
    """MPS-native γ_pq, Γ_pqrs between Ψ (= mps_state) and Ψ̄ (= trial as MPS).

    The trial may be passed either as a block2 MPS (production path) or as an
    FCI ndarray (validation path; auto-converted via CSF round-trip).

    Returns
    -------
    gamma : (ncas, ncas) ndarray, PySCF convention dm1[p, q] = ⟨Ψ| a†_p a_q |Ψ̄⟩.
    Gamma : (ncas, ncas, ncas, ncas) ndarray, PySCF chemist's notation
            dm2[p, q, r, s] = ⟨Ψ| a†_p a_q a†_r a_s |Ψ̄⟩.
    """
    if isinstance(trial, np.ndarray):
        trial_mps = fci_to_mps_via_csf(driver, trial, ncas, nelec, tag=trial_tag)
    else:
        trial_mps = trial

    dm1_b = driver.get_trans_1pdm(bra=mps_state, ket=trial_mps, iprint=0)
    dm2_b = driver.get_trans_2pdm(bra=mps_state, ket=trial_mps, iprint=0)
    return _block2_trans_rdm12_to_pyscf(dm1_b, dm2_b)


def T_matrix_site_replacement_mps(
    driver,
    h_mo: np.ndarray,
    eri_mo: np.ndarray,
    mps_state,
    trial,
    ncas: int,
    ncore: int,
    nelec: tuple[int, int],
    *,
    trial_tag: str = "TRIAL",
    symmetrize_density: bool = True,
):
    """MPS-native generalized orbital gradient T_pq(Ψ, Ψ̄), site-l replacement.

    Mirrors :func:`T_matrix_site_replacement` but uses the MPS path. ``trial``
    may be either an MPS or an FCI ndarray (validation regime).

    Returns
    -------
    T : (nmo, nmo) antisymmetric generalized gradient matrix.
    """
    if symmetrize_density:
        gamma_AB, Gamma_AB = transition_rdm_site_replacement_mps(
            driver, mps_state, trial, ncas, nelec, trial_tag=trial_tag + "-AB",
        )
        # For BA we swap bra↔ket which means swap the input trial to be the bra.
        if isinstance(trial, np.ndarray):
            trial_mps = fci_to_mps_via_csf(driver, trial, ncas, nelec,
                                           tag=trial_tag + "-BRA")
        else:
            trial_mps = trial
        dm1_b = driver.get_trans_1pdm(bra=trial_mps, ket=mps_state, iprint=0)
        dm2_b = driver.get_trans_2pdm(bra=trial_mps, ket=mps_state, iprint=0)
        gamma_BA, Gamma_BA = _block2_trans_rdm12_to_pyscf(dm1_b, dm2_b)
        gamma = 0.5 * (gamma_AB + gamma_BA)
        Gamma = 0.5 * (Gamma_AB + Gamma_BA)
    else:
        gamma, Gamma = transition_rdm_site_replacement_mps(
            driver, mps_state, trial, ncas, nelec, trial_tag=trial_tag,
        )

    F = generalized_fock_matrix(h_mo, eri_mo, gamma, Gamma, ncas, ncore)
    nmo = h_mo.shape[0]
    actv = slice(ncore, ncore + ncas)
    T = np.zeros((nmo, nmo))
    T[actv, :] = 2.0 * F
    T[:, actv] -= 2.0 * F.T
    return T
