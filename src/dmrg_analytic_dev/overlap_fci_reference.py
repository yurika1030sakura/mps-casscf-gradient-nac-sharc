"""Cross-geometry FCI/CAS active-space wavefunction overlap (reference).

Implements the exact overlap between two CAS wavefunctions whose active orbital
bases differ — i.e. the bases are connected by

    S_act = C_a^T @ S_AO_ab @ C_b

where S_AO_ab is the AO overlap matrix evaluated with the basis at geometry R_a
on the bra side and basis at R_b on the ket side. C_a, C_b are the active MO
coefficients.

This is **Milestone 1** of `ANALYTIC_DMRG_NAC_BACKEND_PLAN.md`. It is a
reference implementation against which the MPS/DMRG cross-geometry overlap
implementation must agree to numerical precision in small CAS
spaces.

Math
----
For determinants D_a = |i_1 ... i_n>_a (occupied orbital indices i_k of basis a)
and D_b = |j_1 ... j_n>_b, with both spins doubly counted, the overlap follows
from the Lowdin formula:

    <D_a | D_b> = det(S_act[occ_a, occ_b])_alpha * det(S_act[occ_a, occ_b])_beta

For a CAS state |Psi> = sum_{I_a,I_b} C_{I_a,I_b} |I_a>|I_b>, the overlap is

    <Psi_a | Psi_b> = sum_{ik}^alpha sum_{jl}^beta C_a*[i,j] C_b[k,l]
                    * det(S_act[occ_a^i, occ_b^k])  (alpha block)
                    * det(S_act[occ_a^j, occ_b^l])  (beta block)

Here we use PySCF's `cistring` to enumerate occupations.

Performance note
----------------
Cost scales as (na * nb)^2 * O(ncas^3) for naive implementation. Adequate for
CAS(2,2)..CAS(6,6) validation. Production large-CAS code must use the standard
non-orthogonal CI overlap algorithms (cofactor-based) — this reference is
correctness, not speed.
"""

from __future__ import annotations

import numpy as np
from pyscf.fci import cistring


def alpha_beta_string_lists(ncas: int, nelec: tuple[int, int]):
    """Return PySCF occupation lists for alpha and beta strings."""
    nelec_a, nelec_b = nelec
    occ_a = cistring.gen_occslst(range(ncas), nelec_a)
    occ_b = cistring.gen_occslst(range(ncas), nelec_b)
    return np.asarray(occ_a, dtype=int), np.asarray(occ_b, dtype=int)


def determinant_overlap(S_act: np.ndarray, occ_a: np.ndarray, occ_b: np.ndarray) -> float:
    """<D_a | D_b> for a single-spin block of two Slater determinants.

    D_a occupies orbitals listed in occ_a in basis a, D_b occupies occ_b in basis b.
    """
    if len(occ_a) == 0 and len(occ_b) == 0:
        return 1.0
    if len(occ_a) != len(occ_b):
        return 0.0
    sub = S_act[np.ix_(occ_a, occ_b)]
    return float(np.linalg.det(sub))


def overlap_fci(
    ci_a: np.ndarray,
    ci_b: np.ndarray,
    S_act: np.ndarray,
    ncas: int,
    nelec: tuple[int, int],
) -> float:
    """Compute <Psi_a | Psi_b> exactly for CAS wavefunctions on different
    active orbital bases.

    Parameters
    ----------
    ci_a, ci_b
        FCI vectors with shape (na, nb), na/nb = number of alpha/beta strings.
    S_act
        Active orbital overlap (ncas, ncas), S_act[p,q] = <phi_a^p | phi_b^q>.
    ncas
        Number of active orbitals (assumed identical on both sides).
    nelec
        (nelec_alpha, nelec_beta).
    """
    occ_a_strings, occ_b_strings = alpha_beta_string_lists(ncas, nelec)
    na = occ_a_strings.shape[0]
    nb = occ_b_strings.shape[0]

    if ci_a.shape != (na, nb) or ci_b.shape != (na, nb):
        raise ValueError(
            f"CI shape mismatch: ci_a={ci_a.shape}, ci_b={ci_b.shape}, expected ({na},{nb})"
        )

    # Pre-compute alpha-spin determinant overlaps S_alpha[i,k] = det(S_act[occ_i, occ_k]).
    S_alpha = np.zeros((na, na))
    for i in range(na):
        for k in range(na):
            S_alpha[i, k] = determinant_overlap(S_act, occ_a_strings[i], occ_a_strings[k])
    S_beta = np.zeros((nb, nb))
    for j in range(nb):
        for l in range(nb):
            S_beta[j, l] = determinant_overlap(S_act, occ_b_strings[j], occ_b_strings[l])

    # Overlap = sum_{ijkl} ci_a*[i,j] ci_b[k,l] S_alpha[i,k] S_beta[j,l]
    # = sum_{ik} S_alpha[i,k] * ( sum_{jl} ci_a*[i,j] S_beta[j,l] ci_b[k,l] )
    # Use einsum for clarity.
    result = np.einsum("ij,kl,ik,jl->", np.conjugate(ci_a), ci_b, S_alpha, S_beta)
    return float(np.real_if_close(result))


def overlap_matrix_fci(
    ci_a_list: list[np.ndarray],
    ci_b_list: list[np.ndarray],
    S_act: np.ndarray,
    ncas: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    """Compute overlap matrix between two sets of CAS roots.

    Returns an (n_a_roots, n_b_roots) array O[I,J] = <Psi_a^I | Psi_b^J>.
    """
    n_a = len(ci_a_list)
    n_b = len(ci_b_list)
    O = np.zeros((n_a, n_b))
    for I, ci_a in enumerate(ci_a_list):
        for J, ci_b in enumerate(ci_b_list):
            O[I, J] = overlap_fci(ci_a, ci_b, S_act, ncas, nelec)
    return O


def assign_roots_by_overlap(O: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy maximum-overlap assignment of roots in basis b to roots in basis a.

    Returns
    -------
    perm : np.ndarray
        Permutation such that ci_b_list[perm[I]] is the best match to ci_a_list[I].
    signs : np.ndarray
        +1/-1 phases to align the matched roots.
    """
    n_a, n_b = O.shape
    n = min(n_a, n_b)
    perm = -np.ones(n, dtype=int)
    signs = np.ones(n, dtype=int)
    used = np.zeros(n_b, dtype=bool)
    for I in range(n):
        absO = np.abs(O[I])
        absO[used] = -1.0
        J_best = int(np.argmax(absO))
        perm[I] = J_best
        signs[I] = int(np.sign(O[I, J_best])) or 1
        used[J_best] = True
    return perm, signs


def cross_geometry_S_act(
    mol_a,
    mol_b,
    C_a: np.ndarray,
    C_b: np.ndarray,
    ncas: int,
    ncore: int = 0,
) -> np.ndarray:
    """Build S_act = C_a[:, active]^T @ S_AO_ab @ C_b[:, active] for two mols.

    Both mols must use the same basis set and same atom ordering, only
    geometries differ. The cross-geometry AO overlap is built via PySCF's
    `intor_cross`.
    """
    from pyscf import gto

    # AO overlap between basis_a (R_a) and basis_b (R_b)
    S_AO_ab = gto.intor_cross("int1e_ovlp", mol_a, mol_b)  # shape (nao_a, nao_b)
    actv = slice(ncore, ncore + ncas)
    return C_a[:, actv].T @ S_AO_ab @ C_b[:, actv]
