"""Single-site sigma vector kernel for Freitag-Reiher CP-DMRG-CASSCF.

Implements σ = H_loc · v at one chosen "linear response site" l. This is the
new DMRG primitive needed for the Freitag-Reiher 2019 approximate analytic
gradient/NAC formulation.

Two implementations
-------------------
1. **`single_site_sigma_fci_fallback`** — FCI projection. Converts the MPS to
   an FCI ndarray, applies PySCF's FCI sigma vector, projects back. Exact for
   CAS small enough that the FCI tensor fits, which is the validation regime
   (CAS(2,2)/CAS(4,4) where DMRG = FCI). This is what the validation tests
   need to verify the rest of the Freitag-Reiher response solver.

2. **`single_site_sigma_mps_native`** — TODO. Production-scale path that uses
   pyblock2's MovingEnvironment + EffectiveHamiltonian internals to construct
   H_loc directly without going through FCI. Initial attempt segfaulted at
   the SimplifiedMPO/NoTransposeRule construction step; a proper fix requires
   matching block2's internal symmetry-handling contract more carefully than
   the high-level docs cover. Documented here as the natural extension once
   the FCI-fallback validates the algorithm structure.

For the Freitag-Reiher algorithm itself, the validation regime where we need
to compare against PySCF's FCI baseline IS the CAS(2,2)/(4,4) regime, so the
FCI fallback is sufficient to validate Steps 2-5 of the implementation plan.
The MPS-native version is required only for production large-CAS NAC.
"""

from __future__ import annotations

import numpy as np
from pyscf import fci


def single_site_sigma_fci_fallback(
    h1_act: np.ndarray,
    h2_act: np.ndarray,
    ci_v: np.ndarray,
    ncas: int,
    nelec: tuple[int, int],
    fcisolver=None,
) -> np.ndarray:
    """σ = (H_2e + h_1e absorbed) · v in the FCI ndarray representation.

    This is the standard FCI sigma vector. For CAS small enough that DMRG = FCI,
    this is the EXACT single-site sigma vector that Freitag-Reiher needs (the
    "single site" is the entire FCI tensor when ncas is small enough that the
    MPS hasn't compressed anything).

    Parameters
    ----------
    h1_act, h2_act
        Active-space integrals (MO basis, Mulliken (12|34) for h2).
    ci_v
        Trial CI vector as PySCF FCI ndarray, shape (na, nb).
    ncas, nelec
        Active orbitals and electron tuple.
    fcisolver
        Optional FCI solver instance (e.g. `mc.fcisolver`). When supplied, its
        `absorb_h1e`/`contract_2e` are used so spin penalties / state-average
        machinery from the parent CASSCF carry over. Defaults to
        `pyscf.fci.direct_spin1`. Newton_casscf uses `mc.fcisolver` for its
        `_Hci`; passing the same fcisolver here is what makes the FR backend
        element-wise reproduce newton_casscf when spin penalties are active.

    Returns
    -------
    sigma : (na, nb) ndarray, σ = H · ci_v.
    """
    solver = fcisolver if fcisolver is not None else fci.direct_spin1
    op = solver.absorb_h1e(h1_act, h2_act, ncas, nelec, 0.5)
    return solver.contract_2e(op, np.asarray(ci_v), ncas, nelec)


def single_site_sigma_mps_native(
    driver,
    mpo,
    mps,
    *,
    out_tag: str = "SIGMA",
    bra_bond_dim: int | None = None,
    n_sweeps: int = 10,
    tol: float = 1e-10,
    iprint: int = 0,
    M_compress: int | None = None,
):
    """MPS-native sigma vector σ = MPO · |mps⟩, returned as a *new* MPS.

    Implementation
    --------------
    The cleanest production path bypasses the SimplifiedMPO/Rule plumbing that
    crashed in the earlier attempt. Instead we use the high-level
    ``DMRGDriver.multiply`` method, which fits

        |sigma⟩ ≈ MPO · |mps⟩

    by sweeping a Linear solver over a ``MovingEnvironment``. ``driver.multiply``
    handles all of the symmetry/rule attachments internally — it is the same
    API surface used by block2's compression / addition machinery, so it does
    NOT require us to drive the SimplifiedMPO contract directly.

    Parameters
    ----------
    driver
        ``pyblock2.driver.core.DMRGDriver`` previously initialized via
        ``initialize_system`` for the active space.
    mpo
        Block2 MPO from ``driver.get_qc_mpo`` (active-space H).
    mps
        Block2 MPS (the "ket") to apply the MPO to.
    out_tag
        Tag for the output MPS in driver scratch space. Must differ from the
        input MPS tag.
    bra_bond_dim
        Bond dimension of the output MPS. Defaults to that of ``mps``. For
        validation regimes (DMRG = FCI) the output bond dimension matches the
        input. For larger production cases users may want to pass a larger
        bond dimension to absorb the bond-dimension growth from MPO·MPS.
    M_compress
        Maximum bond dimension for the output sigma MPS. After ``driver.multiply``
        completes the fit, the result is compressed in-place to ``M_compress``
        via ``driver.compress_mps``. If None, no post-fit compression is applied.
        Important for non-eigenstate trial vectors in CP iterations where
        MPO·|ψ⟩ can transiently grow the bond dimension above what is needed
        for the response Hessian-vector product.
    n_sweeps
        Maximum number of fitting sweeps. Default 10.
    tol
        Convergence tolerance for ``|sigma|``. Default 1e-10.
    iprint
        Verbosity. 0 = silent.

    Returns
    -------
    sigma_mps : block2 MPS
        New MPS representing σ = MPO · |mps⟩ (NOT normalized; carries the norm
        ‖σ‖ = √⟨σ|σ⟩, i.e. encodes the magnitude of MPO|mps⟩ correctly).
    norm : float
        ‖σ‖. For an eigenstate of MPO with eigenvalue E, this equals |E|.

    Validation regime
    -----------------
    For a converged DMRG eigenstate ``mps`` of energy E (active-space MPO,
    ecore = 0), ``single_site_sigma_mps_native(driver, mpo, mps)`` should
    yield σ ≈ E · mps to fitting precision. Cross-check by comparing
    expectations or via FCI conversion (see ``test_single_site_sigma_mps``).
    """
    if mps.info.tag == out_tag:
        raise ValueError(
            f"out_tag {out_tag!r} collides with input MPS tag; pick a different out_tag."
        )

    # Initial guess: a copy of the input MPS, retagged. driver.multiply
    # overwrites its bra in-place with the fitted result.
    sigma_mps = driver.copy_mps(mps, tag=out_tag)

    # Bond dimension of the output. For an eigenstate σ = E·ψ the bond
    # dimension is identical to ψ. For non-eigenstate trials, MPO·MPS can in
    # principle have a larger bond dim; the user can pass bra_bond_dim to
    # accommodate.
    if bra_bond_dim is None:
        bra_bond_dim = max(mps.info.bond_dim, sigma_mps.info.bond_dim)

    norm = driver.multiply(
        sigma_mps,
        mpo,
        mps,
        n_sweeps=n_sweeps,
        tol=tol,
        bond_dims=[mps.info.bond_dim],
        bra_bond_dims=[int(bra_bond_dim)],
        iprint=iprint,
    )
    # Optional post-fit compression to a target bond-dim cap. This is
    # important in CP iterations where the trial vector is not an eigenstate
    # of MPO and ``multiply`` can leave the bra MPS with a bond dim larger
    # than needed for the next Krylov step.
    if M_compress is not None and sigma_mps.info.bond_dim > int(M_compress):
        driver.compress_mps(sigma_mps, max_bond_dim=int(M_compress))
    return sigma_mps, float(norm)


def single_site_sigma_mps_to_fci(
    driver,
    sigma_mps,
    ncas: int,
    nelec: tuple[int, int],
):
    """Convert a sigma MPS produced by ``single_site_sigma_mps_native`` to an
    FCI ndarray, for cross-checking against the FCI fallback.

    Hardcoded to CAS(2,2) singlet via the existing CSF→FCI mapping in
    ``dmrg_fcisolver``.
    """
    return site_tensor_to_fci(driver, sigma_mps, ncas, nelec)


def site_tensor_to_fci(driver, mps, ncas, nelec):
    """Convert pyblock2 MPS to PySCF FCI ndarray via CSF coefficients.

    Hardcoded to CAS(2,2) singlet for now; uses the same CSF→FCI mapping as
    `dmrg_fcisolver.py`.
    """
    from dmrg_fcisolver import _csf_to_fci22_singlet
    csfs, coefs = driver.get_csf_coefficients(mps, cutoff=0.0, iprint=0)
    if ncas == 2 and nelec == (1, 1):
        return _csf_to_fci22_singlet(csfs, coefs)
    raise NotImplementedError(
        f"CSF→FCI conversion only implemented for CAS(2,2) singlet; got "
        f"ncas={ncas}, nelec={nelec}."
    )
