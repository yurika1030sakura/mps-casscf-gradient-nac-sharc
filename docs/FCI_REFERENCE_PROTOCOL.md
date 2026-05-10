# FCI Reference Protocol

Public benchmark errors are reported against a fixed-orbital active-space FCI
reference.  The reference is generated the same way for every benchmark system:

1. Run RHF to `1e-12`.
2. Run equal-weight SA(2)-CASSCF with the PySCF FCI active-space solver to
   define the orbital basis.
3. Freeze the converged CASSCF orbitals.
4. Rediagonalize the singlet active-space Hamiltonian at those fixed orbitals
   with PySCF `direct_spin0` FCI for multiple roots.
5. Use the two lowest spin-adapted singlet roots and record `S^2` plus residual
   diagnostics as QC checks.  Same-energy singlet clusters are identified with
   the uniform `BVOE_FCI_DEGENERACY_TOL` threshold, currently `1e-4` Eh.
6. Store those FCI CI vectors, state energies, gradients, NACs, root scan, and
   residual diagnostics in `data_phase2/*_FCI.json`.
7. For every DMRG bond dimension, match converted MPS roots to these FCI roots.
   High-confidence isolated roots are locked first by a uniform overlap and
   margin criterion.  If a selected FCI root belongs to a same-energy cluster,
   the remaining DMRG candidate-root subspace is projected onto the fixed FCI
   gauge before evaluating derivative errors.

The root-matching rule is molecule independent.  The companion note
`ROOT_MATCHING_PROTOCOL.md` lists the special cases covered by the synthetic
regression test, including root-order permutations, phase flips, shared
degenerate target clusters, and the case where a degenerate candidate pool
could otherwise consume an obvious isolated root.

Convergence plots use only points with minimum target overlap after root or
subspace alignment >= 0.98 and a converged corresponding Lagrange-response
solve.  The full scan is still stored in JSON, and `audit_bvoe_references.py`
reports which low-M points are excluded from accuracy claims by these numerical
QC criteria.

This protocol avoids using spin-penalty FCI roots as derivative references.
Spin penalties are useful for steering a calculation, but the penalized
Hamiltonian can leave small residuals with respect to the unpenalized
Hamiltonian.  NACs near avoided crossings can amplify those residuals.  The
benchmark reference therefore comes from the spin-adapted singlet Hamiltonian
itself, with spin used as a recorded diagnostic.

The public JSON schema records the reference construction in
`fci_polish_diagnostics`:

- `mode`: expected value is `spin_adapted_singlet_fci`.
- `root_scan`: energies and `S^2` diagnostics for scanned roots.
- `selected_roots`: the two roots used as validation references.
- `selected_root_clusters`: same-energy singlet clusters used to decide whether
  single-root assignment or subspace alignment is the appropriate comparison.
- `before` and `after`: fixed-orbital FCI residual diagnostics.

The FCI reference is a validation device for benchmark systems where exact
active-space diagonalization is available.  Large-active-space production
root following uses previous-step wave-function overlap, root buffers,
state-character diagnostics, and bond-dimension convergence checks.
