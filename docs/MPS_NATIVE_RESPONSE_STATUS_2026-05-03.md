# MPS-Native Response Status, 2026-05-03

## What was tested

Passing tests:

- `test_mps_krylov_response.py`
  - The CI Krylov vectors are stored as block2 MPS objects.
  - Gradient and NAC response RHS vectors are built directly in MPS form
    from MPS state/transition RDMs.
  - A custom Arnoldi/GMRES solver uses MPS overlaps for orthogonalization and
    block2 MPS addition for Krylov linear combinations.
  - The state-density cache used by `H_OC`/`H_CO` is built from block2
    `get_1pdm`/`get_2pdm` on the state MPSes rather than from FCI
    `make_rdm12`.
  - The MPS-only initializer can solve the response equation from MPS roots
    without dense CI roots stored on the response object.
  - Matvec and state-0 solve agree with the FCI response backend on
    HeH+ CAS(2,2) to about `1e-15`.
- `test_mps_krylov_sharc_interface.py`
  - The SHARC-facing analytic CP helper can use `dmrg-response-mode=mps-krylov`.
  - MPS-Krylov gradients and NACs match the projected-CI SHARC-facing path on
    the exact CAS(2,2) validation system to machine precision.
- `test_mps_only_fixed_orbital_sharc.py`
  - The SHARC-facing fixed-orbital MPS-only facade produces energies,
    dipoles, gradients, and NACs from block2 MPS roots.
  - The runtime `mc.ci` entries are one-element placeholders rather than
    active-space determinant CI roots.
- `test_mps_lagrange_assembly.py`
  - The final PySCF CI-Lagrange nuclear derivative contraction is reproduced
    from weighted MPS transition 1/2-RDMs.
  - The full Lagrange nuclear derivative contribution is reproduced by
    combining the dense orbital Lagrange term with the MPS CI-Lagrange term.
  - This validates the post-solve assembly step without storing the solved CI
    Lagrange vector as a dense ndarray.
- `test_single_site_sigma_mps.py`
  - MPS expectation and MPO sigma application agree with the FCI fallback.
- `test_site_replacement_density_mps.py`
  - MPS transition 1/2-RDMs and the site-replacement `T` matrix agree with
    the FCI reference to about `1e-15`.
- `test_step6c_mps_response_class.py`
  - `CPDMRGCASSCFResponseMPS` reproduces the FCI response class for
    HeH+ CAS(2,2), including `H_CC`, `H_OC`, `H_CO`, and full `solve`.

## Bug fixed

State-specific split-root refinement can leave block2 single-root MPS objects in
`CR`/two-site canonical form.  For SU2 transition NPDMs this can silently return
zero transition densities, and the reverse 2PDM can segfault inside block2.

Fixes included here:

- Before MPS transition-density contractions, copy each MPS and canonicalize the
  copy to one-site form with `driver.adjust_mps(..., dot=1)`.
- In `CPDMRGCASSCFResponseMPS.H_OC_apply`, align the stored root MPS phase
  against the corresponding `mc.ci` vector before accumulating transition
  densities.

## Current interpretation

The MPS-native primitives now work for the validation-scale CAS(2,2) response
test, and `H_OC` now exercises the MPS transition-density primitive rather than
silently using the FCI transition-density fallback.

The new `CPDMRGCASSCFResponseMPSKrylov` class is now an MPS-valued response
backend for the active-space response vector: the RHS, Krylov basis, state
RDM cache, Hessian-vector products, and post-solve Lagrange nuclear assembly
can all be evaluated from MPS quantities.  Small-CAS validation helpers still
convert random test vectors to FCI arrays so the implementation can be
compared against FCI.

The new `mps_lagrange_assembly.py` helper and
`CPDMRGCASSCFResponseMPSKrylov.LdotJnuc_mps` method also remove a downstream
dense-CI assumption: after an MPS-valued response solve, the nuclear derivative
Lagrange contribution can be contracted from MPS transition RDMs and matches
the PySCF dense-CI path in the CAS(2,2) regression test.

The SHARC-facing interface now exposes an `mps-krylov` response mode in
addition to the projected-CI validation mode.  The public large-active-space
benchmark driver `benchmarks/large_active_space/run_mps_only_hchain_response.py`
starts from block2 MPS roots, uses the MPS-only initializer, and records
wall-time diagnostics without converting the active-space response vector
through an FCI array.

The H10/STO-3G CAS(10,10) fixed-orbital endpoint completed with `M=200`
without an FCI reference or dense active-space response vector.  The recorded
timings were about 3.8 s for the DMRG root solve, 442 s for the state-0
gradient response, and 751 s for the NAC response.  A clean rerun with the
updated absolute-or-relative MPS-GMRES convergence check is submitted as
`mps_h10_clean`.
