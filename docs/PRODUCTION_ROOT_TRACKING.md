# Production Root Tracking

FCI references are used only for validation benchmarks, where the goal is to
measure an absolute error against the exact active-space result. Production
DMRG-SHARC calculations use previous-step wave-function overlap for root
continuity.

For a new large-active-space system, use continuity against the previous
accepted wave function:

1. At the initial geometry, solve more roots than the number propagated when
   roots are dense or symmetry-related.  In `PYSCF.template`, this is the
   general setting `dmrg-root-buffer`; for example, `roots 2` with
   `dmrg-root-buffer 4` solves six candidate roots and returns the two roots
   that are continuous with the accepted state set.
2. Choose the chemically relevant state set using energy order, spin,
   occupation diagnostics, transition properties, and active-orbital
   character.
3. At each following CASSCF macro iteration or trajectory step, compute
   overlaps between current DMRG roots and the previous accepted roots.
4. Assign current roots by maximum overlap and align their phases before
   returning gradients, transition properties, or NACs.
5. Near degeneracies, track the state subspace rather than insisting on a
   unique single-state label.
6. Check production convergence by increasing bond dimension, number of
   sweeps, and number of solved roots; use finite-difference or neighboring
   geometry consistency checks when no FCI reference exists.

The public BVOE benchmark uses FCI overlaps as validation data; the production
algorithm uses the previous accepted wave functions from the trajectory.

Implementation notes:

- `MPSAsFCISolver(root_buffer=N)` always returns only the requested target
  roots to PySCF/SHARC; the extra roots are candidates used for assignment.
- `SHARC_PYSCF_ext.py` caches the previous CI vectors from `pyscf.old.chk`
  into `solver.fcisolver._last_ci`, so trajectory steps use previous-step
  overlap rather than any exact reference.
- The solver records diagnostics in `_root_assignment`,
  `_root_assigned_abs_overlaps`, `_root_min_overlap`, and
  `_root_min_energy_gap`.  SHARC prints warnings for low overlaps or very
  small candidate-root gaps.
- If a new system shows root flips, increase `dmrg-root-buffer`, inspect the
  state characters, and rerun a short static scan before starting dynamics.

Minimal `PYSCF.template` block for a two-state singlet SHARC run:

```text
method dmrg-casscf
roots 2
ncas 8
nelecas 8
dmrg-maxm 500
dmrg-nsteps 50
dmrg-sweep-tol 1.0e-8
dmrg-root-buffer 4
analytic-nac true
```

For a different system, change only the active space, basis, and convergence
knobs.  The root-buffer mechanism remains the same.
