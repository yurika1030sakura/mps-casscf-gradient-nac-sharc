# Root Matching And Gauge Protocol

This project uses one root-matching protocol across benchmark systems.  The
rules do not depend on molecule names.

## Benchmark Mode

Benchmark mode is used only when an active-space FCI reference is available.
It is a validation protocol, not a production dependency: FCI is used to define
an exact error and a fixed comparison gauge.

1. Build a fixed-orbital spin-adapted singlet FCI reference with PySCF
   `direct_spin0`.
2. Scan enough FCI roots to identify the target singlet roots and their
   same-energy clusters.
3. Convert each buffered DMRG MPS root to a PySCF CI vector.
4. Compute all DMRG/FCI CI overlaps.
5. Lock high-confidence isolated target roots first.  A root is locked only
   when its best overlap is at least `BVOE_ROOT_LOCK_OVERLAP` and exceeds the
   second-best overlap by at least `BVOE_ROOT_LOCK_MARGIN`.
6. For any target root in a same-energy cluster, select a DMRG candidate-root
   subspace and rotate that subspace to the fixed FCI gauge by polar
   alignment.
7. Phase-align and normalize the CI vectors before calling the analytic
   gradient/NAC machinery.
8. Treat low overlap or unconverged Lagrange solves as QC failures.  These
   points remain in JSON but are not used as accuracy claims.

The protocol covers the following special cases:

- ordinary isolated roots with arbitrary phase;
- roots returned in a different order from the reference;
- extra buffered candidate roots;
- one target root belonging to a same-energy cluster while another target is
  isolated;
- two target roots sharing the same near-degenerate cluster;
- larger candidate subspaces where raw root labels are not meaningful.

The synthetic regression test
`benchmarks/bvoe_convergence_study/test_root_matching_protocol.py` exercises
these cases without using any molecule-specific data.

## FCI-Free Selection Check

For claims about replacing FCI, the benchmark set should also include an
endpoint check in which the DMRG calculation does not use FCI for root
selection.  In that check:

1. DMRG solves buffered roots and selects states by energy order at the first
   endpoint or by previous accepted DMRG roots along a scan.
2. Gradients and NACs are evaluated from the selected DMRG roots.
3. The FCI result is loaded only after the DMRG result exists, to report
   offline errors and overlaps.

This check should be run for the representative problematic classes:
ordinary isolated roots, small-NAC symmetry cases, avoided crossings, and
near-degenerate clusters.  For near-degenerate clusters the comparison should
emphasize subspace continuity because an individual raw root label is not a
gauge-invariant object.

## Production Mode

Production DMRG-SHARC calculations do not have an FCI reference.  They use the
same continuity principle, but the reference is the previous accepted
wave-function set from the trajectory or scan:

1. solve buffered candidate roots;
2. match current roots to previous accepted roots by CI/MPS overlap;
3. phase-align before returning properties;
4. near degeneracies, monitor and propagate the state subspace rather than
   interpreting an arbitrary raw root label as chemically unique;
5. confirm robustness by increasing bond dimension, sweep count, and root
   buffer.

This distinction is important: FCI references are for reporting benchmark
errors, while production root following is based on continuity along the
actual geometry path.
