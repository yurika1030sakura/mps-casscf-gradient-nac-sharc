"""Regression tests for DMRG root/phase tracking.

These tests are intentionally small and do not run DMRG.  They validate the
production bookkeeping used after each eigensolve: DMRG roots may be returned
in a different order or with arbitrary signs, and the solver must map them
onto the previous/root-reference gauge before PySCF gradient/NAC code sees
the CI list.
"""

import numpy as np

from dmrg_fcisolver import MPSAsFCISolver


def test_root_permutation_and_phase_tracking():
    solver = MPSAsFCISolver(track_roots=True)

    ref0 = np.array([1.0, 0.0, 0.0])
    ref1 = np.array([0.0, 1.0, 0.0])
    solver._last_ci = [ref0, ref1]

    # New DMRG order is swapped and root 1 has the opposite sign.
    new_roots = [-ref1.copy(), ref0.copy()]
    tracked = solver._track_and_store_ci(new_roots)

    assert solver._root_assignment == [1, 0]
    assert np.allclose(tracked[0], ref0)
    assert np.allclose(tracked[1], ref1)


def test_root_tracking_uses_ci0_before_cached_roots():
    solver = MPSAsFCISolver(track_roots=True)

    cached0 = np.array([1.0, 0.0])
    cached1 = np.array([0.0, 1.0])
    ci0_ref0 = np.array([0.0, 1.0])
    ci0_ref1 = np.array([1.0, 0.0])
    solver._last_ci = [cached0, cached1]

    tracked = solver._track_and_store_ci(
        [cached0.copy(), cached1.copy()],
        ci0=[ci0_ref0, ci0_ref1],
    )

    assert solver._root_assignment == [1, 0]
    assert np.allclose(tracked[0], ci0_ref0)
    assert np.allclose(tracked[1], ci0_ref1)


def test_extra_candidate_roots_are_selected_by_overlap():
    solver = MPSAsFCISolver(track_roots=True, root_buffer=1)

    ref0 = np.array([1.0, 0.0, 0.0])
    ref1 = np.array([0.0, 1.0, 0.0])
    distractor = np.array([0.0, 0.0, 1.0])
    solver._last_ci = [ref0, ref1]

    # Candidate roots contain one extra state, and the two target states are
    # neither in energy/root order nor in the same arbitrary phase.
    tracked = solver._track_and_store_ci(
        [distractor.copy(), -ref1.copy(), ref0.copy()],
        target_nroots=2,
    )

    assert solver._root_assignment == [2, 1]
    assert np.allclose(tracked[0], ref0)
    assert np.allclose(tracked[1], ref1)
    assert np.allclose(solver._root_assigned_abs_overlaps, [1.0, 1.0])
    assert solver._root_min_overlap == 1.0


def test_extra_candidates_without_reference_use_energy_order():
    solver = MPSAsFCISolver(track_roots=True, root_buffer=2)

    ref0 = np.array([1.0, 0.0, 0.0])
    ref1 = np.array([0.0, 1.0, 0.0])
    extra = np.array([0.0, 0.0, 1.0])
    tracked = solver._track_and_store_ci(
        [ref0.copy(), ref1.copy(), extra.copy()],
        target_nroots=2,
    )

    assert solver._root_assignment == [0, 1]
    assert np.allclose(tracked[0], ref0)
    assert np.allclose(tracked[1], ref1)
    assert "no previous CI reference" in solver._root_tracking_warnings[0]


if __name__ == "__main__":
    test_root_permutation_and_phase_tracking()
    test_root_tracking_uses_ci0_before_cached_roots()
    test_extra_candidate_roots_are_selected_by_overlap()
    test_extra_candidates_without_reference_use_energy_order()
    print("test_dmrg_root_tracking.py: all tests passed")
