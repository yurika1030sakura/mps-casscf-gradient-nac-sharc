"""System-independent regression tests for benchmark root matching.

These tests cover root-order permutations, arbitrary phases, exact/near
degenerate target clusters, and the case where a degenerate candidate subspace
could otherwise consume an obvious isolated root.  They intentionally use
synthetic CI vectors so that the protocol is tested without any molecule-
specific heuristic.
"""

from __future__ import annotations

import numpy as np

from run_bvoe_phase2 import _match_and_align_roots


def _unit(index: int, size: int = 8) -> np.ndarray:
    vec = np.zeros((size, 1))
    vec[index, 0] = 1.0
    return vec


def _rot(a: np.ndarray, b: np.ndarray, sign: float = 1.0) -> np.ndarray:
    return (a + sign * b) / np.sqrt(2.0)


def _clusters(*roots: list[int]) -> list[dict]:
    return [
        {"target_index": i, "cluster_roots": list(root_list)}
        for i, root_list in enumerate(roots)
    ]


def _assert_unit_overlap(assigned: list[float], tol: float = 1.0e-12) -> None:
    assert min(float(x) for x in assigned) > 1.0 - tol, assigned


def test_isolated_roots_allow_permutation_and_phase() -> None:
    e0, e1, e2 = _unit(0), _unit(1), _unit(2)
    aligned, assignment, overlap, assigned, diagnostics = _match_and_align_roots(
        [-e1.copy(), e0.copy(), e2.copy()],
        [e0, e1],
        _clusters([0], [1]),
    )

    assert assignment == [1, 0]
    assert [row["mode"] for row in diagnostics] == [
        "confident_isolated_root_lock",
        "confident_isolated_root_lock",
    ]
    assert np.allclose(aligned[0], e0)
    assert np.allclose(aligned[1], e1)
    _assert_unit_overlap(assigned)


def test_degenerate_target_subspace_is_aligned_to_fci_gauge() -> None:
    e0, e1, e2 = _unit(0), _unit(1), _unit(2)
    raw = [_rot(e0, e1), _rot(e0, e1, sign=-1.0), e2]
    aligned, assignment, overlap, assigned, diagnostics = _match_and_align_roots(
        raw,
        [e0, e1],
        _clusters([0, 1], [0, 1]),
    )

    assert assignment == [0, 1]
    assert [row["mode"] for row in diagnostics] == [
        "degenerate_subspace_projection",
        "degenerate_subspace_projection",
    ]
    assert np.allclose(aligned[0], e0)
    assert np.allclose(aligned[1], e1)
    _assert_unit_overlap(assigned)


def test_degenerate_pool_does_not_steal_confident_isolated_root() -> None:
    e0, e1, e2, e3 = _unit(0), _unit(1), _unit(2), _unit(3)
    aligned, assignment, overlap, assigned, diagnostics = _match_and_align_roots(
        [e0.copy(), e1.copy(), e2.copy(), e3.copy()],
        [e0, e1],
        _clusters([0], [1, 2]),
    )

    assert assignment == [0, [1, 2]]
    assert diagnostics[0]["mode"] == "confident_isolated_root_lock"
    assert diagnostics[1]["mode"] == "degenerate_subspace_projection"
    assert 0 not in diagnostics[1]["raw_roots"]
    assert np.allclose(aligned[0], e0)
    assert np.allclose(aligned[1], e1)
    _assert_unit_overlap(assigned)


def test_larger_degenerate_candidate_pool_uses_subspace_not_root_label() -> None:
    e0, e1, e2, e3 = _unit(0), _unit(1), _unit(2), _unit(3)
    raw = [_rot(e0, e2), _rot(e1, e2), e2.copy(), e3.copy()]
    aligned, assignment, overlap, assigned, diagnostics = _match_and_align_roots(
        raw,
        [e0, e1],
        _clusters([0, 1, 2], [0, 1, 2]),
    )

    assert diagnostics[0]["mode"] == "degenerate_subspace_projection"
    assert diagnostics[1]["mode"] == "degenerate_subspace_projection"
    assert len(diagnostics[0]["raw_roots"]) == 3
    assert len(diagnostics[1]["raw_roots"]) == 3
    assert np.allclose(aligned[0], e0)
    assert np.allclose(aligned[1], e1)
    _assert_unit_overlap(assigned)


def main() -> None:
    test_isolated_roots_allow_permutation_and_phase()
    test_degenerate_target_subspace_is_aligned_to_fci_gauge()
    test_degenerate_pool_does_not_steal_confident_isolated_root()
    test_larger_degenerate_candidate_pool_uses_subspace_not_root_label()
    print("test_root_matching_protocol.py: all tests passed")


if __name__ == "__main__":
    main()
