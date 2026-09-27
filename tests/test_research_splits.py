"""Tests for aurum.research.splits: leak-free CV splits (walk-forward, purged k-fold, CPCV)."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from aurum.research.splits import (
    contiguous_blocks,
    cpcv_splits,
    purge_train,
    purged_kfold,
    walk_forward_splits,
)


def _assert_disjoint_sorted(train: np.ndarray, test: np.ndarray, n: int) -> None:
    assert train.dtype == np.int64 and test.dtype == np.int64
    assert np.intersect1d(train, test).size == 0
    assert np.all(np.diff(train) > 0) and np.all(np.diff(test) > 0)
    if train.size:
        assert train.min() >= 0 and train.max() < n
    assert test.min() >= 0 and test.max() < n


def _assert_gaps(train: np.ndarray, test: np.ndarray, purge: int, embargo: int) -> None:
    """No training index within ``purge`` before, or ``purge + embargo`` after, any test block."""
    tr = set(train.tolist())
    for a, b in contiguous_blocks(test):
        before = set(range(max(0, a - purge), a))
        after = set(range(b + 1, b + 1 + purge + embargo))
        assert not (tr & before), f"purge violated before block [{a}, {b}]"
        assert not (tr & after), f"purge/embargo violated after block [{a}, {b}]"


# ---------------------------------------------------------------------------- walk-forward
@pytest.mark.parametrize("anchored", [False, True])
def test_walk_forward_basic_properties(anchored: bool) -> None:
    n, train, test, purge, embargo = 1000, 300, 100, 5, 3
    splits = walk_forward_splits(n, train, test, anchored=anchored, purge=purge, embargo=embargo)
    assert len(splits) == 7  # tests start at 308, 408, ..., 908 (last one truncated to 92)
    prev_test_end = -1
    for tr, te in splits:
        _assert_disjoint_sorted(tr, te, n)
        # all training strictly precedes test, with gap = purge + embargo
        assert tr.max() < te.min() - purge - embargo
        assert te.min() - tr.max() - 1 == purge + embargo
        if anchored:
            assert tr[0] == 0
        else:
            assert tr.size == train
        # contiguous, non-overlapping OOS blocks with default step
        assert te.min() == prev_test_end + 1 or prev_test_end == -1
        prev_test_end = te.max()
    assert splits[-1][1].max() == n - 1
    if anchored:
        sizes = [tr.size for tr, _ in splits]
        assert sizes == sorted(sizes) and sizes[0] == train


def test_walk_forward_step_and_min_test() -> None:
    splits = walk_forward_splits(500, 100, 50, step=25)
    starts = [te[0] for _, te in splits]
    assert np.all(np.diff(starts) == 25)
    full_only = walk_forward_splits(520, 100, 100, min_test=100)
    assert all(te.size == 100 for _, te in full_only)
    assert walk_forward_splits(100, 200, 10) == []


def test_walk_forward_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(100, 0, 10)
    with pytest.raises(ValueError):
        walk_forward_splits(100, 10, 10, purge=-1)
    with pytest.raises(ValueError):
        walk_forward_splits(100, 10, 10, step=0)


# ---------------------------------------------------------------------------- purged k-fold
@pytest.mark.parametrize("purge,embargo", [(0, 0), (10, 0), (0, 7), (12, 5)])
def test_purged_kfold_no_overlap_and_gaps(purge: int, embargo: int) -> None:
    n, k = 1003, 5
    splits = purged_kfold(n, k, purge=purge, embargo=embargo)
    assert len(splits) == k
    all_test = np.concatenate([te for _, te in splits])
    np.testing.assert_array_equal(np.sort(all_test), np.arange(n))  # tests partition the sample
    for tr, te in splits:
        _assert_disjoint_sorted(tr, te, n)
        _assert_gaps(tr, te, purge, embargo)
        # exactly the purged/embargoed observations are missing, nothing more
        a, b = te[0], te[-1]
        expected_removed = set(range(max(0, a - purge), min(n, b + 1 + purge + embargo)))
        kept = set(range(n)) - expected_removed
        assert set(tr.tolist()) == kept


def test_purged_kfold_label_end_overrides_purge() -> None:
    n = 200
    label_end = np.minimum(np.arange(n) + 3, n - 1)
    label_end[40] = 120  # one long-horizon label reaching deep into the second fold
    splits = purged_kfold(n, 4, purge=0, label_end=label_end)
    tr1, te1 = splits[1]  # test = [50, 99]
    assert te1[0] == 50
    assert 40 not in tr1  # its label [40, 120] overlaps the test block
    assert 47 not in tr1 and 46 in tr1  # label [47, 50] overlaps the test start; [46, 49] does not
    # test labels reach up to 102 (99 + 3) -> training obs 100..102 purged, 103 kept
    assert not {100, 101, 102} & set(tr1.tolist()) and 103 in tr1


def test_purge_train_rejects_bad_label_end() -> None:
    with pytest.raises(ValueError):
        purge_train(np.arange(5), np.arange(5, 10), n=10, label_end=np.arange(10) - 1)
    with pytest.raises(ValueError):
        purge_train(np.arange(5), np.arange(5, 10), n=10, label_end=np.arange(9))


# ---------------------------------------------------------------------------- CPCV
@pytest.mark.parametrize("n_groups,k_test", [(6, 2), (8, 3), (5, 1), (10, 2)])
def test_cpcv_counts_and_paths(n_groups: int, k_test: int) -> None:
    n, purge, embargo = 1200, 4, 6
    res = cpcv_splits(n, n_groups=n_groups, k_test=k_test, purge=purge, embargo=embargo)
    assert res.n_splits == comb(n_groups, k_test)
    phi = k_test * comb(n_groups, k_test) // n_groups
    assert res.n_paths == phi == comb(n_groups - 1, k_test - 1)
    assert res.paths.shape == (phi, n_groups)
    for s, (tr, te) in enumerate(res.splits):
        _assert_disjoint_sorted(tr, te, n)
        _assert_gaps(tr, te, purge, embargo)
        expected_test = np.concatenate([res.groups[g] for g in res.test_groups[s]])
        np.testing.assert_array_equal(te, expected_test)
    # Each (split, group) test assignment is used by exactly one path; every path covers
    # every group once with a split that actually tests that group.
    used = set()
    for p in range(res.n_paths):
        for g in range(n_groups):
            s = int(res.paths[p, g])
            assert g in res.test_groups[s]
            assert (s, g) not in used
            used.add((s, g))
    assert len(used) == res.n_splits * k_test
    splits, paths = res  # unpacking convenience
    assert splits is res.splits and paths is res.paths


def test_cpcv_assemble_paths_reconstructs_full_series() -> None:
    n = 300
    res = cpcv_splits(n, n_groups=6, k_test=2, purge=2, embargo=2)
    # prediction of split s at index i encodes both, so we can check provenance
    preds = [te * 1000.0 + s for s, (_, te) in enumerate(res.splits)]
    paths = res.assemble_paths(preds)
    assert paths.shape == (res.n_paths, n)
    assert np.isfinite(paths).all()
    idx = np.arange(n)
    for p in range(res.n_paths):
        np.testing.assert_array_equal(np.floor(paths[p] / 1000.0), idx)
        split_used = np.round(paths[p] - idx * 1000.0).astype(int)
        for g, grp in enumerate(res.groups):
            assert set(split_used[grp].tolist()) == {int(res.paths[p, g])}
    with pytest.raises(ValueError):
        res.assemble_paths(preds[:-1])


def test_cpcv_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        cpcv_splits(100, n_groups=4, k_test=4)
    with pytest.raises(ValueError):
        cpcv_splits(3, n_groups=6, k_test=2)


def test_contiguous_blocks() -> None:
    assert contiguous_blocks(np.array([], dtype=int)) == []
    assert contiguous_blocks(np.array([3, 4, 5, 9, 10, 20])) == [(3, 5), (9, 10), (20, 20)]


# ---------------------------------------------------------------------------- review: adversarial
def test_cpcv_iteration_is_unpacking_only_and_len_is_undefined() -> None:
    """Regression: ``len()`` used to return n_splits while iteration yields (splits, paths).

    With N=2, k=1 (two splits) ``for tr, te in cpcv_splits(...)`` then silently bound
    ``tr = splits`` list-element 0 and ``te`` = element 1. ``len`` is now undefined so the
    object cannot masquerade as a list of splits.
    """
    res = cpcv_splits(100, n_groups=2, k_test=1)
    with pytest.raises(TypeError):
        len(res)  # type: ignore[arg-type]
    items = list(res)
    assert len(items) == 2 and items[0] is res.splits and items[1] is res.paths
    assert res.n_splits == 2
    for tr, te in res.splits:  # the supported way to loop
        assert np.intersect1d(tr, te).size == 0


def test_purge_train_rejects_out_of_range_indices() -> None:
    """An index >= n would shrink the contaminated span (le[a:b+1] past n) and under-purge."""
    with pytest.raises(ValueError):
        purge_train(np.arange(0, 50), np.arange(90, 120), n=100, purge=5)
    with pytest.raises(ValueError):
        purge_train(np.array([-1, 0, 1]), np.arange(50, 60), n=100)


@pytest.mark.parametrize("anchored", [False, True])
def test_walk_forward_label_end_variable_horizon(anchored: bool) -> None:
    """Variable-horizon labels (triple barrier): no training label may resolve at or after
    ``test_start - embargo``; with ``label_end = i + h`` it equals integer ``purge = h``."""
    n, train, test, embargo = 600, 150, 60, 4
    rng = np.random.default_rng(0)
    label_end = np.minimum(np.arange(n) + rng.integers(0, 25, n), n - 1)
    splits = walk_forward_splits(n, train, test, anchored=anchored, purge=0, embargo=embargo,
                                 label_end=label_end)
    assert splits
    for tr, te in splits:
        _assert_disjoint_sorted(tr, te, n)
        assert np.all(label_end[tr] < te[0] - embargo)
        # a strictly shorter horizon than the max is kept when its label is resolved in time
        cutoff = te[0] - embargo  # purge=0 -> training window ends at the embargo zone
        start = 0 if anchored else max(0, cutoff - train)
        kept_expected = [i for i in range(start, cutoff) if label_end[i] < cutoff]
        assert tr.tolist() == kept_expected
    h = 7
    fixed = walk_forward_splits(n, train, test, anchored=anchored, purge=h, embargo=embargo)
    via_le = walk_forward_splits(n, train, test, anchored=anchored, purge=h, embargo=embargo,
                                 label_end=np.arange(n) + h)
    assert len(fixed) == len(via_le)
    for (a_tr, a_te), (b_tr, b_te) in zip(fixed, via_le, strict=True):
        np.testing.assert_array_equal(a_tr, b_tr)
        np.testing.assert_array_equal(a_te, b_te)


def test_walk_forward_randomised_no_lookahead() -> None:
    """Property check over random configurations: train strictly precedes test by the gap,
    test blocks are in chronological order and every label of a training row is resolved
    before the embargo zone."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        n = int(rng.integers(20, 400))
        train, test = int(rng.integers(1, 80)), int(rng.integers(1, 60))
        purge, embargo = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        step = int(rng.integers(1, 60))
        anchored = bool(rng.integers(0, 2))
        splits = walk_forward_splits(n, train, test, step=step, anchored=anchored,
                                     purge=purge, embargo=embargo)
        starts = [te[0] for _, te in splits]
        assert starts == sorted(starts)
        for tr, te in splits:
            assert tr.size >= 1 and te.size >= 1
            assert tr.max() + purge < te.min() - embargo  # label [i, i+purge] ends before gap
            assert te.max() < n and tr.min() >= 0


def test_purged_splitters_warn_on_empty_training_set(caplog) -> None:
    with caplog.at_level("WARNING", logger="aurum.research.splits"):
        splits = purged_kfold(20, 2, purge=15, embargo=5)
    assert any(tr.size == 0 for tr, _ in splits)
    assert "EMPTY training set" in caplog.text
