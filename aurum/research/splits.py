"""Leak-free cross-validation splits for financial time series.

Why ordinary K-fold is wrong for trading research
-------------------------------------------------
Financial observations are neither independent nor identically distributed, and labels are
built from *future* prices (a label at bar ``i`` typically depends on bars ``i..i+h``).
Shuffled or naive K-fold therefore lets information from the test period leak into training
in two ways (López de Prado, *Advances in Financial Machine Learning* (AFML), 2018, ch. 7):

1. **Label overlap** - a training observation whose label horizon ``[i, i+h]`` intersects the
   test period has "seen" test prices. Such observations must be **purged**.
2. **Serial correlation** - training observations that immediately *follow* a test block are
   correlated with the tail of the test block (and the test labels extend into them). AFML
   recommends an additional **embargo** of bars after every test block.

Conventions used throughout this module
---------------------------------------
* Observations are integer positions ``0..n-1`` in chronological order (e.g. bar rows).
* Every observation ``i`` carries a *label interval* ``[i, label_end[i]]`` (inclusive).
  By default ``label_end[i] = i + purge`` - i.e. ``purge`` is the label horizon in bars.
  An explicit ``label_end`` array (e.g. triple-barrier touch positions) can be supplied and
  is combined with ``purge`` via ``max``.
* For a contiguous test block ``[a, b]`` the *contaminated span* is
  ``[a, max(label_end[a..b]) + embargo]``. A training observation ``i`` is removed when its
  label interval intersects that span: ``i <= span_end and label_end[i] >= a``. With the
  default integer purge this removes ``[a - purge, a)`` before the block and
  ``(b, b + purge + embargo]`` after it - exactly AFML's purge + embargo.
* Returned indices are sorted ``np.int64`` arrays; train and test never intersect.

References
----------
* M. López de Prado (2018), *Advances in Financial Machine Learning*, Wiley - ch. 7
  (purged K-fold, embargo) and ch. 12 (combinatorial purged cross-validation).
* D. Bailey, J. Borwein, M. López de Prado, Q. Zhu (2017), "The Probability of Backtest
  Overfitting", *Journal of Computational Finance* 20(4) - motivation for CPCV/CSCV.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np

logger = logging.getLogger(__name__)

Split = tuple[np.ndarray, np.ndarray]

__all__ = [
    "Split",
    "CPCVSplits",
    "walk_forward_splits",
    "purged_kfold",
    "cpcv_splits",
    "purge_train",
    "contiguous_blocks",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _check_nonneg_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _check_pos_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _label_end(n: int, purge: int, label_end: Sequence[int] | np.ndarray | None) -> np.ndarray:
    """Effective inclusive label end per observation: ``max(i + purge, label_end[i])``."""
    pos = np.arange(n, dtype=np.int64)
    le = pos + purge
    if label_end is not None:
        arr = np.asarray(label_end)
        if arr.shape != (n,):
            raise ValueError(f"label_end must have shape ({n},), got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            if not np.all(np.isfinite(arr)) or np.any(arr != np.round(arr)):
                raise ValueError("label_end must contain integer positions")
            arr = arr.astype(np.int64)
        if np.any(arr < pos):
            raise ValueError("label_end[i] must be >= i (labels cannot end before they start)")
        le = np.maximum(le, arr.astype(np.int64))
    return le


def _warn_empty_train(where: str, splits: list[Split]) -> None:
    empty = [i for i, (tr, _) in enumerate(splits) if tr.size == 0]
    if empty:
        logger.warning("%s: purge/embargo left %d split(s) with an EMPTY training set: %s",
                       where, len(empty), empty[:10])


def contiguous_blocks(idx: np.ndarray) -> list[tuple[int, int]]:
    """Split a sorted integer index into maximal runs of consecutive values.

    Returns a list of inclusive ``(start, end)`` pairs. Used to treat a (possibly
    non-contiguous, e.g. CPCV) test set as a union of contiguous blocks for purging.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    idx = np.sort(idx)
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def purge_train(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n: int,
    purge: int = 0,
    embargo: int = 0,
    label_end: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Remove from ``train_idx`` every observation contaminated by ``test_idx``.

    For each contiguous test block ``[a, b]`` the contaminated span is
    ``[a, max(label_end[a..b]) + embargo]`` (test labels reach forward, then the embargo
    adds a safety margin for serial correlation). A training observation ``i`` with label
    interval ``[i, label_end[i]]`` is dropped if the two intervals intersect (AFML
    snippet 7.1 "getTrainTimes" generalised to integer positions).
    """
    n = _check_pos_int("n", n)
    purge = _check_nonneg_int("purge", purge)
    embargo = _check_nonneg_int("embargo", embargo)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    for name, arr in (("train_idx", train_idx), ("test_idx", test_idx)):
        if arr.size and (arr.min() < 0 or arr.max() >= n):
            # An out-of-range test index would silently shrink the contaminated span
            # (``le[a:b+1]`` slicing past ``n``) and under-purge.
            raise ValueError(f"{name} must lie in [0, {n}), got [{arr.min()}, {arr.max()}]")
    if train_idx.size == 0 or test_idx.size == 0:
        return np.sort(train_idx)
    le = _label_end(n, purge, label_end)
    keep = np.ones(train_idx.size, dtype=bool)
    le_train = le[train_idx]
    for a, b in contiguous_blocks(test_idx):
        span_end = int(le[a : b + 1].max()) + embargo
        keep &= ~((train_idx <= span_end) & (le_train >= a))
    out = np.sort(train_idx[keep])
    # Safety net: purging must never leave train/test overlap.
    if np.intersect1d(out, test_idx, assume_unique=False).size:
        raise AssertionError("purge_train produced overlapping train/test indices")
    return out


# --------------------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------------------
def walk_forward_splits(
    n: int,
    train: int,
    test: int,
    step: int | None = None,
    anchored: bool = False,
    purge: int = 0,
    embargo: int = 0,
    *,
    min_test: int = 1,
    label_end: Sequence[int] | np.ndarray | None = None,
) -> list[Split]:
    """Rolling or anchored (expanding) walk-forward splits.

    Fold ``j`` tests on ``[t_j, min(t_j + test, n))`` with ``t_j = train + gap + j * step``
    and ``gap = purge + embargo``. Training uses the ``train`` observations that end
    ``gap`` bars before the test block (rolling) or everything from 0 up to that point
    (``anchored=True``, a.k.a. expanding window).

    Walk-forward is the most "honest" backtest protocol because it mimics live usage:
    the model is only ever trained on the past. Two gaps still matter:

    * ``purge`` - labels of the last training rows look ``purge`` bars ahead and would
      otherwise overlap the test block (AFML ch. 7).
    * ``embargo`` - because all training data precede the test block there is no
      post-test data to embargo in the AFML sense. It is applied as an *additional*
      pre-test gap so that the same ``(purge, embargo)`` configuration can be passed to
      every splitter without ever being less conservative.

    Parameters
    ----------
    n        : number of observations.
    train    : training window length (the first window's length when ``anchored``).
    test     : test window length.
    step     : distance between consecutive test starts (default ``test`` = contiguous,
               non-overlapping OOS blocks that can be stitched into one OOS series).
    anchored : expanding window if True, rolling window of length ``train`` otherwise.
    min_test : the final (possibly truncated) test block is kept only if it has at least
               this many observations.
    label_end: optional inclusive label-end position per observation (e.g. triple-barrier
               first-touch index) for *variable-horizon* labels. A training observation is
               then additionally dropped when ``label_end[i] >= t_j - embargo``, i.e. when
               its label would be resolved inside the embargo zone or the test block.
               Without it, ``purge`` must be the *maximum* label horizon.

    Returns
    -------
    list of ``(train_idx, test_idx)`` int64 arrays in chronological order.
    """
    n = _check_pos_int("n", n)
    train = _check_pos_int("train", train)
    test = _check_pos_int("test", test)
    step = test if step is None else _check_pos_int("step", step)
    purge = _check_nonneg_int("purge", purge)
    embargo = _check_nonneg_int("embargo", embargo)
    min_test = _check_pos_int("min_test", min_test)
    gap = purge + embargo
    le = _label_end(n, purge, label_end) if label_end is not None else None

    splits: list[Split] = []
    t0 = train + gap
    while t0 < n:
        t1 = min(t0 + test, n)
        if t1 - t0 < min_test:
            break
        tr_end = t0 - gap  # exclusive
        tr_start = 0 if anchored else max(0, tr_end - train)
        tr_idx = np.arange(tr_start, tr_end, dtype=np.int64)
        if le is not None:
            # label [i, le[i]] must be fully resolved before the embargo zone starts
            tr_idx = tr_idx[le[tr_idx] < t0 - embargo]
        splits.append((tr_idx, np.arange(t0, t1, dtype=np.int64)))
        t0 += step
    _warn_empty_train("walk_forward_splits", splits)
    if not splits:
        logger.warning(
            "walk_forward_splits produced no folds (n=%d, train=%d, test=%d, gap=%d)",
            n, train, test, gap,
        )
    return splits


# --------------------------------------------------------------------------------------
# purged k-fold
# --------------------------------------------------------------------------------------
def purged_kfold(
    n: int,
    k: int = 5,
    purge: int = 0,
    embargo: int = 0,
    *,
    label_end: Sequence[int] | np.ndarray | None = None,
) -> list[Split]:
    """Purged K-fold cross-validation with embargo (AFML ch. 7.4).

    The sample is cut into ``k`` contiguous, (almost) equally sized test blocks - no
    shuffling. For each block the training set is every other observation minus those
    purged/embargoed by :func:`purge_train`. Training data may lie *after* the test block;
    this is acceptable for model-selection CV (not for backtesting) precisely because the
    purge and embargo remove the observations that could carry test information.

    ``label_end`` optionally gives the inclusive position at which each observation's
    label is determined (e.g. triple-barrier first-touch index); it overrides ``i + purge``
    when larger.
    """
    n = _check_pos_int("n", n)
    k = _check_pos_int("k", k)
    if k < 2 or k > n:
        raise ValueError(f"k must satisfy 2 <= k <= n, got k={k}, n={n}")
    purge = _check_nonneg_int("purge", purge)
    embargo = _check_nonneg_int("embargo", embargo)
    all_idx = np.arange(n, dtype=np.int64)
    splits: list[Split] = []
    for test_idx in np.array_split(all_idx, k):
        test_idx = test_idx.astype(np.int64)
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        train_idx = purge_train(
            train_idx, test_idx, n=n, purge=purge, embargo=embargo, label_end=label_end
        )
        splits.append((train_idx, test_idx))
    _warn_empty_train("purged_kfold", splits)
    return splits


# --------------------------------------------------------------------------------------
# combinatorial purged cross-validation
# --------------------------------------------------------------------------------------
@dataclass
class CPCVSplits:
    """Result of :func:`cpcv_splits`.

    Attributes
    ----------
    splits      : ``C(N, k)`` ``(train_idx, test_idx)`` pairs.
    groups      : the ``N`` contiguous index groups the sample was cut into.
    test_groups : for each split, the tuple of group ids used as test set.
    paths       : int array of shape ``(n_paths, N)``; ``paths[p, g]`` is the index of the
                  split whose OOS prediction for group ``g`` belongs to backtest path ``p``.
                  Each column is a permutation of the ``C(N-1, k-1)`` splits that test ``g``.

    Unpacking yields ``(splits, paths)`` so ``splits, paths = cpcv_splits(...)`` works.
    Iterating the object therefore yields exactly those two items - to loop over the
    train/test pairs use ``for train, test in res.splits``. ``len()`` is deliberately not
    defined (it used to return ``n_splits``, inconsistent with iteration, which made
    ``for train, test in cpcv_splits(n, 2, 1)`` silently mis-assign); use ``n_splits``.
    """

    splits: list[Split]
    groups: list[np.ndarray]
    test_groups: list[tuple[int, ...]]
    paths: np.ndarray
    n: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_splits(self) -> int:
        return len(self.splits)

    @property
    def n_paths(self) -> int:
        return int(self.paths.shape[0])

    def __iter__(self) -> Iterator:
        yield self.splits
        yield self.paths

    def path_segments(self, path: int) -> list[tuple[int, np.ndarray]]:
        """``[(split_id, group_indices), ...]`` making up backtest path ``path``."""
        return [(int(self.paths[path, g]), self.groups[g]) for g in range(len(self.groups))]

    def assemble_paths(self, predictions: Sequence[np.ndarray]) -> np.ndarray:
        """Stitch per-split OOS predictions into ``n_paths`` full-length backtest paths.

        ``predictions[s]`` must be a 1-D array aligned with ``splits[s][1]`` (the test
        indices of split ``s``). Returns an array of shape ``(n_paths, n)``: row ``p`` is a
        complete out-of-sample prediction series in which every observation was predicted
        by a model that never trained on it (nor on its purged/embargoed neighbours).
        """
        if len(predictions) != self.n_splits:
            raise ValueError(f"expected {self.n_splits} prediction arrays, got {len(predictions)}")
        out = np.full((self.n_paths, self.n), np.nan)
        for p in range(self.n_paths):
            for g, grp in enumerate(self.groups):
                s = int(self.paths[p, g])
                test_idx = self.splits[s][1]
                pred = np.asarray(predictions[s], dtype=float)
                if pred.shape != test_idx.shape:
                    raise ValueError(
                        f"predictions[{s}] has shape {pred.shape}, expected {test_idx.shape}"
                    )
                pos = np.searchsorted(test_idx, grp)
                out[p, grp] = pred[pos]
        return out


def cpcv_splits(
    n: int,
    n_groups: int = 6,
    k_test: int = 2,
    purge: int = 0,
    embargo: int = 0,
    *,
    label_end: Sequence[int] | np.ndarray | None = None,
) -> CPCVSplits:
    """Combinatorial Purged Cross-Validation (AFML ch. 12).

    The sample is cut into ``N = n_groups`` contiguous groups; every combination of
    ``k = k_test`` groups is used once as the test set (with the remaining groups, purged
    and embargoed around *each* test block, as training set). That yields

    * ``C(N, k)`` train/test splits, and
    * ``phi = k / N * C(N, k) = C(N-1, k-1)`` complete backtest *paths*,

    because each group is tested in exactly ``C(N-1, k-1)`` splits. Path ``p`` takes, for
    each group ``g``, the ``p``-th split (in lexicographic order of the combinations) that
    tests ``g`` - the reconstruction of AFML fig. 12.1. Evaluating a strategy on all paths
    gives a *distribution* of OOS Sharpe ratios instead of the single path of walk-forward,
    which is what makes CPCV useful against selection bias.
    """
    n = _check_pos_int("n", n)
    n_groups = _check_pos_int("n_groups", n_groups)
    k_test = _check_pos_int("k_test", k_test)
    if n_groups < 2 or n_groups > n:
        raise ValueError(f"n_groups must satisfy 2 <= n_groups <= n, got {n_groups} (n={n})")
    if k_test >= n_groups:
        raise ValueError(f"k_test must be < n_groups, got k_test={k_test}, n_groups={n_groups}")
    purge = _check_nonneg_int("purge", purge)
    embargo = _check_nonneg_int("embargo", embargo)

    all_idx = np.arange(n, dtype=np.int64)
    groups = [g.astype(np.int64) for g in np.array_split(all_idx, n_groups)]
    combos = list(combinations(range(n_groups), k_test))
    splits: list[Split] = []
    for combo in combos:
        test_idx = np.concatenate([groups[g] for g in combo])
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        train_idx = purge_train(
            train_idx, test_idx, n=n, purge=purge, embargo=embargo, label_end=label_end
        )
        splits.append((train_idx, test_idx))
    _warn_empty_train("cpcv_splits", splits)

    n_paths = comb(n_groups - 1, k_test - 1)
    paths = np.empty((n_paths, n_groups), dtype=np.int64)
    for g in range(n_groups):
        testing = [s for s, combo in enumerate(combos) if g in combo]
        if len(testing) != n_paths:  # pragma: no cover - combinatorial identity
            raise AssertionError("CPCV path construction invariant violated")
        paths[:, g] = testing
    expected = comb(n_groups, k_test) * k_test // n_groups
    if n_paths != expected:  # pragma: no cover - combinatorial identity
        raise AssertionError("CPCV path count mismatch")
    logger.debug("cpcv_splits: n=%d N=%d k=%d -> %d splits, %d paths", n, n_groups, k_test,
                 len(splits), n_paths)
    return CPCVSplits(
        splits=splits,
        groups=groups,
        test_groups=[tuple(c) for c in combos],
        paths=paths,
        n=n,
        meta={"n_groups": n_groups, "k_test": k_test, "purge": purge, "embargo": embargo},
    )
