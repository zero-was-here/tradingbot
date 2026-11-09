"""v2.1 core fixes - ``resample_bars``: a LEADING partial bucket (data that starts inside a
bucket) is not emitted as complete unless its first base bar opens at the bucket open or the
caller passes ``complete_from``; downloads that pass ``complete_until`` keep their previous
output. Point-in-time invariants: HTF bars never depend on where the history starts or ends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.data.resample import align_htf, resample_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.features.multi_timeframe import resample_anchored


@pytest.fixture(scope="module")
def h1() -> pd.DataFrame:
    return make_synthetic_bars(1500, "H1", seed=11, model="regime")


@pytest.fixture(scope="module")
def m15() -> pd.DataFrame:
    return make_synthetic_bars(3000, "M15", seed=5)


def _assert_rows_equal(sub: pd.DataFrame, full: pd.DataFrame) -> None:
    assert sub.index.isin(full.index).all(), "suffix emitted a bar the full history does not have"
    pd.testing.assert_frame_equal(sub, full.loc[sub.index], check_freq=False)


# ---- the head rule ---------------------------------------------------------------------------------
def test_leading_partial_bucket_is_dropped(h1):
    full = resample_bars(h1, "D1")
    sub = resample_bars(h1.iloc[10:], "D1")                # starts at 10:00 inside day 1
    assert full.index[0] == h1.index[0].normalize()
    assert sub.index[0] == full.index[1]                   # the partial first day is gone
    _assert_rows_equal(sub, full)
    # what the old rule emitted: a "day" whose open is the 10:00 open and whose range misses 10h
    legacy = resample_bars(h1.iloc[10:], "D1", complete_from=h1.index[0])
    assert legacy.index[0] == full.index[0]
    assert legacy["open"].iloc[0] == h1["open"].iloc[10] != full["open"].iloc[0]


def test_first_bar_at_bucket_open_is_kept(h1):
    day2 = int(np.flatnonzero(h1.index == h1.index[0].normalize() + pd.Timedelta(days=1))[0])
    sub = resample_bars(h1.iloc[day2:], "D1")
    assert sub.index[0] == h1.index[day2]
    _assert_rows_equal(sub, resample_bars(h1, "D1"))


def test_complete_from_semantics(h1):
    sub = h1.iloc[10:]
    bucket = h1.index[0].normalize()
    # data known complete from the bucket open (market closed before the first bar): kept
    assert resample_bars(sub, "D1", complete_from=bucket).index[0] == bucket
    assert resample_bars(sub, "D1", complete_from=str(bucket.tz_localize(None))).index[0] == bucket
    # complete only from INSIDE the bucket: still partial
    assert resample_bars(sub, "D1", complete_from=sub.index[0]).index[0] == bucket + pd.Timedelta(days=1)
    # explicit None is strict even with complete_until
    strict = resample_bars(sub, "D1", complete_until=sub["available_at"].iloc[-1], complete_from=None)
    assert strict.index[0] == bucket + pd.Timedelta(days=1)


def test_complete_until_callers_keep_the_previous_head_rule(h1):
    """Final historical downloads (dukascopy passes complete_until) trust their head, so the
    datasets they build are unchanged (they drop buckets cut by the request start themselves);
    the shipped data_store files were rebuilt offline with this code and matched their
    manifest hashes."""
    sub = h1.iloc[10:]
    cu = sub["available_at"].iloc[-1]
    legacy = resample_bars(sub, "D1", complete_until=cu)
    assert legacy.index[0] == h1.index[0].normalize()
    # a Sunday-22:00 market open inside a midnight-anchored day (the 2012-01-01 D1 bar)
    idx = pd.date_range("2024-01-07 22:00", periods=30, freq="h", tz="UTC")
    bars = make_synthetic_bars(30, "H1", seed=1, start="2024-01-07 22:00", weekend_gaps=False)
    assert bars.index.equals(idx)
    assert resample_bars(bars, "D1", complete_until=bars["available_at"].iloc[-1]).index[0] == idx[0].normalize()
    assert resample_bars(bars, "D1").index[0] == pd.Timestamp("2024-01-08", tz="UTC")
    assert resample_bars(bars, "D1", complete_from=idx[0].normalize()).index[0] == idx[0].normalize()


def test_head_rule_with_daily_anchor_and_h4(h1):
    for to, anchor in (("H4", 0), ("H4", 22), ("D1", 22), ("D1", 21)):
        full = resample_bars(h1, to, daily_anchor_hour_utc=anchor)
        for k in (0, 1, 3):
            sub = resample_bars(h1.iloc[k:], to, daily_anchor_hour_utc=anchor)
            # no emitted bucket opens before the first base bar (it would be partial)
            assert sub.index[0] >= h1.index[k]
            _assert_rows_equal(sub, full)
    # data starting Monday 00:00 with 22:00-anchored days: the Sunday-22:00 bucket is partial
    d1 = resample_bars(h1, "D1", daily_anchor_hour_utc=22)
    assert h1.index[0] == pd.Timestamp("2020-01-06", tz="UTC")
    assert d1.index[0] == pd.Timestamp("2020-01-06 22:00", tz="UTC")


def test_only_a_partial_bucket_gives_an_empty_frame(h1):
    out = resample_bars(h1.iloc[3:12], "D1")
    assert out.empty and list(out.columns) == ["open", "high", "low", "close", "volume", "spread", "available_at"]
    assert out.attrs["timeframe"] == "D1"


# ---- point-in-time invariants --------------------------------------------------------------------
@pytest.mark.parametrize("to,anchor", [("H4", 0), ("H4", 22), ("D1", 0), ("D1", 22)])
def test_suffix_invariance_any_start(h1, to, anchor):
    """HTF bars computed on ANY suffix of the history equal the full-history bars (a live
    sliding window reproduces research)."""
    full = resample_bars(h1, to, daily_anchor_hour_utc=anchor)
    rng = np.random.default_rng(7)
    for k in sorted({1, 2, 5, 23, 24, 25, *rng.integers(1, 1200, size=12).tolist()}):
        _assert_rows_equal(resample_bars(h1.iloc[k:], to, daily_anchor_hour_utc=anchor), full)


@pytest.mark.parametrize("to,anchor", [("H4", 22), ("D1", 0), ("D1", 22)])
def test_prefix_invariance_any_end(h1, to, anchor):
    """Truncating the FUTURE never changes an emitted bar (and never emits the in-progress one)."""
    full = resample_bars(h1, to, daily_anchor_hour_utc=anchor)
    for k in (30, 101, 333, 777, 1499):
        sub = resample_bars(h1.iloc[:k], to, daily_anchor_hour_utc=anchor)
        _assert_rows_equal(sub, full)
        assert (sub["available_at"] <= h1["available_at"].iloc[k - 1]).all()


def test_window_invariance_m15_to_h1(m15):
    full = resample_bars(m15, "H1")
    for a, b in ((1, 900), (3, 2000), (7, 2999), (13, 1234)):
        _assert_rows_equal(resample_bars(m15.iloc[a:b], "H1"), full)


def test_available_at_and_alignment_are_point_in_time(h1):
    d1 = resample_bars(h1.iloc[7:], "D1", daily_anchor_hour_utc=22)
    assert (d1["available_at"] >= d1.index + pd.Timedelta(days=1)).all()
    aligned = align_htf(h1.iloc[7:], d1, columns=["close"])
    dec = h1["available_at"].iloc[7:]
    # every aligned value comes from a D1 bar that was available at the decision time
    src = d1["available_at"].reindex(d1.index)
    for t in dec.index[::97]:
        v = aligned.loc[t, "close"]
        if np.isfinite(v):
            j = d1.index[d1["close"] == v][0]
            assert src.loc[j] <= dec.loc[t]


def test_resample_anchored_matches_resample_bars_with_the_head_rule(h1):
    for anchor in (0, 21, 22):
        a = resample_anchored(h1.iloc[5:], "D1", anchor)
        b = resample_bars(h1.iloc[5:], "D1", daily_anchor_hour_utc=anchor)
        pd.testing.assert_frame_equal(a, b, check_freq=False)
