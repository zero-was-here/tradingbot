"""Edge cases for aurum.data.pit.asof_join and aurum.data.resample.resample_bars."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aurum.data.pit import asof_join
from aurum.data.resample import align_htf, resample_bars
from aurum.data.schema import make_bars, validate_bars
from aurum.data.synthetic import make_synthetic_bars


def _bars(times: list[str] | pd.DatetimeIndex, tf: str, start_px: float = 2000.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(times)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    n = len(idx)
    close = start_px + np.arange(n, dtype=float)
    df = pd.DataFrame(
        {"open": close - 0.5, "high": close + 1.0, "low": close - 1.0, "close": close,
         "volume": np.ones(n), "spread": np.full(n, 0.3)},
        index=idx,
    )
    return make_bars(df, tf)


# ------------------------------------------------------------------------------------ asof
def test_asof_join_never_uses_future_rows():
    right = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0],
         "available_at": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 12:00", "2024-01-01 14:00"], utc=True)}
    )
    t = pd.DatetimeIndex(["2024-01-01 09:59", "2024-01-01 10:00", "2024-01-01 13:59", "2024-01-01 15:00"], tz="UTC")
    out = asof_join(t, right)
    assert np.isnan(out["value"].iloc[0])           # nothing available yet → NaN, not back-filled
    assert out["value"].tolist()[1:] == [1.0, 2.0, 3.0]  # exact availability instant is usable


def test_asof_join_preserves_order_tolerance_and_nat():
    right = pd.DataFrame({"x": [1.0, 2.0],
                          "available_at": pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True)})
    times = pd.Series(pd.to_datetime(["2024-01-04", "2024-01-02", None], utc=True), index=["a", "b", "c"])
    out = asof_join(times, right, tolerance=pd.Timedelta(days=2))
    assert list(out.index) == ["a", "b", "c"]
    assert out.loc["a", "x"] == 2.0 and out.loc["b", "x"] == 1.0
    assert np.isnan(out.loc["c", "x"])
    stale = asof_join(pd.DatetimeIndex(["2024-01-10"], tz="UTC"), right, tolerance=pd.Timedelta(days=2))
    assert np.isnan(stale["x"].iloc[0])


def test_asof_join_empty_right_and_errors():
    right = pd.DataFrame({"x": pd.Series([], dtype=float),
                          "available_at": pd.Series([], dtype="datetime64[ns, UTC]")})
    out = asof_join(pd.DatetimeIndex(["2024-01-01"], tz="UTC"), right)
    assert out.shape == (1, 1) and np.isnan(out["x"].iloc[0])
    with pytest.raises(ValueError):
        asof_join(pd.DatetimeIndex(["2024-01-01"]), right)  # naive decision times
    with pytest.raises(KeyError):
        asof_join(pd.DatetimeIndex(["2024-01-01"], tz="UTC"), right.drop(columns="available_at"))


def test_asof_join_same_instant_keeps_latest_row():
    right = pd.DataFrame({"x": [1.0, 2.0], "available_at": pd.to_datetime(["2024-01-01", "2024-01-01"], utc=True)})
    out = asof_join(pd.DatetimeIndex(["2024-01-02"], tz="UTC"), right)
    assert out["x"].iloc[0] == 2.0


# ------------------------------------------------------------------------------------ resample
def test_weekend_gap_produces_no_weekend_bars():
    m15 = make_synthetic_bars(2000, "M15", seed=3)
    h1 = resample_bars(m15, "H1")
    validate_bars(h1)
    assert not (h1.index.dayofweek == 5).any()
    # Friday 20:00 is the last H1 bar before the close (market shut at 21:00 UTC)
    fri = h1.index[h1.index.dayofweek == 4]
    assert (fri.hour <= 20).all()
    # no HTF bar is available before all its base bars are
    for t, row in h1.iloc[:50].iterrows():
        base = m15.loc[(m15.index >= t) & (m15.index < t + pd.Timedelta(hours=1))]
        assert row["available_at"] >= base["available_at"].max()
        assert row["open"] == base["open"].iloc[0] and row["close"] == base["close"].iloc[-1]
        assert row["high"] == base["high"].max() and row["low"] == base["low"].min()


def test_early_friday_close_bucket_is_emitted_at_bucket_end():
    # H1 bars Friday 16:00..18:00 (early close), then Sunday 23:00 onwards.
    times = ["2024-01-12 16:00", "2024-01-12 17:00", "2024-01-12 18:00",
             "2024-01-14 23:00", "2024-01-15 00:00", "2024-01-15 01:00", "2024-01-15 02:00", "2024-01-15 03:00"]
    h1 = _bars(times, "H1")
    h4 = resample_bars(h1, "H4")
    assert pd.Timestamp("2024-01-12 16:00", tz="UTC") in h4.index
    row = h4.loc[pd.Timestamp("2024-01-12 16:00", tz="UTC")]
    # bucket 16-20 has only 3 of 4 hours but later data exists → complete; available at 20:00
    assert row["available_at"] == pd.Timestamp("2024-01-12 20:00", tz="UTC")
    assert row["close"] == h1["close"].iloc[2]
    # the Sunday 20:00-24:00 bucket (23:00 only) is complete because Monday data follows
    assert pd.Timestamp("2024-01-14 20:00", tz="UTC") in h4.index
    # Monday 00:00-04:00 is complete (last base bar ends exactly at 04:00)
    assert h4.index[-1] == pd.Timestamp("2024-01-15 00:00", tz="UTC")


def test_trailing_incomplete_bucket_is_dropped():
    times = pd.date_range("2024-01-15 00:00", periods=6, freq="h", tz="UTC")  # 00..05
    h4 = resample_bars(_bars(times, "H1"), "H4")
    assert list(h4.index) == [pd.Timestamp("2024-01-15 00:00", tz="UTC")]
    # but once the bucket is filled it appears
    h4b = resample_bars(_bars(pd.date_range("2024-01-15", periods=8, freq="h", tz="UTC"), "H1"), "H4")
    assert h4b.index[-1] == pd.Timestamp("2024-01-15 04:00", tz="UTC")


@pytest.mark.parametrize("anchor", [0, 21, 22])
def test_d1_anchor_hour(anchor: int):
    h1 = make_synthetic_bars(24 * 20, "H1", seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # pandas must not silently ignore the offset
        d1 = resample_bars(h1, "D1", daily_anchor_hour_utc=anchor)
    validate_bars(d1)
    assert (d1.index.hour == anchor).all()
    assert (d1["available_at"] - d1.index >= pd.Timedelta(hours=24)).all()
    # every D1 bar aggregates exactly the H1 bars in its [anchor, anchor+24h) bucket
    t = d1.index[3]
    base = h1.loc[(h1.index >= t) & (h1.index < t + pd.Timedelta(hours=24))]
    assert d1.loc[t, "open"] == base["open"].iloc[0]
    assert d1.loc[t, "close"] == base["close"].iloc[-1]
    assert d1.loc[t, "volume"] == pytest.approx(base["volume"].sum())


def test_d1_anchor_22_has_no_sunday_stub():
    # Synthetic market: open Sun 22:00 → Fri 21:00. A 22:00 anchor gives 5 bars per week.
    h1 = make_synthetic_bars(24 * 30, "H1", seed=2)
    d_utc = resample_bars(h1, "D1", daily_anchor_hour_utc=0)
    d_ny = resample_bars(h1, "D1", daily_anchor_hour_utc=22)
    assert (d_utc.index.dayofweek == 6).any()      # UTC days: a short Sunday bar exists
    assert not (d_ny.index.dayofweek == 5).any()   # NY-close days: no Saturday-labelled bar
    # the bucket starting Sunday 22:00 contains Monday's session
    assert (d_ny.index.dayofweek == 6).sum() >= 3


def test_h4_anchor_shifts_buckets():
    h1 = make_synthetic_bars(24 * 10, "H1", seed=4)
    h4 = resample_bars(h1, "H4", daily_anchor_hour_utc=22)
    assert set(h4.index.hour) <= {2, 6, 10, 14, 18, 22}


def test_resample_empty_and_bad_anchor():
    h1 = make_synthetic_bars(10, "H1")
    empty = h1.iloc[:0]
    out = resample_bars(empty, "H4")
    assert len(out) == 0 and out.attrs["timeframe"] == "H4"
    with pytest.raises(ValueError):
        resample_bars(h1, "D1", daily_anchor_hour_utc=24)


def test_align_htf_is_point_in_time():
    m15 = make_synthetic_bars(800, "M15", seed=5)
    h1 = resample_bars(m15, "H1")
    aligned = align_htf(m15, h1, columns=["close"])
    # at each M15 decision time, the aligned H1 close must be from an H1 bar already complete
    avail = pd.Series(h1["available_at"].to_numpy(), index=h1["close"].to_numpy())
    for i in range(0, len(m15), 37):
        v = aligned["close"].iloc[i]
        if np.isnan(v):
            continue
        assert avail.loc[v] <= m15["available_at"].iloc[i]
    # the first three M15 bars of every hour cannot see the H1 bar of that same hour
    t = m15.index[m15.index.minute == 30][5]
    hour_start = t.floor("h")
    same_hour_close = h1.loc[hour_start, "close"] if hour_start in h1.index else None
    assert same_hour_close is None or aligned.loc[t, "close"] != same_hour_close


def test_complete_until_keeps_last_friday_bucket_without_early_availability():
    times = pd.date_range("2024-01-12 16:00", periods=5, freq="h", tz="UTC")  # Fri 16:00..20:00
    h1 = _bars(times, "H1")
    assert pd.Timestamp("2024-01-12 20:00", tz="UTC") not in resample_bars(h1, "H4").index
    h4 = resample_bars(h1, "H4", complete_until="2024-01-13")
    row = h4.loc[pd.Timestamp("2024-01-12 20:00", tz="UTC")]
    assert row["available_at"] == pd.Timestamp("2024-01-13 00:00", tz="UTC")  # still the bucket end
    d1 = resample_bars(h1, "D1", complete_until=pd.Timestamp("2024-01-13", tz="UTC"))
    assert d1.index[-1] == pd.Timestamp("2024-01-12", tz="UTC")
    # a complete_until before the bucket end changes nothing
    assert len(resample_bars(h1, "D1", complete_until="2024-01-12 21:00")) == 0


def test_asof_join_empty_decision_series():
    bars = make_synthetic_bars(50, "H1", seed=1)
    right = pd.DataFrame({"x": [1.0], "available_at": pd.to_datetime(["2020-01-06 05:00"], utc=True)})
    out = asof_join(bars["available_at"].iloc[:0], right)
    assert out.shape == (0, 1)
    out2 = asof_join(pd.Series([], dtype=object), right)
    assert out2.shape == (0, 1)
    full = asof_join(bars["available_at"], right)
    assert full.index.equals(bars.index) and np.isnan(full["x"].iloc[0]) and full["x"].iloc[10] == 1.0


# ------------------------------------------------------------------------------------ review
@pytest.mark.parametrize("tf,anchor", [("H1", 0), ("H4", 0), ("H4", 22), ("D1", 0), ("D1", 21), ("D1", 22)])
def test_resample_is_invariant_to_truncating_the_future(tf: str, anchor: int):
    """Adversarial PIT check: every HTF bar already available at time T must be identical
    whether it is computed from the full history or from only the base bars known at T
    (what a live system has). Cuts include Friday closes, weekends and mid-bucket times."""
    m15 = make_synthetic_bars(4 * 24 * 30, "M15", seed=21)
    full = resample_bars(m15, tf, daily_anchor_hour_utc=anchor)
    rng = np.random.default_rng(0)
    fri_last = int(np.flatnonzero(m15.index.dayofweek == 4)[-1])
    cuts = sorted(set(rng.choice(len(m15) - 1, 12, replace=False).tolist()) | {fri_last, fri_last + 1})
    for cut in cuts:
        known_at = m15["available_at"].iloc[cut]
        part = resample_bars(m15.iloc[: cut + 1], tf, daily_anchor_hour_utc=anchor)
        a = full.loc[full["available_at"] <= known_at]
        b = part.loc[part["available_at"] <= known_at]
        pd.testing.assert_frame_equal(b, a, check_freq=False)
        # a live (truncated) run never emits a bar that is not yet available — i.e. it
        # never publishes an in-progress bucket
        assert (part["available_at"] <= known_at).all()


def test_align_htf_matches_live_recomputation():
    """Aligned HTF values at each base decision equal a live recomputation on base[:t+1]."""
    h1 = make_synthetic_bars(24 * 15, "H1", seed=8)
    aligned = align_htf(h1, resample_bars(h1, "H4", daily_anchor_hour_utc=22), columns=["close", "high"])
    for i in range(30, len(h1), 17):
        live = resample_bars(h1.iloc[: i + 1], "H4", daily_anchor_hour_utc=22)
        now = h1["available_at"].iloc[i]
        visible = live.loc[live["available_at"] <= now]
        if len(visible) == 0:
            assert aligned.iloc[i].isna().all()
        else:
            assert aligned["close"].iloc[i] == visible["close"].iloc[-1]
            assert aligned["high"].iloc[i] == visible["high"].iloc[-1]


def test_resample_to_finer_timeframe_raises():
    """Regression: H1 → M15 used to return the H1 bars relabelled 'M15' (available_at an
    hour after the open), silently corrupting any timeframe-based logic downstream."""
    h1 = make_synthetic_bars(50, "H1", seed=1)
    with pytest.raises(ValueError, match="finer"):
        resample_bars(h1, "M15")
    same = resample_bars(h1, "H1")  # identity is allowed
    pd.testing.assert_frame_equal(same, h1, check_freq=False)


def test_asof_join_mixed_datetime_units_and_non_utc_tz():
    """Keys of different resolutions / time zones must align on the instant, not the int."""
    right = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1])
    right["available_at"] = pd.DatetimeIndex(["2024-01-01 10:00", "2024-01-01 12:00"], tz="UTC").as_unit("s")
    t = pd.DatetimeIndex(["2024-01-01 06:59", "2024-01-01 07:00"], tz="America/New_York").as_unit("ns")
    out = asof_join(t, right)  # 06:59 / 07:00 New York = 11:59 / 12:00 UTC
    assert out["x"].tolist() == [1.0, 2.0]


def test_resample_single_bar_and_all_weekend_input():
    one = _bars(["2024-01-15 00:00"], "H1")
    assert len(resample_bars(one, "H4")) == 0  # the only bucket is still in progress
    assert len(resample_bars(one, "H4", complete_until="2024-01-15 04:00")) == 1
    assert resample_bars(one, "H1").index.equals(one.index)
