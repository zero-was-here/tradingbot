"""``mtf`` (higher-timeframe alignment) and ``regime`` groups."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.features.multi_timeframe import htf_compact_features, mtf_features, resample_anchored
from aurum.features.regime import efficiency_ratio, regime_features, variance_ratio
from aurum.features.volatility import atr


@pytest.fixture(scope="module")
def h1() -> pd.DataFrame:
    return make_synthetic_bars(3000, "H1", seed=31)


def test_resample_anchored_matches_manual_grouping(h1: pd.DataFrame) -> None:
    for tf, anchor, floor in (("D1", 22, "1D"), ("H4", 22, "4h"), ("D1", 0, "1D")):
        out = resample_anchored(h1, tf, anchor)
        delta = pd.Timedelta(hours=(24 - anchor) % 24)
        key = (h1.index + delta).floor(floor) - delta
        ref = h1.groupby(key).agg(open=("open", "first"), high=("high", "max"),
                                  low=("low", "min"), close=("close", "last"))
        common = out.index.intersection(ref.index)
        assert len(common) >= len(out) - 1
        pd.testing.assert_frame_equal(out.loc[common, ["open", "high", "low", "close"]],
                                      ref.loc[common], check_names=False, check_freq=False)
        assert (out["available_at"] >= out.index + pd.Timedelta(minutes=240 if tf == "H4" else 1440)).all()


def test_previous_day_levels(h1: pd.DataFrame) -> None:
    out = mtf_features(MarketData(bars=h1))
    a = atr(h1, 14)
    # Wednesday 12:00 bar -> previous broker day = Mon 22:00 .. Tue 22:00 UTC
    i = int(np.flatnonzero((h1.index.dayofweek == 2) & (h1.index.hour == 12))[3])
    t = h1.index[i]
    two = pd.Timedelta(hours=2)
    today_start = (t + two).floor("1D") - two            # Tue 22:00
    prev_start = today_start - pd.Timedelta(days=1)      # Mon 22:00
    prev = h1.loc[(h1.index >= prev_start) & (h1.index < today_start)]
    assert len(prev) == 24
    c = h1["close"].iloc[i]
    assert out["mtf_pdh_dist"].iloc[i] == pytest.approx((c - prev["high"].max()) / a.iloc[i])
    assert out["mtf_pdl_dist"].iloc[i] == pytest.approx((c - prev["low"].min()) / a.iloc[i])
    assert out["mtf_pdc_dist"].iloc[i] == pytest.approx((c - prev["close"].iloc[-1]) / a.iloc[i])


def test_htf_values_only_after_bucket_close(h1: pd.DataFrame) -> None:
    out = mtf_features(MarketData(bars=h1), htfs=("H4",), prev_day=False)
    h4 = resample_anchored(h1, "H4", 22)
    feats = htf_compact_features(h4)
    avail = pd.DatetimeIndex(h1["available_at"])
    j = 200
    t_close = h4["available_at"].iloc[j]
    i_at = int(np.flatnonzero(avail == t_close)[0])
    assert out["mtf_h4_ret_z_1"].iloc[i_at] == pytest.approx(feats["ret_z_1"].iloc[j])
    # one base bar earlier the H4 bar is not complete -> previous H4 value
    assert out["mtf_h4_ret_z_1"].iloc[i_at - 1] == pytest.approx(feats["ret_z_1"].iloc[j - 1])


def test_mtf_skips_non_higher_timeframes() -> None:
    h4 = make_synthetic_bars(600, "H4", seed=2)
    out = mtf_features(MarketData(bars=h4))
    assert not any(c.startswith("mtf_h4_") for c in out.columns)
    assert any(c.startswith("mtf_d1_") for c in out.columns)


def test_efficiency_ratio_extremes() -> None:
    line = pd.Series(np.linspace(100, 110, 50))
    assert efficiency_ratio(line, 20).iloc[-1] == pytest.approx(1.0)
    zigzag = pd.Series(100 + np.tile([0.0, 1.0], 25))
    assert efficiency_ratio(zigzag, 20).iloc[-1] == pytest.approx(0.0)
    assert efficiency_ratio(pd.Series(np.full(30, 5.0)), 10).iloc[-1] == 0.0


def test_variance_ratio_orders_processes() -> None:
    def mean_vr(model: str) -> float:
        b = make_synthetic_bars(6000, "H1", seed=4, model=model, weekend_gaps=False)
        vr = variance_ratio(np.log(b["close"]), 16, 240)
        return float(vr.dropna().mean())

    gbm, trend, mr = mean_vr("gbm"), mean_vr("trend"), mean_vr("mean_revert")
    assert 0.85 < gbm < 1.1
    assert trend > gbm + 0.05
    assert mr < gbm - 0.05


def test_regime_group_values(h1: pd.DataFrame) -> None:
    out = regime_features(MarketData(bars=h1))
    rank = out["regime_vol_pctrank_exp"]
    assert rank.dropna().between(0, 1).all()
    # expanding rank at t equals the empirical CDF of past vols (incl. t)
    rv = np.log(h1["close"]).diff().rolling(24).std()
    t = 1700
    past = rv.iloc[: t + 1].dropna()
    assert rank.iloc[t] == pytest.approx((past < past.iloc[-1]).mean() + 0.5 * (past == past.iloc[-1]).mean()
                                         + 0.5 / len(past), abs=1e-12)
    hurst = out["regime_hurst_16"]
    np.testing.assert_allclose(hurst.dropna(), 0.5 * (1 + np.log(out["regime_vr_16"].dropna()) / math.log(16)))
    assert set(out["regime_trend_state"].dropna().unique()) <= {-1.0, 0.0, 1.0}
    assert set(out["regime_high_vol"].dropna().unique()) <= {0.0, 1.0}


# ---- reviewer: adversarial cases -------------------------------------------------------------
def test_mtf_on_empty_and_tiny_bars(h1: pd.DataFrame) -> None:
    full = mtf_features(MarketData(bars=h1))
    for n in (0, 1, 3):
        out = mtf_features(MarketData(bars=h1.iloc[:n]))
        assert out.index.equals(h1.index[:n])
        assert list(out.columns) == list(full.columns)
        assert out.isna().all().all()


# ---- independent review ---------------------------------------------------------------------------
def test_resample_anchored_equals_shared_resample_bars(h1: pd.DataFrame) -> None:
    """The wrapper must agree with the (now anchor-aware) shared ``resample_bars``."""
    from aurum.data.resample import resample_bars

    for tf in ("H4", "D1"):
        for anchor in (0, 21, 22):
            pd.testing.assert_frame_equal(resample_anchored(h1, tf, anchor),
                                          resample_bars(h1, tf, daily_anchor_hour_utc=anchor))


def test_mtf_lookback_scales_with_bar_size() -> None:
    from aurum.features.multi_timeframe import MTF_LOOKBACK, mtf_lookback

    assert mtf_lookback({}, 60.0) == MTF_LOOKBACK == 24 * 24
    assert mtf_lookback({}, 30.0) == 2 * MTF_LOOKBACK
    assert mtf_lookback({}, 1440.0) == 0  # D1 base: no slower HTF, no previous-day levels
    assert mtf_lookback({}, 240.0) == 24 * 6  # H4 base: D1 context only
    assert mtf_lookback({"ema_span": 50}, 60.0) > MTF_LOOKBACK
