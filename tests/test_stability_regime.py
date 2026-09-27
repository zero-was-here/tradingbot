"""``regime`` feature group: bounded (sliding-window-stable) vol rank and live parity.

The live runner computes features on a sliding window of ``history_multiple * max_lookback``
closed bars (3x by default) while research computes them once on the full history. Every
column of the DEFAULT ``regime`` output must therefore agree between the two after the
group's warm-up — the legacy expanding vol rank never did (it depends on where the history
starts), so it is opt-in (``expanding=True``) and used below as the negative control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.features.pipeline import FeaturePipeline
from aurum.features.regime import (
    DEFAULT_RANK_YEARS,
    REGIME_LOOKBACK,
    rank_window_bars,
    regime_features,
    regime_lookback,
)

#: ``aurum.live.runner.RunnerConfig`` default ``history_multiple``.
LIVE_HISTORY_MULTIPLE = 3


def _worst_start(bars: pd.DataFrame, n_window: int) -> int:
    """Start of a window of at least ``n_window`` trailing bars whose FIRST return is the
    largest one-bar move among nearby candidates (the worst case for any estimator seeded
    with the first observations of its history)."""
    hi = len(bars) - n_window
    lo = max(0, hi - 400)
    r = np.abs(np.diff(np.log(bars["close"].to_numpy())))
    return lo + int(np.argmax(r[lo:hi + 1]))


# ---------------------------------------------------------------------------------------
# warm-up / window sizes
# ---------------------------------------------------------------------------------------
def test_rank_window_is_timeframe_aware() -> None:
    assert rank_window_bars(1.0, 60) == 5796          # 252 days x 23 trading hours
    assert rank_window_bars(2.0, 60) == 2 * 5796
    assert rank_window_bars(1.0, 15) == 4 * 5796
    assert rank_window_bars(1.0, 240) == 1449
    assert rank_window_bars(1.0, 1440) == 252
    with pytest.raises(ValueError):
        rank_window_bars(0.0, 60)
    # lookback = vol_window + W (window FULL), registry value is the H1 default
    assert regime_lookback({}, 60.0) == REGIME_LOOKBACK == 24 + 5796
    assert regime_lookback({"rank_years": 2.0}, 60.0) == 24 + 2 * 5796
    assert regime_lookback({"rank_window": 700}, 60.0) == 24 + 700
    assert regime_lookback({}, 1440.0) == 24 + 252
    assert DEFAULT_RANK_YEARS == 1.0


@pytest.mark.parametrize(("tf", "minutes"), [("H1", 60.0), ("H4", 240.0), ("D1", 1440.0)])
def test_lookback_is_exact_first_valid_row(tf: str, minutes: float) -> None:
    """Defined == stable: the rank is NaN until its window is full, and the group's warm-up
    is exactly the first row where every default column is defined."""
    lb = regime_lookback({}, minutes)
    bars = make_synthetic_bars(lb + 300, tf, seed=5, model="regime")
    out = regime_features(MarketData(bars=bars))
    first = out.notna().to_numpy().argmax(axis=0)
    assert int(first.max()) == lb - 1
    assert out.iloc[lb - 1:].notna().all().all()
    pipe = FeaturePipeline(groups=["regime"])
    pipe.compute(MarketData(bars=bars.iloc[:50]))      # records the bar size
    assert pipe.max_lookback == lb


def test_expanding_rank_is_opt_in() -> None:
    bars = make_synthetic_bars(800, "H1", seed=1)
    default = regime_features(MarketData(bars=bars))
    assert "regime_vol_pctrank_exp" not in default.columns
    assert "regime_vol_pctrank" in default.columns
    legacy = regime_features(MarketData(bars=bars), expanding=True)
    assert "regime_vol_pctrank_exp" in legacy.columns
    assert list(legacy.columns.drop("regime_vol_pctrank_exp")) == list(default.columns)
    assert regime_lookback({"expanding": True, "rank_window": 100}, 60.0) == 24 + 240


# ---------------------------------------------------------------------------------------
# live (sliding window of 3 x lookback) vs research (full history) parity
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(("tf", "minutes", "extra"), [
    ("H1", 60.0, 4000), ("H4", 240.0, 2500), ("D1", 1440.0, 900), ("M15", 15.0, 3000),
])
def test_sliding_window_parity_every_default_column(tf: str, minutes: float, extra: int) -> None:
    """Features computed on the last ``3 x lookback`` bars (what the live runner fetches)
    equal the full-history features on every row past the warm-up, for EVERY column of the
    default ``regime`` output (ranks and flags exactly; the rest to 1e-9)."""
    lb = regime_lookback({}, minutes)
    n_live = LIVE_HISTORY_MULTIPLE * lb
    bars = make_synthetic_bars(n_live + extra, tf, seed=17, model="jump")
    full = regime_features(MarketData(bars=bars))
    a = _worst_start(bars, n_live)
    live = regime_features(MarketData(bars=bars.iloc[a:]))
    assert list(live.columns) == list(full.columns)
    ref = full.iloc[a:]
    past = live.iloc[lb - 1:]
    assert past.notna().all().all(), "live window still warming up after the lookback"
    for c in live.columns:
        np.testing.assert_allclose(past[c].to_numpy(), ref[c].iloc[lb - 1:].to_numpy(),
                                   rtol=0.0, atol=1e-9, err_msg=f"{tf}: {c}")
    for c in ("regime_vol_pctrank", "regime_high_vol", "regime_low_vol", "regime_trend_state"):
        assert np.array_equal(past[c].to_numpy(), ref[c].iloc[lb - 1:].to_numpy()), (tf, c)
    # the live runner's decision row (the last bar) in particular (rolling sums restarted
    # at a different row differ only by floating-point rounding)
    np.testing.assert_allclose(live.iloc[-1].to_numpy(), full.iloc[-1].to_numpy(), rtol=1e-12, atol=1e-12)


def test_expanding_rank_breaks_parity_negative_control() -> None:
    """Why the expanding rank is excluded by default: on the same live window it keeps
    disagreeing with the full-history value long after any warm-up."""
    lb = regime_lookback({"rank_window": 500, "expanding": True}, 60.0)
    bars = make_synthetic_bars(12_000, "H1", seed=3, model="regime")
    full = regime_features(MarketData(bars=bars), rank_window=500, expanding=True)
    a = len(bars) - LIVE_HISTORY_MULTIPLE * max(lb, 1000)
    live = regime_features(MarketData(bars=bars.iloc[a:]), rank_window=500, expanding=True)
    d_exp = np.abs(live["regime_vol_pctrank_exp"].iloc[lb:].to_numpy()
                   - full["regime_vol_pctrank_exp"].iloc[a + lb:].to_numpy())
    assert np.nanmax(d_exp) > 0.05 and np.nanmean(d_exp > 1e-6) > 0.5
    d_bounded = np.abs(live["regime_vol_pctrank"].iloc[lb:].to_numpy()
                       - full["regime_vol_pctrank"].iloc[a + lb:].to_numpy())
    assert np.nanmax(d_bounded) == 0.0


def test_pipeline_live_history_parity() -> None:
    """End to end through FeaturePipeline with the runner's history rule
    ``max(3 * max_lookback, 300)``: the transformed decision row is identical."""
    bars = make_synthetic_bars(22_000, "H1", seed=9, model="regime")
    pipe = FeaturePipeline(groups=["regime"])
    raw = pipe.compute(MarketData(bars=bars))
    pipe.fit(raw.iloc[: len(bars) // 2])
    n_hist = max(LIVE_HISTORY_MULTIPLE * pipe.max_lookback, 300)
    live_raw = pipe.compute(MarketData(bars=bars.iloc[-n_hist:]))
    rep = pipe.parity_report(raw.iloc[-1000:], live_raw.iloc[-1000:], atol=1e-9)
    assert bool(rep["ok"].all()), rep[~rep["ok"]]
    x_full = pipe.transform(raw).iloc[-1]
    x_live = pipe.transform(live_raw).iloc[-1]
    pd.testing.assert_series_equal(x_full, x_live, check_exact=False, atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------------------
# point-in-time with the DEFAULT window (the generic leakage harness runs a short window)
# ---------------------------------------------------------------------------------------
def test_default_window_is_point_in_time() -> None:
    bars = make_synthetic_bars(REGIME_LOOKBACK + 1500, "H1", seed=21, model="regime")
    md = MarketData(bars=bars)
    full = regime_features(md)
    for i, t in enumerate((REGIME_LOOKBACK - 1, REGIME_LOOKBACK + 400, REGIME_LOOKBACK + 1200)):
        alt = make_synthetic_bars(len(bars), "H1", seed=500 + i, model="jump",
                                  start_price=float(bars["close"].iloc[t]) * 1.3, annual_vol=0.5)
        new = pd.concat([bars.iloc[: t + 1], alt.iloc[t + 1:]])
        new.attrs["timeframe"] = "H1"
        pert = regime_features(MarketData(bars=new))
        trunc = regime_features(MarketData(bars=bars.iloc[: t + 1]))
        for other in (pert, trunc):
            x = full.iloc[: t + 1].to_numpy()
            y = other.iloc[: t + 1].to_numpy()
            assert ((x == y) | (np.isnan(x) & np.isnan(y))).all(), t
        assert not np.allclose(full["regime_vol_pctrank"].iloc[t + 1:].to_numpy(),
                               pert["regime_vol_pctrank"].iloc[t + 1:].to_numpy(), equal_nan=True)
