"""Tests for aurum.data.synthetic (schema, determinism, point-in-time macro)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.data.pit import asof_join
from aurum.data.schema import validate_bars
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro


@pytest.mark.parametrize("model", ["gbm", "trend", "mean_revert", "regime", "jump"])
def test_models_produce_valid_bars(model: str):
    b = make_synthetic_bars(1500, "H1", model=model, seed=11)
    validate_bars(b)
    assert len(b) == 1500 and b.attrs["timeframe"] == "H1"
    assert not (b.index.dayofweek == 5).any()


def test_deterministic_by_seed():
    a = make_synthetic_bars(300, "M15", seed=5)
    b = make_synthetic_bars(300, "M15", seed=5)
    c = make_synthetic_bars(300, "M15", seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["close"], c["close"])


def test_synthetic_macro_is_point_in_time():
    """Perturbing gold AFTER time T must not change any macro value available at/before T."""
    bars = make_synthetic_bars(24 * 60, "H1", seed=9)
    base = make_synthetic_macro(bars, seed=9)
    # Cut inside the 21:30-24:00 UTC window: the day's macro print is already public but
    # the UTC day is not over — exactly where a midnight-close construction would leak.
    mid = bars.index[len(bars) // 2:]
    cut = mid[(mid.hour == 22) & (mid.dayofweek < 4)][0]
    future = bars.index > cut
    perturbed = bars.copy()
    k = np.exp(np.linspace(0.02, 0.3, int(future.sum())))
    for c in ("open", "high", "low", "close"):
        perturbed.loc[future, c] = perturbed.loc[future, c] * k
    after = make_synthetic_macro(perturbed, seed=9)
    decision = perturbed.loc[cut, "available_at"]
    for name, frame in base.items():
        known = frame["available_at"] <= decision
        assert known.sum() > 10
        pd.testing.assert_series_equal(frame.loc[known, "value"], after[name].loc[known, "value"])
    # and the perturbation does matter later on (the series really depend on gold)
    later = base["dxy"]["available_at"] > decision + pd.Timedelta(days=2)
    assert not np.allclose(base["dxy"].loc[later, "value"], after["dxy"].loc[later, "value"])


def test_synthetic_macro_correlates_with_gold():
    bars = make_synthetic_bars(24 * 400, "H1", seed=10)
    m = make_synthetic_macro(bars, seed=10)
    avail = m["dxy"]["available_at"]
    gold = asof_join(pd.DatetimeIndex(avail), bars[["close", "available_at"]], columns=["close"])["close"]
    rg = np.diff(np.log(gold.to_numpy(dtype=float)))
    rd = np.diff(np.log(m["dxy"]["value"].to_numpy()))
    ok = np.isfinite(rg) & np.isfinite(rd)
    assert np.corrcoef(rg[ok], rd[ok])[0, 1] < -0.2  # dollar negatively related to gold



# ------------------------------------------------------------------------------------ review
@pytest.mark.parametrize("timeframe,n", [("H1", 24 * 300), ("M15", 4 * 24 * 60), ("D1", 600)])
def test_weekend_open_gaps_are_larger_than_intraweek(timeframe: str, n: int):
    """Regression: the weekend flag compared ``idx.asi8`` (us on pandas 3) with a Timedelta
    in ns, so it never fired and Monday opens had the same tiny gap as any other bar."""
    b = make_synthetic_bars(n, timeframe, seed=1)
    gap = np.log(b["open"] / b["close"].shift(1)).to_numpy()[1:]
    span = b["available_at"].iloc[0] - b.index[0]
    weekend = np.asarray((b.index[1:] - b.index[:-1]) > 2 * span)
    assert weekend.sum() >= 5
    # 1.5 sigma vs 0.05 sigma by construction (~30x); demand a wide margin
    assert np.std(gap[weekend]) > 10 * np.std(gap[~weekend])


@pytest.mark.parametrize("timeframe,n", [("H1", 5), ("M1", 200), ("M15", 3), ("H4", 2)])
def test_short_series_starting_on_a_weekend(timeframe: str, n: int):
    """Regression: a short series starting on a Saturday raised 'timeline too short'."""
    b = make_synthetic_bars(n, timeframe, start="2020-01-04", seed=2)  # Saturday
    validate_bars(b)
    assert len(b) == n and b.index[0] >= pd.Timestamp("2020-01-05 22:00", tz="UTC")
    # the margin does not change the timeline of ordinary calls
    ref = make_synthetic_bars(300, timeframe, seed=2)
    assert ref.index[0] == pd.Timestamp("2020-01-06", tz="UTC")
