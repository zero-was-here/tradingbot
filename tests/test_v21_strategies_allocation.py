"""``managed_gold`` (aurum/strategies/allocation.py): bounds, components, de-risk logic, causality."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.loaders import bars_from_ohlc
from aurum.data.synthetic import make_synthetic_bars
from aurum.strategies.allocation import ManagedGold
from aurum.strategies.base import get_strategy, list_strategies

FAST = {"horizons": (24, 96, 240), "trend_halflife": 160.0, "vol_fast_halflife": 24.0,
        "vol_slow_halflife": 480.0, "vol_halflife": 48.0, "vol_min_periods": 24}


def _bars_from_close(close: np.ndarray, start: str = "2020-01-06") -> pd.DataFrame:
    """Hourly bars (weekdays only) with a given close path."""
    idx = pd.date_range(start, periods=int(len(close) * 1.5) + 100, freq="h", tz="UTC")
    idx = idx[idx.dayofweek < 5][: len(close)]
    close = np.asarray(close, dtype=float)
    opn = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": opn, "high": np.maximum(opn, close) * 1.0005,
                       "low": np.minimum(opn, close) * 0.9995, "close": close,
                       "volume": 100.0}, index=idx)
    return bars_from_ohlc(df, "H1", default_spread=0.3)


def _calm_then_crash(n_calm: int = 1500, n_crash: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r_calm = 0.0003 + 0.002 * rng.standard_normal(n_calm)            # calm uptrend
    r_crash = -0.002 + 0.008 * rng.standard_normal(n_crash)          # turbulent sell-off (4x vol)
    close = 1800.0 * np.exp(np.cumsum(np.r_[r_calm, r_crash]))
    return _bars_from_close(close)


@pytest.fixture(scope="module")
def regime_md() -> MarketData:
    return MarketData(bars=make_synthetic_bars(6000, "H1", seed=11, model="regime"))


def test_registered_with_description_and_defaults() -> None:
    assert "managed_gold" in list_strategies()
    s = get_strategy("managed_gold")
    assert isinstance(s, ManagedGold) and not s.trainable and s.description
    p = s.params
    assert p["base_weight"] == 0.5 and p["tilt_weight"] == 0.5 and p["floor"] == 0.0
    assert p["allow_short"] is False and p["horizons"] == (480, 1440, 2880, 5796)
    assert s.warmup_bars == 5797


@pytest.mark.parametrize("bad", [
    {"base_weight": 1.5}, {"base_weight": -0.2}, {"tilt_weight": -0.1}, {"floor": -0.5},
    {"vol_enter": 1.0, "vol_exit": 1.2}, {"vol_fast_halflife": 500.0, "vol_slow_halflife": 400.0},
    {"horizons": ()}, {"horizons": (0, 5)}, {"trend_halflife": 0.0}, {"vol_min_periods": 1},
    {"base_weight": float("nan")}, {"lenght": 3},
])
def test_rejects_invalid_parameters(bad: dict) -> None:
    with pytest.raises(ValueError):
        ManagedGold(**bad)


def test_short_side_parameters_allowed_only_with_allow_short() -> None:
    ManagedGold(allow_short=True, base_weight=-0.2, floor=-0.5)
    with pytest.raises(ValueError):
        ManagedGold(base_weight=-0.2)


def test_long_only_bounds_and_warmup(regime_md: MarketData) -> None:
    s = ManagedGold(**FAST)
    f = s.generate(regime_md)
    assert f.index.equals(regime_md.bars.index) and f.name == "managed_gold"
    assert np.isfinite(f).all() and f.min() >= 0.0 and f.max() <= 1.0
    assert (f.iloc[: s.warmup_bars] == 0.0).all()
    after = f.iloc[s.warmup_bars:]
    assert after.mean() > 0.2, "long-biased allocation should hold a sizeable average long"
    assert after.nunique() > 50, "the trend tilt should make the forecast continuous"


def test_allow_short_can_go_negative(regime_md: MarketData) -> None:
    s = ManagedGold(**FAST, allow_short=True, base_weight=0.0, tilt_weight=1.0, floor=-1.0)
    f = s.generate(regime_md).iloc[s.warmup_bars:]
    assert f.min() < -0.2 and f.max() > 0.2 and f.between(-1.0, 1.0).all()


def test_components_identity(regime_md: MarketData) -> None:
    s = ManagedGold(**FAST)
    c = s.components(regime_md)
    assert list(c.columns) == ["trend", "tilt", "ema", "below_ema", "vol_ratio", "derisk", "raw", "forecast"]
    w = s.warmup_bars
    body = c.iloc[w:]
    expected_raw = np.where(body["derisk"] > 0, np.minimum(0.5 + body["tilt"], 0.0), 0.5 + body["tilt"])
    np.testing.assert_allclose(body["raw"], expected_raw)
    np.testing.assert_allclose(body["forecast"], np.clip(body["raw"], 0.0, 1.0))
    np.testing.assert_allclose(body["tilt"], 0.5 * body["trend"].fillna(0.0))
    assert set(np.unique(c["derisk"])) <= {0.0, 1.0}
    assert body["trend"].abs().max() <= 1.0
    # vol ratio is scale-free and ~1 on average
    assert 0.7 < body["vol_ratio"].median() < 1.3


def test_trend_tilt_follows_the_trend() -> None:
    up = MarketData(bars=make_synthetic_bars(4000, "H1", seed=3, model="gbm", drift=0.0003, annual_vol=0.10))
    down = MarketData(bars=make_synthetic_bars(4000, "H1", seed=3, model="gbm", drift=-0.0003, annual_vol=0.10))
    s = ManagedGold(**FAST, vol_enter=10.0, vol_exit=9.0)      # de-risk effectively off
    cu, cd = s.components(up).iloc[s.warmup_bars:], s.components(down).iloc[s.warmup_bars:]
    assert cu["trend"].mean() > 0.3 and cd["trend"].mean() < -0.3
    assert cu["forecast"].mean() > 0.6 > 0.4 > cd["forecast"].mean()


def test_zero_tilt_gives_the_base_weight_outside_derisk(regime_md: MarketData) -> None:
    s = ManagedGold(**FAST, tilt_weight=0.0, base_weight=0.4, floor=0.1)
    c = s.components(regime_md).iloc[s.warmup_bars:]
    calm = c["derisk"] == 0
    assert calm.any() and (~calm).any()
    np.testing.assert_allclose(c.loc[calm, "forecast"], 0.4)
    np.testing.assert_allclose(c.loc[~calm, "forecast"], 0.1)


def test_derisk_fires_in_a_turbulent_selloff_and_cuts_to_floor() -> None:
    md = MarketData(bars=_calm_then_crash())
    s = ManagedGold(**FAST, floor=0.0)
    c = s.components(md)
    calm = c.iloc[s.warmup_bars:1500]
    crash = c.iloc[1600:]
    assert calm["derisk"].mean() < 0.05
    assert crash["derisk"].mean() > 0.5
    assert (crash.loc[crash["derisk"] > 0, "forecast"] == 0.0).all()
    assert calm["forecast"].mean() > 0.6           # calm uptrend: above the 0.5 base


def test_derisk_state_has_hysteresis() -> None:
    """Between vol_exit and vol_enter the latched state persists (no flicker)."""
    md = MarketData(bars=_calm_then_crash(seed=4))
    s = ManagedGold(**FAST)
    c = s.components(md).iloc[s.warmup_bars:]
    lc = np.log(md.bars["close"]).iloc[s.warmup_bars:]
    below = (lc < c["ema"]).to_numpy()
    ratio = c["vol_ratio"].to_numpy()
    state = c["derisk"].to_numpy()
    for t in range(1, len(state)):
        if below[t] and ratio[t] >= 1.25:
            assert state[t] == 1.0
        elif (not below[t]) or ratio[t] < 1.0:
            assert state[t] == 0.0
        else:   # in the band: keeps yesterday's state
            assert state[t] == state[t - 1]


def test_causal_under_truncation(regime_md: MarketData) -> None:
    s = ManagedGold(**FAST)
    full = s.generate(regime_md).to_numpy()
    for t in (900, 2500, 4100, 5999):
        part = s.generate(MarketData(bars=regime_md.bars.iloc[: t + 1])).to_numpy()
        np.testing.assert_array_equal(full[: t + 1], part)


def test_deterministic_clone_and_pickle(regime_md: MarketData) -> None:
    s = ManagedGold(**FAST)
    f = s.generate(regime_md).to_numpy()
    for other in (s.clone(), copy.deepcopy(s), pickle.loads(pickle.dumps(s))):
        np.testing.assert_array_equal(f, other.generate(regime_md).to_numpy())


def test_daily_configuration_runs_on_daily_bars() -> None:
    bars = make_synthetic_bars(1500, "D1", seed=2, model="regime")
    s = ManagedGold(horizons=(21, 63, 126, 252), trend_halflife=69, vol_fast_halflife=5,
                    vol_slow_halflife=126, vol_halflife=30, vol_min_periods=20)
    f = s.generate(MarketData(bars=bars))
    assert s.warmup_bars == 253
    assert (f.iloc[:253] == 0).all() and f.iloc[253:].between(0, 1).all() and f.iloc[253:].mean() > 0.1
