"""Technical feature groups: indicator correctness on hand-checkable inputs, bounds, and
structural contract (index, prefixes, float dtype, no inf)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.features.base import get_feature
from aurum.features.technical import (
    adx,
    ema,
    rolling_linreg,
    rsi,
    stochastic,
    streak,
    tsmom_response,
)

GROUPS = ("returns", "trend", "momentum", "meanrev", "range")


def _bars_from_close(close: np.ndarray, spread: float = 0.2) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="h", tz="UTC")
    o = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, close) + 0.5,
                       "low": np.minimum(o, close) - 0.5, "close": close}, index=idx)
    return make_bars(df, "H1", default_spread=spread)


@pytest.fixture(scope="module")
def md() -> MarketData:
    return MarketData(bars=make_synthetic_bars(2000, "H1", seed=5, model="trend"))


@pytest.mark.parametrize("name", GROUPS)
def test_group_contract(name: str, md: MarketData) -> None:
    out = get_feature(name).compute(md)
    assert out.index.equals(md.bars.index)
    assert all(out.dtypes == np.float64)
    vals = out.to_numpy()
    assert not np.isinf(vals).any()
    assert all(c.startswith(f"{name}_") for c in out.columns)
    lb = get_feature(name).lookback
    tail = out.iloc[lb + 5:]
    assert tail.notna().all().all(), tail.columns[tail.isna().any()].tolist()


def test_rsi_extremes_and_reference() -> None:
    up = pd.Series(np.linspace(100, 200, 60))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(up[::-1].reset_index(drop=True), 14).iloc[-1] == pytest.approx(0.0)
    assert rsi(pd.Series(np.full(30, 5.0)), 14).iloc[-1] == pytest.approx(50.0)
    x = pd.Series(np.random.default_rng(0).normal(0, 1, 300).cumsum() + 100)
    d = x.diff().to_numpy()
    au = ad = np.nan
    ref = np.full(len(x), np.nan)
    for i in range(1, len(x)):
        u, v = max(d[i], 0.0), max(-d[i], 0.0)
        au = u if np.isnan(au) else au + (u - au) / 14
        ad = v if np.isnan(ad) else ad + (v - ad) / 14
        if i >= 14:
            ref[i] = 100 * au / (au + ad)
    np.testing.assert_allclose(rsi(x, 14).to_numpy()[14:], ref[14:], rtol=1e-10)


def test_stochastic_and_adx_bounds(md: MarketData) -> None:
    k, d = stochastic(md.bars, 14, 3)
    assert k.dropna().between(0, 100).all() and d.dropna().between(0, 100).all()
    a, p, m = adx(md.bars, 14)
    for s in (a, p, m):
        assert s.dropna().between(0, 100).all()


def test_rolling_linreg_exact_line() -> None:
    y = pd.Series(3.0 + 0.01 * np.arange(200))
    slope, r2 = rolling_linreg(y, 50)
    assert slope.iloc[:49].isna().all()
    np.testing.assert_allclose(slope.iloc[49:], 0.01, rtol=1e-9)
    np.testing.assert_allclose(r2.iloc[49:], 1.0, atol=1e-9)
    noise = pd.Series(np.random.default_rng(1).normal(size=500))
    s2, r22 = rolling_linreg(noise, 20)
    ref = np.polyfit(np.arange(20), noise.iloc[-20:].to_numpy(), 1)[0]
    assert s2.iloc[-1] == pytest.approx(ref, rel=1e-9)
    assert r22.dropna().between(0, 1).all()


def test_tsmom_response_shape() -> None:
    z = np.linspace(-6, 6, 1201)
    phi = tsmom_response(z)
    assert phi.max() == pytest.approx(math.sqrt(2) * math.exp(-0.5) / 0.89, rel=1e-4)
    assert z[np.argmax(phi)] == pytest.approx(math.sqrt(2), abs=0.01)
    np.testing.assert_allclose(phi, -tsmom_response(-z))


def test_streak_counts() -> None:
    r = pd.Series([np.nan, 1, 2, -1, -3, -2, 0, 5])
    np.testing.assert_array_equal(streak(r).to_numpy(), [np.nan, 1, 2, -1, -2, -3, 0, 1])


def test_ema_matches_recursion() -> None:
    x = pd.Series(np.arange(30, dtype=float))
    e = ema(x, 10)
    alpha, ref = 2 / 11, [0.0]
    for v in x.iloc[1:]:
        ref.append(ref[-1] + alpha * (v - ref[-1]))
    np.testing.assert_allclose(e.iloc[9:], np.array(ref)[9:])
    assert e.iloc[:9].isna().all()


def test_returns_z_definition(md: MarketData) -> None:
    out = get_feature("returns").compute(md)
    lc = np.log(md.bars["close"])
    r = lc.diff()
    sig = np.sqrt((r * r).ewm(halflife=48.0, adjust=False, min_periods=24).mean())
    t = 1500
    expect = (lc.iloc[t] - lc.iloc[t - 12]) / (sig.iloc[t] * math.sqrt(12))
    assert out["returns_z_12"].iloc[t] == pytest.approx(expect, rel=1e-12)


def test_trend_on_rising_prices_is_positive() -> None:
    bars = _bars_from_close(1800 * np.exp(0.0005 * np.arange(600)))  # > 480-bar horizon
    out = get_feature("trend").compute(MarketData(bars=bars))
    last = out.iloc[-1]
    assert last["trend_ema_dist_50"] > 0 and last["trend_ema_slope_50"] > 0
    assert last["trend_linreg_t_50"] > 10 and last["trend_linreg_r2_50"] > 0.99
    assert last["trend_di_diff_14"] > 0
    mom = get_feature("momentum").compute(MarketData(bars=bars)).iloc[-1]
    assert mom["momentum_rsi_14"] == pytest.approx(1.0)
    assert mom["momentum_tsmom_agg"] == pytest.approx(1.0)


def test_range_anatomy(md: MarketData) -> None:
    out = get_feature("range").compute(md)
    total = out["range_body"].abs() + out["range_upper_wick"] + out["range_lower_wick"]
    np.testing.assert_allclose(total, 1.0, atol=1e-9)
    for n in (20, 55):
        assert out[f"range_donchian_pos_{n}"].dropna().between(-0.5, 0.5).all()
    assert out["range_clv"].between(-1, 1).all()
    flags = out[["range_inside_bar", "range_outside_bar", "range_nr7"]].dropna()
    assert set(np.unique(flags.to_numpy())) <= {0.0, 1.0}


def test_meanrev_bollinger_and_z(md: MarketData) -> None:
    out = get_feature("meanrev").compute(md)
    c = md.bars["close"]
    t = 900
    win = c.iloc[t - 19:t + 1]
    pctb = (c.iloc[t] - win.mean()) / (4 * win.std(ddof=0))
    assert out["meanrev_bb_pctb_20"].iloc[t] == pytest.approx(pctb, rel=1e-9)
    z = (c.iloc[t] - c.iloc[t - 49:t + 1].mean()) / c.iloc[t - 49:t + 1].std(ddof=1)
    assert out["meanrev_z_50"].iloc[t] == pytest.approx(z, rel=1e-8)


def test_overrides_change_columns(md: MarketData) -> None:
    out = get_feature("returns").compute(md, horizons=(1, 5))
    assert list(out.columns) == ["returns_log_1", "returns_z_1", "returns_z_5"]
