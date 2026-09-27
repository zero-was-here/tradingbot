"""Unit tests for the SPEC §6 rule-based strategies (trend, mean reversion, breakout, macro,
seasonal): registry metadata, parameter validation, forecast scaling, hand-built scenarios
for every rule, DST handling, point-in-time macro alignment and speed on 90k bars.

Generic look-ahead checks for every registered strategy live in
``tests/test_strategies_leakage.py``; random-walk / positive-control statistics in
``tests/test_strategies_rules_randomwalk.py``.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro
from aurum.features.volatility import atr
from aurum.strategies.base import get_strategy, list_strategies
from aurum.strategies.breakout import OpeningRangeBreakout, VolatilitySqueeze
from aurum.strategies.macro import MacroFactor, RiskOff, clean_macro
from aurum.strategies.mean_reversion import RSI2, BollingerRevert, ZScoreFade
from aurum.strategies.seasonal import IntradaySeasonality, bucket_keys
from aurum.strategies.trend import (
    EXPECTED_ABS_BAZ,
    MAX_FDM,
    DonchianBreakout,
    EMACrossover,
    KalmanTrend,
    TimeSeriesMomentum,
    diversification_multiplier,
    ema_spread_variance,
    latch,
    llt_steady_state,
    overlap_correlation,
)

RULES = ("tsmom", "ema_cross", "donchian", "kalman_trend", "zscore_fade", "rsi2",
         "bollinger_revert", "vol_squeeze", "orb", "macro_factor", "risk_off",
         "intraday_seasonality")
CONTINUOUS = ("tsmom", "ema_cross", "kalman_trend", "macro_factor")


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def bars_from_close(close, *, index: pd.DatetimeIndex | None = None, start: str = "2024-01-08",
                    freq: str = "1h", tf: str = "H1", wick: float = 0.05,
                    highs=None, lows=None) -> pd.DataFrame:
    """Canonical bars whose open is the previous close and whose wicks are ``wick``."""
    c = np.asarray(close, dtype=float)
    idx = index if index is not None else pd.date_range(start, periods=len(c), freq=freq, tz="UTC")
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + wick if highs is None else np.asarray(highs, dtype=float)
    lo = np.minimum(o, c) - wick if lows is None else np.asarray(lows, dtype=float)
    df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "spread": 0.3}, index=idx)
    return make_bars(df, tf)


def md_of(bars: pd.DataFrame, macro: dict | None = None) -> MarketData:
    return MarketData(bars=bars, macro=macro or {})


def macro_frame(values, *, start: str = "2020-01-01", lag_hours: float = 22.5,
                index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    idx = index if index is not None else pd.date_range(start, periods=len(values), freq="B", tz="UTC")
    f = pd.DataFrame({"value": np.asarray(values, dtype=float)}, index=idx)
    f.index.name = "date"
    f["available_at"] = f.index + pd.Timedelta(hours=lag_hours)
    return f


@pytest.fixture(scope="module")
def gbm_md() -> MarketData:
    bars = make_synthetic_bars(30_000, "H1", seed=21, model="gbm")
    return MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=21))


# ---------------------------------------------------------------------------------------
# registry, metadata, validation
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", RULES)
def test_registry_and_metadata(name: str) -> None:
    reg = list_strategies()
    assert name in reg
    cls = reg[name]
    assert cls.name == name
    assert len(cls.description) > 20
    doc = cls.__doc__ or ""
    assert "Rationale" in doc, f"{name}: class docstring must state the economic rationale"
    assert re.search(r"\((19|20)\d\d\)", doc), f"{name}: class docstring must cite a reference"
    strat = get_strategy(name)
    assert isinstance(strat.warmup_bars, int) and strat.warmup_bars >= 0
    assert strat.trainable == (name == "intraday_seasonality")
    with pytest.raises(ValueError, match="unknown parameter"):
        get_strategy(name, definitely_not_a_param=1)


@pytest.mark.parametrize("name, params", [
    ("tsmom", {"response": "cubic"}),
    ("tsmom", {"horizons": ()}),
    ("tsmom", {"weights": (1.0, 2.0)}),
    ("ema_cross", {"fast": 64, "slow": 32}),
    ("donchian", {"level": 1.5}),
    ("kalman_trend", {"obs_noise": 0.0}),
    ("zscore_fade", {"entry_z": 1.0, "exit_z": 1.5}),
    ("rsi2", {"lower": 95.0}),
    ("bollinger_revert", {"k": 3.0, "stop_k": 2.0}),
    ("vol_squeeze", {"hold": 0}),
    ("orb", {"sessions": ("tokyo",)}),
    ("orb", {"range_minutes": 600}),
    ("macro_factor", {"series": {}}),
    ("risk_off", {"entry_z": 0.5, "exit_z": 1.0}),
    ("intraday_seasonality", {"shrinkage": "lots"}),
    ("intraday_seasonality", {"significance": 2.0}),
])
def test_invalid_parameters_raise(name: str, params: dict) -> None:
    with pytest.raises(ValueError):
        get_strategy(name, **params)


def test_warmup_follows_parameters() -> None:
    assert TimeSeriesMomentum(horizons=(10, 50)).warmup_bars == 121  # vol_min_periods=120 dominates
    assert TimeSeriesMomentum(horizons=(10, 500)).warmup_bars == 501
    assert EMACrossover(fast=10, slow=100).warmup_bars == 301
    assert DonchianBreakout(entry_n=30, exit_n=10, atr_n=20).warmup_bars == 61
    assert KalmanTrend(lookback=100).warmup_bars == 301
    assert RSI2(trend_n=None).warmup_bars == 11
    assert OpeningRangeBreakout().warmup_bars == 0
    assert OpeningRangeBreakout(max_range_atr=3.0).warmup_bars == 25


# ---------------------------------------------------------------------------------------
# forecast-scaling machinery
# ---------------------------------------------------------------------------------------
def test_latch_matches_reference_loop() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    el, es, xl, xs = (rng.random(n) < p for p in (0.03, 0.03, 0.1, 0.1))
    ref = np.zeros(n)
    state = 0
    for t in range(n):
        if el[t] and es[t]:
            state = 0                     # contradictory entries cancel
        elif el[t]:
            state = 1
        elif es[t]:
            state = -1
        elif (state == 1 and xl[t]) or (state == -1 and xs[t]):
            state = 0
        ref[t] = state
    assert (el & es).any()
    assert np.array_equal(latch(el, es, xl, xs), ref)


def test_diversification_multiplier() -> None:
    assert diversification_multiplier(np.ones((3, 3)), [1 / 3] * 3) == pytest.approx(1.0)
    assert diversification_multiplier(np.eye(4), [0.25] * 4) == pytest.approx(2.0)
    assert diversification_multiplier(np.eye(16), [1 / 16] * 16) == MAX_FDM
    c = overlap_correlation((1, 4, 16))
    assert c[0, 1] == pytest.approx(0.5) and c[0, 2] == pytest.approx(0.25)


def test_ema_spread_variance_matches_simulation() -> None:
    rng = np.random.default_rng(1)
    y = pd.Series(np.cumsum(rng.standard_normal(400_000)))
    for fast, slow in ((4, 16), (32, 128)):
        d = y.ewm(span=fast, adjust=False).mean() - y.ewm(span=slow, adjust=False).mean()
        emp = float(d.iloc[5 * slow:].var())
        assert emp == pytest.approx(ema_spread_variance(fast, slow), rel=0.06)


def test_baz_expected_abs_constant() -> None:
    z = np.random.default_rng(2).standard_normal(2_000_000)
    phi = z * np.exp(-z * z / 4) / 0.89
    assert np.abs(phi).mean() == pytest.approx(EXPECTED_ABS_BAZ, rel=0.01)


@pytest.mark.parametrize("name", CONTINUOUS)
def test_continuous_forecasts_average_half_on_random_walk(name: str, gbm_md: MarketData) -> None:
    """Carver's convention: E|forecast| ~ 0.5 under the null (analytic scalars, no fitting)."""
    s = get_strategy(name)
    f = s.generate(gbm_md).to_numpy()[s.warmup_bars:]
    f = f[f != 0.0]
    assert len(f) > 10_000
    assert 0.38 < np.abs(f).mean() < 0.62, np.abs(f).mean()
    assert np.abs(f).max() <= 1.0


@pytest.mark.parametrize("response", ["linear", "sign", "baz"])
def test_tsmom_responses_and_direction(response: str) -> None:
    rng = np.random.default_rng(3)
    n = 2000
    up = 1800 * np.exp(np.cumsum(0.0006 + 0.001 * rng.standard_normal(n)))
    down = 1800 * np.exp(np.cumsum(-0.0006 + 0.001 * rng.standard_normal(n)))
    s = TimeSeriesMomentum(horizons=(24, 96, 384), response=response)
    fu = s.generate(md_of(bars_from_close(up))).to_numpy()[s.warmup_bars:]
    fd = s.generate(md_of(bars_from_close(down))).to_numpy()[s.warmup_bars:]
    if response == "baz":  # very stretched trends are faded by design
        assert np.mean(fu) > -0.5 and np.abs(fu).max() <= 1.0
    else:
        assert (fu > 0).mean() > 0.95 and (fd < 0).mean() > 0.95
    if response == "sign":
        w, fdm = s.combination()
        assert fdm > 1.0 and np.abs(fu).max() <= 1.0


def test_ema_cross_direction_and_zero_warmup() -> None:
    c = np.r_[np.full(500, 1800.0), 1800 + np.arange(1, 501) * 0.5]
    s = EMACrossover(fast=8, slow=32, vol_halflife=48, vol_min_periods=24)
    b = bars_from_close(c + 0.2 * np.sin(np.arange(1000)))
    f = s.generate(md_of(b))
    assert (f.iloc[: s.warmup_bars] == 0).all()
    assert (f.iloc[600:] > 0.5).all()


# ---------------------------------------------------------------------------------------
# kalman_trend
# ---------------------------------------------------------------------------------------
def _kalman_loop(y: np.ndarray, gain: np.ndarray) -> np.ndarray:
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    x = np.zeros(2)
    out = np.empty(len(y))
    for t, yt in enumerate(y):
        xp = f @ x
        x = xp + gain.ravel() * (yt - xp[0])
        out[t] = x[1]
    return out


def test_kalman_lfilter_equals_explicit_filter() -> None:
    s = KalmanTrend(lookback=60)
    rng = np.random.default_rng(4)
    close = 1800 * np.exp(np.cumsum(0.001 * rng.standard_normal(3000)))
    b = bars_from_close(close)
    k, _, _ = llt_steady_state(1.0, 1.0 / 60**2, 0.1)
    y = np.log(close)
    np.testing.assert_allclose(s.slope(b).to_numpy(), _kalman_loop(y - y[0], k), atol=1e-12)


def test_kalman_slope_is_unbiased_on_linear_trend_and_normalised() -> None:
    s = KalmanTrend(lookback=50)
    close = 1800 * np.exp(0.0004 * np.arange(3000))
    assert s.slope(bars_from_close(close)).iloc[-1] == pytest.approx(0.0004, rel=1e-6)
    # null std of the slope / sigma == ||g||: check by simulating a unit-variance random walk
    b, a, g_norm = s.filter_coefficients()
    r = np.random.default_rng(5).standard_normal(300_000)
    nu = signal.lfilter(b, a, np.cumsum(r))
    assert nu[5000:].std() == pytest.approx(g_norm, rel=0.05)


# ---------------------------------------------------------------------------------------
# donchian
# ---------------------------------------------------------------------------------------
def _donchian_path() -> np.ndarray:
    base = 100 + 0.1 * np.where(np.arange(200) % 2 == 0, 1.0, -1.0)   # tight range
    rise = 100 + 0.5 * np.arange(1, 61)                               # bars 200..259
    fall = rise[-1] - 1.0 * np.arange(1, 81)                          # bars 260..339
    return np.r_[base, rise, fall]


def test_donchian_long_exit_then_short() -> None:
    s = DonchianBreakout(entry_n=50, exit_n=20, atr_n=14, stop_atr=2.0)
    pos = s.positions(bars_from_close(_donchian_path()))
    assert (pos[:200] == 0).all()
    assert pos[200] == 1 and (pos[200:262] == 1).all()
    first_flat = 200 + int(np.argmax(pos[200:] != 1))
    assert 262 < first_flat < 285 and pos[first_flat] == 0
    assert (pos[300:] == -1).all()
    f = s.generate(md_of(bars_from_close(_donchian_path())))
    assert set(np.unique(f.to_numpy())) <= {-0.8, 0.0, 0.8}


def test_donchian_atr_stop() -> None:
    c = _donchian_path()[:202].copy()
    b = bars_from_close(c)
    s = DonchianBreakout(entry_n=50, exit_n=200, atr_n=14, stop_atr=1.0)
    stop = c[200] - 1.0 * float(atr(b, 14).iloc[200])
    lower = float(b["low"].iloc[151:201].min())
    assert lower < stop - 0.02          # scenario precondition: the stop sits above the channel
    c[201] = stop - 0.01                # below the stop, above the short-entry channel
    pos = s.positions(bars_from_close(c))
    assert pos[200] == 1 and pos[201] == 0


# ---------------------------------------------------------------------------------------
# mean reversion
# ---------------------------------------------------------------------------------------
def _range_path(n: int = 400, seed: int = 6) -> np.ndarray:
    return 100 + 0.2 * np.random.default_rng(seed).standard_normal(n)


def test_zscore_fade_enters_on_stretch_and_exits_on_reversion() -> None:
    c = _range_path()
    c[300] = 97.5
    s = ZScoreFade()
    f = s.generate(md_of(bars_from_close(c))).to_numpy()
    assert f[299] == 0.0 and f[300] > 0.4
    first_flat = 300 + int(np.argmax(f[300:] <= 0))
    assert 300 < first_flat < 312                              # reverted -> flat soon after
    assert (f[300:first_flat] > 0).all()
    c[300] = 102.5
    assert s.generate(md_of(bars_from_close(c))).iloc[300] < -0.4


def test_zscore_fade_is_gated_in_trends() -> None:
    c = 100 + 0.05 * np.arange(400) + 0.01 * np.random.default_rng(7).standard_normal(400)
    c[300] -= 4.0                                  # a big dip inside a clean trend
    gated = ZScoreFade().generate(md_of(bars_from_close(c))).to_numpy()
    ungated = ZScoreFade(er_max=None).generate(md_of(bars_from_close(c))).to_numpy()
    assert gated[300] == 0.0 and ungated[300] > 0.0


def test_rsi2_pullback_in_uptrend() -> None:
    c = 100 + 0.1 * np.arange(300)
    c = np.r_[c, c[-1] - 1.0, c[-1] - 2.0, c[-1] - 1.5, c[-1] + 1.0, c[-1] + 2.0, c[-1] + 2.5]
    s = RSI2()
    f = s.generate(md_of(bars_from_close(c))).to_numpy()
    assert f[299] == 0.0
    assert f[300] >= 0.5 and f[301] >= 0.5
    assert f[-1] == 0.0                                  # closed above SMA(5)
    # the same pullback BELOW the 200-bar SMA is not bought
    d = np.r_[100 - 0.1 * np.arange(300), 70.1 - 1.0, 70.1 - 2.0]
    g = s.generate(md_of(bars_from_close(d))).to_numpy()
    assert (g[-2:] <= 0).all()


def test_bollinger_reentry_confirmation() -> None:
    c = _range_path(seed=8)
    c[300] = 99.3          # outside the lower band (but not beyond the 3.5 sd stop)
    c[301] = 99.75         # back inside, below the middle band
    b = bars_from_close(c)
    conf = BollingerRevert().generate(md_of(b)).to_numpy()
    raw = BollingerRevert(confirm=False).generate(md_of(b)).to_numpy()
    assert conf[300] == 0.0 and conf[301] > 0.0
    assert raw[300] > 0.0


# ---------------------------------------------------------------------------------------
# vol_squeeze
# ---------------------------------------------------------------------------------------
def test_vol_squeeze_release_trades_momentum_direction() -> None:
    rng = np.random.default_rng(9)
    noisy = 100 + np.cumsum(0.5 * rng.standard_normal(200))
    quiet = noisy[-1] + 0.02 * rng.standard_normal(60)
    burst = quiet[-1] + 0.8 * np.arange(1, 21)
    c = np.r_[noisy, quiet, burst, burst[-1] + 0.3 * rng.standard_normal(60)]
    s = VolatilitySqueeze(hold=10)
    on = s.squeeze_on(bars_from_close(c))
    assert on[230:258].mean() > 0.8                    # compression detected
    f = s.generate(md_of(bars_from_close(c))).to_numpy()
    assert (f[260:280] > 0).any() and (f[255:285] >= 0).all()
    held = np.flatnonzero(f[258:] != 0)
    assert held.size and np.diff(held).max(initial=1) >= 1 and held.size <= 10 + 20


# ---------------------------------------------------------------------------------------
# orb (DST)
# ---------------------------------------------------------------------------------------
def _day_bars(day: str, closes: dict[int, float], *, or_hour: int, or_hi: float, or_lo: float,
              base: float = 2000.0) -> pd.DataFrame:
    """24 H1 bars on ``day`` (UTC); close ``base`` except the given hours; the OR bar at
    ``or_hour`` UTC has the given high/low."""
    idx = pd.date_range(f"{day} 00:00", periods=24, freq="1h", tz="UTC")
    c = np.array([closes.get(h, base) for h in range(24)], dtype=float)
    o = np.r_[base, c[:-1]]
    hi = np.maximum(o, c) + 0.1
    lo = np.minimum(o, c) - 0.1
    hi[or_hour], lo[or_hour] = or_hi, or_lo
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "spread": 0.3}, index=idx)
    return make_bars(df, "H1")


@pytest.mark.parametrize("day, london_or_utc", [("2024-01-10", 8), ("2024-07-10", 7),
                                                 ("2024-03-13", 8)])
def test_orb_london_is_dst_aware(day: str, london_or_utc: int) -> None:
    h = london_or_utc
    closes = {h + 1: 2000.5, h + 2: 2003.0, h + 3: 2004.0}
    b = _day_bars(day, closes, or_hour=h, or_hi=2002.0, or_lo=1998.0)
    s = OpeningRangeBreakout(sessions=("london",))
    pos = pd.Series(s.session_positions(b, "london"), index=b.index.hour)
    assert pos.loc[h + 1] == 0            # inside the range
    assert pos.loc[h + 2] == 1            # first close above the OR high
    last_hold = h + 6                     # decided at 15:00 London; flat from 16:00
    assert (pos.loc[h + 2:last_hold] == 1).all()
    assert (pos.loc[last_hold + 1:] == 0).all() and (pos.loc[:h] == 0).all()


@pytest.mark.parametrize("day, ny_or_utc", [("2024-01-10", 13), ("2024-07-10", 12),
                                            ("2024-03-13", 12)])   # 03-13: US on DST, UK not
def test_orb_new_york_is_dst_aware_and_stops(day: str, ny_or_utc: int) -> None:
    h = ny_or_utc
    closes = {h + 1: 1997.0, h + 2: 1996.0, h + 3: 2003.0, h + 4: 1990.0}
    b = _day_bars(day, closes, or_hour=h, or_hi=2002.0, or_lo=1998.0)
    s = OpeningRangeBreakout(sessions=("new_york",))
    pos = pd.Series(s.session_positions(b, "new_york"), index=b.index.hour)
    assert pos.loc[h] == 0 and pos.loc[h + 1] == -1 and pos.loc[h + 2] == -1
    assert pos.loc[h + 3] == 0                         # stopped at the opposite side
    assert (pos.loc[h + 4:] == 0).all()                # one trade per session
    rev = OpeningRangeBreakout(sessions=("new_york",), allow_reversal=True)
    pr = pd.Series(rev.session_positions(b, "new_york"), index=b.index.hour)
    assert pr.loc[h + 3] == 1 and pr.loc[h + 4] == -1


def test_orb_sessions_add_and_m15_range() -> None:
    closes = {9: 2003.0, 10: 2003.0, 11: 2003.0, 12: 2003.0, 13: 2003.0, 14: 2004.0}
    b = _day_bars("2024-01-10", closes, or_hour=8, or_hi=2002.0, or_lo=1998.0)
    b.loc[b.index[13], "high"] = 2003.6       # NY OR bar (13:00 UTC in winter)
    b.loc[b.index[13], "low"] = 2002.5
    f = OpeningRangeBreakout().generate(md_of(b))
    assert f.iloc[9] == 0.5 and f.iloc[14] == 1.0     # London long, then NY long too
    # M15: the London OR spans the four quarter-hours 08:00-08:45 UTC (winter)
    idx = pd.date_range("2024-01-10 06:00", periods=24, freq="15min", tz="UTC")
    c = np.full(24, 2000.0)
    c[13] = 2002.5                                    # 09:15 close above the OR high
    hi, lo = c + 0.1, c - 0.1
    hi[10], lo[9] = 2002.0, 1998.0                    # 08:30 high, 08:15 low
    df = pd.DataFrame({"open": c, "high": hi, "low": lo, "close": c, "spread": 0.3}, index=idx)
    m15 = make_bars(df, "M15")
    pos = OpeningRangeBreakout(sessions=("london",)).session_positions(m15, "london")
    assert pos[12] == 0 and pos[13] == 1


def test_orb_range_width_filter() -> None:
    closes = {9: 2000.5, 10: 2003.0}
    b = _day_bars("2024-01-10", closes, or_hour=8, or_hi=2002.0, or_lo=1998.0)
    pre = bars_from_close(np.full(48, 2000.0), start="2024-01-08", wick=0.1)
    both = make_bars(pd.concat([pre, b]).drop(columns="available_at"), "H1")
    wide = OpeningRangeBreakout(sessions=("london",), max_range_atr=2.0)
    assert (wide.session_positions(both, "london") == 0).all()     # OR 4.0 >> 2 x ATR ~0.3
    ok = OpeningRangeBreakout(sessions=("london",), max_range_atr=50.0)
    assert (ok.session_positions(both, "london") == 1).any()


# ---------------------------------------------------------------------------------------
# macro
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def macro_bars() -> pd.DataFrame:
    return make_synthetic_bars(4000, "H1", seed=10, model="gbm", start="2020-06-01")


def _trend_macro(slope_dxy: float, slope_ry: float, n: int = 400) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    dxy = 100 * np.exp(np.cumsum(slope_dxy + 0.003 * rng.standard_normal(n)))
    ry = 1.0 + np.cumsum(slope_ry + 0.03 * rng.standard_normal(n))       # crosses zero: fine
    return {"dxy": macro_frame(dxy), "real10y": macro_frame(ry, lag_hours=45.5)}


def test_macro_factor_sign(macro_bars: pd.DataFrame) -> None:
    s = MacroFactor()
    up = s.generate(md_of(macro_bars, _trend_macro(-0.003, -0.03))).to_numpy()
    down = s.generate(md_of(macro_bars, _trend_macro(0.003, 0.03))).to_numpy()
    assert (up > 0).mean() > 0.9 and (down < 0).mean() > 0.9


def test_macro_factor_missing_and_stale_series(macro_bars: pd.DataFrame) -> None:
    s = MacroFactor()
    assert (s.generate(md_of(macro_bars)) == 0).all()          # no macro at all
    only_dxy = {"dxy": _trend_macro(-0.003, 0.0)["dxy"]}
    assert (s.generate(md_of(macro_bars, only_dxy)) > 0).mean() > 0.9
    short = {k: v.loc[: "2020-09-01"] for k, v in _trend_macro(-0.003, -0.03).items()}
    f = s.generate(md_of(macro_bars, short))
    late = f.loc[pd.DatetimeIndex(macro_bars["available_at"]) > pd.Timestamp("2020-09-20", tz="UTC")]
    assert (late == 0).all()                                  # stale feed -> no signal


def test_macro_factor_uses_available_at_not_observation_date(macro_bars: pd.DataFrame) -> None:
    macro = _trend_macro(-0.003, -0.03)
    s = MacroFactor(series={"dxy": -1.0})
    base = s.generate(md_of(macro_bars, macro))
    day = pd.Timestamp("2020-08-14", tz="UTC")
    shocked = {k: v.copy() for k, v in macro.items()}
    shocked["dxy"].loc[day, "value"] *= 1.2                    # huge dollar spike that day
    alt = s.generate(md_of(macro_bars, shocked))
    known = pd.DatetimeIndex(macro_bars["available_at"]) >= shocked["dxy"].loc[day, "available_at"]
    diff = (base != alt).to_numpy()
    assert diff.any() and not diff[~known].any()
    assert (alt[known].iloc[:5] < base[known].iloc[:5]).all()  # stronger dollar -> less long


def test_clean_macro_monotone_availability() -> None:
    f = macro_frame([1.0, 2.0, 3.0, 4.0])
    f.iloc[1, f.columns.get_loc("available_at")] = f["available_at"].iloc[2] + pd.Timedelta(hours=5)
    out = clean_macro(f, log_values=True)
    assert pd.DatetimeIndex(out["available_at"]).is_monotonic_increasing
    assert out["available_at"].iloc[2] == out["available_at"].iloc[1]


def _vix_frames(spike_day: int, dxy_spike: bool) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(12)
    n = 250
    # calm, persistent VIX (a slow wave: |z| stays below ~1.45 by construction), then a spike
    lv = np.log(15) + 0.05 * np.sin(np.arange(n) / 6.0) + 0.002 * rng.standard_normal(n)
    lv[spike_day:spike_day + 6] += np.log(2.2)         # VIX more than doubles for a week
    dxy = 100 * np.exp(np.cumsum(0.002 * rng.standard_normal(n)))
    if dxy_spike:
        dxy[spike_day:] *= 1.04                        # dash for cash: dollar +4% overnight
    return {"vix": macro_frame(np.exp(lv), lag_hours=21.5), "dxy": macro_frame(dxy)}


def test_risk_off_spike_long_only_and_dash_for_cash(macro_bars: pd.DataFrame) -> None:
    s = RiskOff(entry_z=2.0)
    frames = _vix_frames(150, dxy_spike=False)
    z_calm = s.daily_signal(frames["vix"])["z"].iloc[:150]
    assert z_calm.max() < 2.0                          # scenario precondition
    f = s.generate(md_of(macro_bars, frames))
    spike_avail = frames["vix"]["available_at"].iloc[150]
    after = pd.DatetimeIndex(macro_bars["available_at"]) >= spike_avail
    assert (f >= 0).all() and (f[~after] == 0).all()
    assert f[after].iloc[:24].min() > 0.5
    squeeze = _vix_frames(150, dxy_spike=True)
    g = s.generate(md_of(macro_bars, squeeze))
    dxy_known = pd.DatetimeIndex(macro_bars["available_at"]) >= squeeze["dxy"]["available_at"].iloc[150]
    assert (g[dxy_known].iloc[:24] == 0).all()         # stands aside once the USD spike is known
    assert (g[after & ~dxy_known] > 0).all()           # ...but not before it is published
    assert (s.generate(md_of(macro_bars)) == 0).all()  # no VIX -> no signal


# ---------------------------------------------------------------------------------------
# intraday_seasonality
# ---------------------------------------------------------------------------------------
def test_bucket_keys_follow_local_clock() -> None:
    t = pd.DatetimeIndex(["2024-01-10 13:00", "2024-07-10 12:00", "2024-07-10 13:00"], tz="UTC")
    k = bucket_keys(t, "America/New_York", 60)
    assert k[0] == k[1] != k[2]
    assert k[0] == 2 * 1440 + 8 * 60


def _seasonal_bars(n: int, seed: int, drift_sigma: float) -> pd.DataFrame:
    """GBM whose bars OPENING at 10:00 New York drift up and at 14:00 drift down."""
    base = make_synthetic_bars(n, "H1", seed=seed, model="gbm")
    ny_hour = base.index.tz_convert("America/New_York").hour
    sigma = 0.16 / math.sqrt(252 * 23)
    r = np.random.default_rng(seed).standard_normal(n) * sigma
    r += drift_sigma * sigma * ((ny_hour == 10).astype(float) - (ny_hour == 14).astype(float))
    return bars_from_close(1800 * np.exp(np.cumsum(r)), index=base.index, wick=0.2)


def test_seasonality_recovers_injected_pattern() -> None:
    bars = _seasonal_bars(24_000, seed=13, drift_sigma=0.5)
    s = IntradaySeasonality()
    with pytest.raises(RuntimeError):
        s.generate(md_of(bars))
    s.fit(md_of(bars.iloc[:15_000]))
    assert s.fit_summary_["heterogeneous"] and s.fit_summary_["tau2"] > 0
    f = s.generate(md_of(bars))
    decision_ny = pd.DatetimeIndex(bars["available_at"]).tz_convert("America/New_York").hour
    assert f[decision_ny == 10].mean() > 0.5       # holding the 10:00 bar -> long
    assert f[decision_ny == 14].mean() < -0.5
    other = f[(decision_ny != 10) & (decision_ny != 14)]
    # a single normal prior (one tau^2) under-shrinks the 118 null buckets when only two
    # carry an effect; they stay well below the real ones after scaling
    assert other.abs().mean() < 0.4
    assert other.abs().mean() < 0.5 * min(f[decision_ny == 10].mean(), -f[decision_ny == 14].mean())
    # generate is a function of the clock only
    pert = bars_from_close(bars["close"].to_numpy()[::-1], index=bars.index)
    assert np.array_equal(f.to_numpy(), s.generate(md_of(pert)).to_numpy())


def test_seasonality_shrinks_noise_to_zero() -> None:
    bars = make_synthetic_bars(12_000, "H1", seed=0, model="gbm")
    s = IntradaySeasonality().fit(md_of(bars))
    assert not s.fit_summary_["heterogeneous"]
    assert (s.generate(md_of(bars)) == 0).all()
    s2 = IntradaySeasonality(significance=None, shrinkage=1e9).fit(md_of(bars))
    assert s2.fit_summary_["scalar"] > 0                 # scaled back to avg |f| = 0.5
    f2 = s2.generate(md_of(bars))
    assert f2.abs().mean() == pytest.approx(0.5, abs=0.1)
    s3 = IntradaySeasonality(significance=None, shrinkage=1e9, dead_zone=0.5).fit(md_of(bars))
    f3 = s3.generate(md_of(bars))
    assert ((f3 == 0) | (f3.abs() >= 0.5)).all() and (f3 == 0).mean() > (f2 == 0).mean()


# ---------------------------------------------------------------------------------------
# review (adversarial): train/serve skew, seed-dominated volatility, calibrated pre-test
# ---------------------------------------------------------------------------------------
#: ``aurum.live.runner`` defaults: history = max(history_multiple * max warm-up, min_history_bars)
LIVE_HISTORY_MULTIPLE, LIVE_MIN_HISTORY = 3.0, 300
PRICE_RULES = ("tsmom", "ema_cross", "donchian", "kalman_trend", "zscore_fade", "rsi2",
               "bollinger_revert", "vol_squeeze", "orb")


def _live_history(strat) -> int:
    return max(int(math.ceil(LIVE_HISTORY_MULTIPLE * max(strat.warmup_bars, 1))), LIVE_MIN_HISTORY)


def test_bar_volatility_is_not_dominated_by_the_first_return() -> None:
    """Regression: an ``adjust=False`` EWMA is seeded with ONE squared return, which still
    carries 71% of the weight after ``min_periods=120`` bars at half-life 240. A 1% gap as
    the first return of a history (a weekend gap, a live window starting on a news bar)
    then doubled the volatility estimate and halved every forecast for hundreds of bars."""
    from aurum.strategies.trend import bar_volatility, log_close

    rng = np.random.default_rng(31)
    r = 0.002 * rng.standard_normal(1500)
    r[1] = 0.01                                            # first return: a 5-sigma gap
    lc = log_close(bars_from_close(1800 * np.exp(np.cumsum(r))))
    vol = bar_volatility(lc, 240, 120).to_numpy()
    for t in (121, 385, 721):                              # first forecasts of the rules
        ratio = vol[t] / 0.002
        assert 0.75 < ratio < 1.3, f"bar {t}: vol estimate {ratio:.2f} x the true vol"


@pytest.mark.parametrize("name", PRICE_RULES)
def test_live_history_parity(name: str) -> None:
    """Train/serve skew: the live runner generates on ``max(3 * warmup_bars, 300)`` closed
    bars; its last forecast must match the research forecast computed on the full history.
    Windows are made to START on the largest jumps of a jump-diffusion path, the worst case
    for estimators seeded with the first observation."""
    s = get_strategy(name)
    bars = make_synthetic_bars(9000, "H1", seed=7, model="jump")
    full = s.generate(md_of(bars)).to_numpy()
    n_hist = _live_history(s)
    r = np.abs(np.diff(np.log(bars["close"].to_numpy())))
    starts = [int(j) for j in np.argsort(r)[::-1] if 0 <= j and j + n_hist <= len(bars)][:25]
    assert len(starts) == 25
    worst = 0.0
    for a in starts:
        f = s.generate(md_of(bars.iloc[a:a + n_hist])).to_numpy()
        worst = max(worst, abs(f[-1] - full[a + n_hist - 1]))
    assert worst <= 0.05, f"{name}: live forecast differs from research by {worst:.3f}"


def test_donchian_warmup_covers_a_consolidation_after_a_breakout() -> None:
    """A breakout long followed by a ~500-bar converging triangle: the full-history position
    stays long (lows keep rising above the exit channel, the stop is far below) while a run
    that starts inside the triangle never sees a breakout. The live history must reach
    back to the breakout; with the old warm-up (``max(windows) + 1``) it did not."""
    base = 100 + 0.1 * np.where(np.arange(300) % 2 == 0, 1.0, -1.0)
    rise = 100 + 0.5 * np.arange(1, 21)                     # breakout: bars 300..319
    t = np.arange(500)
    upper, lower = 110.0 - 1.0 * t / 500, 108.0 + 0.9 * t / 500
    tri = np.where(t % 2 == 0, upper - 0.05, lower + 0.05)  # converging triangle
    close = np.r_[base, rise, tri]
    bars = bars_from_close(close)
    s = DonchianBreakout()
    full = s.positions(bars)
    assert full[300] == 1 and (full[300:] == 1).all()        # scenario precondition
    n_hist = _live_history(s)
    live = s.positions(bars.iloc[-n_hist:])
    assert live[-1] == full[-1], f"live history {n_hist} bars misses the open trade"


class _TableSeasonality(IntradaySeasonality):
    """Seasonality fitted on a given (key, y) label table (test-only, not registered)."""

    def __init__(self, frame: pd.DataFrame, **params) -> None:
        super().__init__(**params)
        self._frame = frame

    def training_targets(self, md: MarketData) -> pd.DataFrame:
        return self._frame


def _hetero_labels(seed: int, effect: float = 0.0, n_per: int = 150) -> pd.DataFrame:
    """24 hourly buckets with equal means (plus ``effect`` at 10:00 and -``effect`` at 14:00),
    three "hot" hours with 5x the volatility (gold's London open / US data / COMEX open
    spikes) and t(4) tails."""
    rng = np.random.default_rng(seed)
    keys = np.repeat(np.arange(24) * 60, n_per)
    sd = np.repeat(np.where(np.isin(np.arange(24), [8, 13, 14]), 5.0, 1.0), n_per)
    y = sd * rng.standard_t(4, keys.size) / math.sqrt(2.0)
    y += effect * ((keys == 600).astype(float) - (keys == 840).astype(float))
    return pd.DataFrame({"key": keys, "y": y})


def test_seasonality_pretest_is_calibrated_under_heteroskedasticity() -> None:
    """Regression: the Cochran Q pre-test used one POOLED variance, so volatile hours looked
    like mean effects (18% false positives here at a nominal 5%; 15% on real 2012-15 gold
    labels under a sign-flip null) while effects in quiet hours were under-weighted (power
    0.6). Welch's heteroskedastic ANOVA with per-bucket standard errors fixes both."""
    md = md_of(bars_from_close(np.full(10, 1800.0)))
    rej = [_TableSeasonality(_hetero_labels(seed)).fit(md).fit_summary_["heterogeneous"]
           for seed in range(200)]
    assert np.mean(rej) < 0.12, f"false-positive rate {np.mean(rej):.3f} at nominal 0.05"
    power = [_TableSeasonality(_hetero_labels(seed, effect=0.5)).fit(md).fit_summary_["heterogeneous"]
             for seed in range(20)]
    assert np.mean(power) > 0.9
    s = _TableSeasonality(_hetero_labels(1, effect=0.5)).fit(md)
    assert s.fit_summary_["test"] == "welch"
    table = pd.Series(s.table_)
    assert table[600] > 0 > table[840]
    # empirical-Bayes weights use per-bucket standard errors: a hot (noisy) hour is shrunk
    # harder than a quiet one
    assert s.fit_summary_["shrinkage_by_key"][480] < s.fit_summary_["shrinkage_by_key"][0]


def test_orb_warns_when_bars_are_too_coarse_for_the_opening_range(caplog) -> None:
    """Silent failure mode: on H4/D1 no bar fits inside a 60-minute opening range, so the
    rule is identically 0; it must say so (once per instance) rather than stay silent."""
    h4 = make_synthetic_bars(300, "H4", seed=3, model="gbm")
    s = OpeningRangeBreakout()
    with caplog.at_level("WARNING", logger="aurum.strategies.breakout"):
        f1 = s.generate(md_of(h4))
        f2 = s.generate(md_of(h4))
    assert (f1 == 0).all() and (f2 == 0).all()
    msgs = [r for r in caplog.records if "opening range" in r.getMessage()]
    assert len(msgs) == 1
    assert OpeningRangeBreakout(sessions="london").params["sessions"] == ("london",)
    # H1 bars stamped at :30 (some broker feeds) never fit an 08:00-09:00 range either
    caplog.clear()
    h1 = make_synthetic_bars(500, "H1", seed=3, model="gbm").drop(columns="available_at")
    h1.index = h1.index + pd.Timedelta(minutes=30)
    s30 = OpeningRangeBreakout()
    with caplog.at_level("WARNING", logger="aurum.strategies.breakout"):
        f = s30.generate(md_of(make_bars(h1, "H1")))
        s30.generate(md_of(make_bars(h1, "H1")))
    assert (f == 0).all()
    msgs = [r.getMessage() for r in caplog.records if "no bar fits" in r.getMessage()]
    assert len(msgs) == 2 and "london" in msgs[0] and "new_york" in msgs[1]
    # correctly aligned H1 bars: no warning
    caplog.clear()
    with caplog.at_level("WARNING", logger="aurum.strategies.breakout"):
        OpeningRangeBreakout().generate(md_of(make_synthetic_bars(500, "H1", seed=3, model="gbm")))
    assert not caplog.records


def test_macro_rules_warn_about_missing_or_stale_drivers(macro_bars: pd.DataFrame, caplog) -> None:
    """Silent failure mode: a configured driver that is absent, or whose feed died more than
    ``stale_days`` ago, silently dropped out of the live signal (INFO / no log at all)."""
    macro = _trend_macro(-0.003, -0.03)
    last_avail = pd.Timestamp(macro_bars["available_at"].iloc[-1])
    dead = {"dxy": macro["dxy"],
            "real10y": macro["real10y"].loc[macro["real10y"].index < last_avail - pd.Timedelta(days=20)]}
    s = MacroFactor()
    with caplog.at_level("WARNING", logger="aurum.strategies.macro"):
        f = s.generate(md_of(macro_bars, dead))
        s.generate(md_of(macro_bars, dead))                    # once per instance, not per bar
    stale = [r.getMessage() for r in caplog.records if "latest bar" in r.getMessage()]
    assert len(stale) == 1 and "real10y" in stale[0]
    assert f.iloc[-1] != 0.0                                  # dxy still drives the forecast
    caplog.clear()
    with caplog.at_level("WARNING", logger="aurum.strategies.macro"):
        MacroFactor().generate(md_of(macro_bars, {"dxy": macro["dxy"]}))
    assert any("'real10y' not in md.macro" in r.getMessage() for r in caplog.records)
    caplog.clear()
    fresh = MacroFactor()
    with caplog.at_level("WARNING", logger="aurum.strategies.macro"):
        fresh.generate(md_of(macro_bars, macro))
    assert not caplog.records                                 # healthy feeds: silent
    vix = _vix_frames(150, dxy_spike=False)
    vix["vix"] = vix["vix"].iloc[:100]                        # VIX feed died long ago
    caplog.clear()
    with caplog.at_level("WARNING", logger="aurum.strategies.macro"):
        RiskOff().generate(md_of(macro_bars, vix))
    assert any("'vix'" in r.getMessage() and "latest bar" in r.getMessage() for r in caplog.records)


def test_macro_momentum_z_is_not_dominated_by_the_first_change() -> None:
    """Same seeding defect in the macro EWMAs: a 1% first DXY change (3 sigma) inflated the
    daily vol estimate ~2.5x for the first quarter of history."""
    from aurum.strategies.macro import macro_momentum_z

    rng = np.random.default_rng(32)
    d = 0.003 * rng.standard_normal(400)
    d[1] = 0.01
    x = pd.Series(np.cumsum(d))
    z = macro_momentum_z(x, (1,), vol_halflife=60, vol_min_obs=20).to_numpy()
    disp = np.std(z[40:120])
    assert 0.75 < disp < 1.3, f"null z dispersion {disp:.2f} (should be ~1)"


# ---------------------------------------------------------------------------------------
# speed: 90k bars (the whole H1 history) in < 3 s CPU per strategy
# ---------------------------------------------------------------------------------------
def test_generate_speed_on_90k_bars() -> None:
    bars = make_synthetic_bars(90_000, "H1", seed=14, model="gbm")
    md = MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=14))
    slow = {}
    for name in RULES:
        s = get_strategy(name)
        if s.trainable:
            s.fit(md.slice(end=bars.index[20_000]))
        t0 = time.process_time()
        f = s.generate(md)
        dt = time.process_time() - t0
        assert len(f) == len(bars)
        if dt >= 3.0:
            slow[name] = round(dt, 2)
    assert not slow, f"generate() slower than 3 s CPU on 90k bars: {slow}"
