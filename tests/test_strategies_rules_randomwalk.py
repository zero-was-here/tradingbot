"""Random-walk null and positive controls for the rule-based strategies.

* **Null**: on a driftless GBM (``make_synthetic_bars(model="gbm")``, with the point-in-time
  synthetic macro) nothing is predictable, so the gross (pre-cost) per-bar strategy return
  ``forecast[t] * r[t+1]`` must have mean zero. Under H0 it is a martingale-difference
  sequence, so its t-statistic is ~N(0, 1) whatever the forecast's serial correlation. We
  require the mean t over five seeds < 2 (the mean of 5 N(0,1) has sd 0.45) and every seed
  < 4 — a look-ahead bug produces t-stats in the tens. Trainable strategies are fitted on
  the first 60% and scored on the last 40% only. A leaky control shows the check has power.
* **Positive controls**: trend rules must earn on ``model="trend"`` (AR(1) returns) and
  mean-reversion rules on ``model="mean_revert"`` (OU log price). The synthetic trend model
  is short-memory (autocorrelation ``phi`` at lag 1 only), so the trend rules are run with
  horizons matched to it (a few bars) — the test checks the SIGN logic of each rule, not
  its default horizon. RSI(2) is run without its 200-bar trend filter: on an OU process a
  price above its long average is expected to FALL, so the filter is (correctly)
  counter-productive there.
"""

from __future__ import annotations

import numpy as np
import pytest

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro
from aurum.strategies.base import Strategy, get_strategy

SEEDS = (0, 1, 2, 3, 4)
N_RW = 12_000
TRAIN_FRAC = 0.6
RULES = ("tsmom", "ema_cross", "donchian", "kalman_trend", "zscore_fade", "rsi2",
         "bollinger_revert", "vol_squeeze", "orb", "macro_factor", "risk_off",
         "intraday_seasonality")
#: extra null cases: seasonality with its pre-test and cost gate disabled (so a non-trivial
#: table is traded: with costs the fitted positions on noise are flat) and risk_off with low
#: thresholds (the default rarely fires on synthetic VIX).
EXTRA_NULL = {
    "intraday_seasonality/forced": ("intraday_seasonality", {"significance": None, "cost_multiplier": 0.0}),
    "risk_off/active": ("risk_off", {"entry_z": 0.5, "exit_z": -0.5}),
}


def gross_tstat(forecast: np.ndarray, close: np.ndarray, start: int) -> float:
    """t-stat of ``f[t] * ln(C[t+1]/C[t])`` over ``t >= start`` (0 if the forecast is flat)."""
    r = np.diff(np.log(close))
    p = (forecast[:-1] * r)[start:]
    sd = p.std(ddof=1)
    if not sd > 0:
        return 0.0
    return float(p.mean() / sd * np.sqrt(len(p)))


def evaluate(strat: Strategy, md: MarketData) -> float:
    start = strat.warmup_bars
    if strat.trainable:
        cut = int(TRAIN_FRAC * len(md.bars))
        avail = md.bars["available_at"].iloc[cut - 1]
        strat.fit(MarketData(bars=md.bars.iloc[:cut],
                             macro={k: v[v["available_at"] <= avail] for k, v in md.macro.items()}))
        start = max(start, cut)          # score out-of-sample only
    f = strat.generate(md).to_numpy()
    return gross_tstat(f, md.bars["close"].to_numpy(), start)


@pytest.fixture(scope="module")
def rw_markets() -> list[MarketData]:
    out = []
    for seed in SEEDS:
        bars = make_synthetic_bars(N_RW, "H1", seed=100 + seed, model="gbm")
        out.append(MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=100 + seed)))
    return out


def _null_case(case: str) -> tuple[str, dict]:
    return EXTRA_NULL.get(case, (case, {}))


@pytest.mark.parametrize("case", list(RULES) + list(EXTRA_NULL))
def test_no_edge_on_random_walk(case: str, rw_markets: list[MarketData]) -> None:
    name, params = _null_case(case)
    ts = np.array([evaluate(get_strategy(name, **params), md) for md in rw_markets])
    assert ts.mean() < 2.0, f"{case}: mean gross t-stat {ts.mean():.2f} on a random walk ({ts.round(2)})"
    assert ts.max() < 4.0, f"{case}: a seed has t-stat {ts.max():.2f} on a random walk ({ts.round(2)})"


class _PeekingStrategy(Strategy):
    """Negative control: trades the sign of the next bar's return (weakly, via noise)."""

    name = "__peeking_rw"

    def generate(self, md: MarketData, features=None):
        c = np.log(md.bars["close"].to_numpy())
        nxt = np.r_[np.diff(c), 0.0]
        noise = np.random.default_rng(0).standard_normal(len(c)) * 20 * np.std(nxt)
        return self._finalize(np.sign(nxt + noise) * 0.5, md.bars.index)


def test_random_walk_check_has_power(rw_markets: list[MarketData]) -> None:
    ts = np.array([evaluate(_PeekingStrategy(), md) for md in rw_markets])
    assert ts.mean() > 2.0, ts


# ---------------------------------------------------------------------------------------
# positive controls
# ---------------------------------------------------------------------------------------
N_POS = 20_000
POS_SEEDS = (0, 1, 2)
_FAST_VOL = {"vol_halflife": 48, "vol_min_periods": 24}
TREND_CASES = {
    "tsmom": {"horizons": (2, 8, 32), **_FAST_VOL},
    "ema_cross": {"fast": 2, "slow": 8, **_FAST_VOL},
    "donchian": {"entry_n": 10, "exit_n": 5, "atr_n": 10},
    "kalman_trend": {"lookback": 8, **_FAST_VOL},
    "vol_squeeze": {"mom_n": 3, "hold": 6, "min_squeeze": 3},
}
MEAN_REVERT_CASES = {
    "zscore_fade": {},
    "rsi2": {"trend_n": None},
    "bollinger_revert": {},
}


@pytest.fixture(scope="module")
def trend_markets() -> list[MarketData]:
    return [MarketData(bars=make_synthetic_bars(N_POS, "H1", seed=s, model="trend",
                                                regime_params={"phi": 0.15}))
            for s in POS_SEEDS]


@pytest.fixture(scope="module")
def mean_revert_markets() -> list[MarketData]:
    return [MarketData(bars=make_synthetic_bars(N_POS, "H1", seed=s, model="mean_revert"))
            for s in POS_SEEDS]


@pytest.mark.parametrize("name", list(TREND_CASES))
def test_trend_rules_earn_on_trending_market(name: str, trend_markets: list[MarketData]) -> None:
    ts = np.array([evaluate(get_strategy(name, **TREND_CASES[name]), md) for md in trend_markets])
    assert ts.mean() > 2.5 and ts.min() > 1.0, f"{name}: gross t-stats {ts.round(2)}"


@pytest.mark.parametrize("name", list(MEAN_REVERT_CASES))
def test_mean_reversion_rules_earn_on_mean_reverting_market(
        name: str, mean_revert_markets: list[MarketData]) -> None:
    ts = np.array([evaluate(get_strategy(name, **MEAN_REVERT_CASES[name]), md)
                   for md in mean_revert_markets])
    assert ts.mean() > 2.5 and ts.min() > 1.0, f"{name}: gross t-stats {ts.round(2)}"


def test_fast_trend_rule_loses_on_mean_reversion(
        mean_revert_markets: list[MarketData]) -> None:
    """Sign sanity: a fast trend follower must LOSE (gross) on an OU process."""
    ts = np.array([evaluate(get_strategy("ema_cross", **TREND_CASES["ema_cross"]), md)
                   for md in mean_revert_markets])
    assert ts.mean() < -2.0, ts
