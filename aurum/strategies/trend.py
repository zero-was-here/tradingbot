"""Trend-following strategies: ``tsmom``, ``ema_cross``, ``donchian``, ``kalman_trend``.

Economic rationale
------------------
Time-series momentum (a market's own past excess return predicts its future return) is one
of the best documented anomalies across asset classes and more than a century of data
(Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017). Proposed mechanisms are
behavioural (initial under-reaction to news followed by delayed over-reaction, herding,
disposition effect — Barberis, Shleifer & Vishny 1998; Hong & Stein 1999) and structural
(slow-moving capital, central-bank smoothing, hedging demand). Gold is a macro asset whose
drivers (real rates, the dollar, reserve demand) move in persistent regimes, which is why
trend rules have historically worked on gold at weekly-to-monthly horizons. At intraday
horizons the edge is thinner and costs matter, so the defaults below (for H1 bars) favour
multi-day horizons.

Forecast scaling (shared by every rule-based module)
----------------------------------------------------
Forecasts follow Carver's convention (*Systematic Trading*, 2015, ch. 7): a forecast is a
*risk-adjusted* expected return, scaled so that its long-run average absolute value is half
the cap (10 on a ±20 scale → 0.5 on [-1, 1]) and then capped. Signals here are built as
dimensionless statistics that are ~N(0, 1) under a driftless random walk (a vol-normalised
return, a vol-normalised EMA spread, a Kalman slope t-stat). Multiplying such a statistic by
``Z_FORECAST_SCALAR = 0.5 / E|Z| = 0.6267`` gives ``E|forecast| = 0.5`` exactly under the
null, with no data-fitted scalar (so no look-ahead and no dependence on the sample).
Several correlated sub-signals are averaged and multiplied by a forecast-diversification
multiplier ``1 / sqrt(w' C w)`` computed from their correlation ``C`` UNDER THE NULL
(analytic: ``sqrt(h_min / h_max)`` for overlapping horizons), capped at 2.5 as in Carver.

The helpers of that machinery (``bar_volatility``, ``ewma_mean``, ``momentum_z``,
``z_to_forecast``, ``diversification_multiplier``, ``latch`` …) live in this module and are
imported by ``mean_reversion``, ``breakout``, ``macro`` and ``seasonal``. Volatility
estimates use the normalised EWMA (``adjust=True``), never one seeded with the first
observation, so a live run on a shorter history agrees with research on the full history
(see :func:`ewma_mean`).

References
----------
* Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). "Time series momentum". *Journal of
  Financial Economics* 104(2), 228-250.
* Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). "A Century of Evidence on Trend-Following
  Investing". *Journal of Portfolio Management* 44(1), 15-29.
* Baz, J., Granger, N., Harvey, C. R., Le Roux, N. & Rattray, S. (2015). "Dissecting
  Investment Strategies in the Cross Section and Time Series". SSRN 2695101.
* Carver, R. (2015). *Systematic Trading*. Harriman House — EWMAC rule, forecast scalars,
  forecast diversification multiplier.
* Faith, C. (2007). *Way of the Turtle*. McGraw-Hill — Donchian breakouts with N (ATR) stops.
* Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*.
  CUP — the local linear trend model; Durbin & Koopman (2012), *Time Series Analysis by
  State Space Methods*, 2nd ed., OUP.
* Benhamou, E. (2016). "Trend without hiccups: a Kalman filter approach". SSRN 2747102.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, signal

from aurum.core.types import MarketData
from aurum.features.technical import tsmom_response
from aurum.features.volatility import atr, safe_div
from aurum.strategies.base import Strategy, register_strategy

logger = logging.getLogger(__name__)

__all__ = [
    "EXPECTED_ABS_BAZ",
    "EXPECTED_ABS_NORMAL",
    "MAX_FDM",
    "TARGET_ABS_FORECAST",
    "Z_FORECAST_SCALAR",
    "DonchianBreakout",
    "EMACrossover",
    "KalmanTrend",
    "TimeSeriesMomentum",
    "bar_volatility",
    "check_params",
    "diversification_multiplier",
    "ema_spread_variance",
    "ewma_mean",
    "latch",
    "llt_steady_state",
    "log_close",
    "momentum_z",
    "overlap_correlation",
    "sign_correlation",
    "z_to_forecast",
]

# ---------------------------------------------------------------------------------------
# forecast-scaling constants
# ---------------------------------------------------------------------------------------
#: ``E|Z|`` for ``Z ~ N(0, 1)``.
EXPECTED_ABS_NORMAL: float = math.sqrt(2.0 / math.pi)
#: Carver's convention: average |forecast| = 10 on a ±20 scale, i.e. 0.5 on [-1, 1].
TARGET_ABS_FORECAST: float = 0.5
#: Multiply an ~N(0,1) statistic by this to get ``E|forecast| = 0.5`` (before the ±1 cap).
Z_FORECAST_SCALAR: float = TARGET_ABS_FORECAST / EXPECTED_ABS_NORMAL
#: ``E|phi(Z)|`` for the Baz et al. response ``phi(z) = z exp(-z^2/4) / 0.89``, Z ~ N(0,1):
#: ``2/sqrt(2 pi) * integral_0^inf z exp(-3 z^2 / 4) dz / 0.89 = 4 / (3 sqrt(2 pi)) / 0.89``.
EXPECTED_ABS_BAZ: float = 4.0 / (3.0 * math.sqrt(2.0 * math.pi)) / 0.89
#: Cap on the forecast diversification multiplier (Carver 2015, ch. 8).
MAX_FDM: float = 2.5

_RESPONSES = ("linear", "sign", "baz")


# ---------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------
def check_params(strategy: Strategy) -> None:
    """Reject unknown parameter names (typos would otherwise be silently ignored)."""
    unknown = set(strategy.params) - set(strategy.default_params())
    if unknown:
        raise ValueError(f"{strategy.name}: unknown parameter(s) {sorted(unknown)}; "
                         f"valid: {sorted(strategy.default_params())}")


def log_close(bars: pd.DataFrame) -> pd.Series:
    """Natural log of the close (bars are validated to be finite and positive)."""
    return pd.Series(np.log(bars["close"].to_numpy(dtype=float)), index=bars.index, name="log_close")


def ewma_mean(x: pd.Series, halflife: float, min_periods: int) -> pd.Series:
    """Normalised exponentially weighted mean ``sum_k w^k x_{t-k} / sum_k w^k`` (causal).

    Uses ``adjust=True``: every available observation is weighted by ``w^k`` and the
    weights are normalised by their sum. The recursive ``adjust=False`` form is instead
    *seeded with the first observation*, which keeps weight ``w^t`` — 71% after 120 bars at
    a 240-bar half-life. For a squared return that seed is a single chi-square(1) draw, so
    the estimate (and every forecast divided by it) would be dominated by whichever bar
    happens to start the history: a weekend gap or a news bar at the start of a live
    window changed the forecasts by up to 0.6 (review finding). The normalised form is an
    honest estimate from however much history there is, so research on the full history
    and live on a shorter window agree once both windows are a few half-lives long.
    """
    return x.ewm(halflife=float(halflife), adjust=True, min_periods=int(min_periods)).mean()


def bar_volatility(log_price: pd.Series, halflife: float, min_periods: int) -> pd.Series:
    """Per-bar volatility of log returns: zero-mean EWMA ``sqrt(EWM(r^2))`` (RiskMetrics).

    Causal (row ``t`` uses returns up to and including ``r_t``). The EWMA is the
    normalised (``adjust=True``) form, see :func:`ewma_mean`. Zero volatility (a flat
    history, e.g. frozen 2012 holiday quotes) is mapped to NaN so that callers dividing by
    it get NaN (→ a zero forecast) instead of ``inf``.
    """
    r = log_price.diff()
    vol = np.sqrt(ewma_mean(r * r, halflife, min_periods))
    return vol.where(vol > 0.0)


def momentum_z(log_price: pd.Series, sigma: pd.Series, horizon: int) -> pd.Series:
    """Vol-normalised ``horizon``-bar log return ``ln(C_t/C_{t-h}) / (sigma_t sqrt(h))``.

    ~N(0, 1) under a random walk with volatility ``sigma`` (the TSMOM "signal strength" of
    Moskowitz et al. 2012 before taking the sign).
    """
    h = int(horizon)
    ret = log_price - log_price.shift(h)
    return pd.Series(safe_div(ret, sigma * math.sqrt(h)), index=log_price.index)


def z_to_forecast(z: pd.Series | np.ndarray, scalar: float = Z_FORECAST_SCALAR) -> np.ndarray:
    """Scale an ~N(0,1) statistic to Carver's convention and cap to [-1, 1] (NaN stays NaN)."""
    return np.clip(np.asarray(z, dtype=float) * scalar, -1.0, 1.0)


def overlap_correlation(horizons: Sequence[int]) -> np.ndarray:
    """Correlation of vol-normalised returns over overlapping horizons under a random walk.

    For ``h_i <= h_j`` ending at the same bar, ``corr = h_i / sqrt(h_i h_j) = sqrt(h_i/h_j)``.
    """
    h = np.asarray([float(x) for x in horizons])
    lo = np.minimum.outer(h, h)
    hi = np.maximum.outer(h, h)
    return np.sqrt(lo / hi)


def sign_correlation(rho: np.ndarray) -> np.ndarray:
    """Correlation of ``sign(X), sign(Y)`` for bivariate normal ``X, Y`` (Sheppard / arcsine law)."""
    return (2.0 / math.pi) * np.arcsin(np.clip(rho, -1.0, 1.0))


def diversification_multiplier(corr: np.ndarray, weights: Sequence[float] | np.ndarray,
                               cap: float = MAX_FDM) -> float:
    """Carver's forecast diversification multiplier ``1 / sqrt(w' C w)``, capped at ``cap``.

    Restores the average |forecast| of a weighted mean of imperfectly correlated forecasts
    (each already scaled to the common convention). Negative correlations are floored at 0
    as Carver recommends (prevents an unstable, huge multiplier).
    """
    w = np.asarray(weights, dtype=float)
    c = np.clip(np.asarray(corr, dtype=float), 0.0, 1.0)
    var = float(w @ c @ w)
    if not var > 0:
        return 1.0
    return float(min(cap, 1.0 / math.sqrt(var)))


def ema_spread_variance(fast: float, slow: float) -> float:
    """Variance of ``EMA_fast(y) - EMA_slow(y)`` per unit return variance, for a random walk ``y``.

    With ``EMA_t = sum_k alpha (1-alpha)^k y_{t-k}`` and ``y`` a random walk with iid
    increments ``r``, ``EMA_t = y_t - sum_{j>=0} (1-alpha)^{j+1} r_{t-j}``, hence
    ``EMA_f - EMA_s = sum_{k>=1} (a^k - b^k) r_{t-k+1}`` with ``a = 1 - alpha_slow`` and
    ``b = 1 - alpha_fast``, whose variance is
    ``a^2/(1-a^2) + b^2/(1-b^2) - 2ab/(1-ab)`` times ``Var(r)``.
    Spans use the ``alpha = 2 / (span + 1)`` convention.
    """
    a = 1.0 - 2.0 / (float(slow) + 1.0)
    b = 1.0 - 2.0 / (float(fast) + 1.0)
    return a * a / (1.0 - a * a) + b * b / (1.0 - b * b) - 2.0 * a * b / (1.0 - a * b)


def latch(entry_long: np.ndarray, entry_short: np.ndarray, exit_long: np.ndarray,
          exit_short: np.ndarray) -> np.ndarray:
    """Vectorised two-sided position state machine with entry priority → {-1, 0, +1}.

    A long is opened on ``entry_long`` and held until ``exit_long`` or ``entry_short``; a
    short symmetrically. When an entry and an exit fire on the same bar the entry wins;
    contradictory simultaneous long and short entries cancel (flat). Implemented with
    forward-filled event markers, so row ``t`` depends on rows ``<= t`` only (identical to
    the obvious loop, see ``tests/test_strategies_rules.py``).
    """
    el0 = np.asarray(entry_long, dtype=bool)
    es0 = np.asarray(entry_short, dtype=bool)
    both = el0 & es0
    el, es = el0 & ~both, es0 & ~both
    xl = np.asarray(exit_long, dtype=bool) | both
    xs = np.asarray(exit_short, dtype=bool) | both

    def _one(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
        ev = np.full(entry.shape[0], np.nan)
        ev[exit_] = 0.0
        ev[entry] = 1.0
        return pd.Series(ev).ffill().fillna(0.0).to_numpy()

    return _one(el, xl | es) - _one(es, xs | el)


def _as_int_tuple(values: Sequence[Any], what: str) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if not out or any(v < 1 for v in out):
        raise ValueError(f"{what} must be a non-empty sequence of positive integers, got {values!r}")
    return out


def _weights(weights: Sequence[float] | None, k: int) -> np.ndarray:
    if weights is None:
        return np.full(k, 1.0 / k)
    w = np.asarray([float(x) for x in weights], dtype=float)
    if w.shape != (k,) or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"weights must be {k} non-negative numbers with a positive sum")
    return w / w.sum()


# ---------------------------------------------------------------------------------------
# tsmom
# ---------------------------------------------------------------------------------------
@register_strategy
class TimeSeriesMomentum(Strategy):
    """Multi-horizon time-series momentum (TSMOM), vol-scaled.

    Rationale: Moskowitz, Ooi & Pedersen (2012) show that the sign of an asset's own past
    12-month excess return predicts the next month's return in 58 liquid futures including
    gold; Hurst, Ooi & Pedersen (2017) find the effect in every decade since 1880, with
    blends of 1-, 3- and 12-month lookbacks being the most robust. Under-reaction to
    information and slow-moving capital let trends persist; at extremes they tend to
    reverse, which the optional Baz et al. (2015) response ``z exp(-z^2/4)`` captures.

    Signal: for each horizon ``h`` the vol-normalised return
    ``z_h = ln(C_t / C_{t-h}) / (sigma_t sqrt(h))`` (``sigma`` = EWMA per-bar volatility),
    passed through a response (``linear`` = ``z``, ``sign`` = MOP's ``sign(z)``, ``baz``),
    scaled so each has ``E|.| = 0.5`` under a random walk, averaged with ``weights`` and
    multiplied by the forecast diversification multiplier implied by the analytic
    null correlation of overlapping horizons (``sqrt(h_i/h_j)``; arcsine-transformed for
    ``sign``). Result capped to [-1, 1]. Sizing by volatility happens downstream
    (``VolTargetSizer``), exactly as MOP scale positions by ``1 / sigma``.

    Defaults (H1 bars, ~23 bars/day): horizons 120 / 480 / 1440 bars ≈ 1 week, 1 month,
    3 months; volatility half-life 240 bars (~2 weeks).

    References: Moskowitz, Ooi & Pedersen (2012) JFE 104(2); Hurst, Ooi & Pedersen (2017)
    JPM 44(1); Baz et al. (2015) SSRN 2695101; Carver (2015) *Systematic Trading*.
    """

    name = "tsmom"
    description = ("Multi-horizon time-series momentum (1w/1m/3m on H1): vol-normalised past "
                   "returns, Carver-scaled; long when gold has been rising.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "horizons": (120, 480, 1440),
            "weights": None,
            "response": "linear",
            "vol_halflife": 240.0,
            "vol_min_periods": 120,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        p["horizons"] = _as_int_tuple(p["horizons"], "horizons")
        if p["response"] not in _RESPONSES:
            raise ValueError(f"response must be one of {_RESPONSES}, got {p['response']!r}")
        _weights(p["weights"], len(p["horizons"]))
        if not float(p["vol_halflife"]) > 0 or int(p["vol_min_periods"]) < 2:
            raise ValueError("vol_halflife must be > 0 and vol_min_periods >= 2")

    @property
    def warmup_bars(self) -> int:
        return max(max(self.params["horizons"]), int(self.params["vol_min_periods"])) + 1

    def combination(self) -> tuple[np.ndarray, float]:
        """(normalised weights, forecast diversification multiplier) under the null."""
        p = self.params
        w = _weights(p["weights"], len(p["horizons"]))
        corr = overlap_correlation(p["horizons"])
        if p["response"] == "sign":
            corr = sign_correlation(corr)
        return w, diversification_multiplier(corr, w)

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        p = self.params
        bars = md.bars
        lc = log_close(bars)
        sigma = bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"])
        w, fdm = self.combination()
        combined = np.zeros(len(bars))
        for wi, h in zip(w, p["horizons"], strict=True):
            z = momentum_z(lc, sigma, h).to_numpy()
            if p["response"] == "linear":
                resp = z * Z_FORECAST_SCALAR
            elif p["response"] == "sign":
                resp = np.sign(z) * TARGET_ABS_FORECAST
            else:
                resp = tsmom_response(z) * (TARGET_ABS_FORECAST / EXPECTED_ABS_BAZ)
            combined = combined + wi * resp
        return self._finalize(pd.Series(np.clip(combined * fdm, -1.0, 1.0), index=bars.index),
                              bars.index)


# ---------------------------------------------------------------------------------------
# ema_cross
# ---------------------------------------------------------------------------------------
@register_strategy
class EMACrossover(Strategy):
    """Fast/slow exponential moving-average crossover with a continuous forecast (EWMAC).

    Rationale: the spread between a fast and a slow EMA of price is a smoothed momentum
    measure — a linear filter that weights recent returns positively with a hump-shaped
    kernel (Carver 2015; Levine & Pedersen 2016 show that moving-average crossovers and
    time-series momentum are both linear filters of past returns). It captures the same
    under-reaction/persistence premium as TSMOM with less turnover from single-bar noise.

    Signal: ``d_t = EMA_fast(ln C) - EMA_slow(ln C)`` normalised by its exact standard
    deviation under a random walk, ``sigma_t * sqrt(V)`` with
    ``V = a²/(1-a²) + b²/(1-b²) - 2ab/(1-ab)`` (``a = 1 - 2/(slow+1)``, ``b = 1 - 2/(fast+1)``;
    see :func:`ema_spread_variance`). This replaces Carver's empirically fitted forecast
    scalar with an analytic one, so ``E|forecast| = 0.5`` under the null. Defaults (H1):
    32/128 bars (≈1.4 / 5.6 trading days), a 1:4 ratio as in Carver's EWMAC family.

    References: Carver (2015) *Systematic Trading*, ch. 7 & App. B (EWMAC);
    Levine, A. & Pedersen, L. H. (2016). "Which Trend Is Your Friend?" FAJ 72(3), 51-66.
    """

    name = "ema_cross"
    description = ("EWMAC fast/slow EMA crossover (32/128 H1 bars); continuous forecast = EMA "
                   "spread normalised by its random-walk standard deviation.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"fast": 32, "slow": 128, "vol_halflife": 240.0, "vol_min_periods": 120}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        p["fast"], p["slow"] = int(p["fast"]), int(p["slow"])
        if not 1 <= p["fast"] < p["slow"]:
            raise ValueError("need 1 <= fast < slow")

    @property
    def warmup_bars(self) -> int:
        # EMAs are seeded with the first price; after 3 slow spans the seed weight is < 0.3%.
        return max(3 * int(self.params["slow"]), int(self.params["vol_min_periods"])) + 1

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        p = self.params
        bars = md.bars
        lc = log_close(bars)
        sigma = bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"])
        fast = lc.ewm(span=p["fast"], adjust=False).mean()
        slow = lc.ewm(span=p["slow"], adjust=False).mean()
        norm = sigma * math.sqrt(ema_spread_variance(p["fast"], p["slow"]))
        z = safe_div(fast - slow, norm)
        return self._finalize(pd.Series(z_to_forecast(z), index=bars.index), bars.index)


# ---------------------------------------------------------------------------------------
# donchian
# ---------------------------------------------------------------------------------------
@register_strategy
class DonchianBreakout(Strategy):
    """Turtle-style Donchian channel breakout with an ATR ("N") stop and an exit channel.

    Rationale: a close above the highest high of the last ``entry_n`` bars signals that
    supply at previous resistance has been absorbed; breakout systems are a non-linear
    trend filter that is long only once a move is established, and cut losers quickly via
    the stop/exit channel (positive skew). Donchian breakouts were the core of the Turtle
    Traders' system (Faith 2007) and remain a staple of CTA trend following; their
    returns are highly correlated with moving-average and TSMOM rules (Hurst et al. 2017).

    Rules (stateful, evaluated at each bar close):
      * flat → long when ``close_t > max(high_{t-entry_n..t-1})``; short symmetric;
      * long → flat when ``close_t < min(low_{t-exit_n..t-1})`` (exit channel) or
        ``close_t < entry_close - stop_atr * ATR_entry`` (Turtle 2N stop, fixed at entry);
      * a stop-out may immediately reverse if the opposite breakout fires on the same bar.

    The forecast is ``±level`` while in a position. ``level = 0.8`` is chosen so that the
    long-run average |forecast| on a driftless random walk (≈62% time in the market with
    the defaults) is close to Carver's 0.5 convention. Defaults (H1): entry 120 bars
    (~1 week), exit 60 bars, ATR(48), 2.5 N stop.

    Warm-up: the position is path dependent (a trade opened long ago stays open while the
    exit channel and the stop hold), so a run that starts later can be flat while the
    full-history run is long. ``warmup_bars = 2 * max(windows) + 1`` makes the live
    runner's history (3 x warm-up) reach back far enough: on 2012-19 H1 gold, a history of
    ``3 * (max(windows) + 1)`` = 363 bars disagreed with the full-history position on 0.56%
    of bars, and 484+ bars never did (1,243 sampled end points).

    References: Donchian, R. (1960). "High finance in copper". FAJ 16(6); Faith (2007)
    *Way of the Turtle*; Hurst, Ooi & Pedersen (2017) JPM 44(1).
    """

    name = "donchian"
    description = ("Turtle-style Donchian breakout (120-bar entry, 60-bar exit channel, 2.5 ATR "
                   "stop on H1); stateful, forecast ±0.8 while in a position.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"entry_n": 120, "exit_n": 60, "atr_n": 48, "stop_atr": 2.5, "level": 0.8}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        for k in ("entry_n", "exit_n", "atr_n"):
            p[k] = int(p[k])
            if p[k] < 1:
                raise ValueError(f"{k} must be >= 1")
        if not 0.0 < float(p["level"]) <= 1.0:
            raise ValueError("level must be in (0, 1]")
        if p["stop_atr"] is not None and not float(p["stop_atr"]) > 0:
            raise ValueError("stop_atr must be positive or None")

    @property
    def warmup_bars(self) -> int:
        # Channels are valid after max(windows) bars; the second max(windows) lets the
        # path-dependent position converge to the full-history one (see class docstring).
        p = self.params
        return 2 * max(p["entry_n"], p["exit_n"], p["atr_n"]) + 1

    def positions(self, bars: pd.DataFrame) -> np.ndarray:
        """Raw position state in {-1, 0, +1} for every bar (causal state machine)."""
        p = self.params
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        close = bars["close"].to_numpy(dtype=float).tolist()
        upper = high.rolling(p["entry_n"], min_periods=p["entry_n"]).max().shift(1).tolist()
        lower = low.rolling(p["entry_n"], min_periods=p["entry_n"]).min().shift(1).tolist()
        exit_hi = high.rolling(p["exit_n"], min_periods=p["exit_n"]).max().shift(1).tolist()
        exit_lo = low.rolling(p["exit_n"], min_periods=p["exit_n"]).min().shift(1).tolist()
        n_atr = atr(bars, p["atr_n"]).tolist()
        k = float(p["stop_atr"]) if p["stop_atr"] is not None else math.inf
        out = [0.0] * len(close)
        state = 0
        stop = math.nan
        for t, c in enumerate(close):
            if state == 1 and (c < exit_lo[t] or c < stop):
                state = 0
            elif state == -1 and (c > exit_hi[t] or c > stop):
                state = 0
            if state == 0:
                a = n_atr[t]
                if not a > 0:          # NaN during warm-up (or a frozen market): no entries
                    out[t] = 0.0
                    continue
                if c > upper[t]:
                    state, stop = 1, c - k * a
                elif c < lower[t]:
                    state, stop = -1, c + k * a
            out[t] = float(state)
        return np.asarray(out)

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        bars = md.bars
        pos = self.positions(bars) * float(self.params["level"])
        return self._finalize(pd.Series(pos, index=bars.index), bars.index)


# ---------------------------------------------------------------------------------------
# kalman_trend
# ---------------------------------------------------------------------------------------
def llt_steady_state(level_noise: float, slope_noise: float, obs_noise: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steady-state Kalman filter of the local linear trend model (unit return variance).

    Model (Harvey 1989): ``y_t = mu_t + eps_t``, ``mu_{t+1} = mu_t + nu_t + eta_t``,
    ``nu_{t+1} = nu_t + zeta_t`` with variances ``obs_noise``, ``level_noise`` and
    ``slope_noise`` (all as multiples of the per-bar return variance, which cancels out of
    the gain). Solves the discrete algebraic Riccati equation for the prior covariance
    ``P`` and returns ``(K, A, P)`` with gain ``K = P H' / (H P H' + R)`` and the filtered
    state recursion ``x_{t|t} = A x_{t-1|t-1} + K y_t``, ``A = (I - K H) F``.
    """
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    q = np.diag([float(level_noise), float(slope_noise)])
    r = np.array([[float(obs_noise)]])
    p = linalg.solve_discrete_are(f.T, h.T, q, r)
    s = float((h @ p @ h.T)[0, 0] + r[0, 0])
    k = (p @ h.T) / s
    a = (np.eye(2) - k @ h) @ f
    return k, a, p


@register_strategy
class KalmanTrend(Strategy):
    """Local-linear-trend Kalman filter: forecast = t-statistic of the filtered slope.

    Rationale: model log price as a level plus a slowly time-varying drift (the "trend")
    buried in noise. The Kalman filter is the optimal linear estimator of that drift given
    the past (Harvey 1989; Durbin & Koopman 2012); its slope estimate is a trend-following
    linear filter like an EMA crossover but with weights derived from an explicit
    signal-to-noise model instead of ad-hoc spans (Benhamou 2016 applies it to CTA trend
    following). Trading the drift's sign exploits the same persistence premium as TSMOM.

    Implementation: the steady-state gain is solved once from the DARE (process variances
    are multiples of the per-bar return variance, so the gain is scale-free);
    ``slope_noise = 1 / lookback²`` sets the filter's memory (the slope's random walk moves
    by one return-sigma over ``lookback`` bars). The slope filter is converted to an ARMA
    recursion and run with ``scipy.signal.lfilter`` on ``ln C_t - ln C_0`` (zero initial
    state = level at the first price, zero slope). The forecast is the slope divided by its
    exact standard deviation under a random-walk null, ``sigma_t * ||g||`` where ``g`` is
    the filter's response to one unit return — a proper t-statistic, ~N(0,1) under the
    null — times ``Z_FORECAST_SCALAR``. Defaults (H1): lookback 240 bars (~2 weeks),
    level noise 1.0, observation noise 0.1.

    References: Harvey (1989) CUP; Durbin & Koopman (2012) OUP; Benhamou (2016) SSRN 2747102.
    """

    name = "kalman_trend"
    description = ("Local-linear-trend Kalman filter on log price (≈240-bar memory on H1); "
                   "forecast = filtered slope t-stat under a random-walk null.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"lookback": 240, "level_noise": 1.0, "obs_noise": 0.1,
                "vol_halflife": 240.0, "vol_min_periods": 120}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        p["lookback"] = int(p["lookback"])
        if p["lookback"] < 2:
            raise ValueError("lookback must be >= 2")
        if not float(p["level_noise"]) > 0 or not float(p["obs_noise"]) > 0:
            raise ValueError("level_noise and obs_noise must be positive")
        self._coeffs: tuple[np.ndarray, np.ndarray, float] | None = None

    @property
    def warmup_bars(self) -> int:
        return max(3 * int(self.params["lookback"]), int(self.params["vol_min_periods"])) + 1

    def filter_coefficients(self) -> tuple[np.ndarray, np.ndarray, float]:
        """``(b, a, g_norm)``: ARMA coefficients mapping log price to the filtered slope and
        the L2 norm of the slope's response to a unit return (random-walk null std / sigma)."""
        if self._coeffs is None:
            p = self.params
            k, a_mat, _ = llt_steady_state(p["level_noise"], 1.0 / p["lookback"] ** 2, p["obs_noise"])
            e2 = np.array([[0.0, 1.0]])
            # s_{t+1} = A s_t + K y_t (s_t = x_{t-1|t-1});  nu_t = e2 A s_t + e2 K y_t
            num, den = signal.ss2tf(a_mat, k, e2 @ a_mat, e2 @ k)
            b = np.asarray(num, dtype=float).ravel()
            a = np.asarray(den, dtype=float).ravel()
            rho = float(np.max(np.abs(np.linalg.eigvals(a_mat))))
            m = int(min(2_000_000, math.ceil(math.log(1e-14) / math.log(max(rho, 1e-6))) + 10))
            g = signal.lfilter(b, a, np.ones(m))   # slope response to a unit step in price
            self._coeffs = (b, a, float(math.sqrt(np.dot(g, g))))
        return self._coeffs

    def slope(self, bars: pd.DataFrame) -> pd.Series:
        """Filtered slope of log price per bar (steady-state Kalman filter, causal)."""
        b, a, _ = self.filter_coefficients()
        y = np.log(bars["close"].to_numpy(dtype=float))
        if y.size == 0:
            return pd.Series(np.zeros(0), index=bars.index)
        nu = signal.lfilter(b, a, y - y[0])
        return pd.Series(nu, index=bars.index, name="kalman_slope")

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        p = self.params
        bars = md.bars
        _, _, g_norm = self.filter_coefficients()
        sigma = bar_volatility(log_close(bars), p["vol_halflife"], p["vol_min_periods"])
        z = safe_div(self.slope(bars), sigma * g_norm)
        return self._finalize(pd.Series(z_to_forecast(z), index=bars.index), bars.index)
