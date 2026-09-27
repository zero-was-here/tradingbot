"""Value-at-Risk, Expected Shortfall, VaR back-testing and stress scenarios (SPEC §7).

Conventions
    * Inputs are simple returns (fractions). VaR and ES are reported as POSITIVE loss
      fractions: ``var = 0.02`` means "with probability ``level`` the loss over one period
      does not exceed 2%". ES (a.k.a. CVaR) is the expected loss beyond VaR, so ES >= VaR.
    * ``level`` is the confidence level (0.95, 0.99); the tail probability is
      ``alpha = 1 - level``.

Methods
    * Historical simulation: empirical ``alpha``-quantile and the mean of the returns at
      or below it.
    * Gaussian (variance-covariance): ``VaR = -(mu + sigma z_alpha)``,
      ``ES = -(mu - sigma phi(z_alpha) / alpha)``.
    * Cornish-Fisher (modified VaR, Zangari 1996; Favre & Galeano 2002): the Gaussian
      quantile is corrected for sample skewness ``S`` and excess kurtosis ``K``,
      ``z_cf = z + (z^2-1) S/6 + (z^3-3z) K/24 - (2z^3-5z) S^2/36``. The matching modified ES
      (Boudt, Peterson & Croux, 2008, J. of Risk 11(2)) is computed here in closed form as
      ``E[g(Z) | Z <= z_alpha]`` with ``g`` the CF polynomial, using the truncated normal
      moments ``E[Z|.] = -phi/alpha``, ``E[Z^2|.] = 1 - z phi/alpha``,
      ``E[Z^3|.] = -(z^2+2) phi/alpha``. CF is only reliable for moderate skew/kurtosis
      (the expansion can become non-monotone), so ES is floored at VaR.
    * VaR back-test: Kupiec (1995) proportion-of-failures likelihood-ratio test on a
      rolling, strictly out-of-sample VaR (the estimate for day t uses days < t only).

Stress tests translate price shocks into USD P&L for the current position, including an
optional spread-widening exit cost — the relevant risk for gold around weekend gaps and
major releases, which neither VaR nor the intrabar stop logic captures.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from aurum.core.instrument import XAUUSD, Instrument

logger = logging.getLogger(__name__)

VAR_METHODS = ("historical", "gaussian", "cornish_fisher")

#: Price shocks (fractional moves of the gold price) used by :func:`stress_test`.
#: The named historical episodes are ROUNDED, APPROXIMATE close-to-close moves of spot
#: gold, provided as plausible magnitudes — override with your own data where precision
#: matters.
DEFAULT_STRESS_SCENARIOS: dict[str, float] = {
    "gap_down_3pct": -0.03,
    "gap_up_3pct": 0.03,
    "gap_down_5pct": -0.05,
    "gap_up_5pct": 0.05,
    "gap_down_10pct": -0.10,
    "gap_up_10pct": 0.10,
    "2013-04-15 crash (approx -9%)": -0.09,
    "2011-09-23 selloff (approx -6%)": -0.06,
    "2016-06-24 Brexit spike (approx +5%)": 0.05,
}


def _clean(returns: pd.Series | np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    return r[np.isfinite(r)]


def _alpha(level: float) -> float:
    if not 0.5 < level < 1.0:
        raise ValueError(f"level must be in (0.5, 1), got {level}")
    return 1.0 - level


# ----------------------------------------------------------------------------------------
# VaR / ES estimators
# ----------------------------------------------------------------------------------------
def historical_var_es(returns: pd.Series | np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Historical-simulation VaR and ES (positive loss fractions)."""
    a = _alpha(level)
    r = _clean(returns)
    if r.size == 0:
        return math.nan, math.nan
    q = float(np.quantile(r, a))
    tail = r[r <= q]
    es = -float(tail.mean()) if tail.size else -q
    var = -q
    return var, max(es, var)


def gaussian_var_es(
    returns: pd.Series | np.ndarray | None = None,
    level: float = 0.95,
    *,
    mu: float | None = None,
    sigma: float | None = None,
) -> tuple[float, float]:
    """Gaussian VaR and ES from sample moments (or explicit ``mu``/``sigma``)."""
    a = _alpha(level)
    if mu is None or sigma is None:
        if returns is None:
            raise ValueError("pass returns or both mu and sigma")
        r = _clean(returns)
        if r.size < 2:
            return math.nan, math.nan
        mu = float(r.mean()) if mu is None else mu
        sigma = float(r.std(ddof=1)) if sigma is None else sigma
    z = stats.norm.ppf(a)
    var = -(mu + sigma * z)
    es = -(mu - sigma * stats.norm.pdf(z) / a)
    return float(var), float(es)


def cornish_fisher_quantile(alpha: float, skew: float, excess_kurt: float) -> float:
    """Cornish-Fisher adjusted standard-normal quantile at tail probability ``alpha``."""
    z = stats.norm.ppf(alpha)
    s, k = skew, excess_kurt
    return float(
        z + (z**2 - 1) * s / 6.0 + (z**3 - 3 * z) * k / 24.0 - (2 * z**3 - 5 * z) * s**2 / 36.0
    )


def cornish_fisher_var_es(
    returns: pd.Series | np.ndarray | None,
    level: float = 0.95,
    *,
    mu: float | None = None,
    sigma: float | None = None,
    skew: float | None = None,
    excess_kurt: float | None = None,
) -> tuple[float, float]:
    """Modified (Cornish-Fisher) VaR and ES; moments from ``returns`` unless given."""
    a = _alpha(level)
    r = _clean(returns) if returns is not None else np.empty(0)
    if mu is None or sigma is None or skew is None or excess_kurt is None:
        if r.size < 4:
            return math.nan, math.nan
    mu = float(r.mean()) if mu is None else mu
    sigma = float(r.std(ddof=1)) if sigma is None else sigma
    s = float(stats.skew(r, bias=False)) if skew is None else skew
    k = float(stats.kurtosis(r, fisher=True, bias=False)) if excess_kurt is None else excess_kurt
    if not (math.isfinite(s) and math.isfinite(k)):
        s, k = 0.0, 0.0
    z = stats.norm.ppf(a)
    zcf = cornish_fisher_quantile(a, s, k)
    phi = stats.norm.pdf(z)
    e1 = -phi / a
    e2 = 1.0 - z * phi / a
    e3 = -(z**2 + 2.0) * phi / a
    tail_mean = e1 + (e2 - 1.0) * s / 6.0 + (e3 - 3.0 * e1) * k / 24.0 - (2.0 * e3 - 5.0 * e1) * s**2 / 36.0
    var = -(mu + sigma * zcf)
    es = -(mu + sigma * tail_mean)
    return float(var), float(max(es, var))


def var_es(
    returns: pd.Series | np.ndarray,
    level: float = 0.95,
    method: str = "historical",
    *,
    horizon: int = 1,
) -> dict[str, float]:
    """VaR and ES by ``method`` for a ``horizon``-period holding (iid aggregation).

    Parametric methods aggregate moments (mean x h, sigma x sqrt(h), skew / sqrt(h),
    excess kurtosis / h). Historical simulation is scaled by sqrt(h) — a common but
    approximate rule (it ignores the drift and serial dependence).
    """
    if method not in VAR_METHODS:
        raise ValueError(f"method must be one of {VAR_METHODS}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    r = _clean(returns)
    if method == "historical":
        var, es = historical_var_es(r, level)
        if horizon > 1:
            var, es = var * math.sqrt(horizon), es * math.sqrt(horizon)
    elif method == "gaussian":
        if r.size < 2:
            var, es = math.nan, math.nan
        else:
            var, es = gaussian_var_es(level=level, mu=float(r.mean()) * horizon,
                                      sigma=float(r.std(ddof=1)) * math.sqrt(horizon))
    else:
        if r.size < 4:
            var, es = math.nan, math.nan
        else:
            s = float(stats.skew(r, bias=False))
            k = float(stats.kurtosis(r, fisher=True, bias=False))
            var, es = cornish_fisher_var_es(
                r, level, mu=float(r.mean()) * horizon, sigma=float(r.std(ddof=1)) * math.sqrt(horizon),
                skew=s / math.sqrt(horizon), excess_kurt=k / horizon,
            )
    return {"var": float(var), "es": float(es), "level": float(level), "horizon": int(horizon),
            "method": method, "n_obs": int(r.size)}


# ----------------------------------------------------------------------------------------
# VaR back-testing
# ----------------------------------------------------------------------------------------
def kupiec_pof(n_obs: int, n_breaches: int, level: float) -> dict[str, float]:
    """Kupiec (1995) proportion-of-failures LR test; H0: breach rate == 1 - level."""
    p = _alpha(level)
    n, x = int(n_obs), int(n_breaches)
    if n <= 0:
        return {"lr": math.nan, "p_value": math.nan, "breach_rate": math.nan, "expected_rate": p}
    phat = x / n

    def _ll(prob: float) -> float:
        out = 0.0
        if n - x > 0:
            out += (n - x) * math.log(1.0 - prob) if prob < 1.0 else -math.inf
        if x > 0:
            out += x * math.log(prob) if prob > 0.0 else -math.inf
        return out

    lr = max(-2.0 * (_ll(p) - _ll(phat)), 0.0)
    pval = float(1.0 - stats.chi2.cdf(lr, df=1))
    return {"lr": float(lr), "p_value": pval, "breach_rate": float(phat), "expected_rate": p}


def rolling_var_backtest(
    returns: pd.Series,
    level: float = 0.95,
    *,
    window: int = 250,
    method: str = "historical",
) -> dict:
    """Out-of-sample VaR back-test: VaR for period t from the ``window`` periods before t."""
    if method not in ("historical", "gaussian"):
        raise ValueError("rolling back-test supports 'historical' or 'gaussian'")
    a = _alpha(level)
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) <= window + 5:
        return {"n": 0, "breaches": 0, "expected": 0.0, "window": window, "method": method,
                **kupiec_pof(0, 0, level)}
    if method == "historical":
        q = r.rolling(window, min_periods=window).quantile(a).shift(1)
    else:
        z = stats.norm.ppf(a)
        q = (r.rolling(window, min_periods=window).mean()
             + z * r.rolling(window, min_periods=window).std()).shift(1)
    ok = q.notna()
    breaches = int((r[ok] < q[ok]).sum())
    n = int(ok.sum())
    return {"n": n, "breaches": breaches, "expected": n * a, "window": window, "method": method,
            **kupiec_pof(n, breaches, level)}


# ----------------------------------------------------------------------------------------
# position-level risk
# ----------------------------------------------------------------------------------------
def position_var(
    position_lots: float,
    price: float,
    vol_ann: float,
    *,
    level: float = 0.99,
    horizon_days: float = 1.0,
    instrument: Instrument = XAUUSD,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Gaussian USD VaR/ES of a position from an annualised vol forecast (zero drift)."""
    notional = abs(instrument.notional(position_lots, price))
    sigma = vol_ann * math.sqrt(horizon_days / periods_per_year)
    var, es = gaussian_var_es(level=level, mu=0.0, sigma=sigma)
    return {"notional": notional, "var_usd": var * notional, "es_usd": es * notional,
            "var_frac_of_notional": var, "level": level, "horizon_days": horizon_days}


def gap_shock(
    position_lots: float,
    price: float,
    pct: float,
    *,
    instrument: Instrument = XAUUSD,
    spread: float = 0.0,
) -> float:
    """USD P&L of an instantaneous price gap of ``pct`` (e.g. -0.05) on a signed position.

    ``spread`` (price units) adds the cost of exiting at the (possibly widened) spread:
    ``|lots| * contract_size * spread / 2``. Negative result = loss.
    """
    move = instrument.pnl(position_lots, price, price * (1.0 + pct))
    exit_cost = abs(position_lots) * instrument.contract_size * max(spread, 0.0) / 2.0
    return float(move - exit_cost)


def stress_test(
    position_lots: float,
    price: float,
    equity: float,
    *,
    scenarios: Mapping[str, float] | None = None,
    instrument: Instrument = XAUUSD,
    spread: float = 0.0,
) -> pd.DataFrame:
    """Apply each scenario's price shock to the position; rows sorted worst first."""
    sc = dict(DEFAULT_STRESS_SCENARIOS if scenarios is None else scenarios)
    rows = []
    for name, pct in sc.items():
        pnl = gap_shock(position_lots, price, pct, instrument=instrument, spread=spread)
        rows.append({
            "scenario": name,
            "shock_pct": float(pct),
            "pnl_usd": pnl,
            "pnl_pct_equity": pnl / equity if equity and equity > 0 else math.nan,
            "equity_after": equity + pnl,
        })
    cols = ["scenario", "shock_pct", "pnl_usd", "pnl_pct_equity", "equity_after"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("pnl_usd", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------
def daily_returns_from_equity(equity: pd.Series, *, fold_weekends: bool = True) -> pd.Series:
    """Simple daily returns from an equity curve indexed by bar OPEN time (UTC).

    The day's equity is the last mark of the bars opened on that UTC date. With
    ``fold_weekends`` (default) Saturday/Sunday bars are folded into the following Monday —
    the same trading-day rule as ``aurum.backtest.metrics`` (so ``risk_report`` and
    ``compute_metrics`` agree on daily VaR). Gold reopens Sunday ~22:00 UTC; without the
    fold the one-to-two-hour Sunday stubs would count as ~50 extra low-variance "days" a
    year and bias daily VaR/ES downward. Days without bars are absent (not zero returns).
    The first day has no previous close and is dropped.
    """
    eq = pd.Series(equity, dtype=float).dropna()
    if not isinstance(eq.index, pd.DatetimeIndex):
        raise ValueError("equity must have a DatetimeIndex")
    idx = eq.index.tz_localize("UTC") if eq.index.tz is None else eq.index.tz_convert("UTC")
    dates = idx.normalize()
    if fold_weekends and len(dates):
        wd = dates.weekday.to_numpy()
        shift = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
        if shift.any():
            dates = dates + pd.to_timedelta(shift, unit="D")
    daily = pd.Series(eq.to_numpy(), index=dates).groupby(level=0, sort=True).last()
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = daily / daily.shift(1) - 1.0
    return rets.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna().rename("daily_return")


def _moments(r: np.ndarray) -> dict[str, float]:
    if r.size < 4:
        return {"mean": math.nan, "std": math.nan, "skew": math.nan, "excess_kurtosis": math.nan}
    return {
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "skew": float(stats.skew(r, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
    }


def risk_report(
    result,
    *,
    level: float = 0.95,
    price: float | None = None,
    instrument: Instrument = XAUUSD,
    var_window: int = 250,
    scenarios: Mapping[str, float] | None = None,
) -> dict:
    """JSON-friendly risk summary of a :class:`aurum.backtest.result.BacktestResult`.

    Accepts a ``BacktestResult`` (uses ``equity`` and, if present, ``positions`` /
    ``meta["last_price"]`` / the last fill price for the stress test) or a bare equity
    Series. Contains daily and per-bar VaR/ES by all three methods, return moments, max
    drawdown, a rolling out-of-sample VaR back-test with Kupiec's test, USD figures at the
    last equity and — when a price is available — a stress test of the final position.
    """
    equity = result.equity if hasattr(result, "equity") else result
    equity = pd.Series(equity, dtype=float).dropna()
    report: dict = {"level": level, "notes": []}
    if len(equity) < 3:
        report["notes"].append("equity too short for a risk report")
        return report
    last_equity = float(equity.iloc[-1])
    report["last_equity"] = last_equity
    bar_r = equity.pct_change().dropna()
    report["per_bar"] = {m: var_es(bar_r, level, m) for m in VAR_METHODS}
    report["moments_bar"] = _moments(_clean(bar_r))

    dd = 1.0 - equity / equity.cummax()
    report["max_drawdown"] = float(dd.max())
    report["current_drawdown"] = float(dd.iloc[-1])

    daily = None
    if isinstance(equity.index, pd.DatetimeIndex):
        daily = daily_returns_from_equity(equity)
    if daily is not None and len(daily) >= 20:
        report["n_days"] = int(len(daily))
        report["daily"] = {m: var_es(daily, level, m) for m in VAR_METHODS}
        report["moments_daily"] = _moments(_clean(daily))
        report["worst_day"] = float(daily.min())
        report["best_day"] = float(daily.max())
        report["daily_usd"] = {
            m: {"var": v["var"] * last_equity, "es": v["es"] * last_equity}
            for m, v in report["daily"].items()
        }
        window = int(min(var_window, max(20, len(daily) // 2)))
        report["var_backtest"] = rolling_var_backtest(daily, level, window=window)
    else:
        report["notes"].append("fewer than 20 daily observations: daily VaR omitted")

    # The position still open at the end is the one held at the last CLOSE: prefer
    # ``position_close`` (after intrabar stop/take-profit exits) over ``positions`` (held
    # during the bar), otherwise a stop hit in the last bar would stress a phantom position.
    positions = getattr(result, "position_close", None)
    if positions is None or not len(positions):
        positions = getattr(result, "positions", None)
    if positions is not None and len(positions):
        lots = float(pd.Series(positions, dtype=float).fillna(0.0).iloc[-1])
        px = price
        if px is None:
            meta = getattr(result, "meta", None) or {}
            px = meta.get("last_price")
        if px is None:
            fills = getattr(result, "fills", None)
            if isinstance(fills, pd.DataFrame) and len(fills) and "price" in fills.columns:
                px = float(fills["price"].iloc[-1])
                report["notes"].append("stress price taken from the last fill")
        if px is not None and math.isfinite(float(px)) and float(px) > 0:
            px = float(px)
            st = stress_test(lots, px, last_equity, scenarios=scenarios, instrument=instrument)
            report["position"] = {
                "lots": lots,
                "price": px,
                "notional": float(instrument.notional(lots, px)),
                "leverage": float(abs(instrument.notional(lots, px)) / last_equity) if last_equity > 0 else math.nan,
                "stress": st.to_dict(orient="records"),
            }
        else:
            report["notes"].append("no price available: position stress test omitted")
    return report
