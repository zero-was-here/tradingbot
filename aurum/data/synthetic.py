"""Synthetic market data for tests, leakage checks and demos.

``model="gbm"`` has NO exploitable structure: any strategy's out-of-sample Sharpe on it must
be statistically indistinguishable from zero (minus costs). That property is used by the
leakage tests — a positive Sharpe on a random walk means the code is peeking at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurum.core.timeframes import get_timeframe, infer_bars_per_year
from aurum.data.schema import make_bars


def _timeline(n: int, timeframe: str, start: str, weekend_gaps: bool) -> pd.DatetimeIndex:
    tf = get_timeframe(timeframe)
    # Generate generously, then filter weekends and cut to n.
    factor = 1.6 if weekend_gaps else 1.0
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=int(n * factor) + 10, freq=tf.freq)
    if weekend_gaps:
        # Gold CFD: closed from Fri 21:00 UTC to Sun 22:00 UTC (approx).
        wd, hr = idx.weekday, idx.hour
        closed = (wd == 5) | ((wd == 4) & (hr >= 21)) | ((wd == 6) & (hr < 22))
        idx = idx[~closed]
    if len(idx) < n:
        raise ValueError("timeline too short; increase factor")
    return idx[:n]


def _log_returns(n: int, sigma: float, model: str, drift: float, rng: np.random.Generator,
                 regime_params: dict | None) -> np.ndarray:
    eps = rng.standard_normal(n)
    if model == "gbm":
        return drift - 0.5 * sigma**2 + sigma * eps
    if model == "trend":
        phi = (regime_params or {}).get("phi", 0.08)
        r = np.empty(n)
        prev = 0.0
        for i in range(n):
            prev = phi * prev + sigma * np.sqrt(1 - phi**2) * eps[i]
            r[i] = drift + prev
        return r
    if model == "mean_revert":
        kappa = (regime_params or {}).get("kappa", 0.02)
        x = np.empty(n)
        level = 0.0
        for i in range(n):
            step = -kappa * level + sigma * eps[i]
            level += step
            x[i] = step
        return x + drift
    if model == "regime":
        p = regime_params or {}
        p_stay = p.get("p_stay", 0.995)
        vols = np.array(p.get("vol_mult", [0.7, 1.8])) * sigma
        drifts = np.array(p.get("drifts", [0.0002 * sigma / 0.002, -0.0003 * sigma / 0.002]))
        state = 0
        r = np.empty(n)
        u = rng.random(n)
        for i in range(n):
            if u[i] > p_stay:
                state = 1 - state
            r[i] = drifts[state] + vols[state] * eps[i]
        return r
    if model == "jump":
        p = regime_params or {}
        lam = p.get("jump_prob", 0.002)
        jump_sigma = p.get("jump_sigma", 8 * sigma)
        jumps = (rng.random(n) < lam) * rng.normal(0, jump_sigma, n)
        return drift + sigma * eps + jumps
    raise ValueError(f"unknown synthetic model {model!r}")


def make_synthetic_bars(
    n: int = 5000,
    timeframe: str = "H1",
    *,
    seed: int = 0,
    model: str = "gbm",
    start: str = "2020-01-06",
    annual_vol: float = 0.16,
    drift: float = 0.0,
    spread: float = 0.30,
    start_price: float = 1800.0,
    weekend_gaps: bool = True,
    regime_params: dict | None = None,
) -> pd.DataFrame:
    """Canonical bars (see ``aurum.data.schema``) following a chosen return process."""
    rng = np.random.default_rng(seed)
    idx = _timeline(n, timeframe, start, weekend_gaps)
    bpy = infer_bars_per_year(idx) if n > 2 else 252 * 23
    sigma = annual_vol / np.sqrt(bpy)
    r = _log_returns(n, sigma, model, drift, rng, regime_params)
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.empty(n)
    open_[0] = start_price
    # small open gap vs previous close (larger after weekends)
    gap_sigma = np.full(n, 0.05 * sigma)
    if weekend_gaps and n > 1:
        big = np.r_[False, np.diff(idx.asi8) > get_timeframe(timeframe).delta.value * 2]
        gap_sigma[big] = 1.5 * sigma
    open_[1:] = close[:-1] * np.exp(rng.normal(0, gap_sigma[1:]))
    # High/low: extend beyond open/close by a half-normal excursion scaled to bar vol.
    exc_hi = np.abs(rng.normal(0, 0.6 * sigma, n)) * close
    exc_lo = np.abs(rng.normal(0, 0.6 * sigma, n)) * close
    high = np.maximum(open_, close) + exc_hi
    low = np.minimum(open_, close) - exc_lo
    low = np.maximum(low, 0.01)
    spr = np.maximum(0.01, spread * np.exp(rng.normal(0, 0.25, n)))
    volume = np.round(rng.gamma(2.0, 500.0, n) * (1 + 50 * np.abs(r) / max(sigma, 1e-12) / 10))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "spread": spr},
        index=idx,
    )
    return make_bars(df, timeframe)


def make_synthetic_macro(bars: pd.DataFrame, *, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Daily macro frames (value + available_at) loosely correlated with gold returns."""
    rng = np.random.default_rng(seed + 1)
    days = pd.date_range(bars.index[0].normalize(), bars.index[-1].normalize(), freq="B", tz="UTC")
    gold_daily = bars["close"].resample("1D").last().reindex(days).ffill()
    g = np.log(gold_daily).diff().fillna(0.0).to_numpy()
    n = len(days)
    specs = {
        "dxy": dict(start=100.0, vol=0.005, beta=-0.35, kind="price"),
        "spx": dict(start=4000.0, vol=0.011, beta=0.1, kind="price"),
        "vix": dict(start=18.0, vol=0.06, beta=0.2, kind="price"),
        "us10y": dict(start=3.0, vol=0.05, beta=-2.0, kind="yield"),
        "real10y": dict(start=1.0, vol=0.05, beta=-3.0, kind="yield"),
    }
    out = {}
    for name, s in specs.items():
        eps = rng.standard_normal(n) * s["vol"]
        if s["kind"] == "price":
            val = s["start"] * np.exp(np.cumsum(s["beta"] * g + eps))
        else:
            val = s["start"] + np.cumsum(s["beta"] * g + eps)
        frame = pd.DataFrame({"value": val}, index=days)
        frame.index.name = "date"
        frame["available_at"] = frame.index + pd.Timedelta(hours=21, minutes=30)
        out[name] = frame
    return out


def make_synthetic_events(start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DataFrame:
    """Monthly NFP-like (first Friday 13:30 UTC) and 8-per-year FOMC-like (19:00 UTC) events."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    rows = []
    for m in pd.date_range(start.normalize().replace(day=1), end, freq="MS"):
        first_friday = m + pd.Timedelta(days=(4 - m.weekday()) % 7)
        rows.append((first_friday + pd.Timedelta(hours=13, minutes=30), "NFP"))
        rows.append((m + pd.Timedelta(days=12, hours=13, minutes=30), "CPI"))
        if m.month in (1, 3, 5, 6, 7, 9, 11, 12):
            rows.append((m + pd.Timedelta(days=17, hours=19), "FOMC"))
    ev = pd.DataFrame(rows, columns=["time", "name"])
    ev = ev[(ev["time"] >= start) & (ev["time"] <= end)].sort_values("time").reset_index(drop=True)
    ev["currency"] = "USD"
    ev["importance"] = 3
    ev["source"] = "synthetic"
    ev["approximate"] = True
    return ev
