"""Synthetic market data for tests, leakage checks and demos.

``model="gbm"`` has NO exploitable structure: any strategy's out-of-sample Sharpe on it must
be statistically indistinguishable from zero (minus costs). That property is used by the
leakage tests — a positive Sharpe on a random walk means the code is peeking at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurum.core.timeframes import get_timeframe, infer_bars_per_year
from aurum.data.pit import asof_join
from aurum.data.schema import make_bars

_MACRO_LAG = pd.Timedelta(hours=21, minutes=30)


def _timeline(n: int, timeframe: str, start: str, weekend_gaps: bool) -> pd.DatetimeIndex:
    tf = get_timeframe(timeframe)
    # Generate generously, then filter weekends and cut to n. The fixed margin covers one
    # whole closed weekend (Fri 21:00 → Sun 22:00 = 49h) so a short series starting on a
    # Saturday still fits; the first n open stamps do not depend on the margin.
    factor = 1.6 if weekend_gaps else 1.0
    margin = int(np.ceil(pd.Timedelta(hours=50) / tf.delta)) + 10 if weekend_gaps else 10
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=int(n * factor) + margin, freq=tf.freq)
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
        # Compare Timedeltas, not raw integers: ``asi8`` is in the index's own unit (us on
        # pandas 3, ns on pandas 2) while ``Timedelta.value`` is always ns, so an integer
        # comparison silently never flagged a weekend on pandas 3.
        steps = idx[1:] - idx[:-1]
        big = np.r_[False, np.asarray(steps > 2 * get_timeframe(timeframe).delta)]
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
    avail = days + _MACRO_LAG
    # Point-in-time: the day-d macro print (available at d + 21:30 UTC) may only co-move with
    # gold returns known by then. Using the UTC-midnight close here would embed up to 2.5h
    # of *future* gold moves in the macro series — a leak a macro strategy could exploit,
    # breaking the "no edge on gbm" property that the leakage tests rely on.
    gold_asof = asof_join(pd.DatetimeIndex(avail), bars[["close", "available_at"]], columns=["close"])["close"]
    g = np.log(gold_asof.ffill().to_numpy(dtype=float))
    g = np.nan_to_num(np.diff(g, prepend=g[0]), nan=0.0)
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
        frame["available_at"] = frame.index + _MACRO_LAG
        out[name] = frame
    return out


def _ny(day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    """New York wall-clock time on ``day`` → UTC (DST-aware)."""
    return pd.Timestamp(f"{day.date()} {hhmm}").tz_localize("America/New_York").tz_convert("UTC")


def make_synthetic_events(start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DataFrame:
    """Rule-shaped USD event calendar in the SPEC §3.5 format.

    * NFP-like: first Friday of each month, 08:30 New York (12:30/13:30 UTC by US DST);
    * CPI-like: the 13th of the month (next weekday if it falls on a weekend), 08:30 NY;
    * FOMC-like: 8 per year (Jan, Mar, May, Jun, Jul, Sep, Nov, Dec) on the Wednesday
      falling in days 15-21, 14:00 New York (18:00/19:00 UTC) — real statements are
      (almost always) Wednesdays in the second half of the month.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    rows = []
    for m in pd.date_range(start.tz_localize(None).normalize().replace(day=1), end.tz_localize(None), freq="MS"):
        first_friday = m + pd.Timedelta(days=(4 - m.weekday()) % 7)
        rows.append((_ny(first_friday, "08:30"), "NFP"))
        cpi = m + pd.Timedelta(days=12)
        if cpi.weekday() >= 5:
            cpi += pd.Timedelta(days=7 - cpi.weekday())
        rows.append((_ny(cpi, "08:30"), "CPI"))
        if m.month in (1, 3, 5, 6, 7, 9, 11, 12):
            d15 = m + pd.Timedelta(days=14)
            wed = d15 + pd.Timedelta(days=(2 - d15.weekday()) % 7)
            rows.append((_ny(wed, "14:00"), "FOMC"))
    ev = pd.DataFrame(rows, columns=["time", "name"])
    ev["time"] = pd.DatetimeIndex(ev["time"]).tz_convert("UTC")
    ev = ev[(ev["time"] >= start) & (ev["time"] <= end)].sort_values("time").reset_index(drop=True)
    ev["currency"] = "USD"
    ev["importance"] = 3
    ev["source"] = "synthetic"
    ev["approximate"] = True
    return ev
