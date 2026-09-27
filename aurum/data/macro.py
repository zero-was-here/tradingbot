"""Daily macro series from Yahoo Finance and FRED with explicit availability times (SPEC §3.3).

Gold's best-documented macro drivers are the US dollar and US *real* yields (gold is a
zero-coupon, dollar-denominated real asset: its opportunity cost is the real rate), with
risk sentiment (VIX, equities) and inflation expectations as secondary factors — see e.g.
Erb & Harvey (2013) "The Golden Dilemma", FAJ 69(4), and Baur & Lucey (2010) on gold as a
safe haven.

Point-in-time contract
----------------------
Every returned frame has:

* index: observation DATE as tz-aware UTC midnight, named ``date``;
* ``value``: the close / level used by features (plus optional extras such as
  ``open, high, low, volume``);
* ``available_at``: tz-aware UTC instant from which the row may be used.

A daily close is only known after the market closes, so ``available_at = date + lag``:

* Yahoo cash indices (``^GSPC``, ``^VIX``, ``^TNX``): 16:00-16:15 New York close → lag
  **21:30 UTC** (safe in both EST and EDT) — the SPEC default.
* Yahoo futures (``=F``) and ICE (``.NYB``, e.g. the dollar index): their daily session
  ends at 17:00 New York = **22:00 UTC in winter**, so the SPEC's 21:30 would be 30 minutes
  *early* for half the year. We use **22:30 UTC** for them (stricter than the SPEC).
* FRED: observations for day D are published on the NEXT US BUSINESS DAY (the Fed's H.15
  daily update, ~16:15 New York, carries the previous business day's Treasury/TIPS
  yields; the NY Fed publishes the effective fed funds rate at 09:00 the next business
  day) → lag **1 US business day + 21:30 UTC**. The whole-day part of a FRED lag is
  counted in US federal business days (``pandas`` ``USFederalHolidayCalendar``): with
  plain calendar days (the SPEC's "date + 1 day") a *Friday* print would be "available"
  on Saturday and thus used from the Sunday-evening open, although H.15 only publishes it
  on Monday afternoon (a FRED download on a Saturday indeed ends on Thursday for DFII10) —
  ~1 trading day of look-ahead every week and around every holiday, plus a backtest/live
  mismatch because the live FRED feed cannot have that value yet. Weekend/holiday rows of
  7-day series (DFF) roll back to the preceding business day's publication slot, so a
  Friday/Saturday/Sunday DFF row all become usable on Monday 21:30 UTC. FRED series can be
  revised; the lag handles publication delay, not revisions (use ALFRED vintages if
  revisions matter).

``availability_lag`` accepts a single ``Timedelta`` (all series) or a ``{name: Timedelta}``
mapping to override these defaults (for FRED the days part of an override is again counted
in US business days).

Rows whose ``available_at`` lies in the future at fetch time (e.g. today's still-trading
Yahoo bar when ``end`` is today) are *provisional*: they are dropped and such a download is
not written to the raw cache, so a later backtest can never read an intraday snapshot as if
it were the final close.
"""

from __future__ import annotations

import io
import logging
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.data.store import _harmonise_datetime_units, load_frame, save_frame

logger = logging.getLogger(__name__)

DEFAULT_YAHOO: dict[str, str] = {
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "vix": "^VIX",
    "spx": "^GSPC",
    "silver": "SI=F",
    "gold_fut": "GC=F",
    "oil": "CL=F",
}
DEFAULT_FRED: dict[str, str] = {"real10y": "DFII10", "breakeven10y": "T10YIE", "fedfunds": "DFF"}

YAHOO_CASH_LAG = pd.Timedelta(hours=21, minutes=30)
YAHOO_FUTURES_LAG = pd.Timedelta(hours=22, minutes=30)
#: FRED publication lag; its whole days are counted in US federal business days.
FRED_LAG = pd.Timedelta(days=1, hours=21, minutes=30)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"

LagSpec = pd.Timedelta | Mapping[str, pd.Timedelta] | None


# ------------------------------------------------------------------------------------
# frame construction (pure)
# ------------------------------------------------------------------------------------
def default_yahoo_lag(ticker: str) -> pd.Timedelta:
    """Conservative availability lag for a Yahoo daily bar (see module doc)."""
    t = ticker.upper()
    if t.endswith("=F") or t.endswith(".NYB") or t.endswith("=X"):
        return YAHOO_FUTURES_LAG
    return YAHOO_CASH_LAG


def _resolve_lag(lag: LagSpec, name: str, default: pd.Timedelta) -> pd.Timedelta:
    if lag is None:
        return default
    if isinstance(lag, Mapping):
        return pd.Timedelta(lag.get(name, default))
    return pd.Timedelta(lag)


def us_business_day_offset(dates: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """``n``-th US federal business day after each date (tz-aware UTC midnights in/out).

    Non-business dates (weekends, federal holidays) are first rolled BACK to the preceding
    business day, so ``n=1`` maps Fri, Sat and Sun alike to the next Monday (or Tuesday if
    that Monday is a holiday) — the day a publication covering them appears. ``n=0`` is the
    identity (same-day lags keep their calendar date).
    """
    idx = pd.DatetimeIndex(dates)
    if n == 0 or len(idx) == 0:
        return idx
    from pandas.tseries.holiday import USFederalHolidayCalendar

    days = idx.tz_localize(None).normalize().values.astype("datetime64[D]")
    lo = pd.Timestamp(days.min()) - pd.Timedelta(days=14)
    hi = pd.Timestamp(days.max()) + pd.Timedelta(days=14 + 3 * abs(int(n)))
    hol = USFederalHolidayCalendar().holidays(lo, hi).values.astype("datetime64[D]")
    shifted = pd.DatetimeIndex(np.busday_offset(days, int(n), roll="backward", holidays=hol).astype("datetime64[ns]"))
    if idx.tz is not None:
        shifted = shifted.tz_localize("UTC")
    return shifted.as_unit(idx.unit)


def lagged_available_at(dates: pd.DatetimeIndex, lag: pd.Timedelta, *, business_days: bool = False) -> pd.DatetimeIndex:
    """``available_at`` for observation ``dates`` under ``lag``.

    ``business_days=False``: ``date + lag``. ``True``: the whole days of ``lag`` are US
    federal business days (:func:`us_business_day_offset`), the remainder is added on top —
    ``1 day 21:30`` = next business day at 21:30 UTC.
    """
    lag = pd.Timedelta(lag)
    if not business_days:
        return pd.DatetimeIndex(dates) + lag
    whole = int(lag // pd.Timedelta(days=1))
    return us_business_day_offset(pd.DatetimeIndex(dates), whole) + (lag - pd.Timedelta(days=whole))


def _date_index(values: Any) -> pd.DatetimeIndex:
    """Observation dates → tz-aware UTC midnight (the *local* calendar date is kept)."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # keep exchange-local wall date, drop the zone
    return pd.DatetimeIndex(idx.normalize(), name="date").tz_localize("UTC")


def to_macro_frame(
    values: pd.Series | pd.DataFrame,
    *,
    availability_lag: pd.Timedelta,
    value_col: str = "close",
    source: str = "",
    business_days: bool = False,
) -> pd.DataFrame:
    """Build a SPEC §3.3 macro frame from a date-indexed Series/DataFrame.

    For a DataFrame, ``value_col`` becomes ``value`` and other numeric columns are kept as
    extras. Rows with a missing/non-finite value are dropped; duplicate dates keep the last.
    ``available_at`` = :func:`lagged_available_at` (``business_days=True`` counts the whole
    days of the lag in US federal business days — the FRED default). The lag must be
    positive: a daily observation cannot be known at the very start of its own date.
    """
    if not pd.Timedelta(availability_lag) > pd.Timedelta(0):
        raise ValueError(f"availability_lag must be positive, got {availability_lag!r}")
    if isinstance(values, pd.Series):
        df = pd.DataFrame({"value": values.to_numpy()}, index=values.index)
    else:
        cols = {c: str(c).strip().lower().replace(" ", "_") for c in values.columns}
        df = values.rename(columns=cols)
        if value_col not in df.columns:
            raise KeyError(f"value column {value_col!r} not in {list(df.columns)}")
        df = df.rename(columns={value_col: "value"})
        keep = ["value"] + [c for c in ("open", "high", "low", "volume") if c in df.columns]
        df = df[keep]
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    df.index = _date_index(df.index)
    df = df[np.isfinite(df["value"].to_numpy())]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["available_at"] = lagged_available_at(df.index, availability_lag, business_days=business_days)
    lag_txt = str(pd.Timedelta(availability_lag)) + (" (US business days)" if business_days else "")
    df.attrs.update({"source": source, "availability_lag": lag_txt})
    return df


def _drop_provisional(frame: pd.DataFrame, now: pd.Timestamp, label: str) -> tuple[pd.DataFrame, int]:
    """Drop rows not yet final at ``now`` (``available_at > now``); returns (frame, n_dropped)."""
    future = (frame["available_at"] > now).to_numpy()
    n = int(future.sum())
    if n:
        logger.warning("%s: dropping %d provisional row(s) not final before %s", label, n, now)
        frame = frame.loc[~future]
    return frame, n


def parse_fred_csv(text: str, series_id: str | None = None) -> pd.Series:
    """Parse FRED's ``fredgraph.csv`` (``observation_date|DATE,<ID>``; missing = ".")."""
    df = pd.read_csv(io.StringIO(text), na_values=["."], dtype=str)
    if df.shape[1] < 2:
        raise ValueError("unexpected FRED CSV layout")
    date_col = df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df[date_col])),
        name=str(val_col),
    )
    return s.dropna()


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _date_str(x: Any) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def _utcnow() -> pd.Timestamp:
    """Wall-clock "now" (UTC); a seam so tests can pin the fetch time."""
    return pd.Timestamp.now(tz="UTC")


# ------------------------------------------------------------------------------------
# Yahoo
# ------------------------------------------------------------------------------------
def _yahoo_history(ticker: str, start: str, end_incl: str) -> pd.DataFrame:
    """Raw daily OHLCV from yfinance (lazy import; network)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("fetch_yahoo_daily needs `pip install aurum[data]` (yfinance)") from exc
    end_excl = (pd.Timestamp(end_incl) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hist = yf.Ticker(ticker).history(start=start, end=end_excl, interval="1d", auto_adjust=False, actions=False)
    if hist is None or hist.empty:
        raise ValueError(f"Yahoo returned no data for {ticker}")
    hist = hist.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    hist.index = pd.DatetimeIndex(hist.index)
    hist.index = _date_index(hist.index)
    return hist[[c for c in ("open", "high", "low", "close", "volume") if c in hist.columns]]


def fetch_yahoo_daily(
    tickers: Mapping[str, str] | None = None,
    start: Any = "2011-01-01",
    end: Any = None,
    cache_dir: str | Path | None = "cache/macro",
    *,
    availability_lag: LagSpec = None,
    refresh: bool = False,
    skip_errors: bool = True,
) -> dict[str, pd.DataFrame]:
    """Daily closes for ``{name: yahoo_ticker}`` as SPEC §3.3 macro frames.

    Raw histories are cached as parquet under ``cache_dir/yahoo`` keyed by ticker and date
    range; ``refresh=True`` re-downloads. With ``skip_errors`` a failing ticker is logged
    and omitted instead of aborting the whole batch. Rows whose ``available_at`` is still in
    the future (today's in-progress session when ``end`` reaches today) are provisional
    intraday snapshots: they are dropped, and such a download is not cached.
    """
    tickers = dict(DEFAULT_YAHOO if tickers is None else tickers)
    s = _date_str(start)
    e = _date_str(end if end is not None else _utcnow().normalize() - pd.Timedelta(days=1))
    out: dict[str, pd.DataFrame] = {}
    for name, ticker in tickers.items():
        path = Path(cache_dir) / "yahoo" / f"{_safe_name(ticker)}_{s}_{e}.parquet" if cache_dir else None
        lag = _resolve_lag(availability_lag, name, default_yahoo_lag(ticker))
        try:
            fresh = not (path is not None and path.exists() and not refresh)
            raw = _yahoo_history(ticker, s, e) if fresh else load_frame(path)
            frame = to_macro_frame(raw, availability_lag=lag, value_col="close", source=f"yahoo:{ticker}")
        except Exception as exc:
            if not skip_errors:
                raise
            logger.warning("Yahoo %s (%s) failed: %s", name, ticker, exc)
            continue
        frame = frame.loc[(frame.index >= pd.Timestamp(s, tz="UTC")) & (frame.index <= pd.Timestamp(e, tz="UTC"))]
        frame, n_provisional = _drop_provisional(frame, _utcnow(), f"Yahoo {name} ({ticker})")
        if fresh and path is not None and n_provisional == 0:
            save_frame(raw, path, metadata={"ticker": ticker})
        frame.attrs.update({"source": f"yahoo:{ticker}", "availability_lag": str(lag), "name": name})
        out[name] = frame
        logger.info("Yahoo %s (%s): %d rows %s..%s", name, ticker, len(frame),
                    frame.index.min().date() if len(frame) else None, frame.index.max().date() if len(frame) else None)
    return out


# ------------------------------------------------------------------------------------
# FRED
# ------------------------------------------------------------------------------------
def _fred_text(series_id: str, start: str, end: str, timeout: float = 60.0) -> str:
    """Download one fredgraph.csv. Note: FRED's edge silently stalls requests carrying
    browser-like or unknown User-Agent strings (observed 2026-09), so the library default
    (``python-requests/x.y``) is deliberately kept."""
    import requests

    url = FRED_URL.format(series=series_id, start=start, end=end)
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.text.strip():
                return r.text
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except Exception as exc:  # network
            last = exc
        time.sleep(1.5 * 2**attempt)
    raise RuntimeError(f"FRED download failed for {series_id}: {last}") from last


def fetch_fred(
    series: Mapping[str, str] | None = None,
    start: Any = "2011-01-01",
    end: Any = None,
    cache_dir: str | Path | None = "cache/macro",
    *,
    availability_lag: LagSpec = None,
    refresh: bool = False,
    skip_errors: bool = True,
) -> dict[str, pd.DataFrame]:
    """FRED series ``{name: series_id}`` via ``fredgraph.csv`` as SPEC §3.3 macro frames.

    Raw CSV text is cached under ``cache_dir/fred``. Default availability is the NEXT US
    federal business day after the observation date, 21:30 UTC (:data:`FRED_LAG` with its
    whole days counted in business days — see the module doc for why calendar days leak
    Friday and pre-holiday prints). A per-series ``availability_lag`` override is applied
    the same way (its days part in US business days).
    """
    series = dict(DEFAULT_FRED if series is None else series)
    s = _date_str(start)
    e = _date_str(end if end is not None else _utcnow().normalize())
    out: dict[str, pd.DataFrame] = {}
    for name, sid in series.items():
        path = Path(cache_dir) / "fred" / f"{_safe_name(sid)}_{s}_{e}.csv" if cache_dir else None
        try:
            if path is not None and path.exists() and not refresh:
                text = path.read_text()
            else:
                text = _fred_text(sid, s, e)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
            values = parse_fred_csv(text, sid)
        except Exception as exc:
            if not skip_errors:
                raise
            logger.warning("FRED %s (%s) failed: %s", name, sid, exc)
            continue
        lag = _resolve_lag(availability_lag, name, FRED_LAG)
        frame = to_macro_frame(values, availability_lag=lag, source=f"fred:{sid}", business_days=True)
        frame = frame.loc[(frame.index >= pd.Timestamp(s, tz="UTC")) & (frame.index <= pd.Timestamp(e, tz="UTC"))]
        frame.attrs.update({"source": f"fred:{sid}", "availability_lag": f"{lag} (US business days)", "name": name})
        out[name] = frame
        logger.info("FRED %s (%s): %d rows", name, sid, len(frame))
    return out


# ------------------------------------------------------------------------------------
# persistence
# ------------------------------------------------------------------------------------
def validate_macro_frame(df: pd.DataFrame, name: str = "") -> None:
    """Raise ``ValueError`` unless ``df`` follows the SPEC §3.3 frame contract.

    ``available_at`` must be set for every row and lie strictly AFTER the observation
    date's midnight: a daily close/level stamped available at 00:00 of its own date (lag 0)
    would be used for the whole day before it was observed.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError(f"macro {name!r}: index must be a tz-aware DatetimeIndex")
    for c in ("value", "available_at"):
        if c not in df.columns:
            raise ValueError(f"macro {name!r}: missing column {c!r}")
    av = df["available_at"]
    if not isinstance(av.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"macro {name!r}: available_at must be tz-aware")
    if len(df):
        avail = pd.DatetimeIndex(av)
        if avail.isna().any():
            raise ValueError(f"macro {name!r}: available_at has missing values")
        if (avail <= df.index).any():
            raise ValueError(f"macro {name!r}: available_at not after the observation date")


def save_macro_dir(frames: Mapping[str, pd.DataFrame], path: str | Path) -> Path:
    """Write each frame to ``<path>/<name>.parquet`` (tz, available_at and attrs preserved)."""
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        validate_macro_frame(df, name)
        # one datetime unit per frame (index vs available_at), see store._harmonise_datetime_units
        save_frame(_harmonise_datetime_units(df, ["available_at"]), d / f"{_safe_name(name)}.parquet",
                   metadata={"name": name})
    logger.info("saved %d macro frames to %s", len(frames), d)
    return d


def load_macro_dir(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load every ``*.parquet`` in ``path`` into ``{name: frame}`` (see :func:`save_macro_dir`)."""
    d = Path(path)
    out: dict[str, pd.DataFrame] = {}
    for f in sorted(d.glob("*.parquet")):
        df = load_frame(f)
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        df.index.name = "date"
        name = (df.attrs.get("_meta") or {}).get("name", f.stem)
        validate_macro_frame(df, name)
        out[name] = df
    return out


__all__ = [
    "DEFAULT_FRED",
    "DEFAULT_YAHOO",
    "FRED_LAG",
    "YAHOO_CASH_LAG",
    "YAHOO_FUTURES_LAG",
    "default_yahoo_lag",
    "lagged_available_at",
    "us_business_day_offset",
    "fetch_fred",
    "fetch_yahoo_daily",
    "load_macro_dir",
    "parse_fred_csv",
    "save_macro_dir",
    "to_macro_frame",
    "validate_macro_frame",
]
