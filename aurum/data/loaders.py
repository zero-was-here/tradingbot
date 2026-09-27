"""Bar loaders for broker exports and generic CSVs, plus a data-quality report (SPEC §3.1).

MetaTrader 5 "server time"
--------------------------
MT5 exports timestamps in the *broker server's* clock, which is almost never UTC. Most
retail FX/metals brokers run "New-York-close" servers: server midnight = 17:00
America/New_York, i.e. UTC+2 in (US) winter and UTC+3 in (US) summer, switching on the US
DST dates (not the EU ones). We call that convention ``"NY+7"``: server time = New York
local time + 7 hours. Getting this wrong shifts every bar by an hour for ~8 months a year,
which silently breaks session features, event blackouts and multi-source joins.

Accepted ``server_tz`` / ``tz`` specifications:

* ``"UTC"`` (or ``"GMT"``, ``"Z"``);
* any IANA zone name, e.g. ``"Europe/Athens"`` (EET/EEST, EU DST rules), ``"Etc/GMT-2"``
  (fixed UTC+2 — note POSIX sign inversion: ``Etc/GMT-2`` *is* UTC+2);
* fixed offsets ``"UTC+2"``, ``"UTC-05:00"``, ``"+03:00"``;
* ``"NY+7"`` (DST-aware New-York-close server, see above).

Local times that are ambiguous or non-existent because of a DST switch (only possible on
Sunday early-morning New York time, when metals markets are closed) are resolved to the
DST interpretation / shifted forward respectively; resulting duplicates are dropped with
a warning.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.timeframes import Timeframe, get_timeframe
from aurum.data.schema import PRICE_COLS, SchemaError, make_bars

logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"
_OFFSET_RE = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


# ------------------------------------------------------------------------------------
# timezone handling
# ------------------------------------------------------------------------------------
def _localize(naive: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Localize naive wall-clock times to ``tz`` robustly across DST transitions."""
    try:
        return naive.tz_localize(tz, ambiguous="raise", nonexistent="shift_forward")
    except Exception:  # AmbiguousTimeError and friends (class names differ across versions)
        # Ambiguous fall-back hour: take the DST (first) occurrence. Only Sunday early
        # morning in New York / Europe — outside metals trading hours.
        amb = np.ones(len(naive), dtype=bool)
        return naive.tz_localize(tz, ambiguous=amb, nonexistent="shift_forward")


def to_utc_index(times: Iterable[Any] | pd.DatetimeIndex, tz: str = "UTC") -> pd.DatetimeIndex:
    """Convert wall-clock timestamps in ``tz`` (see module doc for accepted specs) to UTC.

    Already tz-aware input is simply converted to UTC (``tz`` is ignored for it).
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    spec = str(tz).strip()
    upper = spec.upper()
    if upper in ("UTC", "GMT", "Z", "ETC/UTC", "ETC/GMT"):
        return idx.tz_localize("UTC")
    if upper == "NY+7":
        ny_local = idx - pd.Timedelta(hours=7)
        return _localize(ny_local, NY_TZ).tz_convert("UTC")
    m = _OFFSET_RE.match(spec)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        offset = pd.Timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return (idx - sign * offset).tz_localize("UTC")
    try:
        return _localize(idx, spec).tz_convert("UTC")
    except Exception as exc:
        raise ValueError(f"unrecognised timezone spec {tz!r}") from exc


def parse_timestamps(values: Iterable[Any], tz: str = "UTC") -> pd.DatetimeIndex:
    """Parse timestamp strings/objects to a UTC DatetimeIndex.

    Values carrying an explicit UTC offset are converted directly; naive values are
    interpreted as wall-clock time in ``tz`` (any spec accepted by :func:`to_utc_index`).
    Mixed inputs (some with offsets, some without) are handled element-wise.
    """
    vals = list(values) if not isinstance(values, (pd.Series, pd.Index, np.ndarray)) else values
    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(vals, format="mixed"))
        return to_utc_index(parsed, tz)
    except ValueError:
        pass
    stamps = [pd.Timestamp(v) for v in vals]
    naive_pos = [i for i, t in enumerate(stamps) if t.tz is None]
    out = [t.tz_convert("UTC") if t.tz is not None else None for t in stamps]
    if naive_pos:
        conv = to_utc_index(pd.DatetimeIndex([stamps[i] for i in naive_pos]), tz)
        for i, t in zip(naive_pos, conv, strict=True):
            out[i] = t
    return pd.DatetimeIndex(out).tz_convert("UTC")


def _finalize_bars(
    df: pd.DataFrame, timeframe: str | Timeframe, default_spread: float | None, source: str
) -> pd.DataFrame:
    """Sort, de-duplicate and validate into canonical bars."""
    nat = df.index.isna()
    if nat.any():
        # an empty / unparseable timestamp cannot be placed in time; without this the frame
        # failed validation with a misleading "index must be strictly increasing" error
        logger.warning("%s: dropping %d rows with missing/unparseable timestamps", source, int(nat.sum()))
        df = df[~nat]
    df = df.sort_index(kind="stable")
    dup = df.index.duplicated(keep="last")
    if dup.any():
        logger.warning("%s: dropping %d duplicate timestamps (kept last)", source, int(dup.sum()))
        df = df[~dup]
    for c in PRICE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    bad = df[PRICE_COLS].isna().any(axis=1) | (df[PRICE_COLS] <= 0).any(axis=1)
    if bad.any():
        logger.warning("%s: dropping %d rows with missing/non-positive prices", source, int(bad.sum()))
        df = df[~bad]
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    if "spread" in df.columns:
        spread = pd.to_numeric(df["spread"], errors="coerce").astype(float)
        if spread.isna().any():
            if default_spread is None:
                raise SchemaError(f"{source}: {int(spread.isna().sum())} missing spread values and no default_spread")
            spread = spread.fillna(float(default_spread))
        df["spread"] = spread.clip(lower=0.0)
    bars = make_bars(df, timeframe, default_spread=default_spread)
    bars.attrs["source"] = source
    return bars


# ------------------------------------------------------------------------------------
# MT5
# ------------------------------------------------------------------------------------
_MT5_ALIASES = {
    "date": "date", "time": "time", "open": "open", "high": "high", "low": "low", "close": "close",
    "tickvol": "tickvol", "tick_volume": "tickvol", "tickvolume": "tickvol",
    "vol": "vol", "volume": "vol", "real_volume": "vol",
    "spread": "spread",
}
_MT5_POSITIONAL = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]


def _read_text(path_or_buffer: str | Path | io.IOBase) -> str:
    if hasattr(path_or_buffer, "read"):
        data = path_or_buffer.read()
    else:
        data = Path(path_or_buffer).read_bytes()
    if isinstance(data, bytes):
        # MT5 writes UTF-16 LE with BOM when exporting from some terminals.
        if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return data.decode("utf-16")
        return data.decode("utf-8-sig")
    return str(data)


def load_mt5_csv(
    path: str | Path | io.IOBase,
    timeframe: str | Timeframe,
    *,
    server_tz: str = "Etc/GMT-2",
    point_size: float = 0.01,
    default_spread: float | None = None,
) -> pd.DataFrame:
    """Parse an MT5 History Center / "Bars" export into canonical UTC bars.

    Handles the headers ``<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>``
    (tab-, comma- or semicolon-separated; ``<TIME>`` absent for D1 exports), plain-word
    headers, and header-less MT4-style files (positional ``date,time,o,h,l,c,vol``). Dates
    may be ``YYYY.MM.DD`` or ``YYYY-MM-DD``.

    * ``volume`` = ``<TICKVOL>`` (tick count; real ``<VOL>`` is usually 0 for CFDs) — if
      only ``<VOL>`` exists it is used instead.
    * ``spread`` = ``<SPREAD>`` (integer points) × ``point_size``. Note MT5 stores the
      *minimum* spread seen in the bar, so it understates typical costs — prefer a cost
      model floor (``CostModel.min_spread``). Without a ``<SPREAD>`` column,
      ``default_spread`` (price units) is required.
    * timestamps are converted from ``server_tz`` to UTC (see module doc; ``"NY+7"`` for
      New-York-close brokers).
    """
    text = _read_text(path)
    src = str(getattr(path, "name", path))
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    sep = "\t" if "\t" in first else (";" if first.count(";") > first.count(",") else ",")
    has_header = bool(re.search(r"[A-Za-z]", first.split(sep)[0].replace(".", "").replace("-", "")))
    if has_header:
        raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skipinitialspace=True)
        cols = {}
        for c in raw.columns:
            key = re.sub(r"[<>\s]", "", str(c)).lower()
            if key in _MT5_ALIASES:
                cols[c] = _MT5_ALIASES[key]
        raw = raw.rename(columns=cols)[list(cols.values())]
    else:
        raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, header=None, skipinitialspace=True)
        n = raw.shape[1]
        if n == 6:  # date,o,h,l,c,vol (D1 without time)
            names = ["date", "open", "high", "low", "close", "tickvol"]
        else:
            names = _MT5_POSITIONAL[:n]
        raw.columns = names
    missing = [c for c in ["date", "open", "high", "low", "close"] if c not in raw.columns]
    if missing:
        raise SchemaError(f"{src}: MT5 export lacks columns {missing}")

    date = raw["date"].str.strip().str.replace(".", "-", regex=False)
    stamp = date + " " + raw["time"].str.strip() if "time" in raw.columns else date
    naive = pd.DatetimeIndex(pd.to_datetime(stamp, format="mixed"))
    idx = to_utc_index(naive, server_tz)

    out = pd.DataFrame(index=idx)
    for c in PRICE_COLS:
        out[c] = pd.to_numeric(raw[c].to_numpy(), errors="coerce")
    if "tickvol" in raw.columns:
        out["volume"] = pd.to_numeric(raw["tickvol"].to_numpy(), errors="coerce")
    elif "vol" in raw.columns:
        out["volume"] = pd.to_numeric(raw["vol"].to_numpy(), errors="coerce")
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0).astype(float)
    if "spread" in raw.columns:
        pts = pd.to_numeric(raw["spread"].to_numpy(), errors="coerce")
        spread = pd.Series(pts, index=idx).astype(float) * float(point_size)
        if spread.isna().any():
            if default_spread is None:
                raise SchemaError(f"{src}: missing <SPREAD> values and no default_spread")
            spread = spread.fillna(float(default_spread))
        out["spread"] = spread.clip(lower=0.0)
    out.index.name = "time"
    return _finalize_bars(out, timeframe, default_spread, f"mt5:{Path(src).name}")


# ------------------------------------------------------------------------------------
# generic CSV / frames
# ------------------------------------------------------------------------------------
def load_csv(
    path: str | Path | io.IOBase,
    timeframe: str | Timeframe,
    *,
    time_col: str = "time",
    tz: str = "UTC",
    default_spread: float | None = None,
) -> pd.DataFrame:
    """Load a generic OHLC CSV (``time,open,high,low,close[,volume][,spread]``).

    Column names are matched case-insensitively. Timestamps must be bar OPEN times;
    naive ones are interpreted in ``tz`` (any spec accepted by :func:`to_utc_index`), while
    timestamps with explicit offsets are converted directly. ``spread`` is in price units.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = time_col.lower()
    if tcol not in df.columns:
        raise SchemaError(f"CSV lacks time column {time_col!r}; columns={list(df.columns)}")
    idx = parse_timestamps(df[tcol].astype(str), tz)
    out = df.drop(columns=[tcol])
    out.index = idx
    out.index.name = "time"
    keep = [c for c in out.columns if c in PRICE_COLS + ["volume", "spread"]]
    out = out[keep]
    name = Path(str(getattr(path, "name", path))).name
    return _finalize_bars(out, timeframe, default_spread, f"csv:{name}")


def bars_from_ohlc(
    df: pd.DataFrame,
    timeframe: str | Timeframe,
    default_spread: float | None = None,
    *,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Thin wrapper around ``make_bars`` for in-memory OHLC frames.

    Accepts a DatetimeIndex (or a ``time`` column) of bar OPEN times; a naive index is
    interpreted in ``tz``. Column names are lower-cased. ``default_spread`` fills a missing
    ``spread`` column (price units).
    """
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if not isinstance(out.index, pd.DatetimeIndex):
        if "time" not in out.columns:
            raise SchemaError("bars_from_ohlc needs a DatetimeIndex or a 'time' column")
        out = out.set_index("time")
    out.index = to_utc_index(out.index, tz)
    out.index.name = "time"
    out = out.drop(columns=[c for c in ("available_at",) if c in out.columns])
    return make_bars(out, timeframe, default_spread=default_spread)


# ------------------------------------------------------------------------------------
# data quality
# ------------------------------------------------------------------------------------
def _classify_gap(t0: pd.Timestamp, t1: pd.Timestamp) -> str:
    """Classify a no-data window [t0, t1) (previous bar end → next bar open).

    * ``weekend``: spans the Friday close → Sunday reopen (contains a UTC Saturday);
    * ``daily_break``: the ~1h metals maintenance break (starts 20:00-22:59 UTC, <= 2h);
    * ``intraweek``: anything else (holidays, early closes, feed outages).
    """
    dur = t1 - t0
    if dur <= pd.Timedelta(days=4) and t0.weekday() in (4, 5) and t1.weekday() in (5, 6, 0):
        return "weekend"
    if dur <= pd.Timedelta(hours=2) and 20 <= t0.hour <= 22:
        return "daily_break"
    return "intraweek"


def quality_report(
    bars: pd.DataFrame,
    *,
    gap_bars: int = 3,
    outlier_sigma: float = 12.0,
    vol_window: int = 500,
    top: int = 10,
) -> dict[str, Any]:
    """Summarise data-quality issues of a canonical bars frame.

    * **gaps**: windows of at least ``gap_bars`` bar-lengths between a bar's end and the next
      bar's open, classified as ``weekend`` / ``daily_break`` (expected) or ``intraweek``
      (holidays, early closes, feed outages — the ones worth inspecting);
    * **spreads**: zero-spread share (suspicious for an OTC quote feed), quantiles, median
      by calendar year;
    * **outliers**: bars whose close-to-close log return exceeds ``outlier_sigma`` times a
      robust *trailing* volatility (rolling median |return| / 0.6745 over ``vol_window``
      bars, lagged one bar) — causal, so the flag never uses future data. Returns across a
      gap are marked ``after_gap`` (weekend gaps are legitimately larger);
    * **flat bars**: O=H=L=C share (stale quotes).
    """
    tf_name = bars.attrs.get("timeframe")
    rep: dict[str, Any] = {"n_rows": len(bars), "timeframe": tf_name}
    if len(bars) == 0:
        return rep
    rep["start"] = str(bars.index[0])
    rep["end"] = str(bars.index[-1])
    idx = bars.index
    delta = get_timeframe(tf_name).delta if tf_name else pd.Series(idx).diff().median()
    prev_end = pd.DatetimeIndex(bars["available_at"]).tz_convert("UTC")[:-1]
    missing = pd.Series(idx[1:] - prev_end, index=idx[1:])
    big = missing[missing >= delta * gap_bars]
    kinds = pd.Series([_classify_gap(t - d, t) for t, d in big.items()], index=big.index, dtype=object)
    rep["n_gaps"] = int(len(big))
    for k in ("weekend", "daily_break", "intraweek"):
        rep[f"n_{k}_gaps"] = int((kinds == k).sum())
    intraweek = big[(kinds == "intraweek").to_numpy()] if len(big) else big
    rep["largest_intraweek_gaps"] = [
        {"from": str(t - d), "to": str(t), "hours": round(d.total_seconds() / 3600, 2)}
        for t, d in intraweek.sort_values(ascending=False).head(top).items()
    ]
    spread = bars["spread"].astype(float)
    rep["zero_spread_frac"] = float((spread <= 0).mean())
    rep["spread_quantiles"] = {str(q): float(spread.quantile(q)) for q in (0.01, 0.5, 0.99, 0.999)}
    rep["max_spread"] = {"time": str(spread.idxmax()), "value": float(spread.max())}
    rep["median_spread_by_year"] = {int(y): float(v) for y, v in spread.groupby(idx.year).median().items()}
    o, h, lo, c = (bars[k].to_numpy(dtype=float) for k in PRICE_COLS)
    rep["flat_bar_frac"] = float(((o == h) & (h == lo) & (lo == c)).mean())
    r = np.log(bars["close"].astype(float)).diff()
    robust_sigma = r.abs().rolling(vol_window, min_periods=max(20, vol_window // 5)).median().shift(1) / 0.6745
    z = (r / robust_sigma).replace([np.inf, -np.inf], np.nan)
    flagged = z[z.abs() > outlier_sigma].dropna()
    rep["n_return_outliers"] = int(len(flagged))
    after_gap = set(big.index)
    rep["top_return_outliers"] = [
        {"time": str(t), "log_return": float(r.loc[t]), "robust_z": float(z.loc[t]), "after_gap": t in after_gap}
        for t in flagged.abs().sort_values(ascending=False).head(top).index
    ]
    return rep


__all__ = ["bars_from_ohlc", "load_csv", "load_mt5_csv", "parse_timestamps", "quality_report", "to_utc_index"]
