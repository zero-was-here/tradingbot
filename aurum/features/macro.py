"""Cross-asset macro features (``macro``).

Economic rationale: gold is a zero-coupon, dollar-denominated real asset. Its main
documented drivers are the US dollar (inverse relation through the numeraire), US real
yields (the opportunity cost of holding a non-yielding asset — Erb & Harvey 2013, "The
Golden Dilemma", FAJ 69(4)), inflation expectations, and risk-off demand (VIX, equities).
The *sign and strength* of these relations drift through time (e.g. the 2022-24 breakdown
of the gold/real-yield link), so rolling correlations and betas are features in their own
right.

Point-in-time discipline (SPEC §0-1, §3.3):

* Transformations (changes, z-scores) are computed on each series' own observation
  sequence, so a derived row only uses observations up to and including that row.
* Each derived row becomes available at the running maximum of the ``available_at`` of
  every input it uses (``cummax``) — robust to vendors whose availability is not monotone.
* Rows are mapped onto bars with ``asof_join(bars.available_at, ...)`` (latest row with
  ``available_at <= decision time``), with a staleness tolerance so a dead feed becomes NaN
  instead of being forward-filled forever.
* The daily gold series for correlation/beta is built from the bars themselves via
  ``resample_bars(..., "D1")`` (through ``multi_timeframe.resample_anchored``: complete
  daily bars only, each with its own ``available_at``), never from a calendar-day
  ``resample().last()``.
* Whether a series is a yield (bp changes) or a price (log changes) is decided from its
  name/attrs, never from its values (a data-dependent choice would be full-sample).

Schema stability: the column set depends only on the configuration and on WHICH series
names are present in ``md.macro`` — never on how many rows are visible. A live runner
working on a short window (or a history cut before the first print) therefore gets the same
columns as research, filled with NaN where nothing is known yet.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.data.pit import asof_join
from aurum.features.base import register_feature
from aurum.features.multi_timeframe import resample_anchored
from aurum.features.volatility import bar_minutes, log_pos, safe_div

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_YIELD_SERIES", "MACRO_LOOKBACK", "macro_features", "macro_lookback",
           "session_date"]

#: Series quoted in percent (yields/rates): changes are reported in basis points.
DEFAULT_YIELD_SERIES: frozenset[str] = frozenset(
    {"us2y", "us5y", "us10y", "us30y", "real10y", "real5y", "breakeven10y", "breakeven5y",
     "fedfunds", "t10y2y", "sofr"}
)

#: Calendar days of slack on top of the observation warm-up: publication lag (FRED is
#: available the next day), the intraday availability time and a few exchange holidays.
_MACRO_SLACK_DAYS = 5


def _warmup_days(z_min_periods: int, corr_min_periods: int, change_days: Sequence[int]) -> int:
    """Calendar days until every macro column can be valid when macro history starts
    together with the bars: observations are business days, so ``n`` observations span up
    to ``ceil(7n/5)`` calendar days, plus ``_MACRO_SLACK_DAYS``."""
    n_obs = max(int(z_min_periods), int(corr_min_periods) + 1,
                max((int(k) for k in change_days), default=0) + 1)
    return math.ceil(n_obs * 7 / 5) + _MACRO_SLACK_DAYS


def macro_lookback(params: Mapping[str, Any], bar_minutes: float) -> int:
    """Warm-up in BARS of the ``macro`` group for bars of ``bar_minutes`` minutes.

    The warm-up is a number of *days* (daily macro observations), so the bar count scales
    with the bar size: at most ``1440 / bar_minutes`` bars fit in one calendar day.
    """
    days = _warmup_days(params.get("z_min_periods", 60), params.get("corr_min_periods", 40),
                        params.get("change_days", (1, 5, 20)))
    return math.ceil(days * 1440.0 / float(bar_minutes))


#: H1 value of :func:`macro_lookback` with default parameters (89 days -> 2,136 bars).
MACRO_LOOKBACK = macro_lookback({}, 60.0)


def session_date(index: pd.DatetimeIndex, anchor_hour_utc: int) -> pd.DatetimeIndex:
    """Trading date of D1 buckets labelled by their open time with a given UTC anchor.

    With an evening anchor (>= 12h, e.g. 22 = New York close) the bucket opening at
    ``d-1 22:00`` is trading day ``d``; with a morning anchor the bucket belongs to its
    open date.
    """
    idx = pd.DatetimeIndex(index)
    if anchor_hour_utc >= 12:
        idx = idx + pd.Timedelta(hours=24 - anchor_hour_utc)
    return idx.normalize()


def _utc(values: Iterable) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def _running_max_time(*times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Cumulative max over rows of the element-wise max of several UTC time arrays."""
    ns = [_utc(t).as_unit("ns").asi8 for t in times]
    m = ns[0]
    for other in ns[1:]:
        m = np.maximum(m, other)
    m = np.maximum.accumulate(m)
    return pd.DatetimeIndex(m.view("datetime64[ns]")).tz_localize("UTC")


def _clean_series(name: str, frame: pd.DataFrame) -> pd.DataFrame | None:
    """Validate & normalise one macro frame -> ``value``, ``available_at`` (cummax), sorted.

    Returns ``None`` only for a malformed frame (missing columns). A well-formed frame with
    no usable rows (e.g. nothing published yet at a live cutoff) returns an EMPTY frame so
    that the caller still emits the series' columns (as NaN): the schema must not depend on
    the amount of history.
    """
    if "value" not in frame.columns or "available_at" not in frame.columns:
        logger.warning("macro series %r lacks 'value'/'available_at' columns; skipped", name)
        return None
    f = pd.DataFrame(
        {"value": pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)},
        index=_utc(frame.index),
    )
    f["available_at"] = _utc(frame["available_at"])
    f = f.sort_index(kind="stable")
    f = f[~f.index.duplicated(keep="last")]
    f = f.dropna(subset=["value", "available_at"])
    if f.empty:
        logger.info("macro series %r has no usable rows; its columns will be NaN", name)
        return f
    # A derived value is known only once ALL of its inputs are known.
    f["available_at"] = _running_max_time(pd.DatetimeIndex(f["available_at"]))
    return f


def _is_yield(name: str, frame: pd.DataFrame, yield_series: Iterable[str]) -> bool:
    """Whether to difference (bp) rather than log-difference a series.

    Decided from METADATA only (name / ``attrs['kind']``) — never from the data: e.g.
    "any value <= 0" would be a full-sample decision (a future negative print would
    change how past rows are transformed = look-ahead). Non-positive values in a price
    series simply yield NaN log changes.
    """
    kind = frame.attrs.get("kind") if hasattr(frame, "attrs") else None
    if kind is not None:
        return str(kind).lower() in ("yield", "rate", "level", "spread")
    return name.lower() in {s.lower() for s in yield_series}


def _changes(v: pd.Series, k: int, as_yield: bool) -> pd.Series:
    if as_yield:
        return (v - v.shift(k)) * 100.0  # percent -> basis points
    lv = pd.Series(log_pos(v), index=v.index)
    return lv - lv.shift(k)


def _trading_date(open_time: pd.DatetimeIndex, available_at: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC date containing the MIDPOINT of each daily bar.

    Works for any daily anchor without knowing it: a broker D1 bar opening at 21:00/22:00
    UTC of ``d-1`` (MT5 "NY+7" servers) and a UTC-midnight bar of ``d`` both map to ``d``,
    which is the date its macro counterpart (the US close of ``d``) is stamped with. For
    resampled D1 buckets this equals :func:`session_date` with the resampling anchor.
    """
    o = _utc(open_time)
    a = _utc(available_at)
    return (o + (a - o) / 2).normalize()


def _gold_daily(bars: pd.DataFrame, anchor: int) -> pd.DataFrame:
    """Daily gold log returns ``g`` keyed by trading date, with ``g_avail`` per day."""
    daily_input = bar_minutes(bars) >= 1440
    d1 = bars if daily_input else resample_anchored(bars, "D1", anchor)
    lc = pd.Series(np.log(d1["close"].to_numpy(dtype=float)), index=d1.index)
    dates = _trading_date(d1.index, pd.DatetimeIndex(d1["available_at"]))
    out = pd.DataFrame({"g": lc.diff().to_numpy()}, index=dates)
    out["g_avail"] = _utc(d1["available_at"])
    return out[~out.index.duplicated(keep="last")]


@register_feature("macro", family="macro", lookback=MACRO_LOOKBACK)
def macro_features(
    md: MarketData,
    *,
    series: Sequence[str] | None = None,
    change_days: Sequence[int] = (1, 5, 20),
    z_window: int = 250,
    z_min_periods: int = 60,
    corr_window: int = 60,
    corr_min_periods: int = 40,
    beta_series: Sequence[str] = ("dxy", "real10y"),
    yield_series: Iterable[str] = DEFAULT_YIELD_SERIES,
    stale_days: float = 10.0,
    daily_anchor_hour_utc: int = 22,
) -> pd.DataFrame:
    """Per-series changes & level z-scores, and rolling gold correlation/beta.

    Columns (``s`` = series name):

    * ``macro_{s}_chg_{k}d`` — k-observation change: log return for prices, basis-point
      change for yields/rates/spreads (names in ``yield_series`` or
      ``attrs['kind'] in {'yield', 'rate', 'level', 'spread'}``);
    * ``macro_{s}_z`` — rolling ``z_window``-observation z-score of the level (log level for
      prices) with ``z_min_periods`` minimum observations;
    * ``macro_corr_{s}`` / ``macro_beta_{s}`` for ``s`` in ``beta_series`` — rolling
      ``corr_window``-day correlation and OLS beta of daily gold log returns on the series'
      daily change (gold return per unit dollar log-return / per 1bp real-yield move).

    Missing series are skipped (INFO); with no macro data at all an empty frame with the
    bars index is returned. A series that is present but has no row visible yet still
    yields its columns (all NaN), so the schema never depends on the history length. Values
    older than ``stale_days`` at decision time become NaN (``0``/``None`` disables this).
    """
    bars = md.bars
    empty = pd.DataFrame(index=bars.index)
    macro = md.macro or {}
    if not macro:
        logger.info("macro: md.macro is empty; returning no macro features")
        return empty
    names = sorted(macro) if series is None else list(series)
    decision = bars["available_at"]
    tol = pd.Timedelta(days=stale_days) if stale_days else None
    cols: dict[str, np.ndarray] = {}
    clean: dict[str, tuple[pd.DataFrame, bool]] = {}
    for name in names:
        if name not in macro:
            logger.info("macro: series %r not in md.macro; skipping its columns", name)
            continue
        f = _clean_series(name, macro[name])
        if f is None:
            continue
        as_yield = _is_yield(name, macro[name], yield_series)
        clean[name] = (f, as_yield)
        v = f["value"]
        derived: dict[str, pd.Series] = {}
        for k in sorted({int(x) for x in change_days}):
            derived[f"macro_{name}_chg_{k}d"] = _changes(v, k, as_yield)
        level = v if as_yield else pd.Series(log_pos(v), index=v.index)
        roll = level.rolling(z_window, min_periods=z_min_periods)
        derived[f"macro_{name}_z"] = pd.Series(
            safe_div(level - roll.mean(), roll.std(ddof=1)), index=v.index)
        frame = pd.DataFrame(derived, index=v.index)
        frame["available_at"] = f["available_at"]
        aligned = asof_join(decision, frame, tolerance=tol)
        for c in derived:
            cols[c] = aligned[c].to_numpy(dtype=float)

    betas = [s for s in beta_series if s in clean]
    for s in beta_series:
        if s not in clean:
            logger.info("macro: beta series %r unavailable; skipping corr/beta columns", s)
    if betas:
        gold = _gold_daily(bars, daily_anchor_hour_utc)
        for s in betas:
            f, as_yield = clean[s]
            m = pd.DataFrame({"m": _changes(f["value"], 1, as_yield).to_numpy()},
                             index=f.index.normalize())
            m["m_avail"] = pd.DatetimeIndex(f["available_at"])
            m = m[~m.index.duplicated(keep="last")]
            j = gold.join(m, how="inner").dropna(subset=["g", "m"])
            corr_c, beta_c = f"macro_corr_{s}", f"macro_beta_{s}"
            if j.empty:
                # Not enough overlapping history yet: NaN columns keep the schema stable.
                logger.info("macro: no overlapping dates between gold and %r yet", s)
                cols[corr_c] = np.full(len(bars), np.nan)
                cols[beta_c] = np.full(len(bars), np.nan)
                continue
            g_ser, m_ser = j["g"], j["m"]
            corr = g_ser.rolling(corr_window, min_periods=corr_min_periods).corr(m_ser)
            cov = g_ser.rolling(corr_window, min_periods=corr_min_periods).cov(m_ser)
            var = m_ser.rolling(corr_window, min_periods=corr_min_periods).var()
            frame = pd.DataFrame({corr_c: corr.clip(-1.0, 1.0).to_numpy(),
                                  beta_c: safe_div(cov, var)}, index=j.index)
            frame["available_at"] = _running_max_time(pd.DatetimeIndex(j["g_avail"]),
                                                      pd.DatetimeIndex(j["m_avail"]))
            aligned = asof_join(decision, frame, tolerance=tol)
            cols[corr_c] = aligned[corr_c].to_numpy(dtype=float)
            cols[beta_c] = aligned[beta_c].to_numpy(dtype=float)
    if not cols:
        return empty
    out = pd.DataFrame(cols, index=bars.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


macro_features.lookback_fn = macro_lookback  # type: ignore[attr-defined]
