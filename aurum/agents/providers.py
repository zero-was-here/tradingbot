"""Data providers: the ONLY way market/risk/macro data reaches the desk's agents.

A :class:`DeskDataProvider` returns JSON-serialisable snapshots for a decision time ``now``.
Implementations must be point-in-time: a snapshot for ``now`` may only contain information
whose ``available_at <= now`` (SPEC §0-1). Scheduled event *times* are known in advance and
may be shown; event *outcomes* only from their release time onward.

* :class:`StaticDeskDataProvider` — fixed dict snapshots (live wiring, tests, demos).
* :class:`HistoricalDeskDataProvider` — builds snapshots from :class:`~aurum.core.types.MarketData`
  at the latest bar completed by ``now``, for replaying the desk over history.

Look-ahead through model memory (read before trusting a replay backtest)
-----------------------------------------------------------------------
Large language models are trained on text that covers most of gold's price history,
macro releases and market commentary. Even with perfectly point-in-time data, a model shown
"2020-03-16, gold 1,480" may *remember* what happened next — an information leak no
data-side discipline can prevent. Replay results over periods before the model's training
cutoff are therefore optimistic by an unknown amount. ``anonymise=True`` mitigates the most
direct channels: dates are shifted by a whole number of weeks into a fictional future
(weekday and hour-of-day structure preserved), price levels are rebased so the decision
bar's close is 100, macro levels are withheld (only changes and z-scores are shown), and
released economic prints are reduced to their surprise (actual - forecast) because an exact
print identifies the month as surely as a date. It does NOT remove all identifying information — the shape of a price path, the sequence of
events and cross-asset co-movements can still be recognised. The only clean evaluation of
an LLM desk is forward (paper) trading after the model's cutoff.
"""

from __future__ import annotations

import copy
import datetime as _dt
import logging
import math
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from aurum.agents._json import to_jsonable
from aurum.core.timeframes import infer_bars_per_year
from aurum.core.types import MarketData

logger = logging.getLogger(__name__)

__all__ = [
    "SNAPSHOT_KINDS",
    "DeskDataProvider",
    "StaticDeskDataProvider",
    "HistoricalDeskDataProvider",
]

#: snapshot kind -> provider method name
SNAPSHOT_KINDS: dict[str, str] = {
    "market": "market_snapshot",
    "quant_signals": "quant_signals",
    "risk": "risk_status",
    "macro": "macro_snapshot",
    "calendar": "calendar",
    "backtest_stats": "backtest_stats",
    "positions": "positions",
}

_UNAVAILABLE_NOTE = "not provided by the data provider"


@runtime_checkable
class DeskDataProvider(Protocol):
    """Point-in-time snapshots for the desk. All methods return JSON-serialisable dicts."""

    def as_of(self, now: pd.Timestamp) -> str:
        """Decision time as presented to the agents (shifted when anonymised)."""
        ...

    def market_snapshot(self, now: pd.Timestamp) -> dict[str, Any]: ...

    def quant_signals(self, now: pd.Timestamp) -> dict[str, Any]: ...

    def risk_status(self, now: pd.Timestamp) -> dict[str, Any]: ...

    def macro_snapshot(self, now: pd.Timestamp) -> dict[str, Any]: ...

    def calendar(self, now: pd.Timestamp) -> dict[str, Any]: ...

    def backtest_stats(self, now: pd.Timestamp) -> dict[str, Any]: ...

    def positions(self, now: pd.Timestamp) -> dict[str, Any]: ...


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tz is None:
        raise ValueError(f"decision time must be tz-aware (UTC), got {ts!r}")
    return t.tz_convert("UTC")


# ======================================================================================
# Static provider
# ======================================================================================
class StaticDeskDataProvider:
    """Serves fixed snapshots keyed by kind (see :data:`SNAPSHOT_KINDS`).

    ``snapshots`` values may be dicts or callables ``now -> dict`` (e.g. to read live risk
    state lazily). Missing kinds return ``{"available": False, ...}`` so the agents can
    report the gap instead of guessing.
    """

    def __init__(self, snapshots: Mapping[str, Mapping[str, Any] | Callable[[pd.Timestamp], Mapping[str, Any]]]
                 | None = None) -> None:
        snapshots = dict(snapshots or {})
        unknown = sorted(set(snapshots) - set(SNAPSHOT_KINDS))
        if unknown:
            raise KeyError(f"unknown snapshot kinds {unknown}; expected {sorted(SNAPSHOT_KINDS)}")
        self._snapshots = snapshots

    def update(self, kind: str, snapshot: Mapping[str, Any] | Callable[[pd.Timestamp], Mapping[str, Any]]) -> None:
        if kind not in SNAPSHOT_KINDS:
            raise KeyError(f"unknown snapshot kind {kind!r}")
        self._snapshots[kind] = snapshot

    def _get(self, kind: str, now: pd.Timestamp) -> dict[str, Any]:
        snap = self._snapshots.get(kind)
        if snap is None:
            return {"available": False, "note": _UNAVAILABLE_NOTE}
        value = snap(now) if callable(snap) else copy.deepcopy(snap)
        return to_jsonable(dict(value), float_sig=8)

    def as_of(self, now: pd.Timestamp) -> str:
        return _utc(now).isoformat()

    def market_snapshot(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._get("market", now)

    def quant_signals(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._get("quant_signals", now)

    def risk_status(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._get("risk", now)

    def macro_snapshot(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._get("macro", now)

    def calendar(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._get("calendar", now)

    def backtest_stats(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._get("backtest_stats", now)

    def positions(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._get("positions", now)


# ======================================================================================
# Historical (replay) provider
# ======================================================================================
_DEFAULT_YIELD_SERIES = frozenset({"us10y", "real10y", "breakeven10y", "fedfunds", "us2y", "us30y", "us5y"})
_ANON_EPOCH = pd.Timestamp("2101-01-03", tz="UTC")  # a Monday in a fictional future


def _pct(a: float, b: float) -> float | None:
    if not (math.isfinite(a) and math.isfinite(b)) or b <= 0 or a <= 0:
        return None
    return 100.0 * math.log(a / b)


def _float_series(s: pd.Series) -> pd.Series:
    """Element-wise float64 copy with every missing flavour (NaN, ``pd.NA``, None, junk) as NaN.

    Nullable extension dtypes (``Float64``/``Int64``) hold ``pd.NA``, on which ``float()``
    raises — one missing value at the decision bar would otherwise make a whole snapshot
    unavailable to the agents. Element-wise, so it cannot move information across time.
    """
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return pd.Series(vals, index=s.index, name=s.name)


def _float_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for i, col in enumerate(df.columns):
        out[col] = _float_series(df.iloc[:, i]).to_numpy()
    return out


class HistoricalDeskDataProvider:
    """Point-in-time snapshots from historical :class:`MarketData` for replay backtests.

    For a decision time ``now`` the provider uses bar ``t`` = the last bar whose
    ``available_at <= now`` (normally ``now == bars.available_at[t]``, the bar-close decision
    time of SPEC §1). Every statistic is computed from ``bars.iloc[:t+1]`` only; macro rows
    are filtered by their own ``available_at``; calendar outcomes (``actual``/``forecast``/
    ``surprise`` columns) are shown only for events at or before ``now`` (and, when
    anonymising, only as the surprise). Event times must be tz-aware: naive times are
    rejected rather than assumed UTC, because a naive New-York-local release time read as
    UTC would reveal the print 4-5 hours before its real release.

    Parameters
    ----------
    md              : market data (bars + optional macro dict + optional events frame).
    signals         : optional per-strategy forecast frame indexed like ``md.bars`` (columns
                      = strategy names). Must itself be causal (strategies' contract).
    combined        : optional combined forecast series indexed like ``md.bars``.
    vol             : optional causal annualised vol series (e.g. ``ewma_volatility``).
    backtest_stats  : dict or ``now -> dict``; the caller must ensure the stats were computed
                      only on data before ``now`` (e.g. walk-forward OOS up to the fold start).
    risk_status_fn / positions_fn : ``now -> dict`` hooks into the replay's risk manager and
                      simulator state. When anonymising, datetime values in their output (and
                      in ``backtest_stats``) are shifted like every other date, but prices and
                      dates written inside strings pass through — return scale-free values
                      (P&L in % of equity, no price levels) or the anonymisation leaks.
    anonymise       : shift dates and rebase prices (see module docstring for the caveat).
    date_shift_days : explicit shift, a whole number of weeks (enforced when anonymising, so
                      weekday/session structure is preserved); default maps the first bar into
                      year 2101.
    lookback_bars   : history used for statistics (>= 200 for the 200-bar average).
    """

    def __init__(
        self,
        md: MarketData,
        *,
        signals: pd.DataFrame | None = None,
        combined: pd.Series | None = None,
        vol: pd.Series | None = None,
        backtest_stats: Mapping[str, Any] | Callable[[pd.Timestamp], Mapping[str, Any]] | None = None,
        risk_status_fn: Callable[[pd.Timestamp], Mapping[str, Any]] | None = None,
        positions_fn: Callable[[pd.Timestamp], Mapping[str, Any]] | None = None,
        anonymise: bool = False,
        date_shift_days: int | None = None,
        lookback_bars: int = 250,
        recent_bars: int = 12,
        calendar_horizon_hours: float = 72.0,
        calendar_lookback_hours: float = 24.0,
        yield_series: frozenset[str] | set[str] = _DEFAULT_YIELD_SERIES,
    ) -> None:
        bars = md.bars
        if len(bars) == 0:
            raise ValueError("HistoricalDeskDataProvider needs at least one bar")
        self.md = md
        self._bars = bars
        self._avail = pd.DatetimeIndex(bars["available_at"]).tz_convert("UTC")
        if not self._avail.is_monotonic_increasing:
            raise ValueError("bars.available_at must be increasing")
        for name, obj in (("signals", signals), ("combined", combined), ("vol", vol)):
            if obj is not None and not obj.index.equals(bars.index):
                raise ValueError(f"{name} must be indexed exactly like md.bars")
        self.signals = None if signals is None else _float_frame(signals)
        self.combined = None if combined is None else _float_series(combined)
        self.vol = None if vol is None else _float_series(vol)
        self._backtest_stats = backtest_stats
        self._risk_fn = risk_status_fn
        self._positions_fn = positions_fn
        self.anonymise = anonymise
        if date_shift_days is None:
            date_shift_days = int(((_ANON_EPOCH - bars.index[0].normalize()).days // 7) * 7)
        if anonymise and int(date_shift_days) % 7 != 0:
            # Weekday/hour structure carries real information for gold (weekend gaps, sessions,
            # scheduled releases): a non-weekly shift would present a real Friday as another day.
            raise ValueError(f"date_shift_days must be a whole number of weeks, got {date_shift_days}")
        self.date_shift = pd.Timedelta(days=int(date_shift_days))
        self.lookback_bars = max(int(lookback_bars), 2)
        self.recent_bars = max(int(recent_bars), 1)
        self.calendar_horizon = pd.Timedelta(hours=calendar_horizon_hours)
        self.calendar_lookback = pd.Timedelta(hours=calendar_lookback_hours)
        self.yield_series = frozenset(yield_series)
        self._timeframe = bars.attrs.get("timeframe")

    # ---------------------------------------------------------------- time handling
    def bar_index_at(self, now: pd.Timestamp) -> int:
        """Index of the latest bar completed at ``now`` (``available_at <= now``)."""
        t = int(self._avail.searchsorted(_utc(now), side="right")) - 1
        if t < 0:
            raise LookupError(f"no completed bar at {now}")
        return t

    def _show_time(self, ts: pd.Timestamp) -> str:
        ts = _utc(ts)
        return (ts + self.date_shift).isoformat() if self.anonymise else ts.isoformat()

    def as_of(self, now: pd.Timestamp) -> str:
        return self._show_time(now)

    def _window(self, now: pd.Timestamp) -> tuple[int, pd.DataFrame]:
        t = self.bar_index_at(now)
        start = max(0, t - self.lookback_bars + 1)
        return t, self._bars.iloc[start: t + 1]

    def _envelope(self, now: pd.Timestamp, body: dict[str, Any]) -> dict[str, Any]:
        # envelope keys win: a caller-supplied field named "as_of" must not replace the
        # (possibly anonymised) decision time
        out = {**body, "as_of": self.as_of(now), "anonymised": self.anonymise}
        return to_jsonable(out, float_sig=6)

    def _hook_output(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Caller-hook output; when anonymising, datetime values are shifted like every other
        date (an ``entry_time`` in a position summary would otherwise reveal the real date).
        Dates inside free-text strings and price levels cannot be detected: keep hooks
        scale-free (see the class docstring)."""
        out = dict(data)
        return self._shift_datetimes(out) if self.anonymise else out

    def _shift_datetimes(self, obj: Any) -> Any:
        if obj is None or obj is pd.NaT:
            return obj
        if isinstance(obj, np.datetime64):
            return None if np.isnat(obj) else pd.Timestamp(obj) + self.date_shift
        if isinstance(obj, (pd.Timestamp, _dt.datetime)):
            return pd.Timestamp(obj) + self.date_shift
        if isinstance(obj, _dt.date):
            return obj + _dt.timedelta(days=self.date_shift.days)
        if isinstance(obj, Mapping):
            return {k: self._shift_datetimes(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._shift_datetimes(v) for v in obj]
        if isinstance(obj, pd.Series):
            return {str(k): self._shift_datetimes(v) for k, v in obj.items()}
        return obj

    # ---------------------------------------------------------------- snapshots
    def market_snapshot(self, now: pd.Timestamp) -> dict[str, Any]:
        t, w = self._window(now)
        close = w["close"].to_numpy(dtype=float)
        high = w["high"].to_numpy(dtype=float)
        low = w["low"].to_numpy(dtype=float)
        last = w.iloc[-1]
        c = float(close[-1])
        k = 100.0 / c if self.anonymise else 1.0  # price rebasing factor

        def px(x: float | None) -> float | None:
            return None if x is None or not math.isfinite(x) else x * k

        rets = {}
        for h in (1, 4, 24, 120):
            rets[f"{h}_bars"] = _pct(c, float(close[-1 - h])) if len(close) > h else None

        prev_close = np.r_[np.nan, close[:-1]]
        tr = np.nanmax(np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]), axis=0)
        atr14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float("nan")

        sma_dist = {}
        for n in (20, 50, 200):
            if len(close) >= n and math.isfinite(atr14) and atr14 > 0:
                sma_dist[f"{n}"] = (c - float(np.mean(close[-n:]))) / atr14
            else:
                sma_dist[f"{n}"] = None

        n_rng = min(20, len(close))
        hi20, lo20 = float(np.max(high[-n_rng:])), float(np.min(low[-n_rng:]))
        rng_pos = (c - lo20) / (hi20 - lo20) if hi20 > lo20 else None

        if self.vol is not None:
            vol_ann = float(self.vol.iloc[t])
        elif len(close) >= 21:
            lr = np.diff(np.log(close))
            bpy = infer_bars_per_year(w.index) if len(w) > 2 else 252 * 23
            vol_ann = float(np.std(lr[-min(len(lr), 100):], ddof=1) * math.sqrt(bpy))
        else:
            vol_ann = float("nan")

        spreads = w["spread"].to_numpy(dtype=float)
        spread_now = float(spreads[-1])
        recent = w.iloc[-self.recent_bars:]
        bar_open = self._bars.index[t]
        bar_close = self._avail[t]
        body = {
            "timeframe": self._timeframe,
            "bar_open_time": self._show_time(bar_open),
            "bar_close_time": self._show_time(bar_close),
            # staleness of the "latest" bar (weekends, holidays, feed gaps) made explicit
            "hours_since_bar_close": (_utc(now) - bar_close).total_seconds() / 3600.0,
            "bars_in_window": len(w),
            "last_bar": {
                "open": px(float(last["open"])), "high": px(float(last["high"])),
                "low": px(float(last["low"])), "close": px(c),
            },
            "price_basis": "rebased: decision-bar close = 100" if self.anonymise else "USD per troy ounce (mid)",
            "returns_pct": rets,
            "realised_vol_annualised": vol_ann if math.isfinite(vol_ann) else None,
            "atr_14": px(atr14),
            "atr_14_pct_of_price": 100.0 * atr14 / c if math.isfinite(atr14) else None,
            "distance_to_sma_in_atr": sma_dist,
            "range_20_bars": {"high": px(hi20), "low": px(lo20), "position_0_to_1": rng_pos},
            "spread": {
                "current": px(spread_now),
                "median_window": px(float(np.median(spreads))),
                "current_bps": 1e4 * spread_now / c,
            },
            "session": {"hour_utc": int(bar_open.hour), "weekday": bar_open.day_name()},
            "recent_closes": [px(float(x)) for x in recent["close"].to_numpy(dtype=float)],
        }
        return self._envelope(now, body)

    def quant_signals(self, now: pd.Timestamp) -> dict[str, Any]:
        t = self.bar_index_at(now)
        body: dict[str, Any] = {}
        if self.combined is not None:
            hist = self.combined.iloc[max(0, t - 9): t + 1].to_numpy(dtype=float)
            body["combined_forecast"] = float(self.combined.iloc[t])
            body["combined_history_last_10_bars"] = hist.tolist()
        if self.signals is not None and len(self.signals.columns):
            row = self.signals.iloc[t].astype(float)
            prev = self.signals.iloc[max(0, t - 5)].astype(float)
            body["strategies"] = {
                str(name): {"forecast": float(row[name]), "forecast_5_bars_ago": float(prev[name])}
                for name in self.signals.columns
            }
            vals = row.to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                ref = body.get("combined_forecast", float(np.mean(vals)))
                sign = np.sign(ref) if math.isfinite(ref) else 0.0  # NaN combined: no reference
                body["dispersion_std"] = float(np.std(vals))
                body["share_agreeing_with_combined"] = (
                    float(np.mean(np.sign(vals) == sign)) if sign != 0 else None
                )
        if not body:
            return self._envelope(now, {"available": False, "note": _UNAVAILABLE_NOTE})
        return self._envelope(now, body)

    def macro_snapshot(self, now: pd.Timestamp) -> dict[str, Any]:
        now = _utc(now)
        series: dict[str, Any] = {}
        for name, frame in sorted((self.md.macro or {}).items()):
            if frame is None or "available_at" not in frame.columns or "value" not in frame.columns:
                continue
            avail = pd.DatetimeIndex(frame["available_at"]).tz_convert("UTC")
            mask = np.asarray(avail <= now)
            # Point-in-time rows with a value, in the order they became available (storage
            # order is not trusted; the stable sort keeps observation order within ties).
            pit = pd.DataFrame({"value": _float_series(frame["value"]).to_numpy()[mask], "avail": avail[mask]},
                               index=frame.index[mask])
            pit = pit.loc[np.isfinite(pit["value"].to_numpy())].sort_values("avail", kind="stable")
            if pit.empty:
                continue
            is_yield = name in self.yield_series
            v = pit["value"].to_numpy()
            entry: dict[str, Any] = {"kind": "yield_pct" if is_yield else "price"}
            for h in (1, 5, 20):
                if len(v) > h:
                    entry[f"change_{h}obs"] = (
                        100.0 * (v[-1] - v[-1 - h]) if is_yield else _pct(float(v[-1]), float(v[-1 - h]))
                    )
                else:
                    entry[f"change_{h}obs"] = None
            entry["change_units"] = "bp" if is_yield else "log % change"
            tail = v[-250:]
            sd = float(np.std(tail, ddof=1)) if len(tail) > 20 else float("nan")
            entry["zscore_250obs"] = (float(v[-1]) - float(np.mean(tail))) / sd if sd > 0 else None
            if not self.anonymise:
                entry["level"] = float(v[-1])
                entry["observation_date"] = pd.Timestamp(pit.index[-1]).isoformat()
            # staleness of the value actually shown (not of a later row whose value is missing)
            entry["hours_since_available"] = (now - pit["avail"].iloc[-1]).total_seconds() / 3600.0
            series[str(name)] = entry
        if not series:
            return self._envelope(now, {"available": False, "note": "no macro series available at this time"})
        return self._envelope(now, {"series": series})

    def calendar(self, now: pd.Timestamp) -> dict[str, Any]:
        now = _utc(now)
        ev = self.md.events
        if ev is None or len(ev) == 0 or "time" not in ev.columns:
            return self._envelope(now, {"available": False, "note": "no event calendar provided"})
        times = pd.DatetimeIndex(ev["time"])
        if times.tz is None:
            raise ValueError("event 'time' column must be tz-aware UTC (naive times are ambiguous and "
                             "could reveal releases early)")
        times = times.tz_convert("UTC")
        upcoming_mask = np.asarray((times > now) & (times <= now + self.calendar_horizon))
        recent_mask = np.asarray((times <= now) & (times >= now - self.calendar_lookback))
        outcome_cols = [c for c in ("actual", "forecast", "previous", "surprise") if c in ev.columns]

        def row_view(i: int, *, released: bool) -> dict[str, Any]:
            r = ev.iloc[i]
            out = {
                "name": str(r.get("name", "")),
                "currency": r.get("currency"),
                "importance": int(r["importance"]) if "importance" in ev.columns and pd.notna(r["importance"]) else None,
                "time": self._show_time(times[i]),
            }
            if released:
                out["hours_since"] = (now - times[i]).total_seconds() / 3600.0
                # outcomes are known only from release time onward
                if self.anonymise:
                    if outcome_cols:
                        out.update(self._anonymised_outcome(r, outcome_cols))
                else:
                    for c in outcome_cols:
                        out[c] = r[c]
            else:
                out["hours_until"] = (times[i] - now).total_seconds() / 3600.0
            return out

        upcoming = [row_view(i, released=False) for i in np.flatnonzero(upcoming_mask)]
        recent = [row_view(i, released=True) for i in np.flatnonzero(recent_mask)]
        hi = [e for e in upcoming if (e.get("importance") or 0) >= 3]
        body = {
            "horizon_hours": self.calendar_horizon.total_seconds() / 3600.0,
            "upcoming": upcoming,
            "recently_released": recent,
            "next_high_importance_hours": min((e["hours_until"] for e in hi), default=None),
        }
        return self._envelope(now, body)

    @staticmethod
    def _anonymised_outcome(row: pd.Series, outcome_cols: list[str]) -> dict[str, Any]:
        """Released-event outcome without identifying levels.

        An exact print ("CPI actual 9.1", "NFP 263k") pins down the real month as surely as a
        date would, and would let the model recall what gold did next — the very look-ahead
        anonymisation exists to block. Only the surprise (actual - forecast; a given
        ``surprise`` column wins) is shown, which is what moves gold on the release.
        """
        def num(c: str) -> float:
            if c not in outcome_cols:
                return float("nan")
            try:
                return float(row[c])
            except (TypeError, ValueError):
                return float("nan")

        surprise = num("surprise")
        if not math.isfinite(surprise):
            surprise = num("actual") - num("forecast")
        return {"surprise": surprise if math.isfinite(surprise) else None, "outcome_levels_withheld": True}

    def backtest_stats(self, now: pd.Timestamp) -> dict[str, Any]:
        src = self._backtest_stats
        if src is None:
            return self._envelope(now, {"available": False, "note": _UNAVAILABLE_NOTE})
        data = src(now) if callable(src) else src
        return self._envelope(now, self._hook_output(data))

    def risk_status(self, now: pd.Timestamp) -> dict[str, Any]:
        if self._risk_fn is None:
            return self._envelope(now, {"available": False, "note": "risk manager state not wired into this provider"})
        return self._envelope(now, self._hook_output(self._risk_fn(now)))

    def positions(self, now: pd.Timestamp) -> dict[str, Any]:
        if self._positions_fn is None:
            return self._envelope(now, {"available": False, "note": "position state not wired into this provider"})
        return self._envelope(now, self._hook_output(self._positions_fn(now)))
