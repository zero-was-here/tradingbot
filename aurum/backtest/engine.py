"""Backtest engine: forecast -> sizer -> risk manager -> ExecutionSimulator (SPEC §8).

The engine is deliberately thin. All PnL comes from
:class:`aurum.execution.simulator.ExecutionSimulator`; sizing and risk come from the SAME
objects the live runner uses (``aurum.core.interfaces.PositionSizer`` / ``RiskManager``),
so what is backtested is what is traded (SPEC §0.2-0.3).

Point-in-time contract
----------------------
At the close of bar ``t`` (decision time ``available_at[t]``) the engine uses only:

* ``forecast[t]`` (the strategy contract guarantees it is causal),
* ``vol[t]`` — by default :func:`aurum.models.volatility.ewma_volatility` (causal),
* ``ATR[t]`` — Wilder ATR over bars ``<= t`` (distance of protective stops; the level is
  anchored at the entry fill, i.e. the mid open of ``t+1``, once the order has filled),
* ``close[t]``, ``spread[t]`` and the simulator's equity/position (marked at ``close[t]``),
* scheduled calendar events around ``available_at[t]`` (scheduled times are public in
  advance; *outcome* columns such as ``actual``/``surprise`` are stripped from upcoming
  events).

The order then fills at the OPEN of bar ``t+1`` inside the simulator. ``bars_per_year``
(used only to annualise the default vol) is inferred from the timestamps of the whole
sample — calendar density, not price information.

Loop per bar ``t = start .. end-1``::

    risk.on_bar(available_at[t], equity)            # update peak / day-start equity
    requested = sizer.target_lots(forecast[t], vol[t], equity, close[t], instrument,
                                  current_lots=position, drawdown=dd)
    order     = requested, or 0 while a post-stop cooldown blocks that direction
    decision  = risk.evaluate(RiskContext(target_lots=order, ...))  # only reduces (or halts)
    sim.step(decision.approved_lots, stop_distance=k*ATR[t] on entry, then fixed levels)

Risk interventions (approved != requested, halts, or any reasons returned) are recorded in
``BacktestResult.risk_events``. The engine additionally *enforces* the reduce-only rule:
an approval outside ``[min(0, requested), max(0, requested)]`` is clamped and logged.

Forecast hook (LLM-desk replay)
-------------------------------
``run_backtest(..., forecast_hook=fn, hook_every=k)`` calls
``fn(bar_index, decision_time, forecast, state)`` at the close of every ``k``-th bar of the
window (starting with its first bar) and uses the returned value — clipped to [-1, 1] — as
the forecast *before* sizing; between calls the last returned value is held (a desk decision
stands until the next cycle). ``state`` is a fresh dict with the book as seen at that close
(``equity``, ``position``, ``drawdown``, ``peak_equity``, ``halted``, ``price``, ``spread``,
``vol``, ``bar_time``, ``window_bar``). The hook sees only the past (it is called at the
decision time with the causal forecast) and its output goes through the SAME sizer and risk
manager, so it can shape exposure but never bypass risk: while the risk manager is halted
the approved position is 0 whatever the hook returns (SPEC §0.3, §10). Without a hook the
engine's arithmetic is unchanged.

Holding a *forecast* between calls is only right for overlays whose output is itself a
forecast. A policy expressed RELATIVE to the causal forecast (the LLM desk's ``overlay``
mode: scale toward zero or veto) must be re-applied to the current forecast at every bar,
otherwise a held value can outlive a reversal of the underlying signal (flip direction or
add risk). Such callers use ``hook_every=1`` and decide the cadence of their expensive
calls inside the hook (``aurum desk replay`` does).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aurum.backtest.metrics import compute_metrics as _compute_metrics
from aurum.backtest.result import BacktestResult
from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.interfaces import PositionSizer, RiskContext, RiskManager
from aurum.core.timeframes import infer_bars_per_year
from aurum.core.types import MarketData
from aurum.execution.costs import CostModel
from aurum.execution.simulator import ExecutionSimulator
from aurum.models.volatility import ewma_volatility

logger = logging.getLogger(__name__)

RISK_EVENT_COLUMNS: list[str] = [
    "time", "bar_time", "bar", "current", "requested", "approved", "halted", "reasons",
]
#: event-frame columns that reveal (or, like a revised ``previous`` or a late-updated
#: ``forecast``, may be revised at) the release and must not be visible before ``time``. This is
#: a superset of ``aurum.data.calendar.OUTCOME_COLUMNS`` (actual, forecast, previous); names are
#: matched case-insensitively and any column starting with ``actual``/``surprise`` is dropped.
OUTCOME_COLUMNS: tuple[str, ...] = ("actual", "forecast", "previous", "revised", "surprise",
                                    "outcome", "result", "deviation")


def _is_outcome_column(name: object) -> bool:
    low = str(name).strip().lower()
    return low in OUTCOME_COLUMNS or low.startswith(("actual", "surprise"))
_DEFAULT_VOL = 0.20
_EPS = 1e-9

StartEnd = int | str | pd.Timestamp | None


# ---- helpers -------------------------------------------------------------------------------------
def average_true_range(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's Average True Range (Wilder 1978, *New Concepts in Technical Trading Systems*).

    ``TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)`` (first bar:
    ``high - low``), smoothed with Wilder's recursion ``ATR_t = ATR_{t-1} + (TR_t -
    ATR_{t-1}) / n`` (an EWMA with ``alpha = 1/n``) seeded at ``TR_0``. Causal: row ``t`` uses
    bars ``<= t`` only; never NaN, so a stop can always be placed.
    """
    if n < 1:
        raise ValueError("ATR period must be >= 1")
    h = bars["high"].astype(float)
    lo = bars["low"].astype(float)
    pc = bars["close"].astype(float).shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1, skipna=True)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean().rename("atr")


def _unpack(md: MarketData | pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if isinstance(md, MarketData):
        return md.bars, md.events
    if isinstance(md, pd.DataFrame):
        return md, None
    raise TypeError(f"md must be MarketData or a bars DataFrame, got {type(md).__name__}")


def _bound(index: pd.DatetimeIndex, value: StartEnd, *, is_end: bool) -> int:
    n = len(index)
    if value is None:
        return n - 1 if is_end else 0
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        pos = int(value)
        if pos < 0:
            pos += n
        if not 0 <= pos < n:
            raise IndexError(f"{'end' if is_end else 'start'}={value} outside bars")
        return pos
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    if is_end:
        return int(index.searchsorted(ts, side="right")) - 1
    return int(index.searchsorted(ts, side="left"))


def _align(values: pd.Series | float | np.ndarray | None, index: pd.DatetimeIndex, name: str,
           *, fill: float | None, ffill: bool) -> np.ndarray | None:
    """Align a per-bar input onto ``index`` (label-based for Series, positional for arrays)."""
    if values is None:
        return None
    if isinstance(values, (int, float, np.floating, np.integer)) and not isinstance(values, bool):
        return np.full(len(index), float(values))
    if isinstance(values, pd.Series):
        s = values.astype(float)
        if not s.index.equals(index):
            s = s.reindex(index)
            if s.notna().sum() == 0 and len(index):
                raise ValueError(f"{name} index does not overlap the bars index")
        if ffill:
            s = s.ffill()  # causal: carries the last known value forward only
        arr = s.to_numpy(dtype=float, copy=True)  # writable (pandas CoW arrays are read-only)
    else:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (len(index),):
            raise ValueError(f"{name} must have one value per bar ({len(index)}), got {arr.shape}")
        arr = arr.copy()
    if fill is not None:
        bad = ~np.isfinite(arr)
        if bad.any():
            logger.debug("%s: %d non-finite values replaced by %s", name, int(bad.sum()), fill)
            arr[bad] = fill
    return arr


class _EventWindows:
    """Per-bar calendar windows for RiskContext, precomputed with ``searchsorted``.

    upcoming(t): events with ``now <= time <= now + horizon`` (outcome columns removed —
    scheduled times are public, results are not); recent(t): ``now - horizon <= time < now``.
    Slices are cached, so the per-bar cost is O(1) away from events.
    """

    def __init__(self, events: pd.DataFrame, decision_ns: np.ndarray, horizon_ns: int) -> None:
        if "time" not in events.columns:
            raise ValueError("events frame needs a 'time' column (UTC scheduled release)")
        times = pd.DatetimeIndex(events["time"])
        if times.tz is None:
            logger.warning("events 'time' is tz-naive; assuming UTC")
            times = times.tz_localize("UTC")
        ns = times.tz_convert("UTC").as_unit("ns").asi8
        order = np.argsort(ns, kind="stable")
        ev = events.iloc[order].reset_index(drop=True)
        ns = ns[order]
        self._all = ev
        self._public = ev.drop(columns=[c for c in ev.columns if _is_outcome_column(c)])
        self._up_lo = np.searchsorted(ns, decision_ns, side="left")
        self._up_hi = np.searchsorted(ns, decision_ns + horizon_ns, side="right")
        self._rc_lo = np.searchsorted(ns, decision_ns - horizon_ns, side="left")
        self._empty_up = self._public.iloc[0:0]
        self._empty_rc = self._all.iloc[0:0]
        self._cache: dict[tuple[str, int, int], pd.DataFrame] = {}

    def _get(self, kind: str, lo: int, hi: int) -> pd.DataFrame:
        if lo >= hi:
            return self._empty_up if kind == "u" else self._empty_rc
        key = (kind, lo, hi)
        frame = self._cache.get(key)
        if frame is None:
            src = self._public if kind == "u" else self._all
            frame = src.iloc[lo:hi]
            self._cache[key] = frame
        return frame

    def upcoming(self, t: int) -> pd.DataFrame:
        return self._get("u", int(self._up_lo[t]), int(self._up_hi[t]))

    def recent(self, t: int) -> pd.DataFrame:
        return self._get("r", int(self._rc_lo[t]), int(self._up_lo[t]))


def _clamp_reduce_only(requested: float, approved: float) -> tuple[float, bool]:
    lo, hi = min(0.0, requested), max(0.0, requested)
    if approved < lo - _EPS or approved > hi + _EPS or not math.isfinite(approved):
        if not math.isfinite(approved):
            return 0.0, True
        return min(max(approved, lo), hi), True
    return approved, False


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _positive_or_none(x: float) -> float | None:
    """A usable protective distance (finite, > 0) or None (e.g. ATR 0 on a flat series)."""
    return x if (math.isfinite(x) and x > 0.0) else None


# ---- core loop -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Window:
    """The backtest window handed to a decision factory (positions relative to the window)."""

    lo: int                       # first bar of the window in the full bars frame
    hi: int                       # last bar (inclusive)
    bars: pd.DataFrame            # bars_full.iloc[lo:hi+1]
    vol: list[float]              # causal vol forecast per window bar
    close: list[float]            # mid close per window bar
    forecast: list[float] | None = None   # per-bar forecast used for reporting (writable)
    state: dict[str, Any] | None = None   # live loop state (risk halted flag, equity peak)


DecideFn = Callable[[int, float, float, float], float]
#: ``hook(bar_index, decision_time, forecast, state) -> forecast`` (see module docstring).
ForecastHook = Callable[[int, pd.Timestamp, float, dict], float]


def _simulate(
    md: MarketData | pd.DataFrame,
    make_decide: Callable[[_Window], DecideFn],
    *,
    risk: RiskManager | None,
    instrument: Instrument,
    costs: CostModel | None,
    initial_equity: float,
    vol: pd.Series | float | np.ndarray | None,
    forecast: np.ndarray | None,
    stop_atr_mult: float | None,
    take_profit_atr_mult: float | None,
    atr_period: int,
    stop_cooldown_bars: int,
    start: StartEnd,
    end: StartEnd,
    bars_per_year: float | None,
    event_horizon_hours: float,
    compute_metrics: bool,
    meta: dict[str, Any],
) -> BacktestResult:
    """Shared bar loop for :func:`run_backtest` and :func:`run_target_lots`.

    ``make_decide(window)`` returns ``decide(t, equity, current_lots, drawdown)`` giving the
    requested lots at the close of window bar ``t``; ``forecast`` (full-length array) is only
    used for reporting and ``RiskContext.extra``.
    """
    bars_full, events = _unpack(md)
    if len(bars_full) < 2:
        raise ValueError("need at least two bars to backtest")
    for name, k in (("stop_atr_mult", stop_atr_mult), ("take_profit_atr_mult", take_profit_atr_mult)):
        if k is not None and not (math.isfinite(k) and k > 0):
            raise ValueError(f"{name} must be a positive number or None")
    if stop_cooldown_bars < 0:
        raise ValueError("stop_cooldown_bars must be >= 0")
    full_idx = pd.DatetimeIndex(bars_full.index)
    lo = _bound(full_idx, start, is_end=False)
    hi = _bound(full_idx, end, is_end=True)
    if hi - lo < 1:
        raise ValueError(f"backtest window [{start}, {end}] contains fewer than two bars")

    # Causal per-bar inputs, computed on the FULL history so the window start has warm-up.
    # ``bpy`` (realised density) is only used for ex-post reporting; the sizing vol uses the
    # nominal timeframe constant so decisions never depend on the density of future data.
    bpy = bars_per_year or infer_bars_per_year(full_idx)
    if vol is None:
        vol_full = ewma_volatility(bars_full["close"].astype(float), bars_per_year=bars_per_year).to_numpy(float)
    else:
        vol_full = _align(vol, full_idx, "vol", fill=_DEFAULT_VOL, ffill=True)
    protective = stop_atr_mult is not None or take_profit_atr_mult is not None
    atr_full = average_true_range(bars_full, atr_period).to_numpy(float) if protective else None

    bars = bars_full.iloc[lo:hi + 1]
    sl = slice(lo, hi + 1)
    vol_arr = vol_full[sl].tolist()
    atr_arr = atr_full[sl].tolist() if atr_full is not None else None
    fc_arr = forecast[sl].tolist() if forecast is not None else None
    close = bars["close"].to_numpy(dtype=float).tolist()
    loop_state: dict[str, Any] = {"halted": False, "peak_equity": float(initial_equity)}
    decide = make_decide(_Window(lo=lo, hi=hi, bars=bars, vol=vol_arr, close=close, forecast=fc_arr,
                                 state=loop_state))

    sim = ExecutionSimulator(bars, instrument, costs, initial_equity)
    n = sim.n_bars
    spread = bars["spread"].to_numpy(dtype=float).tolist()
    times = list(bars.index)
    dtimes = list(pd.DatetimeIndex(bars["available_at"]))
    windows: _EventWindows | None = None
    if risk is not None and events is not None and len(events):
        dns = pd.DatetimeIndex(bars["available_at"]).tz_convert("UTC").as_unit("ns").asi8
        windows = _EventWindows(events, dns, int(event_horizon_hours * 3600 * 10**9))

    requested_arr = np.full(n, np.nan)
    risk_rows: list[dict[str, Any]] = []
    stop_lvl: float | None = None
    tp_lvl: float | None = None
    cooldown_left = 0
    cooldown_side = 0
    n_clamped = 0
    peak = sim.equity
    round_lots = instrument.round_lots

    for t in range(n - 1):
        eq = sim.equity
        cur = sim.position
        if risk is not None:
            risk.on_bar(dtimes[t], eq)
        if eq > peak:
            peak = eq
        dd = max(0.0, 1.0 - eq / peak) if peak > 0 else 0.0
        loop_state["peak_equity"] = peak
        if risk is not None:
            loop_state["halted"] = bool(getattr(risk, "halted", loop_state["halted"]))

        req = float(decide(t, eq, cur, dd))
        if not math.isfinite(req):
            logger.debug("bar %d: non-finite requested lots -> hold %.2f", t, cur)
            req = cur
        req = round_lots(req)
        requested_arr[t] = req

        # Stop cooldown is part of the ORDER (strategy/execution side), so it is applied before
        # the risk manager: risk must evaluate the order that will actually be sent, otherwise
        # stateful limits (e.g. max_trades_per_day) count re-entries that never happen.
        reasons: list[str] = []
        order = req
        if cooldown_left > 0:
            if order != 0.0 and _sign(order) == cooldown_side:
                order = 0.0
                reasons.append(f"stop_cooldown ({cooldown_left} bars left)")
            cooldown_left -= 1

        approved = order
        halted = False
        if risk is not None:
            ctx = RiskContext(
                time=dtimes[t], equity=eq, current_lots=cur, target_lots=order, price=close[t],
                spread=spread[t], vol_ann=vol_arr[t], bar_index=t,
                upcoming_events=windows.upcoming(t) if windows is not None else None,
                recent_events=windows.recent(t) if windows is not None else None,
                extra={"drawdown": dd, "bar_time": times[t],
                       "forecast": fc_arr[t] if fc_arr is not None else None},
            )
            decision = risk.evaluate(ctx)
            halted = bool(decision.halted)
            loop_state["halted"] = halted
            reasons.extend(str(r) for r in decision.reasons)
            approved = 0.0 if halted else float(decision.approved_lots)
            approved, clamped = _clamp_reduce_only(order, approved)
            if clamped:
                n_clamped += 1
                reasons.append("engine: approval exceeded request; clamped (risk may only reduce)")
                if n_clamped <= 5:
                    logger.warning("bar %d: risk approved %.4f vs requested %.4f; clamped",
                                   t, float(decision.approved_lots), order)
            approved = round_lots(approved)

        changed = abs(approved - req) > _EPS
        if changed or halted or reasons:
            risk_rows.append({
                "time": dtimes[t], "bar_time": times[t], "bar": t, "current": cur,
                "requested": req, "approved": approved, "halted": halted,
                "reasons": "; ".join(reasons),
            })
        # Attribute the fill to risk only when risk (not the signal) caused the de-risking: a
        # halt, or a reduction of the current position that the request itself did not ask
        # for (a capped reversal is still a *signal* exit of the old position).
        risk_driven = halted or (
            changed and abs(approved) < abs(cur) - _EPS and _sign(req) == _sign(cur)
        )
        step_reason = "risk" if risk_driven else "signal"

        stop_dist = tp_dist = None
        if protective:
            s = _sign(approved)
            if s == 0:
                stop_lvl = tp_lvl = None
            elif s != _sign(cur):
                # Entering from flat or reversing: fresh levels at ENTRY -/+ k * ATR[t]. The
                # distance uses the ATR known at this close; the simulator anchors it at the
                # fill (the mid open of t+1), so the stop can never start on the wrong side
                # of the market after a gap. Levels then stay fixed for the position's life.
                a = atr_arr[t]
                stop_lvl = tp_lvl = None
                if stop_atr_mult is not None:
                    stop_dist = _positive_or_none(stop_atr_mult * a)
                if take_profit_atr_mult is not None:
                    tp_dist = _positive_or_none(take_profit_atr_mult * a)

        res = sim.step(approved, stop_price=stop_lvl, take_profit=tp_lvl, reason=step_reason,
                       stop_distance=stop_dist, take_profit_distance=tp_dist)
        if protective:
            if res.position == 0.0:
                stop_lvl = tp_lvl = None
            else:  # keep the levels that protected this bar for the rest of the position
                stop_lvl, tp_lvl = res.stop_price, res.take_profit
        if res.exit_reason == "stop" and stop_cooldown_bars > 0:
            cooldown_left = stop_cooldown_bars
            cooldown_side = _sign(res.position_open)

    if risk is not None:
        risk.on_bar(dtimes[n - 1], sim.equity)
    if n_clamped:
        logger.warning("risk manager approvals clamped on %d bars (reduce-only rule)", n_clamped)

    result = sim.result(compute_metrics=False)
    idx = bars.index
    result.target = pd.Series(requested_arr, index=idx, name="target")
    if fc_arr is not None:
        result.forecast = pd.Series(fc_arr, index=idx, name="forecast", dtype=float)
    result.risk_events = pd.DataFrame(risk_rows, columns=RISK_EVENT_COLUMNS)
    result.meta.update({
        "bars_per_year": float(bpy),
        "stop_atr_mult": stop_atr_mult,
        "take_profit_atr_mult": take_profit_atr_mult,
        "atr_period": atr_period if protective else None,
        "stop_cooldown_bars": stop_cooldown_bars,
        "risk": repr(risk) if risk is not None else None,
        "n_risk_events": len(risk_rows),
        "n_risk_clamped": n_clamped,
        "timeframe": bars_full.attrs.get("timeframe"),
    })
    result.meta.update(meta)
    if compute_metrics:
        result.metrics = _compute_metrics(result, bars_per_year=bpy)
    return result


# ---- public API ----------------------------------------------------------------------------------------
def run_backtest(
    md: MarketData | pd.DataFrame,
    forecast: pd.Series,
    *,
    sizer: PositionSizer,
    risk: RiskManager | None = None,
    instrument: Instrument = XAUUSD,
    costs: CostModel | None = None,
    initial_equity: float = 100_000.0,
    vol: pd.Series | float | None = None,
    stop_atr_mult: float | None = None,
    start: StartEnd = None,
    end: StartEnd = None,
    take_profit_atr_mult: float | None = None,
    atr_period: int = 14,
    stop_cooldown_bars: int = 0,
    bars_per_year: float | None = None,
    event_horizon_hours: float = 24.0,
    compute_metrics: bool = True,
    forecast_hook: ForecastHook | None = None,
    hook_every: int = 1,
) -> BacktestResult:
    """Backtest a forecast series through sizer, risk manager and the execution simulator.

    Parameters
    ----------
    md : ``MarketData`` (events used for risk blackouts) or a bars frame.
    forecast : forecast in [-1, 1] decided at each bar's close, indexed like the bars
        (reindexed by label; missing/NaN -> 0; clipped to [-1, 1]).
    sizer : ``PositionSizer`` (e.g. ``aurum.portfolio.sizing.VolTargetSizer``).
    risk : optional ``RiskManager``; called ``on_bar`` then ``evaluate`` at every close.
    vol : annualised vol forecast per bar (Series/scalar); default ``ewma_volatility(close)``.
        A Series is forward-filled (causal) and remaining gaps set to 0.20.
    stop_atr_mult : when a position is opened or reversed by the decision at the close of
        ``t``, protect it with a stop at ``entry -/+ k * ATR[t]``: the distance uses the ATR
        known at decision time and is anchored at the entry (mid open of ``t+1``, where the
        order fills). Levels stay fixed for the life of the position (no trailing).
        ``take_profit_atr_mult`` likewise for a take-profit (``entry +/- k * ATR[t]``).
    stop_cooldown_bars : after a stop-out, block re-entry in the same direction for this many
        decisions (recorded as ``stop_cooldown`` risk events). 0 = re-enter immediately.
    start, end : inclusive window (bar positions or timestamps). Inputs are computed on the
        full history first, so the window starts with warmed-up vol/ATR.
    forecast_hook : optional ``hook(bar_index, decision_time, forecast, state) -> forecast``
        called at the close of every ``hook_every``-th window bar (first bar included); the
        returned value (clipped to [-1, 1]; non-finite -> 0) replaces the forecast before
        sizing and is held until the next call. ``bar_index`` is the position in the FULL
        bars frame, ``decision_time`` is ``available_at`` of that bar, and ``state`` holds
        ``equity``, ``position``, ``drawdown``, ``peak_equity``, ``halted``, ``price``,
        ``spread``, ``vol``, ``bar_time`` and ``window_bar``. Used to replay the LLM desk
        through the SAME sizer and risk manager (it cannot bypass risk). ``result.forecast``
        then holds the forecasts actually used; ``result.meta["hook"]`` summarises the calls.
    """
    if isinstance(hook_every, bool) or int(hook_every) != hook_every or hook_every < 1:
        raise ValueError(f"hook_every must be a positive integer, got {hook_every!r}")
    if forecast_hook is not None and not callable(forecast_hook):
        raise TypeError("forecast_hook must be callable")
    bars_full, _ = _unpack(md)
    fc = _align(forecast, pd.DatetimeIndex(bars_full.index), "forecast", fill=0.0, ffill=False)
    assert fc is not None
    np.clip(fc, -1.0, 1.0, out=fc)
    hook_stats = {"calls": 0, "overrides": 0, "invalid": 0}

    def make_decide(w: _Window) -> DecideFn:
        f = fc[w.lo:w.hi + 1].tolist()
        v, c = w.vol, w.close

        if forecast_hook is None:
            def decide(t: int, equity: float, current: float, dd: float) -> float:
                return sizer.target_lots(f[t], v[t], equity, c[t], instrument,
                                         current_lots=current, drawdown=dd)

            return decide

        dtimes = list(pd.DatetimeIndex(w.bars["available_at"]))
        btimes = list(w.bars.index)
        spreads = w.bars["spread"].to_numpy(dtype=float).tolist()
        every = int(hook_every)
        loop = w.state if w.state is not None else {}
        held = [0.0]

        def decide_hooked(t: int, equity: float, current: float, dd: float) -> float:
            if t % every == 0:
                state = {
                    "equity": equity, "position": current, "drawdown": dd,
                    "peak_equity": loop.get("peak_equity", equity), "halted": bool(loop.get("halted", False)),
                    "price": c[t], "spread": spreads[t], "vol": v[t], "bar_time": btimes[t],
                    "window_bar": t,
                }
                out = forecast_hook(w.lo + t, dtimes[t], f[t], state)
                hook_stats["calls"] += 1
                try:
                    x = float(out)
                except (TypeError, ValueError):
                    x = math.nan
                if not math.isfinite(x):
                    hook_stats["invalid"] += 1
                    if hook_stats["invalid"] <= 5:
                        logger.warning("forecast_hook returned %r at bar %d; using 0 (flat)", out, w.lo + t)
                    x = 0.0
                held[0] = min(1.0, max(-1.0, x))
            fe = held[0]
            if fe != f[t]:
                hook_stats["overrides"] += 1
            if w.forecast is not None:
                w.forecast[t] = fe  # report (and show risk) the forecast actually used
            return sizer.target_lots(fe, v[t], equity, c[t], instrument,
                                     current_lots=current, drawdown=dd)

        return decide_hooked

    meta: dict[str, Any] = {"engine": "run_backtest", "sizer": repr(sizer)}
    result = _simulate(
        md, make_decide, risk=risk, instrument=instrument, costs=costs,
        initial_equity=initial_equity, vol=vol, forecast=fc, stop_atr_mult=stop_atr_mult,
        take_profit_atr_mult=take_profit_atr_mult, atr_period=atr_period,
        stop_cooldown_bars=stop_cooldown_bars, start=start, end=end, bars_per_year=bars_per_year,
        event_horizon_hours=event_horizon_hours, compute_metrics=compute_metrics,
        meta=meta,
    )
    if forecast_hook is not None:
        result.meta["hook"] = {"hook": repr(forecast_hook), "every": int(hook_every), **hook_stats}
    return result


def run_target_lots(
    md: MarketData | pd.DataFrame,
    target_lots: pd.Series | np.ndarray,
    *,
    risk: RiskManager | None = None,
    instrument: Instrument = XAUUSD,
    costs: CostModel | None = None,
    initial_equity: float = 100_000.0,
    vol: pd.Series | float | None = None,
    stop_atr_mult: float | None = None,
    start: StartEnd = None,
    end: StartEnd = None,
    take_profit_atr_mult: float | None = None,
    atr_period: int = 14,
    stop_cooldown_bars: int = 0,
    bars_per_year: float | None = None,
    event_horizon_hours: float = 24.0,
    compute_metrics: bool = True,
) -> BacktestResult:
    """Backtest a pre-sized lot path: ``target_lots[t]`` = signed lots wanted after the fill at
    the open of ``t+1`` (decided at the close of ``t``). NaN means "no decision, hold".

    Used by the RL evaluation, the LLM-desk replay and benchmarks; the optional ``risk``
    manager and protective stops behave exactly as in :func:`run_backtest`.
    """
    bars_full, _ = _unpack(md)
    tl = _align(target_lots, pd.DatetimeIndex(bars_full.index), "target_lots", fill=None, ffill=False)
    assert tl is not None

    def make_decide(w: _Window) -> DecideFn:
        path = tl[w.lo:w.hi + 1].tolist()

        def decide(t: int, equity: float, current: float, dd: float) -> float:
            x = path[t]
            return current if x != x else x  # NaN -> no decision: hold

        return decide

    return _simulate(
        md, make_decide, risk=risk, instrument=instrument, costs=costs, initial_equity=initial_equity,
        vol=vol, forecast=None, stop_atr_mult=stop_atr_mult,
        take_profit_atr_mult=take_profit_atr_mult, atr_period=atr_period,
        stop_cooldown_bars=stop_cooldown_bars, start=start, end=end, bars_per_year=bars_per_year,
        event_horizon_hours=event_horizon_hours, compute_metrics=compute_metrics,
        meta={"engine": "run_target_lots"},
    )


def buy_and_hold_benchmark(
    md: MarketData | pd.DataFrame,
    lots: float | None = None,
    notional: float | None = None,
    *,
    instrument: Instrument = XAUUSD,
    costs: CostModel | None = None,
    initial_equity: float = 100_000.0,
    start: StartEnd = None,
    end: StartEnd = None,
    frictionless: bool = False,
    compute_metrics: bool = True,
) -> BacktestResult:
    """Long-only benchmark: buy at the open after the first bar and hold to the end.

    Size is ``lots`` or, if given, ``notional`` USD converted at the first bar's close
    (default ``notional = initial_equity``, i.e. 1x leverage). The benchmark is run through
    the same simulator, so by default it pays spread, slippage, commission and the CFD swap
    — the like-for-like comparison for a CFD strategy. ``frictionless=True`` removes all
    costs and swaps (a pure price-return benchmark).
    """
    if lots is not None and notional is not None:
        raise ValueError("pass either lots or notional, not both")
    bars_full, _ = _unpack(md)
    full_idx = pd.DatetimeIndex(bars_full.index)
    lo = _bound(full_idx, start, is_end=False)
    if frictionless:
        import dataclasses

        costs = CostModel.zero()
        instrument = dataclasses.replace(instrument, swap_long_per_lot=0.0, swap_short_per_lot=0.0,
                                         commission_per_lot=0.0)
    if lots is None:
        notional = float(initial_equity if notional is None else notional)
        px0 = float(bars_full["close"].iloc[lo])
        lots = notional / (instrument.contract_size * px0)
    size = instrument.round_lots(float(lots))
    if size == 0.0:
        logger.warning("buy_and_hold_benchmark: size rounds to 0 lots")
    res = run_target_lots(md, pd.Series(size, index=full_idx), instrument=instrument, costs=costs,
                          initial_equity=initial_equity, start=start, end=end,
                          compute_metrics=compute_metrics)
    res.meta.update({"engine": "buy_and_hold_benchmark", "lots": size, "frictionless": frictionless})
    return res
