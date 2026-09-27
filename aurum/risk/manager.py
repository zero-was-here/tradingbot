"""Pre-trade risk manager (SPEC §7).

The risk manager is the last gate between any signal source (quant strategies, the RL
policy or the LLM desk) and the execution layer. Its single invariant is that it can only
*reduce* risk: the approved position always lies between 0 and the requested target
(same sign, smaller or equal magnitude), and when a "no new risk" rule fires it also never
exceeds the current position in the risk-increasing direction.

Rules, in evaluation order
    1. **Kill switch** (persistent): maximum drawdown from the equity peak, daily loss
       versus the day's starting equity, non-positive equity, a manual ``halt()`` or an
       unreadable state file. Sets ``halted=True`` → approved 0 (flatten). The flag is
       persisted to ``state_path`` (JSON, atomic write) and survives restarts until an
       operator calls ``reset_halt(confirm="RESET")``.
    2. **Exposure caps**: ``max_lots``, instrument ``max_lot``, ``max_leverage`` (notional /
       equity) and ``max_margin_utilisation`` (margin / equity) shrink the target toward 0.
    3. **Event blackout** around high-importance scheduled releases (NFP, CPI, FOMC):
       ``blackout_mode="flatten"`` flattens, ``"no_new_risk"`` only blocks new exposure.
       Scheduled release times are published in advance, so using *future* event times is
       point-in-time safe (SPEC §3.5).
    4. **No-new-risk guards**: spread above ``max_spread``, stale data, ``max_trades_per_day``
       reached, invalid price. Positions may still be reduced or closed.
    5. Rounding toward zero on the lot grid.

Every intervention appends a human-readable reason to ``RiskDecision.reasons`` and is
recorded in :meth:`StandardRiskManager.events_frame` (for ``BacktestResult.risk_events``).

Daily loss is measured against the equity at the start of the trading day: UTC midnight
(``daily_reset="utc"``) or the broker rollover hour (``daily_reset="rollover"``, using
``instrument.rollover_hour_utc``). When a decision falls exactly on the boundary its own
equity is the day's start; otherwise the last equity observed before the boundary is used
(i.e. the equity at the open of the day's first bar). A mark exactly on the boundary is
also the CLOSE of the previous day's last bar, so it is first checked against the previous
day's start (the kill fires even if the breach happened in the final bar of the day, which
on D1 bars is the whole day).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import numbers
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.interfaces import RiskContext, RiskDecision

logger = logging.getLogger(__name__)

STATE_VERSION = 1
_EPS = 1e-9


@dataclass
class RiskLimits:
    """Hard limits enforced by :class:`StandardRiskManager` (fractions of equity unless noted).

    max_lots                 : absolute cap on |position| (lots); None = only the instrument cap.
    max_leverage             : cap on |notional| / equity; None = no cap.
    max_daily_loss           : kill when equity falls this fraction below the day's start.
    max_drawdown             : kill when equity falls this fraction below its peak.
    max_spread               : block NEW risk when the current spread (price units) exceeds it.
    event_blackout_before_min / event_blackout_after_min : window around scheduled events.
    event_min_importance     : events with importance >= this trigger the blackout.
    blackout_mode            : "no_new_risk" (default) or "flatten".
    max_trades_per_day       : after this many position changes in a day, only reductions.
    stale_data_seconds       : block new risk when ``ctx.data_age_seconds`` exceeds this.
    max_margin_utilisation   : cap on required margin / equity.
    daily_reset              : "utc" (midnight UTC) or "rollover" (instrument.rollover_hour_utc).
    daily_loss_persistent    : True (SPEC default) = a daily-loss halt is a hard kill that needs
                               ``reset_halt``; False = it clears automatically at the next day.
    event_lookahead_min      : extra minutes added before each event; set to the bar length on
                               coarse timeframes so an event inside the NEXT holding bar
                               triggers the blackout at the preceding close.
    """

    max_lots: float | None = None
    max_leverage: float | None = 3.0
    max_daily_loss: float | None = 0.03
    max_drawdown: float | None = 0.20
    max_spread: float | None = None
    event_blackout_before_min: float = 30.0
    event_blackout_after_min: float = 30.0
    event_min_importance: int = 3
    blackout_mode: str = "no_new_risk"
    max_trades_per_day: int | None = None
    stale_data_seconds: float | None = None
    max_margin_utilisation: float | None = 0.5
    daily_reset: str = "utc"
    daily_loss_persistent: bool = True
    event_lookahead_min: float = 0.0

    def __post_init__(self) -> None:
        def _pos(name: str, v: float | None) -> None:
            # numbers.Real accepts numpy scalars (e.g. limits loaded from a YAML/array config).
            if v is not None and (isinstance(v, bool) or not isinstance(v, numbers.Real) or not v > 0):
                raise ValueError(f"RiskLimits.{name} must be positive or None, got {v!r}")

        for name in ("max_lots", "max_leverage", "max_spread", "stale_data_seconds",
                     "max_margin_utilisation"):
            _pos(name, getattr(self, name))
        for name in ("max_daily_loss", "max_drawdown"):
            v = getattr(self, name)
            if v is not None and not (0.0 < v < 1.0):
                raise ValueError(f"RiskLimits.{name} must be in (0, 1) or None, got {v!r}")
        if self.max_trades_per_day is not None and self.max_trades_per_day < 0:
            raise ValueError("max_trades_per_day must be >= 0 or None")
        if self.blackout_mode not in ("no_new_risk", "flatten"):
            raise ValueError("blackout_mode must be 'no_new_risk' or 'flatten'")
        if self.daily_reset not in ("utc", "rollover"):
            raise ValueError("daily_reset must be 'utc' or 'rollover'")
        if self.event_blackout_before_min < 0 or self.event_blackout_after_min < 0 or self.event_lookahead_min < 0:
            raise ValueError("event blackout windows must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskState:
    """Mutable, persistable equity-path state."""

    halted: bool = False
    halt_kind: str | None = None        # "drawdown" | "daily_loss" | "equity" | "manual" | "state_file"
    halt_reason: str | None = None
    halted_at: str | None = None
    peak_equity: float | None = None
    day_key: str | None = None
    day_start_equity: float | None = None
    trades_today: int = 0
    last_time: str | None = None
    last_equity: float | None = None
    version: int = STATE_VERSION


_STATE_TYPES: dict[str, str] = {
    "halted": "bool",
    "halt_kind": "str",
    "halt_reason": "str",
    "halted_at": "str",
    "peak_equity": "float",
    "day_key": "str",
    "day_start_equity": "float",
    "trades_today": "int",
    "last_time": "str",
    "last_equity": "float",
    "version": "int",
}


def _state_from_json(data: object) -> RiskState:
    """Validate a decoded state file strictly; raise ValueError on anything unexpected.

    The state file guards the kill switch, so a hand-edited or partially written file must
    never be half-trusted: a JSON ``"false"`` string would be truthy, and a non-numeric
    equity would only crash later, mid-session. Any violation makes the caller start HALTED.
    """
    if not isinstance(data, dict):
        raise ValueError(f"state must be a JSON object, got {type(data).__name__}")
    known = {f.name for f in fields(RiskState)}
    out: dict[str, object] = {}
    for key, value in data.items():
        if key not in known:
            continue
        kind = _STATE_TYPES[key]
        if value is None:
            if key in ("halted", "trades_today", "version"):
                raise ValueError(f"state field {key!r} may not be null")
            out[key] = None
        elif kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"state field {key!r} must be a boolean, got {value!r}")
            out[key] = value
        elif kind == "str":
            if not isinstance(value, str):
                raise ValueError(f"state field {key!r} must be a string, got {value!r}")
            out[key] = value
        elif kind == "int":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"state field {key!r} must be a non-negative integer, got {value!r}")
            out[key] = value
        else:  # float
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"state field {key!r} must be a finite number, got {value!r}")
            out[key] = float(value)
    if "halted" not in out:
        raise ValueError("state file lacks the 'halted' flag")
    if out.get("last_time") is not None:
        _utc(str(out["last_time"]))  # raises ValueError if it is not a timestamp
    return RiskState(**out)


def _utc(t: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ns(ts: pd.Timestamp) -> int:
    return int(ts.as_unit("ns").value)


def no_new_risk(approved: float, current: float) -> float:
    """Clamp ``approved`` so it does not add exposure relative to ``current``.

    Same side: ``min(|approved|, |current|)``; a reversal or a new position from flat: 0
    (the existing position may be closed but not reversed/opened).
    """
    if approved == 0.0 or current == 0.0 or (approved > 0) != (current > 0):
        return 0.0
    return math.copysign(min(abs(approved), abs(current)), approved)


def toward_zero(approved: float, target: float) -> float:
    """Enforce the core invariant: approved lies between 0 and target (inclusive)."""
    if target == 0.0 or approved == 0.0 or (approved > 0) != (target > 0):
        return 0.0
    if abs(approved) > abs(target):
        return target
    return approved


class StandardRiskManager:
    """Implements :class:`aurum.core.interfaces.RiskManager` (``evaluate`` + ``on_bar``).

    Parameters
    ----------
    limits     : :class:`RiskLimits` (defaults if None).
    instrument : contract specification (lot grid, contract size, margin rate).
    state_path : optional JSON file for the kill-switch/equity state (live trading). When it
                 exists at construction the state is restored; if it cannot be parsed the
                 manager starts HALTED (fail-safe).
    events     : optional economic-calendar frame (columns ``time``, ``importance``, optional
                 ``name``) checked in addition to ``ctx.upcoming_events``/``ctx.recent_events``.
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        instrument: Instrument = XAUUSD,
        state_path: str | os.PathLike | None = None,
        *,
        events: pd.DataFrame | None = None,
    ) -> None:
        self.limits = limits if limits is not None else RiskLimits()
        self.instrument = instrument
        self.state_path = Path(state_path) if state_path is not None else None
        self.state = RiskState()
        self._log: list[dict] = []
        self._saved: dict | None = None
        self._ev_ns = np.empty(0, dtype=np.int64)
        self._ev_desc: list[str] = []
        self._frame_cache: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
        self._last_time_cache: tuple[str, pd.Timestamp] | None = None
        self.set_events(events)
        if self.state_path is not None and self.state_path.exists():
            self._load()

    # ------------------------------------------------------------------------------------
    # properties / persistence
    # ------------------------------------------------------------------------------------
    @property
    def halted(self) -> bool:
        return self.state.halted

    @property
    def halt_reason(self) -> str | None:
        return self.state.halt_reason

    def _load(self) -> None:
        assert self.state_path is not None
        try:
            self.state = _state_from_json(json.loads(self.state_path.read_text()))
            self._saved = asdict(self.state)
            if self.state.halted:
                logger.warning("Risk state restored HALTED from %s: %s", self.state_path, self.state.halt_reason)
        except (OSError, ValueError, TypeError) as exc:
            logger.error("Unreadable risk state file %s (%s): starting HALTED", self.state_path, exc)
            self.state = RiskState(
                halted=True,
                halt_kind="state_file",
                halt_reason=f"unreadable risk state file {self.state_path}: {exc}",
                halted_at=pd.Timestamp.now(tz="UTC").isoformat(),
            )

    def _persist(self) -> None:
        if self.state_path is None:
            return
        data = asdict(self.state)
        if data == self._saved:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, self.state_path)
        self._saved = data

    # ------------------------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------------------------
    def set_events(self, events: pd.DataFrame | None) -> None:
        """Register a calendar frame (``time``, ``importance``[, ``name``]) for blackouts."""
        if events is None or len(events) == 0:
            self._ev_ns = np.empty(0, dtype=np.int64)
            self._ev_desc = []
            return
        if "time" not in events.columns:
            raise KeyError("events frame needs a 'time' column (UTC scheduled release)")
        ev = events.copy()
        ev["_t"] = pd.to_datetime(ev["time"], utc=True)
        # Same rule as for context frames: non-numeric/missing importance counts as high
        # (fail safe), and string columns such as "3" are accepted.
        if "importance" in ev.columns:
            imp = pd.to_numeric(ev["importance"], errors="coerce").fillna(np.inf)
        else:
            imp = pd.Series(np.inf, index=ev.index)
        ev = ev.loc[(imp >= self.limits.event_min_importance) & ev["_t"].notna()]
        ev = ev.sort_values("_t", kind="stable")
        self._ev_ns = pd.DatetimeIndex(ev["_t"]).as_unit("ns").asi8.copy()
        self._ev_desc = [self._describe_event(row) for _, row in ev.iterrows()]

    @staticmethod
    def _describe_event(row: pd.Series) -> str:
        name = row.get("name", "event")
        imp = row.get("importance", "?")
        t = pd.Timestamp(row["_t"])
        return f"{name} at {t:%Y-%m-%d %H:%M} UTC (importance {imp})"

    def _blackout(self, t: pd.Timestamp, ctx: RiskContext) -> str | None:
        lim = self.limits
        before = pd.Timedelta(minutes=lim.event_blackout_before_min + lim.event_lookahead_min)
        after = pd.Timedelta(minutes=lim.event_blackout_after_min)
        lo, hi = _ns(t - after), _ns(t + before)
        if self._ev_ns.size:
            i = int(np.searchsorted(self._ev_ns, lo, side="left"))
            if i < self._ev_ns.size and self._ev_ns[i] <= hi:
                return self._ev_desc[i]
        for frame in (ctx.upcoming_events, ctx.recent_events):
            if frame is None or len(frame) == 0:
                continue
            times, eligible = self._parse_event_frame(frame)
            mask = eligible & (times >= lo) & (times <= hi)
            if mask.any():
                i = int(np.flatnonzero(mask)[0])
                row = frame.iloc[i].copy()
                row["_t"] = pd.Timestamp(int(times[i]), tz="UTC")
                return self._describe_event(row)
        return None

    def _parse_event_frame(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(event times in ns, eligible-by-importance mask), cached per frame object.

        The backtest engine hands the same cached slice objects to consecutive bars, so
        parsing once per object keeps ``evaluate`` at O(10 µs). Context frames are treated as
        immutable snapshots; the cache holds a reference, so object ids cannot be recycled.
        """
        hit = self._frame_cache.get(id(frame))
        if hit is not None and hit[0] is frame:
            return hit[1], hit[2]
        if "time" not in frame.columns:
            raise KeyError("event frames in RiskContext need a 'time' column")
        times = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8
        if "importance" in frame.columns:
            imp = pd.to_numeric(frame["importance"], errors="coerce").to_numpy(dtype=float)
            imp = np.where(np.isnan(imp), np.inf, imp)  # unknown importance: treat as high
        else:
            imp = np.full(len(frame), np.inf)
        eligible = imp >= self.limits.event_min_importance
        if len(self._frame_cache) >= 512:
            self._frame_cache.clear()
        self._frame_cache[id(frame)] = (frame, times, eligible)
        return times, eligible

    # ------------------------------------------------------------------------------------
    # equity path
    # ------------------------------------------------------------------------------------
    def _day(self, t: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        off = (
            pd.Timedelta(hours=self.instrument.rollover_hour_utc)
            if self.limits.daily_reset == "rollover"
            else pd.Timedelta(0)
        )
        label = (t - off).normalize()
        return label, label + off

    def _halt(self, kind: str, reason: str, t: pd.Timestamp) -> None:
        st = self.state
        st.halted = True
        st.halt_kind = kind
        st.halt_reason = reason
        st.halted_at = t.isoformat()
        logger.critical("RISK KILL SWITCH (%s) at %s: %s", kind, t, reason)
        self._log.append({"time": t, "requested": math.nan, "current": math.nan, "approved": 0.0,
                          "halted": True, "reasons": f"KILL SWITCH ({kind}): {reason}"})

    def _update(self, t: pd.Timestamp, equity: float) -> None:
        st = self.state
        lim = self.limits
        if not math.isfinite(equity):
            logger.warning("Risk manager received non-finite equity %s at %s; state not updated", equity, t)
            return
        last_t = self._last_time()
        if last_t is not None and t < last_t:
            # Out-of-order mark (e.g. an agent previewing with a stale timestamp, or a replay
            # restarted mid-stream). Rolling the day on it would re-base the day-start equity
            # and reset the trade budget, hiding a daily loss. Keep the current day and the
            # latest (time, equity) as the reference, but still run the kill checks on it.
            logger.warning("Risk manager got time %s earlier than the last mark %s; day not rolled", t, last_t)
            self._kill_checks(t, equity)
            return
        label, boundary = self._day(t)
        key = label.isoformat()
        if st.day_key != key:
            # A decision exactly ON the boundary marks the close of the previous day's last
            # bar (bars close at ``available_at``; e.g. every D1 bar, or the 23:00 H1 bar at
            # 00:00). That bar's P&L belongs to the day that is ending, so settle the old
            # day's loss against its own start BEFORE rolling over — otherwise the last bar of
            # every day (all of it on D1) would escape the daily-loss limit.
            if (
                t == boundary
                and not st.halted
                and st.day_key is not None
                and lim.max_daily_loss is not None
                and st.day_start_equity is not None
                and st.day_start_equity > 0
            ):
                day0 = float(st.day_start_equity)
                loss = 1.0 - equity / day0
                if loss >= lim.max_daily_loss - 1e-12:
                    self._halt(
                        "daily_loss",
                        f"daily loss {loss:.2%} vs day start {day0:,.2f} for trading day "
                        f"{st.day_key[:10]} (settled at its closing mark {t}) reached max_daily_loss "
                        f"{lim.max_daily_loss:.2%}",
                        t,
                    )
            start = equity if (t == boundary or st.last_equity is None) else float(st.last_equity)
            st.day_key = key
            st.day_start_equity = start
            st.trades_today = 0
            if st.halted and st.halt_kind == "daily_loss" and not lim.daily_loss_persistent:
                logger.warning("Daily-loss halt cleared at new trading day %s", key)
                st.halted, st.halt_kind, st.halt_reason, st.halted_at = False, None, None, None
        self._kill_checks(t, equity)
        st.last_time = t.isoformat()
        st.last_equity = equity

    def _last_time(self) -> pd.Timestamp | None:
        """Parsed ``state.last_time`` (cached: parsing an ISO string costs a few µs per bar)."""
        raw = self.state.last_time
        if raw is None:
            return None
        if self._last_time_cache is None or self._last_time_cache[0] != raw:
            try:
                self._last_time_cache = (raw, _utc(raw))
            except ValueError:
                logger.error("Unparseable last_time %r in risk state; ignoring it", raw)
                return None
        return self._last_time_cache[1]

    def _kill_checks(self, t: pd.Timestamp, equity: float) -> None:
        """Update the peak and fire the hard kills (equity <= 0, max drawdown, daily loss)."""
        st = self.state
        lim = self.limits
        st.peak_equity = equity if st.peak_equity is None else max(float(st.peak_equity), equity)
        if not st.halted:
            peak = float(st.peak_equity)
            day0 = st.day_start_equity
            if equity <= 0:
                self._halt("equity", f"equity {equity:,.2f} <= 0", t)
            elif lim.max_drawdown is not None and peak > 0 and 1.0 - equity / peak >= lim.max_drawdown - 1e-12:
                dd = 1.0 - equity / peak
                self._halt(
                    "drawdown",
                    f"drawdown {dd:.2%} from peak {peak:,.2f} reached max_drawdown {lim.max_drawdown:.2%}",
                    t,
                )
            elif (
                lim.max_daily_loss is not None
                and day0 is not None
                and day0 > 0
                and 1.0 - equity / day0 >= lim.max_daily_loss - 1e-12
            ):
                loss = 1.0 - equity / day0
                self._halt(
                    "daily_loss",
                    f"daily loss {loss:.2%} vs day start {day0:,.2f} reached max_daily_loss {lim.max_daily_loss:.2%}",
                    t,
                )

    def on_bar(self, time: pd.Timestamp, equity: float) -> None:
        """Update day-start equity, peak and kill checks — call once per bar close."""
        self._update(_utc(time), float(equity))
        self._persist()

    # ------------------------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------------------------
    def _exposure_cap(self, equity: float, price: float) -> tuple[float, str]:
        lim = self.limits
        inst = self.instrument
        caps: list[tuple[float, str]] = [(inst.max_lot, f"instrument max_lot {inst.max_lot:g}")]
        if lim.max_lots is not None:
            caps.append((lim.max_lots, f"max_lots {lim.max_lots:g}"))
        if math.isfinite(price) and price > 0 and math.isfinite(equity) and equity > 0:
            unit = inst.contract_size * price
            if lim.max_leverage is not None:
                caps.append((lim.max_leverage * equity / unit, f"max_leverage {lim.max_leverage:g}x"))
            if lim.max_margin_utilisation is not None and inst.margin_rate > 0:
                caps.append((
                    lim.max_margin_utilisation * equity / (unit * inst.margin_rate),
                    f"max_margin_utilisation {lim.max_margin_utilisation:.0%}",
                ))
        return min(caps, key=lambda c: c[0])

    def evaluate(self, ctx: RiskContext, *, commit: bool = True) -> RiskDecision:
        """Approve (or reduce) ``ctx.target_lots``. ``commit=False`` is a side-effect-free preview."""
        if not commit:
            saved_state, saved_log = copy.deepcopy(self.state), list(self._log)
            try:
                return self._evaluate(ctx, commit=False)
            finally:
                self.state, self._log = saved_state, saved_log
        decision = self._evaluate(ctx, commit=True)
        self._persist()
        return decision

    def _evaluate(self, ctx: RiskContext, *, commit: bool) -> RiskDecision:
        lim = self.limits
        t = _utc(ctx.time)
        equity = float(ctx.equity)
        price = float(ctx.price) if ctx.price is not None else math.nan
        current = float(ctx.current_lots) if math.isfinite(float(ctx.current_lots)) else 0.0
        target = float(ctx.target_lots)
        reasons: list[str] = []
        self._update(t, equity)

        if not math.isfinite(target):
            reasons.append(f"non-finite target lots ({target!r}) -> flatten")
            target = 0.0

        if self.state.halted:
            reasons.append(
                f"HALTED ({self.state.halt_kind}): {self.state.halt_reason}. Flatten; trading disabled "
                "until reset_halt(confirm='RESET')"
            )
            return self._record(t, ctx.target_lots, current, 0.0, True, reasons)

        approved = target
        # 2. exposure caps --------------------------------------------------------------
        cap, cap_desc = self._exposure_cap(equity, price)
        if abs(approved) > cap + _EPS:
            new = math.copysign(cap, approved)
            reasons.append(f"{cap_desc}: {approved:+.2f} -> {new:+.2f} lots")
            approved = new

        # 3. event blackout -------------------------------------------------------------
        event = self._blackout(t, ctx)
        no_new: list[str] = []
        if event is not None:
            if lim.blackout_mode == "flatten":
                if approved != 0.0:
                    reasons.append(f"event blackout (flatten): {event}: {approved:+.2f} -> +0.00 lots")
                    approved = 0.0
            else:
                no_new.append(f"event blackout: {event}")

        # 4. no-new-risk guards ---------------------------------------------------------
        spread = float(ctx.spread) if ctx.spread is not None else math.nan
        if lim.max_spread is not None:
            if not math.isfinite(spread):
                no_new.append("spread unknown")
            elif spread > lim.max_spread:
                no_new.append(f"spread {spread:.2f} > max_spread {lim.max_spread:.2f}")
        if lim.stale_data_seconds is not None and ctx.data_age_seconds is not None:
            age = float(ctx.data_age_seconds)
            if not math.isfinite(age):  # set but unknown (live feed error): fail safe
                no_new.append(f"data age unknown ({age!r})")
            elif age > lim.stale_data_seconds:
                no_new.append(f"stale data: age {age:.0f}s > {lim.stale_data_seconds:.0f}s")
        if lim.max_trades_per_day is not None and self.state.trades_today >= lim.max_trades_per_day:
            no_new.append(f"max_trades_per_day {lim.max_trades_per_day} reached ({self.state.trades_today})")
        if not (math.isfinite(price) and price > 0):
            no_new.append(f"invalid price {price!r}")
        if not (math.isfinite(equity) and equity > 0):
            no_new.append(f"invalid equity {equity!r}")
        if no_new:
            new = no_new_risk(approved, current)
            if abs(new - approved) > _EPS:
                reasons.append(f"no new risk ({'; '.join(no_new)}): {approved:+.2f} -> {new:+.2f} lots")
                approved = new

        # 5. rounding -------------------------------------------------------------------
        rounded = self.instrument.round_lots(approved)
        if abs(rounded - approved) > _EPS:
            reasons.append(f"rounded toward zero on lot grid: {approved:+.4f} -> {rounded:+.2f} lots")
            approved = rounded

        # invariant guard (should never trigger) ------------------------------------------
        guarded = toward_zero(approved, self.instrument.round_lots(target) if target else 0.0)
        if abs(guarded - approved) > _EPS:
            logger.error("risk invariant violated (approved %s vs target %s); clamping", approved, target)
            reasons.append(f"invariant clamp toward target: {approved:+.2f} -> {guarded:+.2f} lots")
            approved = guarded

        if commit and abs(approved - current) > _EPS:
            self.state.trades_today += 1
        return self._record(t, ctx.target_lots, current, approved, False, reasons)

    def _record(self, t: pd.Timestamp, requested: float, current: float, approved: float,
                halted: bool, reasons: list[str]) -> RiskDecision:
        approved = 0.0 if approved == 0 else float(approved)
        if reasons:
            self._log.append({
                "time": t,
                "requested": float(requested) if requested is not None else math.nan,
                "current": float(current),
                "approved": approved,
                "halted": halted,
                "reasons": "; ".join(reasons),
            })
            logger.info("risk intervention at %s: %s", t, "; ".join(reasons))
        return RiskDecision(approved_lots=approved, halted=halted, reasons=list(reasons))

    # ------------------------------------------------------------------------------------
    # operator controls & reporting
    # ------------------------------------------------------------------------------------
    def halt(self, reason: str, *, time: pd.Timestamp | None = None) -> None:
        """Manual kill switch (operator, monitor or the LLM Risk Officer). Persisted."""
        t = _utc(time) if time is not None else pd.Timestamp.now(tz="UTC")
        if not self.state.halted:
            self._halt("manual", reason, t)
        self._persist()

    def reset_halt(self, confirm: str = "", *, equity: float | None = None) -> None:
        """Clear the kill switch. Requires ``confirm="RESET"`` (deliberate human action).

        The equity peak and day start are re-based to ``equity`` (default: last observed
        equity) — otherwise the same drawdown would re-trigger the kill immediately.
        """
        if confirm != "RESET":
            raise ValueError("reset_halt requires confirm='RESET'")
        st = self.state
        eq = equity if equity is not None else st.last_equity
        previous = st.halt_reason
        st.halted, st.halt_kind, st.halt_reason, st.halted_at = False, None, None, None
        if eq is not None and math.isfinite(eq) and eq > 0:
            st.peak_equity = float(eq)
            st.day_start_equity = float(eq)
        logger.warning("Risk halt reset by operator (was: %s); peak/day start re-based to %s", previous, eq)
        self._log.append({"time": pd.Timestamp.now(tz="UTC"), "requested": math.nan, "current": math.nan,
                          "approved": math.nan, "halted": False,
                          "reasons": f"halt reset by operator (was: {previous})"})
        self._persist()

    def events_frame(self) -> pd.DataFrame:
        """All interventions so far: time, requested, current, approved, halted, reasons."""
        cols = ["time", "requested", "current", "approved", "halted", "reasons"]
        if not self._log:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self._log, columns=cols)

    def snapshot(self) -> dict:
        """JSON-friendly state for monitoring dashboards and the LLM desk."""
        st = self.state
        dd = None
        if st.peak_equity and st.last_equity is not None and st.peak_equity > 0:
            dd = 1.0 - st.last_equity / st.peak_equity
        day_pnl = None
        if st.day_start_equity and st.last_equity is not None and st.day_start_equity > 0:
            day_pnl = st.last_equity / st.day_start_equity - 1.0
        return {
            "halted": st.halted,
            "halt_kind": st.halt_kind,
            "halt_reason": st.halt_reason,
            "halted_at": st.halted_at,
            "peak_equity": st.peak_equity,
            "last_equity": st.last_equity,
            "drawdown": dd,
            "day_start_equity": st.day_start_equity,
            "day_return": day_pnl,
            "trades_today": st.trades_today,
            "limits": self.limits.to_dict(),
        }

    def __repr__(self) -> str:
        return f"StandardRiskManager(halted={self.state.halted}, limits={self.limits})"
