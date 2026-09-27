"""Bar-level execution simulator — the single source of truth for PnL (SPEC §0.2, §8).

Research backtests, the RL environment, the paper broker and the LLM-desk replay all drive
this class, so a strategy's economics cannot differ between research and trading.

Timing model (SPEC §1)
----------------------
``step(target)`` is called at the CLOSE of bar ``t = sim.index`` (decision time
``available_at[t]``). It then simulates bar ``t+1``:

1. **Gap segment** ``(close[t] -> open[t+1]]``: the *old* position is marked from
   ``close[t]`` to ``open[t+1]`` and pays swap for any rollover falling strictly after
   ``available_at[t]`` and at/before ``open[t+1]`` (e.g. a daily maintenance break); with
   rate-based financing the notional is valued at ``close[t]``, the last mid before it.
2. **Fill at the open** of ``t+1``: the position is traded to ``round_lots(target)``;
   buys fill at ``open + spread_eff/2 + slippage``, sells at ``open - spread_eff/2 -
   slippage`` (:class:`~aurum.execution.costs.CostModel`). Spread and slippage are booked
   as costs; the position itself is marked from the MID, so
   ``equity change = price PnL (at mid) - costs + swap`` holds *exactly* every bar.
3. **Intrabar protective exits** on bar ``t+1`` for the post-fill position. OHLC are mid
   prices and stop/take-profit levels are mid levels (:func:`intrabar_exit`, shared with
   the paper broker and the live runner). For a long (mirror for shorts):

   * ``open <= stop``  → gap through the stop: market exit at the OPEN;
   * ``open >= tp``    → gap through the take-profit: exit at the OPEN (limit, no slip);
   * ``low <= stop``   → exit at ``stop`` (bid side: ``stop - spread/2 - slippage``);
   * ``high >= tp``    → exit at ``tp`` (limit: ``tp - spread/2``, no slippage).

   If both levels are inside the bar's range the STOP is assumed to fill first (the
   conservative convention — OHLC bars do not reveal the intrabar path). Levels are passed
   either as absolute mid prices or, for a position entered at this open, as distances
   from the entry (``stop_distance`` / ``take_profit_distance``, anchored at ``open[t+1]``).
4. **Mark-to-market** at ``close[t+1]`` and swap for rollovers inside
   ``(open[t+1], available_at[t+1]]`` charged on the position held at the bar's close
   (rate-based financing values the notional at ``close[t+1]``).
   (If a protective exit fired in the same bar the exact intrabar time is unknown; the
   position at the close — i.e. flat — is used. With bars ending exactly at the rollover
   hour, the usual case for M1..H1, this is exact; for H4/D1 bars that contain the
   rollover, the bar close stands in for the price at the rollover.)

Financing (:class:`~aurum.execution.costs.FinancingModel`, ``costs.financing``)
---------------------------------------------------------------------------------
``"fixed"`` charges ``instrument.swap_{long,short}_per_lot`` per night (bit-identical to the
original model); ``"rate"`` charges ``-lots * contract_size * P * (r - lease +/- markup) /
360`` per night, where ``r`` is the benchmark (``rates``; the engine passes ``md.macro``, so
``md.macro["fedfunds"]`` by default) as of each rollover instant ``R``: only observations
with ``available_at <= R`` are used. The rate per bar interval is precomputed at
construction, and a rollover is only settled by the step that simulates it, so a rate is
never used before it was published. ``"none"`` charges nothing.

No information from bar ``t+1`` is available to the caller before it calls ``step``: the
simulator only *consumes* the future to settle orders that were decided at ``t``.

Trade accounting
----------------
A trade is a round trip from flat to flat; a reversal fill is split into a closing leg and
an opening leg with its costs split pro rata by lots. Scaling in/out stays in one trade:
``entry_price``/``exit_price`` are lot-weighted averages of the increasing/decreasing fills
(executed prices, i.e. including spread and slippage), ``lots`` is the maximum absolute
size held. Every USD of price PnL, cost and swap is attributed to exactly one trade, so
``sum(trades.pnl) == equity[-1] - equity[0]`` once the open position is closed virtually at
the last close (``exit_reason="end"``, no liquidation cost charged).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from aurum.backtest.result import BacktestResult
from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.types import Fill, Side, Trade
from aurum.data.schema import validate_bars
from aurum.execution.costs import CostModel, FinancingModel, RateSource, _warn_once, rollover_nights_ns

logger = logging.getLogger(__name__)

_EPS = 1e-9
_STALE_RATE_NS = 14 * 86_400 * 10**9

#: columns of ``BacktestResult.trades``: the fields of ``aurum.core.types.Trade`` (``side`` is
#: stored as int +1/-1) plus bar bookkeeping.
TRADE_COLUMNS: list[str] = [f.name for f in fields(Trade)] + [
    "entry_bar", "exit_bar", "bars_held", "price_pnl",
]
#: columns of ``BacktestResult.fills``.
FILL_COLUMNS: list[str] = [
    "time", "bar", "side", "lots", "price", "mid", "spread_cost", "slippage_cost",
    "commission", "kind", "reason", "position_after",
]


def intrabar_exit(pos: float, o: float, h: float, lo: float, sl: float | None,
                  tp: float | None) -> tuple[float, str, bool] | None:
    """Protective exit of a position of sign ``pos`` within one bar (``o``/``h``/``lo`` MID).

    Returns ``(exit mid level, reason, is_limit)`` or ``None``. A gap through a level exits
    at the OPEN; otherwise the stop (market order, slips) is checked before the take-profit
    (limit order, no slippage), the conservative convention when a bar touches both. Shared
    by :class:`ExecutionSimulator`, :class:`aurum.live.paper.PaperBroker` and the live runner
    so that all of them resolve the same bar identically.
    """
    if pos > 0:
        if sl is not None and o <= sl:
            return o, "stop", False
        if tp is not None and o >= tp:
            return o, "take_profit", True
        if sl is not None and lo <= sl:
            return sl, "stop", False
        if tp is not None and h >= tp:
            return tp, "take_profit", True
    else:
        if sl is not None and o >= sl:
            return o, "stop", False
        if tp is not None and o <= tp:
            return o, "take_profit", True
        if sl is not None and h >= sl:
            return sl, "stop", False
        if tp is not None and lo <= tp:
            return tp, "take_profit", True
    return None


@dataclass
class StepResult:
    """Outcome of one :meth:`ExecutionSimulator.step` (bar ``index`` just settled)."""

    index: int                     # bar that was just simulated (t+1)
    time: pd.Timestamp             # its OPEN time
    equity: float                  # equity marked at its close (USD)
    pnl: float                     # net USD change of equity during this step
    ret: float                     # simple return of equity during this step
    price_pnl: float               # mark-to-market PnL at mid prices (USD)
    costs: dict[str, float]        # spread, slippage, commission (>= 0) and swap (signed)
    fills: list[Fill]
    position: float                # signed lots at the close (after protective exits)
    position_open: float           # signed lots right after the fill at the open
    done: bool                     # True when no further step is possible
    exit_reason: str | None = None  # "stop" | "take_profit" if a protective exit fired
    stop_price: float | None = None   # absolute MID stop level that protected this bar (if any)
    take_profit: float | None = None  # absolute MID take-profit level of this bar (if any)


@dataclass
class _OpenTrade:
    side: int
    entry_time: pd.Timestamp
    entry_bar: int
    entry_value: float
    entry_lots: float
    max_lots: float
    costs: float = 0.0
    exit_value: float = 0.0
    exit_lots: float = 0.0
    price_pnl: float = 0.0
    swap: float = 0.0

    def row(self, exit_time: pd.Timestamp, exit_bar: int, bars_held: int, reason: str,
            exit_value: float | None = None, exit_lots: float | None = None) -> dict:
        ev = self.exit_value if exit_value is None else exit_value
        el = self.exit_lots if exit_lots is None else exit_lots
        return {
            "entry_time": self.entry_time,
            "exit_time": exit_time,
            "side": self.side,
            "lots": self.max_lots,
            "entry_price": self.entry_value / self.entry_lots,
            "exit_price": ev / el if el > 0 else float("nan"),
            "pnl": self.price_pnl + self.swap - self.costs,
            "costs": self.costs,
            "swap": self.swap,
            "exit_reason": reason,
            "entry_bar": self.entry_bar,
            "exit_bar": exit_bar,
            "bars_held": bars_held,
            "price_pnl": self.price_pnl,
        }


class ExecutionSimulator:
    """Event-free, bar-by-bar CFD account simulator (see module docstring for semantics).

    Parameters
    ----------
    bars : canonical bars frame (``aurum.data.schema``), mid OHLC + ``spread`` + ``available_at``.
    instrument : contract specification (lot size, rounding, swaps, rollover hour).
    costs : :class:`CostModel`; ``None`` -> ``CostModel()`` defaults (incl. its financing).
    initial_equity : starting account equity in USD.
    validate : run ``validate_bars`` on construction (cheap, vectorised).
    rates : benchmark-rate source for ``"rate"`` financing: ``md.macro`` (the model's
        ``rate_series`` is picked), a macro frame with ``available_at``, a Series indexed by
        availability time, or a :class:`~aurum.execution.costs.RateCurve`. ``None`` means
        ``financing.fallback_rate`` for every rollover (warned once). Ignored by the
        ``"fixed"`` and ``"none"`` modes.

    Attributes
    ----------
    index : int — the bar whose CLOSE is "now" (decisions are taken here).
    equity : float — equity marked at ``close[index]``.
    position : float — signed lots held now.
    done : bool — ``index`` is the last bar; ``step`` would need data that does not exist.
    peak_equity, bankrupt, n_bars, start.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        instrument: Instrument = XAUUSD,
        costs: CostModel | None = None,
        initial_equity: float = 100_000.0,
        *,
        validate: bool = True,
        rates: RateSource = None,
    ) -> None:
        if validate:
            validate_bars(bars)
        if len(bars) < 1:
            raise ValueError("bars is empty")
        if not (math.isfinite(initial_equity) and initial_equity > 0):
            raise ValueError("initial_equity must be finite and > 0")
        self.bars = bars
        self.instrument = instrument
        self.costs = costs if costs is not None else CostModel()
        self.initial_equity = float(initial_equity)
        n = len(bars)
        self.n_bars = n
        self._index = bars.index
        self._times: list[pd.Timestamp] = list(bars.index)
        # Plain Python lists: scalar access is ~10x faster than numpy item access in the loop.
        self._open: list[float] = bars["open"].to_numpy(dtype=float).tolist()
        self._high: list[float] = bars["high"].to_numpy(dtype=float).tolist()
        self._low: list[float] = bars["low"].to_numpy(dtype=float).tolist()
        self._close: list[float] = bars["close"].to_numpy(dtype=float).tolist()
        self._spread: list[float] = bars["spread"].to_numpy(dtype=float).tolist()
        open_ns = pd.DatetimeIndex(bars.index).tz_convert("UTC").as_unit("ns").asi8
        avail_ns = pd.DatetimeIndex(bars["available_at"]).tz_convert("UTC").as_unit("ns").asi8
        self._avail: list[pd.Timestamp] = list(pd.DatetimeIndex(bars["available_at"]))
        nights_bar = rollover_nights_ns(open_ns, avail_ns, instrument)
        nights_gap = np.zeros(n)
        if n > 1:
            nights_gap[1:] = rollover_nights_ns(avail_ns[:-1], open_ns[1:], instrument)
        self._nights_bar: list[float] = nights_bar.tolist()
        self._nights_gap: list[float] = nights_gap.tolist()
        # Rate-based financing: sum_R w(R) * r(R) per interval, r as of each rollover instant R.
        fin = self.costs.financing
        self.financing: FinancingModel = fin
        self.rate_curve = fin.curve(rates) if fin.uses_rates else None
        rn_bar = np.zeros(n)
        rn_gap = np.zeros(n)
        self._financing_info: dict = {"mode": fin.mode}
        if fin.uses_rates:
            jb = np.flatnonzero(nights_bar != 0)
            if len(jb):
                rn_bar[jb] = fin.rate_nights_ns(open_ns[jb], avail_ns[jb], instrument, self.rate_curve)
            jg = np.flatnonzero(nights_gap != 0)
            if len(jg):
                rn_gap[jg] = fin.rate_nights_ns(avail_ns[jg - 1], open_ns[jg], instrument, self.rate_curve)
            self._financing_info = self._rate_coverage(int(open_ns[0]), int(avail_ns[-1]))
        self._rn_bar: list[float] = rn_bar.tolist()
        self._rn_gap: list[float] = rn_gap.tolist()
        self.reset()

    def _rate_coverage(self, t_first: int, t_last: int) -> dict:
        """Provenance of the benchmark used by rate financing (and one-time warnings)."""
        fin, curve = self.financing, self.rate_curve
        info: dict = {"mode": fin.mode, "rate_series": fin.rate_series,
                      "fallback_rate": fin.fallback_rate, "rate_first_available": None,
                      "rate_last_available": None}
        if curve is None or not len(curve):
            _warn_once(("no_rates", fin.rate_series),
                       "rate financing: no %r series supplied (md.macro / rates=); every rollover "
                       "uses fallback_rate=%.4f", fin.rate_series, fin.fallback_rate)
            return info
        info["rate_first_available"] = str(curve.first_available)
        info["rate_last_available"] = str(curve.last_available)
        if int(curve.available_ns[0]) > t_first:
            _warn_once(("rates_start", fin.rate_series, int(curve.available_ns[0])),
                       "rate financing: %r is first available at %s, after the first bar (%s); "
                       "earlier rollovers use fallback_rate=%.4f", fin.rate_series,
                       curve.first_available, pd.Timestamp(t_first, tz="UTC"), fin.fallback_rate)
        if t_last - int(curve.available_ns[-1]) > _STALE_RATE_NS:
            _warn_once(("rates_stale", fin.rate_series, int(curve.available_ns[-1])),
                       "rate financing: %r ends at %s, more than 14 days before the last bar (%s); "
                       "its last value is carried forward", fin.rate_series, curve.last_available,
                       pd.Timestamp(t_last, tz="UTC"))
        return info

    # ---- lifecycle ---------------------------------------------------------------------------
    def reset(self, start: int = 0, equity: float | None = None) -> None:
        """Start (again) at the close of bar ``start``, flat, with ``equity`` (default initial)."""
        n = self.n_bars
        if not 0 <= int(start) < n:
            raise IndexError(f"start={start} outside [0, {n})")
        start = int(start)
        eq = float(self.initial_equity if equity is None else equity)
        if not (math.isfinite(eq) and eq > 0):
            raise ValueError("equity must be finite and > 0")
        self.start = start
        self.index = start
        self.equity = eq
        self.position = 0.0
        self.peak_equity = eq
        self.bankrupt = False
        nan = float("nan")
        self._eq = [nan] * n
        self._pos_open = [0.0] * n
        self._pos_close = [0.0] * n
        self._price_pnl = [0.0] * n
        self._c_spread = [0.0] * n
        self._c_slip = [0.0] * n
        self._c_comm = [0.0] * n
        self._swap = [0.0] * n
        self._net = [0.0] * n
        self._target = [nan] * n
        self._eq[start] = eq
        self._fills: list[tuple] = []
        self._closed: list[dict] = []
        self._trade: _OpenTrade | None = None

    @property
    def done(self) -> bool:
        return self.index >= self.n_bars - 1

    # ---- read-only views of "now" (all known at the close of bar ``index``) --------------------
    @property
    def time(self) -> pd.Timestamp:
        """OPEN time of the current bar (the bars-index label)."""
        return self._times[self.index]

    @property
    def decision_time(self) -> pd.Timestamp:
        """``available_at`` of the current bar = the moment decisions are taken."""
        return self._avail[self.index]

    @property
    def price(self) -> float:
        """Mid close of the current bar."""
        return self._close[self.index]

    @property
    def spread(self) -> float:
        return self._spread[self.index]

    @property
    def drawdown(self) -> float:
        """Fractional drawdown from the running equity peak (>= 0)."""
        return 0.0 if self.peak_equity <= 0 else max(0.0, 1.0 - self.equity / self.peak_equity)

    @property
    def margin_used(self) -> float:
        return self.instrument.margin_required(self.position, self.price)

    @property
    def free_margin(self) -> float:
        return self.equity - self.margin_used

    def snapshot(self) -> dict:
        """JSON-friendly state for agents, paper broker and logs."""
        return {
            "index": self.index,
            "time": str(self.time),
            "decision_time": str(self.decision_time),
            "price": self.price,
            "spread": self.spread,
            "equity": self.equity,
            "position": self.position,
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown,
            "margin_used": self.margin_used,
            "free_margin": self.free_margin,
            "done": self.done,
            "bankrupt": self.bankrupt,
        }

    # ---- core ------------------------------------------------------------------------------------
    def step(
        self,
        target_lots: float,
        *,
        stop_price: float | None = None,
        take_profit: float | None = None,
        reason: str = "signal",
        stop_distance: float | None = None,
        take_profit_distance: float | None = None,
    ) -> StepResult:
        """Trade to ``target_lots`` at the next open and settle the next bar.

        Called at the close of bar ``index``. ``target_lots`` is rounded with
        ``instrument.round_lots`` (toward zero, clipped to ``max_lot``). ``stop_price`` /
        ``take_profit`` are absolute MID price levels protecting the post-fill position during
        bar ``index+1`` only (pass them again every step to keep them working). ``reason`` tags
        the fill and, if the fill closes a trade, the trade's ``exit_reason``
        (e.g. ``"signal"`` or ``"risk"``).

        ``stop_distance`` / ``take_profit_distance`` (price units, > 0) are the alternative to
        absolute levels for a position being *entered* at this open: the level is anchored at
        the entry, i.e. the mid OPEN of bar ``index+1`` (``open -/+ distance`` for a long,
        mirrored for a short). This is how a live OMS attaches an ATR stop after the fill, and
        it avoids placing a level on the wrong side of the market after a gap (which would
        enter and immediately exit, paying the spread twice). The open is only used after the
        order has filled at that open, so no decision uses it. The resolved levels are
        returned in ``StepResult.stop_price`` / ``take_profit`` so the caller can keep them
        fixed on later steps. Passing both a level and a distance for the same order raises.
        """
        if self.done:
            raise RuntimeError("simulation finished (no next bar); call reset()")
        tgt = float(target_lots)
        if not math.isfinite(tgt):
            raise ValueError(f"target_lots must be finite, got {target_lots!r}")
        for name, lvl in (("stop_price", stop_price), ("take_profit", take_profit),
                          ("stop_distance", stop_distance),
                          ("take_profit_distance", take_profit_distance)):
            if lvl is not None and not (math.isfinite(lvl) and lvl > 0):
                raise ValueError(f"{name} must be a positive finite number or None, got {lvl!r}")
        if stop_price is not None and stop_distance is not None:
            raise ValueError("pass either stop_price or stop_distance, not both")
        if take_profit is not None and take_profit_distance is not None:
            raise ValueError("pass either take_profit or take_profit_distance, not both")
        t = self.index
        i = t + 1
        self._target[t] = tgt
        inst = self.instrument
        cs = inst.contract_size
        cm = self.costs
        target = inst.round_lots(tgt)
        if self.bankrupt and target != 0.0:
            target, reason = 0.0, "risk"

        o = self._open[i]
        h = self._high[i]
        lo = self._low[i]
        c = self._close[i]
        spr = self._spread[i]
        rng = h - lo
        price_pnl = swap = c_spread = c_slip = c_comm = 0.0
        fills: list[Fill] = []
        pos = self.position

        # 1) gap segment: old position from close[t] to open[t+1] (+ swap for rollovers in the gap)
        if pos != 0.0:
            tr = self._trade
            seg = pos * cs * (o - self._close[t])
            price_pnl += seg
            tr.price_pnl += seg
            ng = self._nights_gap[i]
            if ng:  # rate financing values the notional at the last mid before the rollover
                x = cm.swap(pos, ng, instrument=inst, price=self._close[t], rate_nights=self._rn_gap[i])
                swap += x
                tr.swap += x

        # 2) fill at the open
        delta = target - pos
        if abs(delta) > _EPS:
            a, b, d, f = self._execute(i, delta, o, spr, rng, "open", reason, limit=False)
            c_spread += a
            c_slip += b
            c_comm += d
            fills.append(f)
            pos = self.position
        pos_open = pos

        # 3) intrabar protective exits (stop first)
        sl_lvl: float | None = None
        tp_lvl: float | None = None
        if pos != 0.0:
            side = 1.0 if pos > 0 else -1.0
            sl_lvl = stop_price
            tp_lvl = take_profit
            if stop_distance is not None:
                x = o - side * stop_distance
                sl_lvl = x if x > 0.0 else None
            if take_profit_distance is not None:
                x = o + side * take_profit_distance
                tp_lvl = x if x > 0.0 else None
        exit_reason: str | None = None
        if pos != 0.0 and (sl_lvl is not None or tp_lvl is not None):
            ex = intrabar_exit(pos, o, h, lo, sl_lvl, tp_lvl)
            if ex is not None:
                exit_mid, exit_reason, is_limit = ex
                seg = pos * cs * (exit_mid - o)
                price_pnl += seg
                self._trade.price_pnl += seg
                a, b, d, f = self._execute(i, -pos, exit_mid, spr, rng, exit_reason, exit_reason,
                                           limit=is_limit)
                c_spread += a
                c_slip += b
                c_comm += d
                fills.append(f)
                pos = self.position

        # 4) mark to the close and swap for rollovers inside the bar
        if pos != 0.0:
            tr = self._trade
            seg = pos * cs * (c - o)
            price_pnl += seg
            tr.price_pnl += seg
            nb = self._nights_bar[i]
            if nb:
                x = cm.swap(pos, nb, instrument=inst, price=c, rate_nights=self._rn_bar[i])
                swap += x
                tr.swap += x

        net = price_pnl - c_spread - c_slip - c_comm + swap
        prev_eq = self.equity
        self.equity = prev_eq + net
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        if self.equity <= 0.0 and not self.bankrupt:
            self.bankrupt = True
            logger.warning("equity %.2f <= 0 at bar %d (%s): account bankrupt; further steps "
                           "are forced flat", self.equity, i, self._times[i])
        self.index = i
        self._eq[i] = self.equity
        self._pos_open[i] = pos_open
        self._pos_close[i] = pos
        self._price_pnl[i] = price_pnl
        self._c_spread[i] = c_spread
        self._c_slip[i] = c_slip
        self._c_comm[i] = c_comm
        self._swap[i] = swap
        self._net[i] = net
        return StepResult(
            index=i,
            time=self._times[i],
            equity=self.equity,
            pnl=net,
            ret=net / prev_eq if prev_eq != 0 else float("nan"),
            price_pnl=price_pnl,
            costs={"spread": c_spread, "slippage": c_slip, "commission": c_comm, "swap": swap},
            fills=fills,
            position=pos,
            position_open=pos_open,
            done=self.done,
            exit_reason=exit_reason,
            stop_price=sl_lvl,
            take_profit=tp_lvl,
        )

    #: backward-compatible alias of the module-level :func:`intrabar_exit`.
    _intrabar_exit = staticmethod(intrabar_exit)

    def _execute(self, i: int, qty: float, mid: float, spr: float, rng: float, kind: str,
                 reason: str, *, limit: bool) -> tuple[float, float, float, Fill]:
        """Fill ``qty`` signed lots at bar ``i`` around ``mid``; book costs, trade and position."""
        inst = self.instrument
        side = 1 if qty > 0 else -1
        q = abs(qty)
        fp = self.costs.fill_price(side, mid, spr, rng, q, instrument=inst, limit=limit)
        comm = self.costs.commission(q, instrument=inst)
        cost = fp.spread_cost + fp.slippage_cost + comm
        pos_before = self.position
        pos_after = round(pos_before + qty, 8)
        if abs(pos_after) < _EPS:
            pos_after = 0.0
        self._book_trade(i, pos_before, side, q, fp.price, cost, reason, intrabar=(kind != "open"))
        self.position = pos_after
        time = self._times[i]
        self._fills.append((time, i, side, q, fp.price, mid, fp.spread_cost, fp.slippage_cost,
                            comm, kind, reason, pos_after))
        fill = Fill(client_id=f"sim-{i}-{kind}", symbol=inst.symbol, side=Side(side), lots=q,
                    price=fp.price, time=time, commission=comm, slippage_cost=fp.slippage_cost,
                    spread_cost=fp.spread_cost)
        return fp.spread_cost, fp.slippage_cost, comm, fill

    def _book_trade(self, i: int, pos_before: float, side: int, q: float, price: float,
                    cost: float, reason: str, *, intrabar: bool) -> None:
        tr = self._trade
        time = self._times[i]
        if tr is None:
            self._trade = _OpenTrade(side=side, entry_time=time, entry_bar=i, entry_value=q * price,
                                     entry_lots=q, max_lots=q, costs=cost)
            return
        if side == tr.side:  # scale in
            tr.entry_value += q * price
            tr.entry_lots += q
            tr.costs += cost
            tr.max_lots = max(tr.max_lots, abs(pos_before) + q)
            return
        held = abs(pos_before)
        close_q = min(q, held)
        open_q = q - close_q
        if open_q < _EPS:
            open_q, close_q = 0.0, q
        c_close = cost * (close_q / q)
        tr.exit_value += close_q * price
        tr.exit_lots += close_q
        tr.costs += c_close
        if held - close_q < _EPS:  # flat (or reversed): the round trip is complete
            bars_held = i - tr.entry_bar + (1 if intrabar else 0)
            self._closed.append(tr.row(time, i, bars_held, reason))
            self._trade = None
            if open_q > 0.0:
                self._trade = _OpenTrade(side=side, entry_time=time, entry_bar=i,
                                         entry_value=open_q * price, entry_lots=open_q,
                                         max_lots=open_q, costs=cost - c_close)

    # ---- results ---------------------------------------------------------------------------------
    def trades_frame(self, *, include_open: bool = True) -> pd.DataFrame:
        """Closed round trips, plus the open one closed virtually at the current close
        (``exit_reason="end"``, no liquidation cost) when ``include_open``."""
        rows = list(self._closed)
        tr = self._trade
        if include_open and tr is not None:
            i = self.index
            q = abs(self.position)
            mark = self._close[i]
            rows.append(tr.row(self._times[i], i, i - tr.entry_bar + 1, "end",
                               exit_value=tr.exit_value + q * mark, exit_lots=tr.exit_lots + q))
        df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
        if not rows:
            df = df.astype({"pnl": float, "costs": float, "swap": float, "lots": float})
        return df

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._fills, columns=FILL_COLUMNS)

    def result(self, *, compute_metrics: bool = True) -> BacktestResult:
        """Assemble a :class:`BacktestResult` for bars ``start..index`` (inclusive).

        The first row is the starting bar (flat, zero return). ``target`` holds the lots
        passed to ``step`` at each close (NaN at the last bar: no decision is executed).
        """
        s, e = self.start, self.index + 1
        idx = self._index[s:e]
        eq = np.asarray(self._eq[s:e], dtype=float)
        ret = np.zeros_like(eq)
        if len(eq) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                ret[1:] = eq[1:] / eq[:-1] - 1.0
        spread = np.asarray(self._c_spread[s:e])
        slip = np.asarray(self._c_slip[s:e])
        comm = np.asarray(self._c_comm[s:e])
        swap = np.asarray(self._swap[s:e])
        price = np.asarray(self._price_pnl[s:e])
        costs = pd.DataFrame({"spread": spread, "slippage": slip, "commission": comm, "swap": swap},
                             index=idx)
        pnl = pd.DataFrame({"price": price, "costs": spread + slip + comm, "swap": swap,
                            "net": np.asarray(self._net[s:e])}, index=idx)
        res = BacktestResult(
            equity=pd.Series(eq, index=idx, name="equity"),
            returns=pd.Series(ret, index=idx, name="returns"),
            positions=pd.Series(self._pos_open[s:e], index=idx, name="positions", dtype=float),
            costs=costs,
            trades=self.trades_frame(include_open=True),
            fills=self.fills_frame(),
            target=pd.Series(self._target[s:e], index=idx, name="target", dtype=float),
            pnl=pnl,
            position_close=pd.Series(self._pos_close[s:e], index=idx, name="position_close",
                                     dtype=float),
            meta={
                "simulator": "aurum.execution.simulator.ExecutionSimulator",
                "instrument": self.instrument.symbol,
                "contract_size": self.instrument.contract_size,
                "costs": self.costs.to_dict(),
                "initial_equity": float(eq[0]) if len(eq) else self.initial_equity,
                "timeframe": self.bars.attrs.get("timeframe"),
                "start": str(idx[0]) if len(idx) else None,
                "end": str(idx[-1]) if len(idx) else None,
                "n_bars": int(e - s),
                "bankrupt": self.bankrupt,
                "financing": dict(self._financing_info),
            },
        )
        if compute_metrics:
            from aurum.backtest.metrics import compute_metrics as _cm

            res.metrics = _cm(res)
        return res
