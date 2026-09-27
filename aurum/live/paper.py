"""Paper broker and bar feeds (SPEC §11).

:class:`PaperBroker` is a venue that never touches money. It fills market orders with the
SAME :class:`~aurum.execution.costs.CostModel` arithmetic as the research simulator
(``mid ± effective_spread/2 ± slippage``, commission per lot per side, overnight financing
per rollover with the triple-swap weekday) and keeps positions the way a broker does — at their
*executed* price, with commission booked to the balance and swap accrued on the position.
That bookkeeping is algebraically identical to the simulator's "mark at mid, book spread and
slippage as costs" identity, so equity paths agree to floating-point precision
(``tests/test_live_paper.py`` enforces it against
:class:`~aurum.execution.simulator.ExecutionSimulator`).

Feeds
-----
A feed tells the paper broker what the market looks like at clock time ``now``:

* :class:`ReplayFeed` serves a stored bars frame progressively: ``closed_bars(now)`` only
  returns bars with ``available_at <= now``; ``quote(now)`` is the OPEN of the bar in
  progress (``open <= now < available_at``) with that bar's spread — i.e. an order placed
  just after the close of bar ``t`` executes at ``open[t+1]``, the fill convention of the
  simulator (SPEC §1). Between bars (weekend, daily maintenance break) the market is
  closed: ``quote`` returns None and orders are rejected with ``market_closed``, as a real
  MT5 server does (retcode 10018). The high-low range of the execution bar is passed to the
  slippage term exactly as in the simulator; it is a cost-model input only and never reaches
  a decision.
* :class:`BrokerDataFeed` wraps another :class:`~aurum.live.broker.Broker` (e.g. an MT5
  demo terminal) so strategies can be paper-traded on live quotes.

Protective orders
-----------------
Stops and take-profits are MID levels (as in the simulator) evaluated on every bar that
closes after the position was opened with the simulator's own
:func:`aurum.execution.simulator.intrabar_exit`: gap through the level at the open → exit at
the open; otherwise stop at the level (market, slips) before take-profit at the level (limit,
no slippage) — the conservative convention when a bar touches both.

Swap
----
A position is financed for every rollover instant ``R`` (``rollover_hour_utc`` on weekdays,
triple on ``triple_swap_weekday``) with ``open_time < R <= close_time``
(:func:`aurum.execution.costs.rollover_nights_ns`) by ``costs.financing``
(:class:`~aurum.execution.costs.FinancingModel`): ``"fixed"`` per-lot swaps from the
instrument, or ``"rate"`` = benchmark (``rates=``, as of ``R``) +/- markup on the notional
valued at the mid close of the bar ending at/after ``R`` (the previous close for a rollover
in a gap between bars) — exactly the simulator's valuation, so the two stay in parity. Swap
is accrued on the position (included in equity) and realised into the balance on close,
the MT5 convention.

The replay's clock is a :class:`~aurum.live.broker.SimulatedClock`, so hundreds of bars of
paper trading run in well under a second.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.timeframes import get_timeframe, index_bar_minutes
from aurum.core.types import Fill, Side
from aurum.data.schema import validate_bars
from aurum.execution.costs import CostModel, RateSource, _warn_once, rollover_nights_ns
from aurum.execution.simulator import intrabar_exit
from aurum.live.broker import (
    AccountInfo,
    Broker,
    BrokerDeal,
    BrokerError,
    BrokerPosition,
    Clock,
    OrderRequest,
    OrderResult,
    OrderStatusCode,
    Quote,
)
from aurum.live.state import StateCorruptError, atomic_write_json, read_json

logger = logging.getLogger(__name__)

__all__ = ["BarFeed", "BrokerDataFeed", "PaperBroker", "ReplayFeed"]

_EPS = 1e-9
#: MT5-compatible return codes used by the paper broker (so logs read the same).
RETCODE_DONE = 10009
RETCODE_MARKET_CLOSED = 10018
RETCODE_NO_MONEY = 10019
RETCODE_INVALID_VOLUME = 10014
RETCODE_INVALID = 10013
RETCODE_REJECT = 10006
_STATE_FORMAT = "aurum.live.PaperBroker"
_STATE_VERSION = 1


def _ns(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).tz_convert("UTC").as_unit("ns").value)


def _ts(ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(ns), tz="UTC")


# =============================================================================================
# feeds
# =============================================================================================
@runtime_checkable
class BarFeed(Protocol):
    """Market-data source for :class:`PaperBroker`."""

    timeframe: str

    def closed_bars(self, now: pd.Timestamp, n: int | None = None) -> pd.DataFrame: ...

    def quote(self, now: pd.Timestamp) -> Quote | None: ...

    def mark(self, now: pd.Timestamp) -> float | None: ...

    def bars_between(self, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame: ...


class ReplayFeed:
    """Serve CLOSED bars of a stored frame progressively as the clock advances.

    Parameters
    ----------
    bars : canonical bars frame (``aurum.data.schema``).
    timeframe : name (default: ``bars.attrs["timeframe"]`` or inferred from the spacing).
    """

    def __init__(self, bars: pd.DataFrame, timeframe: str | None = None) -> None:
        validate_bars(bars)
        if len(bars) < 2:
            raise ValueError("ReplayFeed needs at least two bars")
        tf = timeframe or bars.attrs.get("timeframe")
        if tf is None:
            minutes = index_bar_minutes(pd.DatetimeIndex(bars.index))
            tf = next((t.name for t in map(get_timeframe, ("M1", "M5", "M15", "M30", "H1", "H4", "D1"))
                       if t.minutes == minutes), None)
            if tf is None:
                raise ValueError(f"cannot infer a timeframe from a {minutes}-minute spacing")
        self.timeframe = get_timeframe(tf).name
        self.bars = bars
        self._open_ns = pd.DatetimeIndex(bars.index).tz_convert("UTC").as_unit("ns").asi8.copy()
        self._avail_ns = pd.DatetimeIndex(bars["available_at"]).tz_convert("UTC").as_unit("ns").asi8.copy()
        if not (np.diff(self._avail_ns) > 0).all():
            raise ValueError("bars.available_at must be strictly increasing")
        self._o = bars["open"].to_numpy(dtype=float)
        self._h = bars["high"].to_numpy(dtype=float)
        self._l = bars["low"].to_numpy(dtype=float)
        self._c = bars["close"].to_numpy(dtype=float)
        self._s = bars["spread"].to_numpy(dtype=float)

    # ---- positions in time ------------------------------------------------------------------
    @property
    def start(self) -> pd.Timestamp:
        """Open time of the first bar."""
        return _ts(self._open_ns[0])

    @property
    def end(self) -> pd.Timestamp:
        """``available_at`` of the last bar (the replay is exhausted afterwards)."""
        return _ts(self._avail_ns[-1])

    def n_closed(self, now: pd.Timestamp) -> int:
        """Number of bars closed at ``now`` (``available_at <= now``)."""
        return int(np.searchsorted(self._avail_ns, _ns(now), side="right"))

    def forming_index(self, now: pd.Timestamp) -> int | None:
        """Index of the bar in progress at ``now`` (``open <= now < available_at``), else None."""
        t = _ns(now)
        i = int(np.searchsorted(self._open_ns, t, side="right")) - 1
        if i < 0 or t >= self._avail_ns[i]:
            return None
        return i

    def decision_time(self, i: int) -> pd.Timestamp:
        return _ts(self._avail_ns[i])

    # ---- BarFeed ------------------------------------------------------------------------------
    def closed_bars(self, now: pd.Timestamp, n: int | None = None) -> pd.DataFrame:
        k = self.n_closed(now)
        lo = 0 if n is None else max(0, k - int(n))
        return self.bars.iloc[lo:k]

    def quote(self, now: pd.Timestamp) -> Quote | None:
        i = self.forming_index(now)
        if i is None:
            return None
        half = 0.5 * float(self._s[i])
        o = float(self._o[i])
        return Quote(time=pd.Timestamp(now).tz_convert("UTC"), bid=o - half, ask=o + half,
                     bar_range=float(self._h[i] - self._l[i]))

    def mark(self, now: pd.Timestamp) -> float | None:
        """Mid close of the last closed bar (the simulator marks equity there at a decision)."""
        k = self.n_closed(now)
        return float(self._c[k - 1]) if k > 0 else None

    def bars_between(self, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
        """Bars with ``t0 < available_at <= t1``."""
        lo = int(np.searchsorted(self._avail_ns, _ns(t0), side="right"))
        hi = int(np.searchsorted(self._avail_ns, _ns(t1), side="right"))
        return self.bars.iloc[lo:hi]

    def next_open_after(self, now: pd.Timestamp) -> pd.Timestamp | None:
        """Open time of the first bar opening strictly after ``now`` (market reopen hint)."""
        i = int(np.searchsorted(self._open_ns, _ns(now), side="right"))
        return _ts(self._open_ns[i]) if i < len(self._open_ns) else None


class BrokerDataFeed:
    """Paper-trade on another broker's live data (e.g. an MT5 demo terminal's quotes).

    The execution bar's range is unknown in real time, so the slippage range term uses the
    last CLOSED bar's range as a proxy. Protective levels are evaluated on closed bars.
    """

    def __init__(self, data_broker: Broker, symbol: str, timeframe: str, *, lookback: int = 64) -> None:
        self.data_broker = data_broker
        self.symbol = symbol
        self.timeframe = get_timeframe(timeframe).name
        self.lookback = int(lookback)
        self._last_range = 0.0

    def closed_bars(self, now: pd.Timestamp, n: int | None = None) -> pd.DataFrame:
        bars = self.data_broker.latest_bars(self.symbol, self.timeframe, int(n or 5000))
        bars = bars.loc[pd.DatetimeIndex(bars["available_at"]) <= now]
        if len(bars):
            self._last_range = float(bars["high"].iloc[-1] - bars["low"].iloc[-1])
        return bars

    def quote(self, now: pd.Timestamp) -> Quote | None:
        q = self.data_broker.quote(self.symbol)
        if q is None:
            return None
        return Quote(time=q.time, bid=q.bid, ask=q.ask, bar_range=q.bar_range or self._last_range)

    def mark(self, now: pd.Timestamp) -> float | None:
        q = self.data_broker.quote(self.symbol)
        if q is not None:
            return q.mid
        bars = self.closed_bars(now, 1)
        return float(bars["close"].iloc[-1]) if len(bars) else None

    def bars_between(self, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
        bars = self.data_broker.latest_bars(self.symbol, self.timeframe, self.lookback)
        avail = pd.DatetimeIndex(bars["available_at"])
        return bars.loc[(avail > t0) & (avail <= t1)]


# =============================================================================================
# paper broker
# =============================================================================================
@dataclass
class _Pos:
    ticket: int
    symbol: str
    lots: float                 # signed
    price_open: float           # executed price (volume-weighted when added to)
    sl: float | None
    tp: float | None
    magic: int
    comment: str
    open_ns: int
    swap: float = 0.0           # accrued, signed (+ received)
    swap_from_ns: int = 0

    def to_state(self) -> dict[str, Any]:
        return {"ticket": self.ticket, "symbol": self.symbol, "lots": self.lots,
                "price_open": self.price_open, "sl": self.sl, "tp": self.tp, "magic": self.magic,
                "comment": self.comment, "open_ns": self.open_ns, "swap": self.swap,
                "swap_from_ns": self.swap_from_ns}


#: backward-compatible alias (``aurum.live.runner`` imports it); the implementation is the
#: simulator's :func:`aurum.execution.simulator.intrabar_exit`, so there is no mirror copy.
_protective_exit = intrabar_exit


class PaperBroker:
    """Simulated venue implementing :class:`~aurum.live.broker.Broker` (always a demo).

    Parameters
    ----------
    feed           : :class:`ReplayFeed` or :class:`BrokerDataFeed`.
    instrument     : contract specification (lot grid, contract size, swaps, margin rate).
    costs          : :class:`CostModel` (default ``CostModel()``, the research default).
    initial_equity : starting balance (USD).
    clock          : source of "now" (a :class:`SimulatedClock` for replays).
    hedging        : MT5 hedging semantics (one ticket per entry; closes target tickets) instead
                     of netting (one position per symbol, deals net against it).
    state_path     : optional JSON file; the book is persisted after every change and restored
                     on construction, so paper trading survives restarts like a real venue.
    rates          : benchmark-rate source for ``"rate"`` financing (``md.macro``, a macro frame
                     with ``available_at``, a Series indexed by availability time or a
                     :class:`~aurum.execution.costs.RateCurve`); read as of each rollover.
                     ``None`` -> ``financing.fallback_rate``. Refresh with :meth:`set_rates`.
    """

    def __init__(
        self,
        feed: BarFeed,
        instrument: Instrument = XAUUSD,
        costs: CostModel | None = None,
        initial_equity: float = 100_000.0,
        *,
        clock: Clock,
        symbol: str | None = None,
        hedging: bool = False,
        currency: str = "USD",
        state_path: str | Path | None = None,
        max_deals: int = 5000,
        rates: RateSource = None,
    ) -> None:
        if not (math.isfinite(initial_equity) and initial_equity > 0):
            raise ValueError("initial_equity must be finite and > 0")
        self.feed = feed
        self.instrument = instrument
        self.costs = costs if costs is not None else CostModel()
        self.clock = clock
        self.symbol = symbol or instrument.symbol
        self.hedging = bool(hedging)
        self.currency = currency
        self.initial_equity = float(initial_equity)
        self.state_path = Path(state_path) if state_path is not None else None
        self.max_deals = int(max_deals)
        self.balance = float(initial_equity)
        self._positions: dict[int, _Pos] = {}
        self._deals: list[BrokerDeal] = []
        self._next_ticket = 1
        self._settled_ns = _ns(clock.now())
        self.n_orders = 0
        self.rate_curve = None
        self.set_rates(rates)
        if self.state_path is not None and self.state_path.exists():
            self._load_state()

    def set_rates(self, rates: RateSource) -> None:
        """(Re)load the benchmark-rate source used by ``"rate"`` financing (e.g. after a macro
        refresh). Only observations with ``available_at <= R`` are used for a rollover ``R``."""
        fin = self.costs.financing
        self.rate_curve = fin.curve(rates) if fin.uses_rates else None
        if fin.uses_rates and (self.rate_curve is None or not len(self.rate_curve)):
            _warn_once(("paper_no_rates", fin.rate_series),
                       "PaperBroker: rate financing without a %r series; using fallback_rate=%.4f "
                       "(pass rates=md.macro or call set_rates)", fin.rate_series, fin.fallback_rate)

    # ------------------------------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------------------------------
    def to_state(self) -> dict[str, Any]:
        return {
            "format": _STATE_FORMAT, "version": _STATE_VERSION, "symbol": self.symbol,
            "hedging": self.hedging, "balance": self.balance, "next_ticket": self._next_ticket,
            "settled_ns": self._settled_ns, "n_orders": self.n_orders,
            "positions": [p.to_state() for p in self._positions.values()],
            "deals": [{
                "ticket": d.ticket, "order": d.order, "position_ticket": d.position_ticket,
                "symbol": d.symbol, "side": int(d.side), "lots": d.lots, "price": d.price,
                "magic": d.magic, "comment": d.comment, "time_ns": _ns(d.time),
                "commission": d.commission, "swap": d.swap, "profit": d.profit,
            } for d in self._deals[-self.max_deals:]],
        }

    def _save(self) -> None:
        if self.state_path is not None:
            atomic_write_json(self.state_path, self.to_state(), indent=None)

    def _load_state(self) -> None:
        assert self.state_path is not None
        data = read_json(self.state_path)
        try:
            if data.get("format") != _STATE_FORMAT or int(data.get("version", -1)) != _STATE_VERSION:
                raise ValueError("unknown format/version")
            if data["symbol"] != self.symbol or bool(data["hedging"]) != self.hedging:
                raise ValueError("state belongs to a different symbol/account mode")
            self.balance = float(data["balance"])
            self._next_ticket = int(data["next_ticket"])
            # Resume from the persisted settle point: bars closed while the process was down
            # are replayed through swap accrual and the protective orders on the next call.
            self._settled_ns = int(data["settled_ns"])
            self.n_orders = int(data.get("n_orders", 0))
            self._positions = {}
            for p in data["positions"]:
                pos = _Pos(ticket=int(p["ticket"]), symbol=str(p["symbol"]), lots=float(p["lots"]),
                           price_open=float(p["price_open"]), sl=p["sl"], tp=p["tp"],
                           magic=int(p["magic"]), comment=str(p["comment"]), open_ns=int(p["open_ns"]),
                           swap=float(p["swap"]), swap_from_ns=int(p["swap_from_ns"]))
                self._positions[pos.ticket] = pos
            self._deals = [BrokerDeal(
                ticket=int(d["ticket"]), order=d["order"], position_ticket=d["position_ticket"],
                symbol=str(d["symbol"]), side=Side(int(d["side"])), lots=float(d["lots"]),
                price=float(d["price"]), magic=int(d["magic"]), comment=str(d["comment"]),
                time=_ts(int(d["time_ns"])), commission=float(d["commission"]), swap=float(d["swap"]),
                profit=float(d["profit"])) for d in data["deals"]]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise StateCorruptError(f"paper broker state {self.state_path} is invalid: {exc}") from exc
        logger.info("PaperBroker restored %d position(s), balance %.2f from %s",
                    len(self._positions), self.balance, self.state_path)

    # ------------------------------------------------------------------------------------------
    # time evolution: swap accrual and broker-side protective orders
    # ------------------------------------------------------------------------------------------
    def _accrue_swap(self, t_ns: int, price: float | None = None) -> None:
        """Accrue financing on every position up to ``t_ns``; ``price`` (mid) values the notional
        for ``"rate"`` financing (the simulator's valuation of the rollovers in that stretch)."""
        fin = self.costs.financing
        for p in self._positions.values():
            if t_ns > p.swap_from_ns:
                nights = float(rollover_nights_ns(p.swap_from_ns, t_ns, self.instrument))
                if nights:
                    rn = None
                    if fin.uses_rates:
                        rn = float(fin.rate_nights_ns(p.swap_from_ns, t_ns, self.instrument, self.rate_curve))
                    p.swap += self.costs.swap(p.lots, nights, instrument=self.instrument, price=price,
                                              rate_nights=rn)
                p.swap_from_ns = t_ns

    def _price_at(self, t: pd.Timestamp) -> float | None:
        """Mid used to value a notional for rate financing at ``t`` (last closed bar's close)."""
        if not self.costs.financing.uses_rates:
            return None
        m = self.feed.mark(t)
        if m is not None and math.isfinite(m):
            return float(m)
        return self._mark(t)

    def _settle(self, now: pd.Timestamp) -> None:
        """Advance the book to ``now``: per closed bar, accrue swap to its open, run the
        protective orders on its OHLC, accrue swap to its close; then accrue to ``now``."""
        now_ns = _ns(now)
        if now_ns <= self._settled_ns:
            return
        changed = False
        if self._positions:
            rate_fin = self.costs.financing.uses_rates
            prev = self._price_at(_ts(self._settled_ns)) if rate_fin else None
            new = self.feed.bars_between(_ts(self._settled_ns), _ts(now_ns))
            if len(new):
                open_ns = pd.DatetimeIndex(new.index).tz_convert("UTC").as_unit("ns").asi8
                avail_ns = pd.DatetimeIndex(new["available_at"]).tz_convert("UTC").as_unit("ns").asi8
                o = new["open"].to_numpy(dtype=float)
                h = new["high"].to_numpy(dtype=float)
                lo = new["low"].to_numpy(dtype=float)
                c = new["close"].to_numpy(dtype=float)
                spr = new["spread"].to_numpy(dtype=float)
                for k in range(len(new)):
                    if not self._positions:
                        break
                    # rollovers in the gap before the bar: valued at the previous close
                    self._accrue_swap(int(open_ns[k]), prev)
                    changed |= self._protective(int(open_ns[k]), int(avail_ns[k]), float(o[k]),
                                                float(h[k]), float(lo[k]), float(spr[k]))
                    # rollovers inside the bar: valued at its close (as the simulator does)
                    self._accrue_swap(int(avail_ns[k]), float(c[k]))
                    prev = float(c[k])
            self._accrue_swap(now_ns, self._price_at(_ts(now_ns)) if rate_fin else None)
            changed = True
        self._settled_ns = now_ns
        if self._positions:
            changed |= self._stop_out(now)
        if changed:
            self._save()

    def _protective(self, bar_open_ns: int, bar_avail_ns: int, o: float, h: float, lo: float,
                    spr: float) -> bool:
        fired = False
        for p in list(self._positions.values()):
            if (p.sl is None and p.tp is None) or p.open_ns >= bar_avail_ns:
                continue
            ex = intrabar_exit(p.lots, o, h, lo, p.sl, p.tp)
            if ex is None:
                continue
            exit_mid, reason, is_limit = ex
            side = Side.SELL if p.lots > 0 else Side.BUY
            fp = self.costs.fill_price(side, exit_mid, spr, h - lo, abs(p.lots), instrument=self.instrument,
                                       limit=is_limit)
            tag = "[sl]" if reason == "stop" else "[tp]"
            # The exit happens inside the bar; stamp it at the bar open (exact time unknown).
            self._close_qty(p, abs(p.lots), fp.price, t_ns=max(bar_open_ns, p.open_ns),
                            comment=f"{tag} {p.comment}"[:31], magic=p.magic)
            logger.info("paper %s hit on ticket %d at %.2f (mid %.2f)", reason, p.ticket, fp.price, exit_mid)
            fired = True
        return fired

    def _stop_out(self, now: pd.Timestamp) -> bool:
        eq = self._equity(now)
        if eq > 0:
            return False
        mark = self._mark(now)
        logger.critical("paper account equity %.2f <= 0: stop-out, closing all positions", eq)
        for p in list(self._positions.values()):
            side = Side.SELL if p.lots > 0 else Side.BUY
            fp = self.costs.fill_price(side, mark, 0.0, 0.0, abs(p.lots), instrument=self.instrument)
            self._close_qty(p, abs(p.lots), fp.price, t_ns=_ns(now), comment="[so]", magic=p.magic)
        return True

    # ------------------------------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------------------------------
    def _new_ticket(self) -> int:
        t = self._next_ticket
        self._next_ticket += 1
        return t

    def _record_deal(self, *, position_ticket: int, side: Side, lots: float, price: float, magic: int,
                     comment: str, t_ns: int, commission: float, swap: float, profit: float) -> int:
        deal = self._new_ticket()
        self._deals.append(BrokerDeal(ticket=deal, order=deal, position_ticket=position_ticket,
                                      symbol=self.symbol, side=side, lots=lots, price=price, magic=magic,
                                      comment=comment, time=_ts(t_ns), commission=commission, swap=swap,
                                      profit=profit))
        if len(self._deals) > 2 * self.max_deals:
            self._deals = self._deals[-self.max_deals:]
        return deal

    def _close_qty(self, p: _Pos, qty: float, price: float, *, t_ns: int, comment: str, magic: int,
                   commission: float | None = None) -> tuple[int, float]:
        """Close ``qty`` lots of ``p`` at executed ``price``; realise price PnL + swap share."""
        cs = self.instrument.contract_size
        vol = abs(p.lots)
        qty = min(qty, vol)
        frac = qty / vol if vol > 0 else 1.0
        sign = 1.0 if p.lots > 0 else -1.0
        pnl = sign * qty * cs * (price - p.price_open)
        swap = p.swap * frac
        comm = self.costs.commission(qty, instrument=self.instrument) if commission is None else commission
        self.balance += pnl + swap - comm
        p.swap -= swap
        remaining = round(vol - qty, 8)
        side = Side.SELL if p.lots > 0 else Side.BUY
        deal = self._record_deal(position_ticket=p.ticket, side=side, lots=qty, price=price, magic=magic,
                                 comment=comment, t_ns=t_ns, commission=comm, swap=swap, profit=pnl)
        if remaining <= _EPS:
            del self._positions[p.ticket]
        else:
            p.lots = sign * remaining
        return deal, pnl

    def _open(self, signed_lots: float, price: float, *, t_ns: int, magic: int, comment: str,
              sl: float | None, tp: float | None, commission: float) -> tuple[int, int]:
        ticket = self._new_ticket()
        self._positions[ticket] = _Pos(ticket=ticket, symbol=self.symbol, lots=signed_lots,
                                       price_open=price, sl=sl, tp=tp, magic=magic, comment=comment,
                                       open_ns=t_ns, swap_from_ns=t_ns)
        self.balance -= commission
        side = Side.BUY if signed_lots > 0 else Side.SELL
        deal = self._record_deal(position_ticket=ticket, side=side, lots=abs(signed_lots), price=price,
                                 magic=magic, comment=comment, t_ns=t_ns, commission=commission, swap=0.0,
                                 profit=0.0)
        return ticket, deal

    def _mark(self, now: pd.Timestamp) -> float:
        m = self.feed.mark(now)
        if m is None or not math.isfinite(m):
            q = self.feed.quote(now)
            if q is not None:
                return q.mid
            if self._positions:
                return next(iter(self._positions.values())).price_open
            return math.nan
        return float(m)

    def _equity(self, now: pd.Timestamp) -> float:
        if not self._positions:
            return self.balance
        mark = self._mark(now)
        cs = self.instrument.contract_size
        return self.balance + sum(p.lots * cs * (mark - p.price_open) + p.swap for p in self._positions.values())

    # ------------------------------------------------------------------------------------------
    # Broker protocol
    # ------------------------------------------------------------------------------------------
    def server_time(self) -> pd.Timestamp:
        return self.clock.now()

    def is_demo(self) -> bool:
        return True

    def is_hedging(self) -> bool:
        return self.hedging

    def account(self) -> AccountInfo:
        now = self.clock.now()
        self._settle(now)
        eq = self._equity(now)
        mark = self._mark(now) if self._positions else math.nan
        margin = sum(self.instrument.margin_required(p.lots, mark) for p in self._positions.values())
        lev = 1.0 / self.instrument.margin_rate if self.instrument.margin_rate > 0 else math.inf
        return AccountInfo(equity=eq, balance=self.balance, margin=margin, free_margin=eq - margin,
                           currency=self.currency, is_demo=True, leverage=lev, hedging=self.hedging,
                           server="paper", name="PaperBroker")

    def positions(self, symbol: str | None = None, magic: int | None = None) -> list[BrokerPosition]:
        now = self.clock.now()
        self._settle(now)
        mark = self._mark(now) if self._positions else math.nan
        cs = self.instrument.contract_size
        out = []
        for p in sorted(self._positions.values(), key=lambda x: (x.open_ns, x.ticket)):
            if symbol is not None and p.symbol != symbol:
                continue
            if magic is not None and p.magic != magic:
                continue
            out.append(BrokerPosition(ticket=p.ticket, symbol=p.symbol, lots=p.lots, price_open=p.price_open,
                                      sl=p.sl, tp=p.tp, magic=p.magic, comment=p.comment, time=_ts(p.open_ns),
                                      swap=p.swap, profit=p.lots * cs * (mark - p.price_open)))
        return out

    def quote(self, symbol: str) -> Quote | None:
        if symbol != self.symbol:
            raise BrokerError(f"PaperBroker trades {self.symbol}, not {symbol}")
        return self.feed.quote(self.clock.now())

    def latest_bars(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        if symbol != self.symbol:
            raise BrokerError(f"PaperBroker trades {self.symbol}, not {symbol}")
        if get_timeframe(timeframe).name != self.feed.timeframe:
            raise BrokerError(f"feed serves {self.feed.timeframe} bars, not {timeframe}")
        now = self.clock.now()
        self._settle(now)
        return self.feed.closed_bars(now, int(n))

    def find_deals(self, client_id: str, *, symbol: str, magic: int,
                   since: pd.Timestamp | None = None) -> list[BrokerDeal]:
        cid = client_id[:31]  # stored like an MT5 comment (31 chars), compared the same way
        return [d for d in self._deals
                if d.comment == cid and d.symbol == symbol and d.magic == magic
                and (since is None or d.time >= since)]

    def deals(self, *, magic: int | None = None) -> list[BrokerDeal]:
        return [d for d in self._deals if magic is None or d.magic == magic]

    def _reject(self, req: OrderRequest, status: OrderStatusCode, retcode: int, msg: str) -> OrderResult:
        logger.warning("paper order %s rejected (%s): %s", req.client_id, status.value, msg)
        return OrderResult(ok=False, retcode=retcode, message=msg, status=status, client_id=req.client_id,
                           position_ticket=req.position_ticket)

    def _valid_volume(self, lots: float) -> bool:
        inst = self.instrument
        if lots < inst.min_lot - _EPS or lots > inst.max_lot + _EPS:
            return False
        steps = lots / inst.lot_step
        return abs(steps - round(steps)) < 1e-6

    def place_order(self, order: OrderRequest) -> OrderResult:
        now = self.clock.now()
        self._settle(now)
        self.n_orders += 1
        req = order
        if req.symbol != self.symbol:
            return self._reject(req, OrderStatusCode.REJECTED, RETCODE_INVALID, f"unknown symbol {req.symbol}")
        if not self._valid_volume(req.lots):
            return self._reject(req, OrderStatusCode.INVALID_VOLUME, RETCODE_INVALID_VOLUME,
                                f"invalid volume {req.lots} (min {self.instrument.min_lot}, step "
                                f"{self.instrument.lot_step}, max {self.instrument.max_lot})")
        q = self.feed.quote(now)
        if q is None:
            return self._reject(req, OrderStatusCode.MARKET_CLOSED, RETCODE_MARKET_CLOSED, "market closed")
        target_pos: _Pos | None = None
        if req.position_ticket is not None:
            target_pos = self._positions.get(int(req.position_ticket))
            if target_pos is None:
                return self._reject(req, OrderStatusCode.REJECTED, RETCODE_INVALID,
                                    f"position {req.position_ticket} not found")
            if target_pos.magic != req.magic or target_pos.symbol != req.symbol:
                return self._reject(req, OrderStatusCode.REJECTED, RETCODE_REJECT,
                                    f"position {req.position_ticket} belongs to magic {target_pos.magic}")
            if (target_pos.lots > 0) == (int(req.side) > 0):
                return self._reject(req, OrderStatusCode.REJECTED, RETCODE_INVALID,
                                    "closing deal must be opposite to the position")
            if req.lots > abs(target_pos.lots) + _EPS:
                return self._reject(req, OrderStatusCode.INVALID_VOLUME, RETCODE_INVALID_VOLUME,
                                    f"close volume {req.lots} exceeds position {abs(target_pos.lots)}")
        # Margin check for exposure-increasing deals (MT5: "no money").
        if target_pos is None:
            eq = self._equity(now)
            used = sum(self.instrument.margin_required(p.lots, q.mid) for p in self._positions.values())
            add = self.instrument.margin_required(req.lots, q.mid)
            netting_reduces = (not self.hedging and self._netting_pos() is not None
                               and (self._netting_pos().lots > 0) != (int(req.side) > 0))
            if not netting_reduces and used + add > eq:
                return self._reject(req, OrderStatusCode.NO_MONEY, RETCODE_NO_MONEY,
                                    f"insufficient margin: need {used + add:,.2f}, equity {eq:,.2f}")

        fp = self.costs.fill_price(req.side, q.mid, q.spread, q.bar_range, req.lots, instrument=self.instrument)
        comm = self.costs.commission(req.lots, instrument=self.instrument)
        t_ns = _ns(now)
        side_sign = float(int(req.side))
        sl = req.stop_loss
        tp = req.take_profit
        if req.sl_distance is not None:
            sl = q.mid - side_sign * req.sl_distance
            sl = sl if sl > 0 else None
        if req.tp_distance is not None:
            tp = q.mid + side_sign * req.tp_distance
            tp = tp if tp > 0 else None
        comment = req.client_id[:31]

        if target_pos is not None:
            ticket = target_pos.ticket
            deal, _ = self._close_qty(target_pos, req.lots, fp.price, t_ns=t_ns, comment=comment,
                                      magic=req.magic, commission=comm)
        elif self.hedging:
            ticket, deal = self._open(side_sign * req.lots, fp.price, t_ns=t_ns, magic=req.magic,
                                      comment=comment, sl=sl, tp=tp, commission=comm)
        else:
            ticket, deal = self._netting_deal(side_sign * req.lots, fp.price, t_ns=t_ns, magic=req.magic,
                                              comment=comment, sl=sl, tp=tp, commission=comm)
        fill = Fill(client_id=req.client_id, symbol=self.symbol, side=req.side, lots=req.lots, price=fp.price,
                    time=now, commission=comm, slippage_cost=fp.slippage_cost, spread_cost=fp.spread_cost,
                    broker_ref=str(deal))
        self._save()
        logger.info("paper fill %s %s %.2f lots @ %.3f (mid %.3f) ticket %s", req.client_id,
                    "BUY" if side_sign > 0 else "SELL", req.lots, fp.price, q.mid, ticket)
        return OrderResult(ok=True, retcode=RETCODE_DONE, message="done", fill=fill, broker_ref=str(deal),
                           status=OrderStatusCode.FILLED, client_id=req.client_id, position_ticket=ticket,
                           requested_price=q.ask if side_sign > 0 else q.bid,
                           extra={"mid": q.mid, "spread": q.spread, "bar_range": q.bar_range, "sl": sl, "tp": tp})

    def _netting_pos(self) -> _Pos | None:
        for p in self._positions.values():
            if p.symbol == self.symbol:
                return p
        return None

    def _netting_deal(self, d: float, price: float, *, t_ns: int, magic: int, comment: str,
                      sl: float | None, tp: float | None, commission: float) -> tuple[int, int]:
        """MT5 netting: one position per symbol; deals add (VWAP), reduce or reverse it."""
        p = self._netting_pos()
        if p is None:
            return self._open(d, price, t_ns=t_ns, magic=magic, comment=comment, sl=sl, tp=tp,
                              commission=commission)
        if (p.lots > 0) == (d > 0):  # add: volume-weighted average entry
            vol = abs(p.lots) + abs(d)
            p.price_open = (abs(p.lots) * p.price_open + abs(d) * price) / vol
            p.lots = round(p.lots + d, 8)
            if sl is not None:
                p.sl = sl
            if tp is not None:
                p.tp = tp
            self.balance -= commission
            side = Side.BUY if d > 0 else Side.SELL
            deal = self._record_deal(position_ticket=p.ticket, side=side, lots=abs(d), price=price, magic=magic,
                                     comment=comment, t_ns=t_ns, commission=commission, swap=0.0, profit=0.0)
            return p.ticket, deal
        vol = abs(p.lots)
        if abs(d) <= vol + _EPS:  # reduce / close
            ticket = p.ticket
            deal, _ = self._close_qty(p, abs(d), price, t_ns=t_ns, comment=comment, magic=magic,
                                      commission=commission)
            return ticket, deal
        # reversal: close the whole position, open the remainder (commission split pro rata)
        rest = abs(d) - vol
        c_close = commission * vol / abs(d)
        self._close_qty(p, vol, price, t_ns=t_ns, comment=comment, magic=magic, commission=c_close)
        return self._open(math.copysign(rest, d), price, t_ns=t_ns, magic=magic, comment=comment, sl=sl,
                          tp=tp, commission=commission - c_close)

    def close_position(self, ticket: int, *, magic: int, lots: float | None = None,
                       client_id: str = "") -> OrderResult:
        p = self._positions.get(int(ticket))
        now = self.clock.now()
        if p is None:
            return OrderResult(ok=False, retcode=RETCODE_INVALID, message=f"position {ticket} not found",
                               status=OrderStatusCode.REJECTED, client_id=client_id)
        side = Side.SELL if p.lots > 0 else Side.BUY
        qty = abs(p.lots) if lots is None else float(lots)
        req = OrderRequest(client_id=client_id or f"close-{ticket}", symbol=p.symbol, side=side, lots=qty,
                           time=now, magic=magic, position_ticket=int(ticket), reason="close")
        return self.place_order(req)

    def close_all(self, symbol: str, magic: int) -> list[OrderResult]:
        return [self.close_position(p.ticket, magic=magic, client_id=f"{magic}-closeall-{p.ticket}")
                for p in self.positions(symbol, magic)]

    # ------------------------------------------------------------------------------------------
    # inspection helpers (not part of the protocol)
    # ------------------------------------------------------------------------------------------
    def deals_frame(self) -> pd.DataFrame:
        cols = ["ticket", "position_ticket", "time", "side", "lots", "price", "magic", "comment",
                "commission", "swap", "profit"]
        rows: Sequence[dict[str, Any]] = [
            {"ticket": d.ticket, "position_ticket": d.position_ticket, "time": d.time, "side": int(d.side),
             "lots": d.lots, "price": d.price, "magic": d.magic, "comment": d.comment,
             "commission": d.commission, "swap": d.swap, "profit": d.profit} for d in self._deals]
        return pd.DataFrame(list(rows), columns=cols)

    def equity(self) -> float:
        now = self.clock.now()
        self._settle(now)
        return self._equity(now)
