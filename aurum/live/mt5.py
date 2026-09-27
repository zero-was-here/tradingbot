"""MetaTrader 5 venue adapter (SPEC §11) — the production execution path.

Uses the official ``MetaTrader5`` Python package, which only exists for Windows and talks
to a locally running MT5 terminal. The package is imported lazily, so importing
``aurum.live`` works everywhere; tests inject a fake module through ``sys.modules``.

Safety properties
-----------------
* **Credentials** come from the environment (``MT5_LOGIN``, ``MT5_PASSWORD``,
  ``MT5_SERVER``, ``MT5_PATH``) and are never logged, stored or put in exceptions.
* **Magic isolation.** ``positions`` filters by symbol AND magic; closing or modifying a
  ticket that belongs to another magic/symbol is refused (the legacy v1 bot filtered by
  symbol only and could close other EAs' trades).
* **Closed bars only.** ``copy_rates_from_pos`` returns the forming bar last; it is dropped
  unless its close time (``open + timeframe``, converted to UTC) has passed.
* **Server time → UTC.** MT5 stamps bars, ticks and deals with the broker *server's* wall
  clock encoded as epoch seconds (most metals brokers run "New-York close" servers:
  UTC+2 in US winter, UTC+3 in US summer — ``"NY+7"`` in :mod:`aurum.data.loaders`). The
  offset is detected from a fresh tick (``tick.time - utc_now`` rounded to 15 minutes) and,
  when it matches NY+7's current offset, the DST-aware ``"NY+7"`` rule is used so a
  history window spanning a DST switch converts correctly. A configured ``server_tz`` that
  disagrees with a fresh detection is a fatal misconfiguration (bars would be shifted by
  hours and a forming bar could be taken for a closed one).
* **Retcodes.** Requote / price-changed / price-off / invalid-price are retried with a
  FRESH tick; an unsupported filling mode falls back FOK → IOC → RETURN; market closed,
  no money, invalid volume/stops, trading disabled are rejections; timeouts and connection
  losses are ``unknown`` (the OMS verifies before any resend).
* **Broker-side protection.** Stop-loss/take-profit levels travel with the order. Levels
  are MID levels (as in the simulator) converted to MT5's trigger side (longs trigger on
  the bid, shorts on the ask) with the current half spread, so the exit price matches the
  simulator's ``level ∓ spread/2`` convention. ``positions()`` converts back (trigger level
  ± current half spread), so ``BrokerPosition.sl/tp`` are MID levels like every other venue
  and a stop re-used for a scale-in lands on the same trigger level.
* **Closing** is always an opposite deal on the specific ticket (``position=ticket``).

Reference: MetaQuotes, "MetaTrader 5 Python integration" (``order_send``,
``copy_rates_from_pos``, ``positions_get``, trade server return codes).
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.timeframes import get_timeframe
from aurum.core.types import Fill, Side
from aurum.data.loaders import NY_TZ, to_utc_index
from aurum.data.schema import make_bars
from aurum.live.broker import (
    AccountInfo,
    BrokerDeal,
    BrokerError,
    BrokerPosition,
    Clock,
    OrderRequest,
    OrderResult,
    OrderStatusCode,
    Quote,
    SystemClock,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MT5Broker",
    "MT5UnavailableError",
    "classify_retcode",
    "detect_server_offset",
    "import_mt5",
    "server_tz_for_offset",
]

#: Fallback values of the MetaTrader5 constants (used when the module lacks an attribute).
MT5_DEFAULTS: dict[str, int] = {
    "TIMEFRAME_M1": 1, "TIMEFRAME_M5": 5, "TIMEFRAME_M15": 15, "TIMEFRAME_M30": 30,
    "TIMEFRAME_H1": 16385, "TIMEFRAME_H4": 16388, "TIMEFRAME_D1": 16408,
    "ORDER_TYPE_BUY": 0, "ORDER_TYPE_SELL": 1, "TRADE_ACTION_DEAL": 1, "TRADE_ACTION_SLTP": 6,
    "ORDER_TIME_GTC": 0, "ORDER_FILLING_FOK": 0, "ORDER_FILLING_IOC": 1, "ORDER_FILLING_RETURN": 2,
    "SYMBOL_FILLING_FOK": 1, "SYMBOL_FILLING_IOC": 2,
    "POSITION_TYPE_BUY": 0, "POSITION_TYPE_SELL": 1, "DEAL_TYPE_BUY": 0, "DEAL_TYPE_SELL": 1,
    "ACCOUNT_TRADE_MODE_DEMO": 0, "ACCOUNT_TRADE_MODE_CONTEST": 1, "ACCOUNT_TRADE_MODE_REAL": 2,
    "ACCOUNT_MARGIN_MODE_RETAIL_NETTING": 0, "ACCOUNT_MARGIN_MODE_EXCHANGE": 1,
    "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING": 2,
    "TRADE_RETCODE_REQUOTE": 10004, "TRADE_RETCODE_REJECT": 10006, "TRADE_RETCODE_CANCEL": 10007,
    "TRADE_RETCODE_PLACED": 10008, "TRADE_RETCODE_DONE": 10009, "TRADE_RETCODE_DONE_PARTIAL": 10010,
    "TRADE_RETCODE_ERROR": 10011, "TRADE_RETCODE_TIMEOUT": 10012, "TRADE_RETCODE_INVALID": 10013,
    "TRADE_RETCODE_INVALID_VOLUME": 10014, "TRADE_RETCODE_INVALID_PRICE": 10015,
    "TRADE_RETCODE_INVALID_STOPS": 10016, "TRADE_RETCODE_TRADE_DISABLED": 10017,
    "TRADE_RETCODE_MARKET_CLOSED": 10018, "TRADE_RETCODE_NO_MONEY": 10019,
    "TRADE_RETCODE_PRICE_CHANGED": 10020, "TRADE_RETCODE_PRICE_OFF": 10021,
    "TRADE_RETCODE_TOO_MANY_REQUESTS": 10024, "TRADE_RETCODE_SERVER_DISABLES_AT": 10026,
    "TRADE_RETCODE_CLIENT_DISABLES_AT": 10027, "TRADE_RETCODE_LOCKED": 10028,
    "TRADE_RETCODE_FROZEN": 10029, "TRADE_RETCODE_INVALID_FILL": 10030,
    "TRADE_RETCODE_CONNECTION": 10031, "TRADE_RETCODE_ONLY_REAL": 10032,
    "TRADE_RETCODE_LIMIT_ORDERS": 10033, "TRADE_RETCODE_LIMIT_VOLUME": 10034,
    "TRADE_RETCODE_INVALID_CLOSE_VOLUME": 10038, "TRADE_RETCODE_LIMIT_POSITIONS": 10040,
    "TRADE_RETCODE_LONG_ONLY": 10042, "TRADE_RETCODE_SHORT_ONLY": 10043,
    "TRADE_RETCODE_CLOSE_ONLY": 10044, "TRADE_RETCODE_FIFO_CLOSE": 10045,
}

_REQUOTE_CODES = (10004, 10015, 10020, 10021)       # retry with a fresh tick
_STATUS_BY_CODE: dict[int, OrderStatusCode] = {
    10009: OrderStatusCode.FILLED, 10010: OrderStatusCode.PARTIAL,
    10008: OrderStatusCode.UNKNOWN,   # "placed" for a market order: execution not confirmed
    10004: OrderStatusCode.REQUOTE, 10015: OrderStatusCode.REQUOTE, 10020: OrderStatusCode.REQUOTE,
    10021: OrderStatusCode.MARKET_CLOSED,  # no quotes (after fresh-tick retries)
    10018: OrderStatusCode.MARKET_CLOSED, 10019: OrderStatusCode.NO_MONEY,
    10014: OrderStatusCode.INVALID_VOLUME, 10034: OrderStatusCode.INVALID_VOLUME,
    10038: OrderStatusCode.INVALID_VOLUME, 10016: OrderStatusCode.INVALID_STOPS,
    10017: OrderStatusCode.TRADE_DISABLED, 10026: OrderStatusCode.TRADE_DISABLED,
    10027: OrderStatusCode.TRADE_DISABLED, 10032: OrderStatusCode.TRADE_DISABLED,
    10042: OrderStatusCode.TRADE_DISABLED, 10043: OrderStatusCode.TRADE_DISABLED,
    10044: OrderStatusCode.TRADE_DISABLED, 10045: OrderStatusCode.TRADE_DISABLED,
    10028: OrderStatusCode.REJECTED, 10029: OrderStatusCode.REJECTED,
    10024: OrderStatusCode.TOO_MANY_REQUESTS,
    10011: OrderStatusCode.UNKNOWN, 10012: OrderStatusCode.UNKNOWN, 10031: OrderStatusCode.UNKNOWN,
}
_INVALID_FILL = 10030
_FILLING_ORDER = ("FOK", "IOC", "RETURN")


class MT5UnavailableError(BrokerError):
    """The MetaTrader5 package cannot be imported (it is Windows-only)."""


def import_mt5() -> Any:
    """Import the ``MetaTrader5`` package (or a test double registered in ``sys.modules``)."""
    try:
        return importlib.import_module("MetaTrader5")
    except ImportError as exc:
        hint = ("the MetaTrader5 package is only available on Windows (it drives a local MT5 "
                "terminal); run the live runner on Windows or use the paper broker"
                if not sys.platform.startswith("win") else "pip install MetaTrader5")
        raise MT5UnavailableError(f"cannot import MetaTrader5: {hint}") from exc


def classify_retcode(retcode: int | None) -> OrderStatusCode:
    """Normalised status of an MT5 trade-server return code."""
    if retcode is None:
        return OrderStatusCode.UNKNOWN
    return _STATUS_BY_CODE.get(int(retcode), OrderStatusCode.REJECTED)


def detect_server_offset(tick_time_s: float, utc_now: pd.Timestamp, *, quantum_s: int = 900,
                         max_tick_age_s: float = 120.0, max_abs_offset_h: float = 14.0) -> pd.Timedelta | None:
    """Server-clock offset from a FRESH tick stamped in server wall-clock epoch seconds.

    ``tick_time_s - utc_now = offset - tick_age``; the offset is rounded to ``quantum_s``
    (15 min: every real-world offset is a multiple) and must lie within ``±max_abs_offset_h``.
    The caller must establish that the tick is fresh (see ``MT5Broker._fresh_tick_time``): an
    old tick aliases onto a wrong quarter-hour (a 62-minute-old tick after the daily break
    would look like a one-hour-smaller offset), which no arithmetic on one timestamp can detect.
    """
    raw = float(tick_time_s) - pd.Timestamp(utc_now).timestamp()
    offset = round(raw / quantum_s) * quantum_s
    age = offset - raw
    if abs(offset) > max_abs_offset_h * 3600 or not -60.0 <= age <= max_tick_age_s:
        return None
    return pd.Timedelta(seconds=offset)


def _ny7_offset(utc_now: pd.Timestamp) -> pd.Timedelta:
    """Current UTC offset of a New-York-close ("NY+7") server."""
    ny = pd.Timestamp(utc_now).tz_convert(NY_TZ)
    return ny.utcoffset() + pd.Timedelta(hours=7)


def server_tz_for_offset(offset: pd.Timedelta, utc_now: pd.Timestamp) -> str:
    """Timezone spec for a detected offset: ``"NY+7"`` if it matches the DST-aware NY-close
    convention now, else a fixed ``"UTC+hh:mm"`` offset."""
    if offset == _ny7_offset(utc_now):
        return "NY+7"
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    h, m = divmod(abs(total) // 60, 60)
    return f"UTC{sign}{h:02d}:{m:02d}"


_OFFSET_RE = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


def _tz_offset_now(server_tz: str, utc_now: pd.Timestamp) -> pd.Timedelta:
    """Offset (server wall clock - UTC) of ``server_tz`` at ``utc_now`` (same specs as
    :func:`aurum.data.loaders.to_utc_index`)."""
    spec = str(server_tz).strip()
    upper = spec.upper()
    if upper in ("UTC", "GMT", "Z", "ETC/UTC", "ETC/GMT"):
        return pd.Timedelta(0)
    if upper == "NY+7":
        return _ny7_offset(utc_now)
    m = _OFFSET_RE.match(spec)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        return sign * pd.Timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
    return pd.Timestamp(utc_now).tz_convert(spec).utcoffset()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class MT5Broker:
    """:class:`~aurum.live.broker.Broker` backed by a MetaTrader 5 terminal.

    Parameters
    ----------
    symbol          : venue symbol (brokers use suffixes, e.g. ``"XAUUSD.a"``).
    instrument      : contract spec (lot grid is refreshed from ``symbol_info`` on connect).
    server_tz       : ``"auto"`` (detect from a fresh tick), or any spec accepted by
                      :func:`aurum.data.loaders.to_utc_index` (e.g. ``"NY+7"``).
    deviation_points: max accepted slippage for market orders, in points.
    price_basis     : MT5 bars are BID bars; ``"bid"`` converts to mid with ``+spread/2``.
    requote_retries : fresh-tick retries on requote / price changed / price off.
    mt5             : injected module (tests); default imports ``MetaTrader5`` lazily.
    env             : mapping to read credentials from (default ``os.environ``).
    clock           : UTC wall clock (NTP-synchronised machine clock).
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        *,
        instrument: Instrument = XAUUSD,
        server_tz: str = "auto",
        deviation_points: int = 20,
        price_basis: str = "bid",
        requote_retries: int = 3,
        requote_delay_s: float = 0.25,
        tz_probe_seconds: float = 5.0,
        mt5: Any | None = None,
        env: dict[str, str] | None = None,
        clock: Clock | None = None,
        connect: bool = True,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if price_basis not in ("bid", "mid"):
            raise ValueError("price_basis must be 'bid' or 'mid'")
        self.symbol = symbol
        self.instrument = instrument
        self.server_tz_config = server_tz
        self.deviation_points = int(deviation_points)
        self.price_basis = price_basis
        self.requote_retries = int(requote_retries)
        self.requote_delay_s = float(requote_delay_s)
        self.tz_probe_seconds = float(tz_probe_seconds)
        self._mt5 = mt5
        self._env = env if env is not None else os.environ
        self.clock = clock or SystemClock()
        self._sleep = sleep if sleep is not None else time.sleep
        self._server_tz: str | None = None if server_tz == "auto" else server_tz
        self._filling: str | None = None
        self._symbol_info: Any = None
        self.point = 0.01
        self.digits = 2
        self._connected = False
        if connect:
            self.connect()

    # ------------------------------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------------------------------
    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            self._mt5 = import_mt5()
        return self._mt5

    def _c(self, name: str) -> int:
        return int(getattr(self.mt5, name, MT5_DEFAULTS[name]))

    def _last_error(self) -> str:
        try:
            return str(self.mt5.last_error())
        except Exception:  # pragma: no cover
            return "unknown error"

    def connect(self) -> None:
        """Initialise the terminal connection, log in from env vars, select the symbol."""
        mt5 = self.mt5
        kwargs: dict[str, Any] = {}
        path = self._env.get("MT5_PATH")
        if path:
            kwargs["path"] = path
        login = self._env.get("MT5_LOGIN")
        if login:
            try:
                kwargs["login"] = int(login)
            except ValueError as exc:
                raise BrokerError("MT5_LOGIN must be an integer account number") from exc
            password = self._env.get("MT5_PASSWORD")
            server = self._env.get("MT5_SERVER")
            if password:
                kwargs["password"] = password
            if server:
                kwargs["server"] = server
        ok = mt5.initialize(**kwargs)
        if not ok:
            # never include kwargs (credentials) in the message
            raise BrokerError(f"MT5 initialize failed: {self._last_error()} "
                              f"(login provided: {'yes' if login else 'no'}, path provided: {'yes' if path else 'no'})")
        logger.info("MT5 terminal initialised (explicit login: %s)", "yes" if login else "no")
        info = mt5.symbol_info(self.symbol)
        if info is None or not _get(info, "visible", True):
            if not mt5.symbol_select(self.symbol, True):
                raise BrokerError(f"MT5 symbol_select({self.symbol!r}) failed: {self._last_error()}")
            info = mt5.symbol_info(self.symbol)
        if info is None:
            raise BrokerError(f"MT5 symbol {self.symbol!r} not found: {self._last_error()}")
        self._symbol_info = info
        self.point = float(_get(info, "point", 0.01) or 0.01)
        self.digits = int(_get(info, "digits", 2) or 2)
        vmin, vstep, vmax = (_get(info, "volume_min"), _get(info, "volume_step"), _get(info, "volume_max"))
        cs = _get(info, "trade_contract_size")
        if cs and abs(float(cs) - self.instrument.contract_size) > 1e-9:
            raise BrokerError(f"venue contract size {cs} != instrument contract size "
                              f"{self.instrument.contract_size}: sizing would be wrong")
        if vmin and vstep and vmax:
            from dataclasses import replace

            self.instrument = replace(self.instrument, min_lot=float(vmin), lot_step=float(vstep),
                                      max_lot=min(float(vmax), self.instrument.max_lot))
        term = mt5.terminal_info() if hasattr(mt5, "terminal_info") else None
        if term is not None and not _get(term, "trade_allowed", True):
            logger.warning("MT5 terminal: AutoTrading is disabled — orders will be rejected (10027)")
        self._connected = True
        self._check_server_tz()

    def reconnect(self) -> None:
        """Re-initialise the terminal connection (called by the runner after venue errors)."""
        try:
            self.shutdown()
        except Exception:  # pragma: no cover
            pass
        self.connect()

    def shutdown(self) -> None:
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            finally:
                self._connected = False

    # ------------------------------------------------------------------------------------------
    # time
    # ------------------------------------------------------------------------------------------
    def _raw_tick(self) -> Any:
        return self.mt5.symbol_info_tick(self.symbol)

    @staticmethod
    def _tick_msc(tick: Any) -> int | None:
        if tick is None:
            return None
        msc = _get(tick, "time_msc")
        if msc:
            return int(msc)
        t = _get(tick, "time")
        return int(t) * 1000 if t else None

    def _fresh_tick_time(self) -> float | None:
        """Server time (s) of a tick seen ARRIVING while we watch (``time_msc`` changes within
        ``tz_probe_seconds``), so its age is at most one poll interval. None when the market is
        quiet or closed — then the last tick's age is unknown and detection is unsafe."""
        first = self._tick_msc(self._raw_tick())
        if first is None:
            return None
        polls = max(1, int(self.tz_probe_seconds / 0.25))
        for _ in range(polls):
            self._sleep(0.25)
            cur = self._tick_msc(self._raw_tick())
            if cur is not None and cur != first:
                return cur / 1000.0
        return None

    def _check_server_tz(self) -> None:
        """Detect the server offset from a fresh tick and reconcile it with the config."""
        tick_s = self._fresh_tick_time()
        now = self.clock.now()
        detected = detect_server_offset(tick_s, now) if tick_s is not None else None
        if detected is None:
            if self._server_tz is None:
                raise BrokerError("cannot detect the MT5 server time offset (no fresh tick: market closed?); "
                                  "set server_tz explicitly (e.g. 'NY+7')")
            logger.info("MT5 server offset not detectable now (no fresh tick); using configured %s", self._server_tz)
            return
        if self._server_tz is None:
            self._server_tz = server_tz_for_offset(detected, now)
            logger.info("MT5 server time zone detected: %s (offset %s)", self._server_tz, detected)
            return
        expected = _tz_offset_now(self._server_tz, now)
        if abs((expected - detected).total_seconds()) >= 900:
            raise BrokerError(f"configured server_tz {self._server_tz!r} implies offset {expected}, but the "
                              f"server clock is at {detected}: fix server_tz (bars would be mis-timed)")

    @property
    def server_tz(self) -> str:
        if self._server_tz is None:
            self._check_server_tz()
        assert self._server_tz is not None
        return self._server_tz

    def to_utc(self, server_seconds: Any) -> pd.DatetimeIndex:
        """Server wall-clock epoch seconds -> UTC timestamps."""
        secs = np.asarray(server_seconds, dtype="int64").reshape(-1)
        naive = pd.DatetimeIndex(pd.to_datetime(secs, unit="s"))
        return to_utc_index(naive, self.server_tz)

    def _to_server_seconds(self, t: pd.Timestamp) -> int:
        off = _tz_offset_now(self.server_tz, t)
        return int((pd.Timestamp(t).tz_convert("UTC") + off).timestamp())

    def server_time(self) -> pd.Timestamp:
        """UTC now. The machine clock (NTP) is authoritative; a fresh tick stamped ahead of it
        reveals clock skew and wins (never act on a bar the server has not closed)."""
        now = self.clock.now()
        try:
            tick = self._raw_tick()
        except Exception:  # pragma: no cover - connection hiccup
            return now
        if tick is not None and _get(tick, "time"):
            t = self.to_utc([int(_get(tick, "time"))])[0]
            if t > now + pd.Timedelta(seconds=30):
                logger.warning("MT5 tick time %s is ahead of the local clock %s: clock skew", t, now)
                return t
        return now

    # ------------------------------------------------------------------------------------------
    # account / positions / market data
    # ------------------------------------------------------------------------------------------
    def _account_raw(self) -> Any:
        info = self.mt5.account_info()
        if info is None:
            raise BrokerError(f"MT5 account_info failed: {self._last_error()}")
        return info

    def account(self) -> AccountInfo:
        a = self._account_raw()
        return AccountInfo(
            equity=float(_get(a, "equity", 0.0)), balance=float(_get(a, "balance", 0.0)),
            margin=float(_get(a, "margin", 0.0)), free_margin=float(_get(a, "margin_free", 0.0)),
            currency=str(_get(a, "currency", "USD")), is_demo=self._is_demo(a),
            leverage=float(_get(a, "leverage", 0.0) or 0.0), hedging=self._is_hedging(a),
            server=_get(a, "server"), name=None,  # the account holder's name is PII: not exported
        )

    def _is_demo(self, a: Any) -> bool:
        return int(_get(a, "trade_mode", -1)) == self._c("ACCOUNT_TRADE_MODE_DEMO")

    def _is_hedging(self, a: Any) -> bool:
        return int(_get(a, "margin_mode", -1)) == self._c("ACCOUNT_MARGIN_MODE_RETAIL_HEDGING")

    def is_demo(self) -> bool:
        return self._is_demo(self._account_raw())

    def is_hedging(self) -> bool:
        return self._is_hedging(self._account_raw())

    def _half_spread(self) -> float:
        """Current half spread (0 when no valid tick): converts venue trigger levels <-> mid."""
        try:
            tick = self._raw_tick()
        except Exception:  # pragma: no cover - connection hiccup
            return 0.0
        bid, ask = float(_get(tick, "bid", 0.0) or 0.0), float(_get(tick, "ask", 0.0) or 0.0)
        return 0.5 * (ask - bid) if bid > 0 and ask >= bid else 0.0

    def _position(self, p: Any, half_spread: float = 0.0) -> BrokerPosition:
        """Venue position -> :class:`BrokerPosition`. ``sl``/``tp`` are reported as MID levels
        (the convention of :class:`~aurum.live.broker.OrderRequest`): MT5 stores trigger levels
        (long exits trigger on the bid = mid - half spread, shorts on the ask), so the current
        half spread is added back. Without this, re-using a position's stop for a scale-in
        would move the stop another half spread away on every add."""
        typ = int(_get(p, "type", 0))
        vol = float(_get(p, "volume", 0.0))
        lots = vol if typ == self._c("POSITION_TYPE_BUY") else -vol
        side = 1.0 if lots > 0 else -1.0
        sl = float(_get(p, "sl", 0.0) or 0.0)
        tp = float(_get(p, "tp", 0.0) or 0.0)
        sl_mid = sl + side * half_spread if sl > 0 else None
        tp_mid = tp + side * half_spread if tp > 0 else None
        return BrokerPosition(
            ticket=int(_get(p, "ticket")), symbol=str(_get(p, "symbol")), lots=lots,
            price_open=float(_get(p, "price_open", 0.0)), sl=sl_mid, tp=tp_mid,
            magic=int(_get(p, "magic", 0)), comment=str(_get(p, "comment", "") or ""),
            time=self.to_utc([int(_get(p, "time", 0))])[0], swap=float(_get(p, "swap", 0.0) or 0.0),
            profit=float(_get(p, "profit", 0.0) or 0.0),
        )

    def positions(self, symbol: str | None = None, magic: int | None = None) -> list[BrokerPosition]:
        mt5 = self.mt5
        raw = mt5.positions_get(symbol=symbol) if symbol is not None else mt5.positions_get()
        if raw is None:
            raise BrokerError(f"MT5 positions_get failed: {self._last_error()}")
        half = self._half_spread() if any(_get(p, "sl") or _get(p, "tp") for p in raw) else 0.0
        out = [self._position(p, half if str(_get(p, "symbol")) == self.symbol else 0.0) for p in raw]
        if symbol is not None:
            out = [p for p in out if p.symbol == symbol]
        if magic is not None:
            out = [p for p in out if p.magic == int(magic)]
        return sorted(out, key=lambda p: (p.time, p.ticket))

    def quote(self, symbol: str) -> Quote | None:
        if symbol != self.symbol:
            raise BrokerError(f"MT5Broker is bound to {self.symbol}, not {symbol}")
        tick = self._raw_tick()
        if tick is None:
            return None
        bid, ask = float(_get(tick, "bid", 0.0)), float(_get(tick, "ask", 0.0))
        if not (bid > 0 and ask > 0):
            return None
        t = self.to_utc([int(_get(tick, "time", 0))])[0]
        if self.clock.now() - t > pd.Timedelta(minutes=10):
            return None  # stale quote: treat the market as closed
        return Quote(time=t, bid=bid, ask=ask)

    def latest_bars(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        """Last ``n`` CLOSED bars as canonical (UTC, mid) bars."""
        if symbol != self.symbol:
            raise BrokerError(f"MT5Broker is bound to {self.symbol}, not {symbol}")
        tf = get_timeframe(timeframe)
        code = self._c(f"TIMEFRAME_{tf.name}")
        rates = self.mt5.copy_rates_from_pos(symbol, code, 0, int(n) + 1)
        if rates is None:
            raise BrokerError(f"MT5 copy_rates_from_pos failed: {self._last_error()}")
        if len(rates) == 0:
            return make_bars(pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread"],
                                          index=pd.DatetimeIndex([], tz="UTC")), tf.name)
        df = pd.DataFrame(np.asarray(rates))
        idx = self.to_utc(df["time"].to_numpy())
        spread_px = df["spread"].to_numpy(dtype=float) * self.point if "spread" in df.columns else np.zeros(len(df))
        adj = spread_px / 2.0 if self.price_basis == "bid" else 0.0
        vol = df["tick_volume"] if "tick_volume" in df.columns else df.get("real_volume", 0.0)
        out = pd.DataFrame({
            "open": df["open"].to_numpy(dtype=float) + adj, "high": df["high"].to_numpy(dtype=float) + adj,
            "low": df["low"].to_numpy(dtype=float) + adj, "close": df["close"].to_numpy(dtype=float) + adj,
            "volume": np.asarray(vol, dtype=float), "spread": spread_px,
        }, index=idx)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        now = self.server_time()
        closed = (out.index + tf.delta) <= now
        dropped = int((~closed).sum())
        if dropped:
            logger.debug("dropped %d forming bar(s) from MT5 rates", dropped)
        bars = make_bars(out.loc[closed], tf.name)
        return bars.iloc[-int(n):] if n else bars.iloc[0:0]

    # ------------------------------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------------------------------
    def _filling_candidates(self) -> list[str]:
        flags = int(_get(self._symbol_info, "filling_mode", 0) or 0)
        cands = []
        for name in _FILLING_ORDER:
            if name == "FOK" and flags and not flags & self._c("SYMBOL_FILLING_FOK"):
                continue
            if name == "IOC" and flags and not flags & self._c("SYMBOL_FILLING_IOC"):
                continue
            cands.append(name)
        if self._filling in cands:  # the mode that worked last time goes first
            cands.remove(self._filling)
            cands.insert(0, self._filling)
        return cands

    def _venue_level(self, mid_level: float | None, position_side: int, kind: str, half_spread: float) -> float:
        """Mid level -> MT5 trigger level (long exits trigger on the bid, short exits on the ask)."""
        if mid_level is None:
            return 0.0
        lvl = mid_level - half_spread if position_side > 0 else mid_level + half_spread
        return round(max(lvl, 0.0), self.digits)

    def _own_position(self, ticket: int, magic: int) -> tuple[BrokerPosition | None, str]:
        raw = self.mt5.positions_get(ticket=int(ticket))
        if not raw:
            return None, f"position {ticket} not found"
        pos = self._position(raw[0])
        if pos.symbol != self.symbol or pos.magic != int(magic):
            return None, (f"position {ticket} belongs to {pos.symbol}/magic {pos.magic}, "
                          f"not {self.symbol}/magic {magic}: refusing to touch it")
        return pos, ""

    def place_order(self, order: OrderRequest) -> OrderResult:
        req = order
        if req.symbol != self.symbol:
            return OrderResult(ok=False, retcode="symbol", message=f"bound to {self.symbol}",
                               status=OrderStatusCode.REJECTED, client_id=req.client_id)
        side = int(req.side)
        pos_side = side  # side of the position the protective levels protect
        if req.position_ticket is not None:
            pos, why = self._own_position(req.position_ticket, req.magic)
            if pos is None:
                logger.error("order %s refused: %s", req.client_id, why)
                return OrderResult(ok=False, retcode="ownership", message=why, status=OrderStatusCode.REJECTED,
                                   client_id=req.client_id, position_ticket=req.position_ticket)
            if (pos.lots > 0) == (side > 0):
                return OrderResult(ok=False, retcode="side", message="closing deal must oppose the position",
                                   status=OrderStatusCode.REJECTED, client_id=req.client_id)
        comment = req.client_id
        if len(comment) > 31:
            logger.warning("client id %r longer than MT5's 31-char comment; truncated", comment)
            comment = comment[:31]
        volume = round(req.lots, 8)
        fillings = self._filling_candidates()
        fi = 0
        requotes = 0
        last: OrderResult | None = None
        while True:
            tick = self._raw_tick()
            if tick is None or not (_get(tick, "bid", 0) and _get(tick, "ask", 0)):
                return OrderResult(ok=False, retcode="no_tick", message=f"no quote: {self._last_error()}",
                                   status=OrderStatusCode.MARKET_CLOSED, client_id=req.client_id)
            bid, ask = float(_get(tick, "bid")), float(_get(tick, "ask"))
            mid, half = 0.5 * (bid + ask), 0.5 * (ask - bid)
            price = ask if side > 0 else bid
            sl_mid, tp_mid = req.stop_loss, req.take_profit
            if req.sl_distance is not None:
                sl_mid = mid - pos_side * req.sl_distance
            if req.tp_distance is not None:
                tp_mid = mid + pos_side * req.tp_distance
            request: dict[str, Any] = {
                "action": self._c("TRADE_ACTION_DEAL"), "symbol": self.symbol, "volume": volume,
                "type": self._c("ORDER_TYPE_BUY") if side > 0 else self._c("ORDER_TYPE_SELL"),
                "price": price, "deviation": self.deviation_points, "magic": int(req.magic),
                "comment": comment, "type_time": self._c("ORDER_TIME_GTC"),
                "type_filling": self._c(f"ORDER_FILLING_{fillings[fi]}"),
            }
            if req.position_ticket is not None:
                request["position"] = int(req.position_ticket)
            else:
                request["sl"] = self._venue_level(sl_mid, pos_side, "sl", half)
                request["tp"] = self._venue_level(tp_mid, pos_side, "tp", half)
            res = self.mt5.order_send(request)
            if res is None:
                msg = f"order_send returned None: {self._last_error()}"
                logger.error("order %s: %s", req.client_id, msg)
                return OrderResult(ok=False, retcode="none", message=msg, status=OrderStatusCode.UNKNOWN,
                                   client_id=req.client_id, requested_price=price)
            code = int(_get(res, "retcode", -1))
            if code == _INVALID_FILL and fi + 1 < len(fillings):
                logger.warning("filling mode %s rejected for %s; trying %s", fillings[fi], self.symbol,
                               fillings[fi + 1])
                fi += 1
                continue
            if code in _REQUOTE_CODES and requotes < self.requote_retries:
                requotes += 1
                logger.warning("order %s requote/price change (%d); retry %d/%d with a fresh tick",
                               req.client_id, code, requotes, self.requote_retries)
                self._sleep(self.requote_delay_s)
                continue
            status = classify_retcode(code)
            if code == _INVALID_FILL:
                status = OrderStatusCode.REJECTED
            last = self._result(req, res, code, status, price, bid, ask, sl_mid, tp_mid)
            if status in (OrderStatusCode.FILLED, OrderStatusCode.PARTIAL):
                self._filling = fillings[fi]
            break
        return last

    def _result(self, req: OrderRequest, res: Any, code: int, status: OrderStatusCode, price: float,
                bid: float, ask: float, sl_mid: float | None, tp_mid: float | None) -> OrderResult:
        comment = str(_get(res, "comment", "") or "")
        ok = status in (OrderStatusCode.FILLED, OrderStatusCode.PARTIAL)
        fill = None
        deal = _get(res, "deal")
        if ok:
            px = float(_get(res, "price", 0.0) or price)
            vol = float(_get(res, "volume", 0.0) or req.lots)
            cs = self.instrument.contract_size
            side = int(req.side)
            commission = self._deal_commission(deal)
            fill = Fill(client_id=req.client_id, symbol=self.symbol, side=Side(side), lots=vol, price=px,
                        time=self.clock.now(), commission=commission,
                        slippage_cost=side * (px - price) * vol * cs,
                        spread_cost=0.5 * (ask - bid) * vol * cs, broker_ref=str(deal) if deal else None)
        ticket = req.position_ticket
        if ticket is None and ok:
            ticket = self._ticket_for_order(_get(res, "order"), deal)
        msg = f"{code} {comment}".strip()
        if not ok:
            logger.error("MT5 order %s failed: retcode %d (%s) -> %s", req.client_id, code, comment, status.value)
        return OrderResult(ok=ok, retcode=code, message=msg, fill=fill, broker_ref=str(deal) if deal else None,
                           status=status, retryable=status == OrderStatusCode.TOO_MANY_REQUESTS,
                           client_id=req.client_id, position_ticket=ticket, requested_price=price,
                           extra={"mid": 0.5 * (bid + ask), "spread": ask - bid, "sl": sl_mid, "tp": tp_mid})

    def _deal_commission(self, deal: Any) -> float:
        if not deal or not hasattr(self.mt5, "history_deals_get"):
            return 0.0
        try:
            deals = self.mt5.history_deals_get(ticket=int(deal))
        except Exception:  # pragma: no cover
            return 0.0
        if not deals:
            return 0.0
        return abs(float(_get(deals[0], "commission", 0.0) or 0.0))

    def _ticket_for_order(self, order: Any, deal: Any) -> int | None:
        """Position ticket created by a market order (MT5: position id == opening order ticket)."""
        if deal and hasattr(self.mt5, "history_deals_get"):
            try:
                deals = self.mt5.history_deals_get(ticket=int(deal))
                if deals:
                    pid = _get(deals[0], "position_id")
                    if pid:
                        return int(pid)
            except Exception:  # pragma: no cover
                pass
        return int(order) if order else None

    def close_position(self, ticket: int, *, magic: int, lots: float | None = None,
                       client_id: str = "") -> OrderResult:
        pos, why = self._own_position(ticket, magic)
        if pos is None:
            logger.error("close of ticket %s refused: %s", ticket, why)
            return OrderResult(ok=False, retcode="ownership", message=why, status=OrderStatusCode.REJECTED,
                               client_id=client_id, position_ticket=int(ticket))
        side = Side.SELL if pos.lots > 0 else Side.BUY
        qty = abs(pos.lots) if lots is None else float(lots)
        req = OrderRequest(client_id=client_id or f"{magic}-close-{ticket}"[:31], symbol=self.symbol, side=side,
                           lots=qty, time=self.clock.now(), magic=int(magic), position_ticket=int(ticket),
                           reason="close")
        return self.place_order(req)

    def close_all(self, symbol: str, magic: int) -> list[OrderResult]:
        return [self.close_position(p.ticket, magic=magic, client_id=f"{magic}-closeall-{p.ticket}"[:31])
                for p in self.positions(symbol, magic)]

    def find_deals(self, client_id: str, *, symbol: str, magic: int,
                   since: pd.Timestamp | None = None) -> list[BrokerDeal]:
        now = self.clock.now()
        start = since if since is not None else now - pd.Timedelta(days=7)
        # server-clock epoch seconds, padded a day each side (offset/DST insensitive)
        t0 = self._to_server_seconds(start) - 86_400
        t1 = self._to_server_seconds(now) + 86_400
        raw = self.mt5.history_deals_get(t0, t1)
        if raw is None:
            raise BrokerError(f"MT5 history_deals_get failed: {self._last_error()}")
        cid = client_id[:31]
        out = []
        for d in raw:
            if str(_get(d, "symbol")) != symbol or int(_get(d, "magic", 0)) != int(magic):
                continue
            if str(_get(d, "comment", "") or "") != cid:
                continue
            typ = int(_get(d, "type", 0))
            out.append(BrokerDeal(
                ticket=int(_get(d, "ticket")), order=_get(d, "order"), position_ticket=_get(d, "position_id"),
                symbol=symbol, side=Side.BUY if typ == self._c("DEAL_TYPE_BUY") else Side.SELL,
                lots=float(_get(d, "volume", 0.0)), price=float(_get(d, "price", 0.0)), magic=int(magic),
                comment=cid, time=self.to_utc([int(_get(d, "time", 0))])[0],
                commission=float(_get(d, "commission", 0.0) or 0.0), swap=float(_get(d, "swap", 0.0) or 0.0),
                profit=float(_get(d, "profit", 0.0) or 0.0)))
        return out

    def modify_protection(self, ticket: int, *, magic: int, stop_loss: float | None,
                          take_profit: float | None) -> OrderResult:
        """Set broker-side SL/TP (MID levels) on one of OUR tickets (TRADE_ACTION_SLTP)."""
        pos, why = self._own_position(ticket, magic)
        if pos is None:
            return OrderResult(ok=False, retcode="ownership", message=why, status=OrderStatusCode.REJECTED)
        tick = self._raw_tick()
        half = 0.5 * (float(_get(tick, "ask", 0.0)) - float(_get(tick, "bid", 0.0))) if tick is not None else 0.0
        side = 1 if pos.lots > 0 else -1
        request = {"action": self._c("TRADE_ACTION_SLTP"), "symbol": self.symbol, "position": int(ticket),
                   "sl": self._venue_level(stop_loss, side, "sl", half),
                   "tp": self._venue_level(take_profit, side, "tp", half), "magic": int(magic)}
        res = self.mt5.order_send(request)
        code = int(_get(res, "retcode", -1)) if res is not None else -1
        status = classify_retcode(code) if res is not None else OrderStatusCode.UNKNOWN
        ok = code == self._c("TRADE_RETCODE_DONE")
        return OrderResult(ok=ok, retcode=code, message=str(_get(res, "comment", "")), status=status,
                           position_ticket=int(ticket))

    def __repr__(self) -> str:  # never include credentials
        return f"MT5Broker(symbol={self.symbol!r}, server_tz={self._server_tz!r}, connected={self._connected})"


def mt5_available() -> bool:
    """True if the MetaTrader5 package can be imported here."""
    try:
        import_mt5()
    except MT5UnavailableError:
        return False
    return True
