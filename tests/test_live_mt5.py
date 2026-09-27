"""MT5Broker against a fake ``MetaTrader5`` module injected via ``sys.modules`` (no terminal)."""

from __future__ import annotations

import logging
import sys
from collections import deque
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import Side
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.live.broker import BrokerError, OrderRequest, OrderStatusCode, SimulatedClock, net_lots
from aurum.live.mt5 import (
    MT5Broker,
    MT5UnavailableError,
    classify_retcode,
    detect_server_offset,
    import_mt5,
    server_tz_for_offset,
)
from aurum.live.oms import OrderManager

NY = "America/New_York"
MAGIC = 7
RATE_DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
              ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]


class FakeMT5(ModuleType):
    """Minimal stateful stand-in for the MetaTrader5 package (hedging account by default)."""

    TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 15, 16385, 16388, 16408
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC = 1, 2
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    DEAL_TYPE_BUY, DEAL_TYPE_SELL = 0, 1
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_CONTEST, ACCOUNT_TRADE_MODE_REAL = 0, 1, 2
    ACCOUNT_MARGIN_MODE_RETAIL_NETTING, ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 0, 2
    TRADE_RETCODE_DONE = 10009

    def __init__(self, bars: pd.DataFrame, clock: SimulatedClock, *, server_tz: str = "NY+7",
                 filling_flags: int = 3, trade_mode: int = 0, margin_mode: int = 2) -> None:
        super().__init__("MetaTrader5")
        self.bars = bars
        self.clock = clock
        self.server_tz = server_tz
        self.filling_flags = filling_flags
        self.trade_mode = trade_mode
        self.margin_mode = margin_mode
        self.retcodes: deque[int | None] = deque()
        self.requests: list[dict[str, Any]] = []
        self.init_kwargs: dict[str, Any] | None = None
        self.init_ok = True
        self.positions: dict[int, SimpleNamespace] = {}
        self.deals: list[SimpleNamespace] = []
        self._id = 1000
        self.tick_calls = 0
        self.tick_bump = 0.0
        self.tick_age = pd.Timedelta(seconds=2)
        self.streaming = True  # new ticks keep arriving (time_msc changes between calls)
        self.shut = False

    # ---- time ---------------------------------------------------------------------------------
    def server_seconds(self, ts: pd.Timestamp) -> int:
        ts = pd.Timestamp(ts).tz_convert("UTC")
        if self.server_tz == "NY+7":
            wall = ts.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=7)
        else:
            hours = float(self.server_tz.replace("UTC", "") or 0)
            wall = ts.tz_localize(None) + pd.Timedelta(hours=hours)
        return int((wall - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))

    def _next(self) -> int:
        self._id += 1
        return self._id

    # ---- API ----------------------------------------------------------------------------------
    def initialize(self, **kwargs: Any) -> bool:
        self.init_kwargs = kwargs
        return self.init_ok

    def shutdown(self) -> None:
        self.shut = True

    def last_error(self) -> tuple[int, str]:
        return (-6, "Terminal: Authorization failed") if not self.init_ok else (1, "Success")

    def terminal_info(self) -> SimpleNamespace:
        return SimpleNamespace(trade_allowed=True)

    def symbol_info(self, symbol: str) -> SimpleNamespace | None:
        if symbol != "XAUUSD":
            return None
        return SimpleNamespace(visible=True, point=0.01, digits=2, volume_min=0.01, volume_step=0.01,
                               volume_max=100.0, trade_contract_size=100.0, filling_mode=self.filling_flags)

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return symbol == "XAUUSD"

    def _forming(self) -> int:
        now = self.clock.now()
        return int(np.searchsorted(pd.DatetimeIndex(self.bars.index), now, side="right")) - 1

    def symbol_info_tick(self, symbol: str) -> SimpleNamespace:
        self.tick_calls += 1
        i = max(self._forming(), 0)
        mid = float(self.bars["open"].iloc[i]) + self.tick_bump * self.tick_calls
        half = float(self.bars["spread"].iloc[i]) / 2
        t = self.server_seconds(self.clock.now() - self.tick_age)
        msc = t * 1000 + (self.tick_calls % 1000 if self.streaming else 0)
        return SimpleNamespace(time=t, bid=mid - half, ask=mid + half, time_msc=msc)

    def account_info(self) -> SimpleNamespace:
        return SimpleNamespace(login=12345678, trade_mode=self.trade_mode, margin_mode=self.margin_mode,
                               equity=100_000.0, balance=100_000.0, margin=0.0, margin_free=100_000.0,
                               currency="USD", leverage=100, server="Fake-Demo", name="Jane Trader")

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start: int, count: int) -> np.ndarray:
        k = self._forming() + 1  # bars opened so far, the last one is still forming
        sel = self.bars.iloc[max(0, k - start - count): k - start]
        out = np.zeros(len(sel), dtype=RATE_DTYPE)
        pts = np.round(sel["spread"].to_numpy() / 0.01).astype(int)
        half = pts * 0.01 / 2
        out["time"] = [self.server_seconds(t) for t in sel.index]
        for c in ("open", "high", "low", "close"):
            out[c] = sel[c].to_numpy() - half  # MT5 bars are BID bars
        out["tick_volume"] = sel["volume"].to_numpy().astype(int)
        out["spread"] = pts
        return out

    def positions_get(self, symbol: str | None = None, ticket: int | None = None) -> tuple:
        ps = list(self.positions.values())
        if symbol is not None:
            ps = [p for p in ps if p.symbol == symbol]
        if ticket is not None:
            ps = [p for p in ps if p.ticket == ticket]
        return tuple(ps)

    def history_deals_get(self, *args: Any, ticket: int | None = None) -> tuple:
        if ticket is not None:
            return tuple(d for d in self.deals if d.ticket == ticket)
        return tuple(self.deals)

    def order_send(self, request: dict[str, Any]) -> SimpleNamespace | None:
        self.requests.append(dict(request))
        code = self.retcodes.popleft() if self.retcodes else 10009
        if code is None:
            return None
        tick = self.symbol_info_tick(request["symbol"])
        if code != 10009:
            return SimpleNamespace(retcode=code, deal=0, order=0, volume=0.0, price=0.0, bid=tick.bid, ask=tick.ask,
                                   comment="rejected")
        now_s = self.server_seconds(self.clock.now())
        typ = request["type"]
        vol = float(request["volume"])
        order_id, deal_id = self._next(), self._next()
        if "position" in request:
            p = self.positions[request["position"]]
            p.volume = round(p.volume - vol, 8)
            if p.volume <= 1e-9:
                del self.positions[p.ticket]
            pid = p.ticket
        else:
            pid = order_id
            self.positions[pid] = SimpleNamespace(
                ticket=pid, time=now_s, type=typ, magic=request["magic"], identifier=pid, volume=vol,
                price_open=request["price"], sl=request.get("sl", 0.0), tp=request.get("tp", 0.0),
                symbol=request["symbol"], comment=request["comment"], swap=0.0, profit=0.0)
        self.deals.append(SimpleNamespace(ticket=deal_id, order=order_id, position_id=pid, symbol=request["symbol"],
                                          type=typ, volume=vol, price=request["price"], magic=request["magic"],
                                          comment=request["comment"], time=now_s, commission=-3.5 * vol, swap=0.0,
                                          profit=0.0))
        return SimpleNamespace(retcode=10009, deal=deal_id, order=order_id, volume=vol, price=request["price"],
                               bid=tick.bid, ask=tick.ask, comment="Request executed")

    def add_position(self, ticket: int, *, magic: int, lots: float, symbol: str = "XAUUSD") -> None:
        self.positions[ticket] = SimpleNamespace(
            ticket=ticket, time=self.server_seconds(self.clock.now()), type=0 if lots > 0 else 1, magic=magic,
            identifier=ticket, volume=abs(lots), price_open=1800.0, sl=0.0, tp=0.0, symbol=symbol,
            comment="other", swap=0.0, profit=0.0)


def _bars(start: str = "2026-07-13", n: int = 200) -> pd.DataFrame:
    b = make_synthetic_bars(n, "H1", seed=2, start=start, weekend_gaps=False)
    b = b.copy()
    b["spread"] = np.round(b["spread"], 2).clip(lower=0.02)
    return make_bars(b.drop(columns=["available_at"]), "H1")


def _setup(monkeypatch: pytest.MonkeyPatch, *, start: str = "2026-07-13", at: int = 150, minutes: int = 30,
           **fake_kw: Any) -> tuple[FakeMT5, SimulatedClock, pd.DataFrame]:
    bars = _bars(start)
    clock = SimulatedClock(bars.index[at] + pd.Timedelta(minutes=minutes))
    fake = FakeMT5(bars, clock, **fake_kw)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    return fake, clock, bars


def _broker(clock: SimulatedClock, **kw: Any) -> MT5Broker:
    kw.setdefault("env", {})
    return MT5Broker("XAUUSD", clock=clock, sleep=lambda s: None, **kw)


def _req(cid: str, side: int, lots: float, clock: SimulatedClock, **kw: Any) -> OrderRequest:
    return OrderRequest(client_id=cid, symbol="XAUUSD", side=Side(side), lots=lots, time=clock.now(), magic=MAGIC, **kw)


# ------------------------------------------------------------------------------------------------
def test_import_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    with pytest.raises(MT5UnavailableError, match="MetaTrader5"):
        import_mt5()
    with pytest.raises(MT5UnavailableError):
        MT5Broker("XAUUSD", env={})


def test_credentials_from_env_never_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    fake, clock, _ = _setup(monkeypatch)
    env = {"MT5_LOGIN": "87654321", "MT5_PASSWORD": "s3cret-Pa55", "MT5_SERVER": "Broker-Demo", "MT5_PATH": "C:/mt5"}
    caplog.set_level(logging.DEBUG)
    b = _broker(clock, env=env)
    assert fake.init_kwargs == {"path": "C:/mt5", "login": 87654321, "password": "s3cret-Pa55", "server": "Broker-Demo"}
    assert "s3cret-Pa55" not in caplog.text and "87654321" not in caplog.text
    assert "s3cret" not in repr(b)
    assert b.account().name is None  # the holder's name is PII
    fake.init_ok = False
    with pytest.raises(BrokerError) as ei:
        _broker(clock, env=env)
    assert "s3cret" not in str(ei.value) and "87654321" not in str(ei.value)


@pytest.mark.parametrize(("start", "tz", "expected"), [
    ("2026-07-13", "NY+7", "NY+7"),        # US summer: UTC+3
    ("2026-01-12", "NY+7", "NY+7"),        # US winter: UTC+2
    ("2026-07-13", "UTC+1", "UTC+01:00"),  # fixed-offset server
    ("2026-07-13", "UTC+0", "UTC+00:00"),
])
def test_server_tz_detection_and_bar_conversion(monkeypatch: pytest.MonkeyPatch, start: str, tz: str,
                                                expected: str) -> None:
    fake, clock, bars = _setup(monkeypatch, start=start, server_tz=tz)
    b = _broker(clock)
    assert b.server_tz == expected
    got = b.latest_bars("XAUUSD", "H1", 50)
    want = bars.loc[bars["available_at"] <= clock.now()].iloc[-50:]
    assert got.index.equals(want.index)                       # UTC open times recovered exactly
    np.testing.assert_allclose(got["close"], want["close"], atol=1e-9)  # bid + spread/2 = mid
    np.testing.assert_allclose(got["spread"], want["spread"], atol=1e-9)
    assert (got["available_at"] <= clock.now()).all()


def test_forming_bar_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, bars = _setup(monkeypatch, at=150, minutes=59)
    b = _broker(clock)
    got = b.latest_bars("XAUUSD", "H1", 10)
    assert got.index[-1] == bars.index[149]  # bar 150 is still forming at :59
    clock.advance_to(bars.index[151] + pd.Timedelta(seconds=5))
    assert b.latest_bars("XAUUSD", "H1", 10).index[-1] == bars.index[150]


def test_dst_spanning_history_with_ny7(monkeypatch: pytest.MonkeyPatch) -> None:
    """A window across the US DST switch (2026-03-08) converts without a one-hour shift."""
    fake, clock, bars = _setup(monkeypatch, start="2026-03-02", at=190)
    b = _broker(clock, server_tz="NY+7")
    got = b.latest_bars("XAUUSD", "H1", 180)
    want = bars.loc[bars["available_at"] <= clock.now()].iloc[-180:]
    assert got.index.equals(want.index)


def test_server_tz_mismatch_and_stale_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, _ = _setup(monkeypatch, server_tz="NY+7")
    with pytest.raises(BrokerError, match="server_tz"):
        _broker(clock, server_tz="UTC")
    fake.streaming = False  # quiet/closed market: no tick arrives while we watch
    with pytest.raises(BrokerError, match="set server_tz"):
        _broker(clock)
    fake.tick_age = pd.Timedelta(minutes=62)  # e.g. just after the daily break: would alias
    with pytest.raises(BrokerError, match="set server_tz"):
        _broker(clock)
    assert _broker(clock, server_tz="NY+7").server_tz == "NY+7"  # config trusted when undetectable


def test_detect_offset_helpers() -> None:
    now = pd.Timestamp("2026-07-15 12:00", tz="UTC")
    tick = (now + pd.Timedelta(hours=3) - pd.Timedelta(seconds=40)).timestamp()
    off = detect_server_offset(tick, now)
    assert off == pd.Timedelta(hours=3) and server_tz_for_offset(off, now) == "NY+7"
    assert detect_server_offset(tick - 2 * 86400, now) is None  # a weekend-old tick: implausible offset
    assert detect_server_offset(tick - 400, now) is None       # ~7 min old: implied age too large
    assert server_tz_for_offset(pd.Timedelta(hours=2), now) == "UTC+02:00"
    assert classify_retcode(10018) == OrderStatusCode.MARKET_CLOSED
    assert classify_retcode(None) == OrderStatusCode.UNKNOWN
    assert classify_retcode(10006) == OrderStatusCode.REJECTED


def test_positions_filtered_by_magic_and_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.add_position(1, magic=MAGIC, lots=1.0)
    fake.add_position(2, magic=8, lots=-2.0)                    # another EA, same symbol
    fake.add_position(3, magic=MAGIC, lots=0.5, symbol="EURUSD")  # same magic, other symbol
    b = _broker(clock)
    mine = b.positions("XAUUSD", MAGIC)
    assert [p.ticket for p in mine] == [1] and net_lots(mine) == 1.0
    assert len(b.positions("XAUUSD")) == 2 and len(b.positions()) == 3
    res = b.close_all("XAUUSD", MAGIC)
    assert len(res) == 1 and res[0].ok
    assert set(fake.positions) == {2, 3}
    r = b.close_position(2, magic=MAGIC)  # not ours: refused, nothing sent
    assert not r.ok and "refusing" in r.message
    n = len(fake.requests)
    r = b.place_order(_req("x", -1, 2.0, clock, position_ticket=2))
    assert not r.ok and len(fake.requests) == n and 2 in fake.positions


def test_requote_retried_with_fresh_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.tick_bump = 0.05
    b = _broker(clock, requote_retries=3)
    fake.retcodes.extend([10004, 10020])
    r = b.place_order(_req("rq", 1, 1.0, clock))
    assert r.ok and r.status == OrderStatusCode.FILLED
    prices = [q["price"] for q in fake.requests]
    assert len(prices) == 3 and len(set(prices)) == 3  # each retry used a new tick
    fake.retcodes.extend([10004] * 5)
    r = b.place_order(_req("rq2", 1, 1.0, clock))
    assert not r.ok and r.status == OrderStatusCode.REQUOTE


def test_filling_mode_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, _ = _setup(monkeypatch, filling_flags=3)
    b = _broker(clock)
    fake.retcodes.extend([10030, 10030])
    r = b.place_order(_req("f1", 1, 1.0, clock))
    assert r.ok
    assert [q["type_filling"] for q in fake.requests] == [0, 1, 2]  # FOK -> IOC -> RETURN
    b.place_order(_req("f2", 1, 1.0, clock))
    assert fake.requests[-1]["type_filling"] == 2  # the working mode is remembered
    fake2, clock2, _ = _setup(monkeypatch, filling_flags=2)  # IOC only
    b2 = _broker(clock2)
    b2.place_order(_req("f3", 1, 1.0, clock2))
    assert fake2.requests[0]["type_filling"] == 1


@pytest.mark.parametrize(("code", "status"), [
    (10018, OrderStatusCode.MARKET_CLOSED), (10019, OrderStatusCode.NO_MONEY),
    (10014, OrderStatusCode.INVALID_VOLUME), (10016, OrderStatusCode.INVALID_STOPS),
    (10012, OrderStatusCode.UNKNOWN), (10031, OrderStatusCode.UNKNOWN), (None, OrderStatusCode.UNKNOWN),
    (10027, OrderStatusCode.TRADE_DISABLED), (10024, OrderStatusCode.TOO_MANY_REQUESTS),
])
def test_retcode_handling(monkeypatch: pytest.MonkeyPatch, code: int | None, status: OrderStatusCode) -> None:
    fake, clock, _ = _setup(monkeypatch)
    b = _broker(clock)
    fake.retcodes.append(code)
    r = b.place_order(_req("rc", 1, 1.0, clock))
    assert not r.ok and r.status == status and r.fill is None
    assert len(fake.requests) == 1  # never blindly resent by the adapter
    assert fake.positions == {}


def test_order_request_fields_and_protection(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, bars = _setup(monkeypatch)
    b = _broker(clock, deviation_points=15)
    r = b.place_order(_req("7-202607191200-0", 1, 1.5, clock, sl_distance=5.0, tp_distance=10.0))
    assert r.ok and r.fill.lots == 1.5 and r.fill.commission == pytest.approx(5.25)
    q = fake.requests[-1]
    tick = SimpleNamespace(bid=q["price"] - bars["spread"].iloc[150], ask=q["price"])
    mid = (tick.bid + tick.ask) / 2
    half = (tick.ask - tick.bid) / 2
    assert q["type"] == 0 and q["magic"] == MAGIC and q["deviation"] == 15 and q["comment"] == "7-202607191200-0"
    assert q["sl"] == pytest.approx(round(mid - 5.0 - half, 2))   # long stop triggers on the bid
    assert q["tp"] == pytest.approx(round(mid + 10.0 - half, 2))
    ticket = r.position_ticket
    assert ticket in fake.positions
    r2 = b.place_order(_req("7-202607191300-0", -1, 1.5, clock, position_ticket=ticket))
    q2 = fake.requests[-1]
    assert r2.ok and q2["position"] == ticket and q2["type"] == 1 and "sl" not in q2
    assert ticket not in fake.positions
    deals = b.find_deals("7-202607191200-0", symbol="XAUUSD", magic=MAGIC)
    assert len(deals) == 1 and deals[0].lots == 1.5 and deals[0].side == Side.BUY


def test_account_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, _ = _setup(monkeypatch, trade_mode=2, margin_mode=0)
    b = _broker(clock)
    assert not b.is_demo() and not b.is_hedging()
    acc = b.account()
    assert not acc.is_demo and acc.equity == 100_000.0 and acc.server == "Fake-Demo"
    fake.trade_mode, fake.margin_mode = 0, 2
    assert b.is_demo() and b.is_hedging()
    b.shutdown()
    assert fake.shut


def test_oms_over_mt5_hedging(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, bars = _setup(monkeypatch)
    fake.add_position(99, magic=8, lots=3.0)  # foreign EA
    b = _broker(clock)
    oms = OrderManager(b, magic=MAGIC)
    t = bars.index[150]
    rep = oms.reconcile(2.0, t, sl_distance=4.0)
    assert rep.status == "filled" and net_lots(b.positions("XAUUSD", MAGIC)) == pytest.approx(2.0)
    rep = oms.reconcile(-1.0, t + pd.Timedelta(hours=1))
    assert rep.status == "filled" and net_lots(b.positions("XAUUSD", MAGIC)) == pytest.approx(-1.0)
    assert [lg.kind for lg in rep.planned] == ["close", "open"]
    assert 99 in fake.positions and fake.positions[99].volume == 3.0  # untouched
    for q in fake.requests:
        assert q["magic"] == MAGIC and len(q["comment"]) <= 31


# ------------------------------------------------------------------------------------------------
# adversarial review: protective levels must round-trip as MID levels
# ------------------------------------------------------------------------------------------------
def test_position_levels_are_mid_and_scale_ins_keep_the_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """MT5 stores trigger levels (long SL on the bid). Reading them back as-is and re-using
    them as a MID ``stop_loss`` for a scale-in moved the stop half a spread further away on
    every add (the runner re-uses the existing stop for adds, like the backtest engine)."""
    fake, clock, bars = _setup(monkeypatch)
    b = _broker(clock)
    r = b.place_order(_req("7-a", 1, 1.0, clock, sl_distance=5.0, tp_distance=9.0))
    assert r.ok
    first = dict(fake.requests[-1])
    tick = fake.symbol_info_tick("XAUUSD")
    mid = 0.5 * (tick.bid + tick.ask)
    pos = b.positions("XAUUSD", MAGIC)[0]
    assert pos.sl == pytest.approx(mid - 5.0, abs=0.006)   # MID level (venue rounding only)
    assert pos.tp == pytest.approx(mid + 9.0, abs=0.006)
    for k in range(3):  # hedging scale-ins inherit the SAME trigger levels
        last = b.positions("XAUUSD", MAGIC)[-1]
        assert b.place_order(_req(f"7-add{k}", 1, 0.5, clock, stop_loss=last.sl, take_profit=last.tp)).ok
        assert fake.requests[-1]["sl"] == pytest.approx(first["sl"])
        assert fake.requests[-1]["tp"] == pytest.approx(first["tp"])
    # shorts: trigger on the ask -> mid = trigger - half spread
    r = b.place_order(_req("7-s", -1, 1.0, clock, sl_distance=4.0))
    short = next(p for p in b.positions("XAUUSD", MAGIC) if p.lots < 0)
    assert short.sl == pytest.approx(mid + 4.0, abs=0.006)
    assert fake.requests[-1]["sl"] == pytest.approx(round(mid + 4.0 + (tick.ask - tick.bid) / 2, 2))
