"""v2.1 MT5 adapter: ``server_tz='auto'`` without a fresh tick (weekend / closed market) falls
back to the persisted or configured offset and refuses to TRADE until a fresh tick confirms it.
Runs against the fake ``MetaTrader5`` module of ``test_live_mt5`` (no terminal)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from test_live_mt5 import MAGIC, _broker, _req, _setup

from aurum.core.instrument import XAUUSD
from aurum.live.broker import BrokerError, OrderStatusCode, net_lots
from aurum.live.mt5 import MT5Broker
from aurum.live.oms import OrderManager


def _state(path: Path, spec: str = "NY+7", server: str | None = "Fake-Demo") -> Path:
    path.write_text(json.dumps({"server_tz": spec, "offset_seconds": 10800, "detected_at": "2026-07-10T21:00:00+00:00",
                                "server": server, "symbol": "XAUUSD"}))
    return path


def _expected_bars(bars: pd.DataFrame, now: pd.Timestamp, n: int) -> pd.DataFrame:
    return bars.loc[bars["available_at"] <= now].iloc[-n:]


# ------------------------------------------------------------------------------------------------
def test_fresh_detection_is_persisted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake, clock, _ = _setup(monkeypatch)
    st = tmp_path / "tz.json"
    b = _broker(clock, tz_state_path=st)
    assert b.tz_confirmed and b.tz_status == {"server_tz": "NY+7", "source": "detected", "confirmed": True,
                                              "config": "auto"}
    saved = json.loads(st.read_text())
    assert saved["server_tz"] == "NY+7" and saved["offset_seconds"] == 3 * 3600   # US summer
    assert saved["server"] == "Fake-Demo" and saved["symbol"] == "XAUUSD"


def test_explicit_server_tz_is_confirmed_by_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.streaming = False
    b = _broker(clock, server_tz="NY+7")
    assert b.tz_confirmed and b.tz_status["source"] == "config"
    assert b.place_order(_req("7-x", 1, 0.1, clock)).ok      # unchanged behaviour: config trusted


def test_no_fresh_tick_uses_the_persisted_offset_unconfirmed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                             caplog: pytest.LogCaptureFixture) -> None:
    fake, clock, bars = _setup(monkeypatch)
    fake.streaming = False                                 # weekend: no tick arrives
    caplog.set_level(logging.WARNING, logger="aurum.live.mt5")
    b = _broker(clock, tz_state_path=_state(tmp_path / "tz.json"))
    assert not b.tz_confirmed
    assert b.tz_status == {"server_tz": "NY+7", "source": "persisted", "confirmed": False, "config": "auto"}
    assert "UNCONFIRMED" in caplog.text and "persisted" in caplog.text
    # data keeps flowing on the provisional zone (dry-run decisions proceed) ...
    got = b.latest_bars("XAUUSD", "H1", 40)
    want = _expected_bars(bars, clock.now(), 40)
    assert got.index.equals(want.index)
    # ... but nothing is traded: market_closed -> the OMS defers, no order reaches the terminal
    r = b.place_order(_req("7-a", 1, 0.5, clock))
    assert not r.ok and r.status == OrderStatusCode.MARKET_CLOSED and r.retcode == "server_tz_unconfirmed"
    assert fake.requests == []


def test_configured_fallback_when_nothing_was_persisted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.streaming = False
    b = _broker(clock, server_tz_fallback="NY+7", tz_state_path=tmp_path / "missing.json")
    assert b.tz_status["source"] == "fallback" and not b.tz_confirmed
    # the persisted offset (observed on this server) wins over the configured guess
    b2 = _broker(clock, server_tz_fallback="UTC+02:00", tz_state_path=_state(tmp_path / "tz.json"))
    assert b2.tz_status["server_tz"] == "NY+7" and b2.tz_status["source"] == "persisted"


def test_without_state_or_fallback_connecting_still_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.streaming = False
    with pytest.raises(BrokerError, match="set server_tz"):
        _broker(clock, tz_state_path=tmp_path / "missing.json")
    # a persisted zone of ANOTHER server is not used
    with pytest.raises(BrokerError, match="set server_tz"):
        _broker(clock, tz_state_path=_state(tmp_path / "other.json", server="Other-Live"))
    # an unreadable / garbled state file is ignored, the fallback applies
    (tmp_path / "bad.json").write_text("{not json")
    assert _broker(clock, tz_state_path=tmp_path / "bad.json", server_tz_fallback="NY+7").tz_status["source"] == \
        "fallback"
    _state(tmp_path / "garbled.json", spec="Mars/Olympus")
    with pytest.raises(BrokerError, match="set server_tz"):
        _broker(clock, tz_state_path=tmp_path / "garbled.json")


def test_fresh_tick_confirms_and_trading_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.streaming = False
    st = _state(tmp_path / "tz.json")
    b = _broker(clock, tz_state_path=st, tz_reprobe_seconds=3600)
    assert b.place_order(_req("7-a", 1, 0.5, clock)).status == OrderStatusCode.MARKET_CLOSED
    fake.streaming = True                                  # the market reopens
    r = b.place_order(_req("7-b", 1, 0.5, clock))
    assert r.ok and r.status == OrderStatusCode.FILLED and len(fake.requests) == 1
    assert b.tz_confirmed and b.tz_status["source"] == "detected" and b.tz_status["server_tz"] == "NY+7"
    assert json.loads(st.read_text())["detected_at"] != "2026-07-10T21:00:00+00:00"   # refreshed


def test_wrong_provisional_offset_is_corrected_and_that_order_refused(monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path) -> None:
    """A stale persisted offset (here UTC+5 for a NY+7 server) would mis-time bars by hours and
    could pass the forming bar off as closed: the first fresh tick corrects it, and the order
    decided on the mis-timed bars is rejected rather than sent."""
    fake, clock, bars = _setup(monkeypatch)
    fake.streaming = False
    st = _state(tmp_path / "tz.json", spec="UTC+05:00")
    b = _broker(clock, tz_state_path=st, tz_reprobe_seconds=3600)
    assert b.tz_status["server_tz"] == "UTC+05:00" and not b.tz_confirmed
    shifted = b.latest_bars("XAUUSD", "H1", 10)
    assert not shifted.index.equals(_expected_bars(bars, clock.now(), 10).index)   # mis-timed (why we refuse)
    fake.streaming = True
    r = b.place_order(_req("7-a", 1, 0.5, clock))
    assert not r.ok and r.status == OrderStatusCode.REJECTED and r.retcode == "server_tz_corrected"
    assert fake.requests == []
    assert b.tz_confirmed and b.tz_status["server_tz"] == "NY+7"
    assert json.loads(st.read_text())["server_tz"] == "NY+7"
    assert b.latest_bars("XAUUSD", "H1", 10).index.equals(_expected_bars(bars, clock.now(), 10).index)
    assert b.place_order(_req("7-b", 1, 0.5, clock)).ok


def test_latest_bars_reprobes_so_decisions_use_the_confirmed_zone(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    fake, clock, bars = _setup(monkeypatch)
    fake.streaming = False
    b = _broker(clock, tz_state_path=_state(tmp_path / "tz.json", spec="UTC+05:00"), tz_reprobe_seconds=0)
    fake.streaming = True
    got = b.latest_bars("XAUUSD", "H1", 20)                # probes first, converts with the detected zone
    assert b.tz_confirmed and b.tz_status["server_tz"] == "NY+7"
    assert got.index.equals(_expected_bars(bars, clock.now(), 20).index)


def test_latest_bars_reprobe_is_rate_limited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.streaming = False
    b = _broker(clock, tz_state_path=_state(tmp_path / "tz.json"), tz_reprobe_seconds=3600)
    n0 = fake.tick_calls
    b.latest_bars("XAUUSD", "H1", 5)                       # first call probes (polls ticks for ~5 s)
    probe_cost = fake.tick_calls - n0
    n1 = fake.tick_calls
    b.latest_bars("XAUUSD", "H1", 5)                       # within tz_reprobe_seconds: no probe
    assert fake.tick_calls - n1 < probe_cost and probe_cost > 5
    # a trade attempt always probes (it must not be sent on an unconfirmed zone)
    n2 = fake.tick_calls
    b.place_order(_req("7-a", 1, 0.1, clock))
    assert fake.tick_calls - n2 >= probe_cost - 1    # latest_bars' cost also includes its server_time() tick


def test_closes_and_protection_changes_are_refused_too(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.add_position(501, magic=MAGIC, lots=1.0)
    fake.streaming = False
    b = _broker(clock, server_tz_fallback="NY+7", tz_reprobe_seconds=3600)
    r = b.close_position(501, magic=MAGIC)
    assert r.status == OrderStatusCode.MARKET_CLOSED and 501 in fake.positions
    r = b.modify_protection(501, magic=MAGIC, stop_loss=1700.0, take_profit=None)
    assert r.status == OrderStatusCode.MARKET_CLOSED and fake.requests == []
    assert all(x.status == OrderStatusCode.MARKET_CLOSED for x in b.close_all("XAUUSD", MAGIC))
    assert fake.requests == [] and fake.positions[501].volume == 1.0


def test_oms_defers_then_executes_once_confirmed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake, clock, bars = _setup(monkeypatch)
    fake.streaming = False
    b = _broker(clock, tz_state_path=_state(tmp_path / "tz.json"), tz_reprobe_seconds=3600)
    t = bars.index[150]
    dry = OrderManager(b, XAUUSD, MAGIC, dry_run=True).reconcile(1.0, t)
    assert dry.status == "dry_run" and dry.planned                  # dry-run decisions proceed
    oms = OrderManager(b, XAUUSD, MAGIC, sleep=lambda s: None)
    rep = oms.reconcile(1.0, t)
    assert rep.status == "deferred" and fake.requests == []          # the runner keeps the intent
    fake.streaming = True
    rep = oms.reconcile(1.0, t)                                       # the runner's retry of the SAME intent
    assert rep.status == "filled" and net_lots(b.positions("XAUUSD", MAGIC)) == pytest.approx(1.0)


def test_reconnect_keeps_a_detected_zone_without_a_fresh_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, _ = _setup(monkeypatch)
    b = _broker(clock)
    assert b.tz_confirmed
    fake.streaming = False
    b.reconnect()                                          # e.g. after a venue error over the weekend
    assert b.tz_confirmed and b.tz_status["source"] == "detected"


@pytest.mark.parametrize(("kw", "match"), [
    ({"server_tz": "NY+7", "server_tz_fallback": "NY+7"}, "only applies"),
    ({"server_tz_fallback": "auto"}, "not a time-zone"),
    ({"server_tz_fallback": "Mars/Olympus"}, "not a time-zone"),
    ({"server_tz_fallback": ""}, "not a time-zone"),
    ({"tz_reprobe_seconds": -1}, "tz_reprobe_seconds"),
])
def test_tz_option_validation(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MT5Broker("XAUUSD", connect=False, env={}, **kw)


def test_repr_shows_tz_confirmation_but_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, clock, _ = _setup(monkeypatch)
    fake.streaming = False
    b = _broker(clock, server_tz_fallback="NY+7", env={"MT5_LOGIN": "1234", "MT5_PASSWORD": "pw-secret"})
    assert "tz_confirmed=False" in repr(b) and "pw-secret" not in repr(b)


def test_unconfirmed_zone_never_lets_a_tick_override_the_machine_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a provisional offset 2 h too small (UTC+1 for a UTC+3 server), a converted tick
    looks 2 h AHEAD of the NTP clock; the old skew rule would then treat forming bars as
    closed. Until the zone is confirmed the machine clock is authoritative."""
    fake, clock, bars = _setup(monkeypatch)
    fake.streaming = False
    b = _broker(clock, server_tz_fallback="UTC+01:00", tz_reprobe_seconds=3600)
    assert not b.tz_confirmed
    assert b.server_time() == clock.now()
    got = b.latest_bars("XAUUSD", "H1", 5)
    assert (got["available_at"] <= clock.now()).all()          # no bar from the "future"
    fake.streaming = True
    assert b.place_order(_req("7-a", 1, 0.1, clock)).retcode == "server_tz_corrected"
    assert b.tz_confirmed and abs((b.server_time() - clock.now()).total_seconds()) < 60
