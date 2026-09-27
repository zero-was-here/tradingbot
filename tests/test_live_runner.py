"""LiveRunner end-to-end on ReplayFeed + PaperBroker (no network, simulated clock)."""

from __future__ import annotations

import json
import logging
import signal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from test_live_support import EmaCrossStub, ExplodingStub, build_artifact, synthetic_bars

from aurum.backtest.engine import run_backtest
from aurum.core.types import MarketData, Side
from aurum.execution.costs import CostModel
from aurum.live.broker import AccountInfo, OrderRequest, SimulatedClock, net_lots
from aurum.live.paper import PaperBroker, ReplayFeed
from aurum.live.runner import (
    LiveConfig,
    LiveDeskDataProvider,
    LiveRunner,
    RealMoneyGuardError,
    load_artifact,
    main,
)
from aurum.live.state import read_json, read_jsonl
from aurum.portfolio.sizing import VolTargetSizer
from aurum.risk.manager import RiskLimits, StandardRiskManager

DELAY = pd.Timedelta(seconds=5)
COSTS = CostModel(commission_per_lot=3.5)
RISK = {"daily_loss_persistent": False, "max_daily_loss": 0.05}


@pytest.fixture(autouse=True)
def _no_network_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never reach a real webhook, Anthropic key or MT5 account from these tests."""
    for var in ("AURUM_ALERT_WEBHOOK_URL", "ANTHROPIC_API_KEY", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER",
                "AURUM_ARTIFACT_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("live_world")
    bars = synthetic_bars(760, weekend_gaps=False)
    gap_bars = synthetic_bars(760, weekend_gaps=True, seed=9)
    return {
        "bars": bars,
        "art": build_artifact(root / "art", bars, n_train=350,
                              backtest_stats={"daily_mean": 0.0003, "daily_std": 0.006}),
        "gap_bars": gap_bars,
        "gap_art": build_artifact(root / "gap_art", gap_bars, n_train=350),
    }


def _runner(tmp: Path, bars: pd.DataFrame, art_path: Path, *, start_i: int, broker: PaperBroker | None = None,
            clock: SimulatedClock | None = None, runner_kw: dict | None = None, **cfg: Any
            ) -> tuple[LiveRunner, PaperBroker, SimulatedClock, ReplayFeed]:
    feed = ReplayFeed(bars) if broker is None else broker.feed
    if clock is None:
        clock = broker.clock if broker is not None else SimulatedClock(bars["available_at"].iloc[start_i])
    if broker is None:
        broker = PaperBroker(feed, costs=COSTS, clock=clock)
    base = {"dry_run": False, "state_dir": str(tmp / "state"), "calendar": None, "risk": dict(RISK)}
    base.update(cfg)
    runner = LiveRunner(LiveConfig(**base), broker=broker, artifact=load_artifact(art_path), clock=clock,
                        **(runner_kw or {}))
    return runner, broker, clock, feed


def _decisions(state_dir: Path) -> list[dict]:
    return [r for r in read_jsonl(state_dir / "decisions.jsonl") if r["type"] == "decision"]


def _backtest(bars: pd.DataFrame, art_path: Path, start_i: int, *, stop: float | None = None,
              sizer_kw: dict | None = None) -> pd.Series:
    art = load_artifact(art_path)
    md = MarketData(bars)
    X = art.features(md)
    comb = art.combine(pd.DataFrame({n: s.generate(md, X) for n, s in art.strategies.items()}))
    res = run_backtest(md, comb, sizer=VolTargetSizer(**{**art.sizer_config, **(sizer_kw or {})}),
                       risk=StandardRiskManager(RiskLimits(**RISK)), costs=COSTS, stop_atr_mult=stop,
                       start=bars.index[start_i], compute_metrics=False)
    return res.equity


# ------------------------------------------------------------------------------------------------
def test_paper_run_matches_backtest_and_logs_everything(world: dict, tmp_path: Path) -> None:
    bars, art = world["bars"], world["art"]
    start = 450
    runner, pb, clock, feed = _runner(tmp_path, bars, art, start_i=start, history_bars=10_000,
                                      stop_atr_mult=3.0, sizer={"target_vol": 0.2})
    results = runner.run(until=feed.end + DELAY + pd.Timedelta(seconds=1), install_signal_handlers=False)
    assert len(results) == len(bars) - start  # one decision per closed bar
    recs = _decisions(tmp_path / "state")
    live = pd.Series({pd.Timestamp(r["bar_time"]): r["equity"] for r in recs})
    bt = _backtest(bars, art, start, stop=3.0, sizer_kw={"target_vol": 0.2}).reindex(live.index)
    assert np.max(np.abs(live.to_numpy() - bt.to_numpy())) < 1e-6
    assert len(pb.deals(magic=runner.config.magic)) > 20
    r = next(x for x in recs if x["execution"] and x["execution"]["legs"])
    for key in ("forecasts", "combined", "final_forecast", "requested_lots", "approved_lots", "equity",
                "risk", "execution", "vol_ann", "sizing", "time", "bar_time"):
        assert key in r
    assert set(r["forecasts"]) == {"ema_cross_stub", "mom_stub", "feature_stub"}
    assert r["execution"]["legs"][0]["fill"]["price"] > 0
    assert r["execution"]["legs"][0]["client_id"].startswith(f"{runner.config.magic}-")
    hb = read_json(tmp_path / "state" / "heartbeat.json")
    assert hb["status"] == "stopped" and hb["n_decisions"] == len(results)
    kinds = [x["type"] for x in read_jsonl(tmp_path / "state" / "decisions.jsonl")]
    assert kinds[0] == "start" and kinds[-1] == "stop"


def test_dry_run_sends_nothing(world: dict, tmp_path: Path) -> None:
    bars, art = world["bars"], world["art"]
    runner, pb, clock, feed = _runner(tmp_path, bars, art, start_i=600, dry_run=True)
    results = runner.run(max_cycles=60, install_signal_handlers=False)
    assert len(results) == 60 and pb.deals() == [] and pb.positions() == []
    assert all(r.status in ("dry_run", "noop") for r in results)
    planned = [r for r in _decisions(tmp_path / "state") if r["execution"]["planned"]]
    assert planned, "dry run must log the orders it would have sent"
    assert all(r["execution"]["status"] == "dry_run" for r in planned)
    assert not (tmp_path / "state" / "oms_state.json").exists()
    assert runner.mode.startswith("DRY-RUN")


class RealAccountBroker:
    """A paper venue that claims to be a real-money account."""

    def __init__(self, inner: PaperBroker) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def is_demo(self) -> bool:
        return False

    def account(self) -> AccountInfo:
        a = self.inner.account()
        return AccountInfo(equity=a.equity, balance=a.balance, margin=a.margin, free_margin=a.free_margin,
                           is_demo=False, server="Real-Server")


def test_real_account_guard(world: dict, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bars, art = world["bars"], world["art"]
    feed = ReplayFeed(bars)
    clock = SimulatedClock(bars["available_at"].iloc[600])
    real = RealAccountBroker(PaperBroker(feed, clock=clock))
    for allow, flag in ((False, False), (True, False), (False, True)):
        r = LiveRunner(LiveConfig(state_dir=str(tmp_path / "s"), calendar=None, allow_live_real=allow),
                       broker=real, artifact=load_artifact(art), clock=clock, i_understand_real_money=flag)
        with pytest.raises(RealMoneyGuardError, match="NOT a demo"):
            r.start()
    caplog.set_level(logging.CRITICAL)
    r = LiveRunner(LiveConfig(state_dir=str(tmp_path / "s"), calendar=None, allow_live_real=True),
                   broker=real, artifact=load_artifact(art), clock=clock, i_understand_real_money=True)
    r.start()
    assert r.real_money and r.config.dry_run and "REAL-MONEY" in caplog.text
    assert r.mode == "DRY-RUN REAL"


def test_risk_halt_persists_across_restart(world: dict, tmp_path: Path) -> None:
    bars, art = world["bars"], world["art"]
    r1, pb, clock, feed = _runner(tmp_path, bars, art, start_i=500, sizer={"target_vol": 0.25})
    r1.run(max_cycles=40, install_signal_handlers=False)
    assert net_lots(pb.positions("XAUUSD", r1.config.magic)) != 0
    r1.risk.halt("operator kill switch (test)", time=clock.now())
    # a brand-new process: same state dir, same venue
    r2, _, _, _ = _runner(tmp_path, bars, art, start_i=500, broker=pb, sizer={"target_vol": 0.25})
    res = r2.run(max_cycles=10, install_signal_handlers=False)
    assert r2.risk.halted and all(x.halted for x in res)
    assert all(x.approved_lots == 0.0 for x in res)
    assert net_lots(pb.positions("XAUUSD", r2.config.magic)) == 0.0  # flattened, stays flat
    alerts = read_jsonl(tmp_path / "state" / "alerts.jsonl")
    assert any(a["kind"] == "risk_halt" for a in alerts)
    with pytest.raises(ValueError):
        r2.risk.reset_halt("yes")


def test_restart_does_not_double_send(world: dict, tmp_path: Path) -> None:
    bars, art = world["bars"], world["art"]
    start, cut, end = 500, 560, 620
    until = bars["available_at"].iloc[end] + DELAY + pd.Timedelta(seconds=1)
    # uninterrupted reference
    ref, pref, _, _ = _runner(tmp_path / "ref", bars, art, start_i=start)
    ref.run(until=until, install_signal_handlers=False)
    # interrupted run: stop at `cut`, lose the runner state (crash before it was saved), restart
    r1, pb, clock, _ = _runner(tmp_path / "a", bars, art, start_i=start)
    r1.run(until=bars["available_at"].iloc[cut] + DELAY + pd.Timedelta(seconds=1), install_signal_handlers=False)
    (tmp_path / "a" / "state" / "runner_state.json").unlink()
    r2, _, _, _ = _runner(tmp_path / "a", bars, art, start_i=start, broker=pb)
    r2.run(until=until, install_signal_handlers=False)
    magic = r1.config.magic
    assert len(pb.deals(magic=magic)) == len(pref.deals(magic=magic))
    assert pb.account().equity == pytest.approx(pref.account().equity, abs=1e-6)
    recs = _decisions(tmp_path / "a" / "state")
    redo = [r for r in recs if pd.Timestamp(r["bar_time"]) == bars.index[cut]]
    assert len(redo) == 2  # decided twice (state lost) ...
    assert redo[1]["execution"]["status"] in ("duplicate", "noop")  # ... executed once
    comments = [d.comment for d in pb.deals(magic=magic)]
    assert len(comments) == len(set(comments))


def test_deferred_orders_over_market_gaps(world: dict, tmp_path: Path) -> None:
    bars, art = world["gap_bars"], world["gap_art"]
    avail = pd.DatetimeIndex(bars["available_at"])
    weekends = np.flatnonzero(bars.index[1:] - avail[:-1] > pd.Timedelta(hours=24))
    start = int(weekends[weekends > 440][0]) - 30  # 30 bars before a weekend
    end = start + 90                              # ... and ~60 bars after it
    until = avail[end] + DELAY + pd.Timedelta(seconds=1)
    runner, pb, _, _ = _runner(tmp_path / "long", bars, art, start_i=start, history_bars=10_000,
                               max_defer_seconds=4 * 86400)
    runner.run(until=until, install_signal_handlers=False)
    log = read_jsonl(tmp_path / "long" / "state" / "decisions.jsonl")
    deferred = [r for r in log if r["type"] == "decision" and r["execution"] and r["execution"]["status"] == "deferred"]
    executed = [r for r in log if r["type"] == "deferred_execution"]
    assert deferred and executed  # weekend decisions filled at the Monday open ...
    live = pd.Series({pd.Timestamp(r["bar_time"]): r["equity"] for r in log if r["type"] == "decision"})
    bt = _backtest(bars, art, start).reindex(live.index)
    assert np.max(np.abs(live.to_numpy() - bt.to_numpy())) < 1e-6  # ... exactly like the simulator
    # with the default max_defer (1.5 bars) the weekend decision expires instead
    r2, _, _, _ = _runner(tmp_path / "short", bars, art, start_i=start)
    r2.run(until=until, install_signal_handlers=False)
    log2 = read_jsonl(tmp_path / "short" / "state" / "decisions.jsonl")
    assert any(r["type"] == "deferred_expired" for r in log2)


def _desk(client: Any, **policy_kw: Any) -> Any:
    from aurum.agents import DecisionPolicy, TradingDesk

    return TradingDesk(LiveDeskDataProvider(), client=client, policy=DecisionPolicy(**policy_kw), journal_dir=None)


def test_desk_forecast_goes_through_sizer_and_risk(world: dict, tmp_path: Path) -> None:
    from aurum.agents.testing import FakeAnthropicClient, decision_call, message, tool_use

    bars, art = world["bars"], world["art"]
    n = 8
    script = []
    for _ in range(n):
        script += [message(tool_use("get_quant_signals")),
                   message(decision_call(action="scale", scale=0.5, confidence=0.8))]
    client = FakeAnthropicClient({"chief": script})
    desk = _desk(client, mode="overlay")
    runner, pb, _, _ = _runner(tmp_path, bars, art, start_i=600, runner_kw={"desk": desk},
                               risk={**RISK, "max_lots": 0.05}, sizer={"target_vol": 0.4})
    res = runner.run(max_cycles=n, install_signal_handlers=False)
    assert client.errors == []
    decided = [r for r in res if r.desk and r.desk["status"] == "decided"]
    assert decided
    for r in decided:
        assert r.final_forecast == pytest.approx(0.5 * r.combined)
        rec = r.record
        assert rec["sizing"]["forecast"] == pytest.approx(r.final_forecast)   # the SIZER saw the desk's forecast
        sized = runner.sizer.target_lots(r.final_forecast, r.vol_ann, r.equity, rec["price"], runner.instrument,
                                         current_lots=r.current_lots, drawdown=rec["drawdown"])
        assert r.requested_lots == pytest.approx(sized)
        assert abs(r.approved_lots) <= min(abs(r.requested_lots), 0.05) + 1e-9  # risk can only reduce
        if r.requested_lots:
            assert np.sign(r.approved_lots) in (0.0, np.sign(r.requested_lots))
    assert any(abs(r.requested_lots) > 0.05 for r in decided)  # the risk cap actually bound
    # the desk read live data through the provider (tool result carries the strategy names)
    second = client.calls_for("chief")[1].kwargs["messages"]
    assert "ema_cross_stub" in json.dumps(second, default=str)


def test_desk_failures_fall_back(world: dict, tmp_path: Path) -> None:
    from aurum.agents.testing import FakeAnthropicClient, refusal

    bars, art = world["bars"], world["art"]
    client = FakeAnthropicClient({"chief": [refusal()] * 5})
    runner, _, _, _ = _runner(tmp_path / "a", bars, art, start_i=600, dry_run=True,
                              runner_kw={"desk": _desk(client, on_failure="veto")})
    for r in runner.run(max_cycles=5, install_signal_handlers=False):
        if r.desk["status"] == "failed":
            assert r.final_forecast == 0.0  # policy.on_failure = veto

    class Broken:
        last_final_forecast = None

        def run_cycle(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("desk exploded")

    for on_error, check in (("follow_quant", lambda r: r.final_forecast == r.combined),
                            ("flat", lambda r: r.final_forecast == 0.0)):
        runner, _, _, _ = _runner(tmp_path / on_error, bars, art, start_i=600, dry_run=True,
                                  runner_kw={"desk": Broken()}, desk={"on_error": on_error})
        res = runner.run(max_cycles=3, install_signal_handlers=False)
        assert all(r.desk["status"] == "error" and check(r) for r in res)


def test_strategy_errors(world: dict, tmp_path: Path) -> None:
    bars = world["bars"]
    art_path = build_artifact(tmp_path / "art_partial", bars, n_train=350, combiner=False,
                              strategies=[EmaCrossStub(), ExplodingStub(explode_from=str(bars.index[620]))])
    runner, pb, _, _ = _runner(tmp_path / "p", bars, art_path, start_i=610)
    res = runner.run(max_cycles=20, install_signal_handlers=False)
    broken = [r for r in res if r.errors]
    assert len(broken) == 10 and all(r.forecasts["exploding_stub"] == 0.0 for r in broken)
    assert all(r.forecasts["exploding_stub"] == 0.5 for r in res if not r.errors)
    assert all(not r.status.startswith("error") for r in res)  # one strategy down: keep trading
    # every strategy fails: hold (default) keeps the existing position, flatten closes it
    art_all = build_artifact(tmp_path / "art_all", bars, n_train=350, combiner=False,
                             strategies=[ExplodingStub(always=True)])
    for on_error, expect in (("hold", 1.0), ("flatten", 0.0)):
        clock = SimulatedClock(bars["available_at"].iloc[610])
        pb = PaperBroker(ReplayFeed(bars), costs=COSTS, clock=clock)
        pb.place_order(OrderRequest("manual", "XAUUSD", Side.BUY, 1.0, clock.now(), 20260926))
        runner, _, _, _ = _runner(tmp_path / on_error, bars, art_all, start_i=610, broker=pb, on_error=on_error)
        res = runner.run(max_cycles=3, install_signal_handlers=False)
        assert all(r.status.startswith("error") for r in res)
        assert net_lots(pb.positions("XAUUSD", 20260926)) == expect
    alerts = read_jsonl(tmp_path / "hold" / "state" / "alerts.jsonl")
    assert any(a["kind"] == "signal_error" and a["level"] == "critical" for a in alerts)


def test_stale_data_and_insufficient_history(world: dict, tmp_path: Path) -> None:
    bars, art = world["bars"], world["art"]
    runner, pb, clock, _ = _runner(tmp_path / "hist", bars, art, start_i=60)
    res = runner.run(max_cycles=3, install_signal_handlers=False)
    assert all(r.status == "insufficient_history" for r in res) and pb.deals() == []
    runner, pb, clock, _ = _runner(tmp_path / "stale", bars, art, start_i=600)
    r = runner.run_once(now=clock.now() + pd.Timedelta(hours=3))
    assert r is not None and r.status == "stale" and pb.deals() == []


def test_graceful_stop_and_heartbeat(world: dict, tmp_path: Path) -> None:
    bars, art = world["bars"], world["art"]

    class StoppingClock(SimulatedClock):
        runner: LiveRunner | None = None
        sleeps = 0

        def sleep(self, seconds: float) -> None:
            super().sleep(seconds)
            self.sleeps += 1
            if self.sleeps == 7 and self.runner is not None:
                self.runner._on_signal(signal.SIGINT, None)  # what the SIGINT handler does

    clock = StoppingClock(bars["available_at"].iloc[600])
    runner, _, _, _ = _runner(tmp_path, bars, art, start_i=600, clock=clock, dry_run=True)
    clock.runner = runner
    res = runner.run(install_signal_handlers=False)
    assert 0 < len(res) <= 7
    hb = read_json(tmp_path / "state" / "heartbeat.json")
    assert hb["status"] == "stopped" and hb["mode"].startswith("DRY-RUN")
    assert read_jsonl(tmp_path / "state" / "decisions.jsonl")[-1]["type"] == "stop"


def test_config_parsing(tmp_path: Path) -> None:
    doc = {"live": {"symbol": "XAUUSD", "timeframe": "H1", "magic": 123, "dry_run": True,
                    "allow_live_real": False, "stop_atr_mult": 2.5},
           "risk": {"max_daily_loss": 0.02}, "desk": {"enabled": False}, "oms": {"max_retries": 5}}
    p = tmp_path / "live.yaml"
    p.write_text(yaml.safe_dump(doc))
    cfg = LiveConfig.from_yaml(p)
    assert cfg.magic == 123 and cfg.risk == {"max_daily_loss": 0.02} and cfg.oms == {"max_retries": 5}
    assert cfg.dry_run and not cfg.allow_live_real
    with pytest.raises(ValueError, match="unknown"):
        LiveConfig.from_mapping({"live": {"dryrun": False}})
    with pytest.raises(ValueError):
        LiveConfig(timeframe="H2")
    assert LiveConfig().dry_run is True  # safe default


def test_cli_paper_replay(world: dict, tmp_path: Path) -> None:
    from aurum.data.store import save_bars

    bars, art = world["bars"], world["art"]
    save_bars(bars, tmp_path / "bars.parquet")
    cfg = {"live": {"artifact_dir": str(art), "broker": "paper", "state_dir": str(tmp_path / "state"),
                    "calendar": None},
           "paper": {"bars_path": str(tmp_path / "bars.parquet"), "warmup_bars": 600}}
    (tmp_path / "live.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["--config", str(tmp_path / "live.yaml"), "--max-cycles", "4", "--log-level", "WARNING"]) == 0
    recs = _decisions(tmp_path / "state")
    assert len(recs) == 4 and all(r["dry_run"] for r in recs)


def test_with_real_rule_strategies_if_available(world: dict, tmp_path: Path) -> None:
    trend = pytest.importorskip("aurum.strategies.trend")
    from aurum.strategies.base import list_strategies

    names = [n for n in ("ema_cross", "tsmom") if n in list_strategies()]
    if not names:
        pytest.skip("rule strategies not registered yet")
    assert trend is not None
    from aurum.strategies.base import get_strategy

    strats = [get_strategy(n) for n in names]
    warm = max(int(s.warmup_bars) for s in strats)
    bars = synthetic_bars(max(760, warm + 520), seed=4)
    art_path = build_artifact(tmp_path / "art", bars, n_train=max(350, warm + 60), strategies=strats,
                              combiner=False)
    lookback = load_artifact(art_path).max_lookback
    runner, _, _, _ = _runner(tmp_path, bars, art_path, start_i=len(bars) - 20, dry_run=True,
                              history_bars=lookback + 200)
    res = runner.run(max_cycles=10, install_signal_handlers=False)
    assert len(res) == 10 and not any(r.errors for r in res)


def test_runner_survives_transient_broker_errors(world: dict, tmp_path: Path) -> None:
    from aurum.live.broker import BrokerError

    bars, art = world["bars"], world["art"]

    class Flaky:
        def __init__(self, inner: PaperBroker) -> None:
            self.inner = inner
            self.fail = 2
            self.reconnects = 0

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def latest_bars(self, *a: Any, **k: Any) -> pd.DataFrame:
            if self.fail > 0:
                self.fail -= 1
                raise BrokerError("terminal disconnected")
            return self.inner.latest_bars(*a, **k)

        def reconnect(self) -> None:
            self.reconnects += 1

    clock = SimulatedClock(bars["available_at"].iloc[600])
    flaky = Flaky(PaperBroker(ReplayFeed(bars), costs=COSTS, clock=clock))
    runner = LiveRunner(LiveConfig(state_dir=str(tmp_path / "state"), calendar=None, dry_run=True),
                        broker=flaky, artifact=load_artifact(art), clock=clock)
    res = runner.run(max_cycles=5, install_signal_handlers=False)
    assert len(res) == 5 and flaky.reconnects == 2
    alerts = read_jsonl(tmp_path / "state" / "alerts.jsonl")
    assert [a["kind"] for a in alerts].count("broker_error") >= 1

    class Buggy(Flaky):
        def latest_bars(self, *a: Any, **k: Any) -> pd.DataFrame:
            raise ZeroDivisionError("a bug must not be swallowed")

    runner = LiveRunner(LiveConfig(state_dir=str(tmp_path / "s2"), calendar=None), broker=Buggy(flaky.inner),
                        artifact=load_artifact(art), clock=clock)
    with pytest.raises(ZeroDivisionError):
        runner.run(max_cycles=2, install_signal_handlers=False)
    assert read_json(tmp_path / "s2" / "heartbeat.json")["status"].startswith("error")


def test_deferred_intent_survives_restart(world: dict, tmp_path: Path) -> None:
    bars, art = world["gap_bars"], world["gap_art"]
    avail = pd.DatetimeIndex(bars["available_at"])
    weekends = np.flatnonzero(bars.index[1:] - avail[:-1] > pd.Timedelta(hours=24))
    w = int(weekends[weekends > 440][0])
    r1, pb, clock, _ = _runner(tmp_path, bars, art, start_i=w - 40, max_defer_seconds=4 * 86400,
                               sizer={"target_vol": 0.3})
    r1.run(until=avail[w] + pd.Timedelta(minutes=30), install_signal_handlers=False)
    if r1._pending is None:
        pytest.skip("no order was needed at this weekend close for this seed")
    target = r1._pending.target
    r2, _, _, _ = _runner(tmp_path, bars, art, start_i=w - 40, broker=pb, max_defer_seconds=4 * 86400,
                          sizer={"target_vol": 0.3})
    r2.start()
    assert r2._pending is not None and r2._pending.target == target
    r2.run(until=bars.index[w + 1] + pd.Timedelta(minutes=30), install_signal_handlers=False)
    assert r2._pending is None
    assert net_lots(pb.positions("XAUUSD", r2.config.magic)) == pytest.approx(target)
    log = read_jsonl(tmp_path / "state" / "decisions.jsonl")
    assert any(r["type"] == "deferred_execution" for r in log)


# ------------------------------------------------------------------------------------------------
# adversarial review
# ------------------------------------------------------------------------------------------------
def test_flatten_on_shutdown_really_flattens(world: dict, tmp_path: Path) -> None:
    """The shutdown flatten used to reuse the last decision's minute key, so a stop within the
    same minute as the last bar close came back ``duplicate`` and left the book open."""
    bars, art = world["bars"], world["art"]
    runner, pb, clock, feed = _runner(tmp_path, bars, art, start_i=600, flatten_on_shutdown=True,
                                      sizer={"target_vol": 0.3})
    res = runner.run(max_cycles=30, install_signal_handlers=False)
    assert any(abs(r.current_lots or 0) > 0 for r in res), "test needs an open position before shutdown"
    assert net_lots(pb.positions("XAUUSD", runner.config.magic)) == 0.0
    flat = [r for r in read_jsonl(tmp_path / "state" / "decisions.jsonl") if r["type"] == "shutdown_flatten"]
    assert len(flat) == 1 and flat[0]["execution"]["status"] in ("filled", "noop")
    comments = [d.comment for d in pb.deals(magic=runner.config.magic)]
    assert len(comments) == len(set(comments))


def test_deferred_intent_is_not_executed_while_halted(world: dict, tmp_path: Path) -> None:
    """A market-closed decision restored after a restart must not ADD risk when the kill
    switch engaged in the meantime: a halted book may only go flat."""
    bars, art = world["gap_bars"], world["gap_art"]
    avail = pd.DatetimeIndex(bars["available_at"])
    weekends = np.flatnonzero(bars.index[1:] - avail[:-1] > pd.Timedelta(hours=24))
    kw: dict[str, Any] = {"max_defer_seconds": 4 * 86400, "sizer": {"target_vol": 0.3}}
    for w in (int(x) for x in weekends[weekends > 440]):
        d = tmp_path / f"w{w}"
        r1, pb, clock, _ = _runner(d, bars, art, start_i=w - 40, **kw)
        r1.run(until=avail[w] + pd.Timedelta(minutes=30), install_signal_handlers=False)
        if r1._pending is not None and abs(r1._pending.target) > 0:
            break
    else:
        pytest.skip("no non-flat order was deferred over a weekend for this seed")
    r1.risk.halt("operator kill switch (test)", time=clock.now())
    r2, _, _, _ = _runner(d, bars, art, start_i=w - 40, broker=pb, **kw)
    r2.start()
    assert r2.risk.halted and r2._pending is not None
    # past the reopen, before the first new bar closes: only the deferred intent can trade
    r2.run(until=bars.index[w + 1] + pd.Timedelta(minutes=10), install_signal_handlers=False)
    assert net_lots(pb.positions("XAUUSD", r2.config.magic)) == 0.0  # flattened, not the stale target
    log = read_jsonl(d / "state" / "decisions.jsonl")
    assert any(r["type"] == "deferred_halted" for r in log)


def test_non_finite_size_holds_instead_of_liquidating(world: dict, tmp_path: Path) -> None:
    """``Instrument.round_lots(nan) == 0.0``: a NaN from the sizer used to flatten the book.
    The backtest engine holds on a non-finite size; live must do the same (and alert)."""
    bars, art = world["bars"], world["art"]
    runner, pb, clock, feed = _runner(tmp_path, bars, art, start_i=600, sizer={"target_vol": 0.3})
    runner.start()
    for i in range(600, 615):
        clock.advance_to(feed.decision_time(i) + DELAY)
        runner.run_once()
    held = net_lots(pb.positions("XAUUSD", runner.config.magic))
    assert held != 0.0, "test needs an open position"

    class NaNSizer:
        def breakdown(self, *a: Any, **k: Any) -> Any:
            class B:
                final_lots = float("nan")

                def to_dict(self) -> dict:
                    return {"final_lots": None}

            return B()

    runner.sizer = NaNSizer()
    res = []
    for i in range(615, 618):
        clock.advance_to(feed.decision_time(i) + DELAY)
        res.append(runner.run_once())
    assert all(r is not None and r.requested_lots == pytest.approx(r.current_lots) for r in res)
    assert all(any("non-finite" in e for e in r.errors) for r in res)
    assert net_lots(pb.positions("XAUUSD", runner.config.magic)) == pytest.approx(held)
    alerts = read_jsonl(tmp_path / "state" / "alerts.jsonl")
    assert any(a["kind"] == "sizing_error" for a in alerts)


def test_cli_accepts_tz_aware_times(world: dict, tmp_path: Path) -> None:
    from aurum.data.store import save_bars

    bars, art = world["bars"], world["art"]
    save_bars(bars, tmp_path / "bars.parquet")
    end = bars.index[700].to_pydatetime()  # tz-aware datetime: YAML round-trips it as such
    cfg = {"live": {"artifact_dir": str(art), "broker": "paper", "state_dir": str(tmp_path / "state"),
                    "calendar": None},
           "paper": {"bars_path": str(tmp_path / "bars.parquet"), "warmup_bars": 600, "end": end}}
    (tmp_path / "live.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["--config", str(tmp_path / "live.yaml"), "--max-cycles", "3", "--log-level", "WARNING",
                 "--until", str(bars["available_at"].iloc[690])]) == 0
    assert len(_decisions(tmp_path / "state")) == 3


def test_second_runner_on_the_same_state_dir_is_refused(world: dict, tmp_path: Path) -> None:
    """Two processes on one state dir would both trade every bar (their OMS states diverge in
    memory): the state directory is locked for the runner's lifetime."""
    from aurum.live.runner import RunnerLockedError

    bars, art = world["bars"], world["art"]
    r1, pb, clock, _ = _runner(tmp_path, bars, art, start_i=600)
    r1.start()
    r2, _, _, _ = _runner(tmp_path, bars, art, start_i=600, broker=pb)
    with pytest.raises(RunnerLockedError):
        r2.start()
    assert not r2._started
    r1.shutdown()
    r2.start()  # released on shutdown
    assert r2._started
    r2.shutdown()


def test_stop_cooldown_matches_the_backtest_engine(world: dict, tmp_path: Path) -> None:
    """``run_backtest(stop_cooldown_bars=k)`` blocks same-side re-entry for k bars after a
    stop; live used to ignore it (silent train/serve skew)."""
    bars, art_path = world["bars"], world["art"]
    start, k, mult = 450, 3, 0.5  # tight stops: many stop-outs followed by re-entry attempts
    runner, pb, clock, feed = _runner(tmp_path, bars, art_path, start_i=start, history_bars=10_000,
                                      stop_atr_mult=mult, stop_cooldown_bars=k, sizer={"target_vol": 0.2})
    runner.run(until=feed.end + DELAY + pd.Timedelta(seconds=1), install_signal_handlers=False)
    recs = _decisions(tmp_path / "state")
    live = pd.Series({pd.Timestamp(r["bar_time"]): r["equity"] for r in recs})
    art = load_artifact(art_path)
    md = MarketData(bars)
    X = art.features(md)
    comb = art.combine(pd.DataFrame({n: s.generate(md, X) for n, s in art.strategies.items()}))
    bt = run_backtest(md, comb, sizer=VolTargetSizer(**{**art.sizer_config, "target_vol": 0.2}),
                      risk=StandardRiskManager(RiskLimits(**RISK)), costs=COSTS, stop_atr_mult=mult,
                      stop_cooldown_bars=k, start=bars.index[start], compute_metrics=False)
    cooled = bt.risk_events["reasons"].str.contains("stop_cooldown").sum()
    assert cooled > 0, "the scenario must exercise the cooldown"
    assert sum("stop_cooldown" in r for r in recs) == cooled
    assert np.max(np.abs(live.to_numpy() - bt.equity.reindex(live.index).to_numpy())) < 1e-6


def test_halted_book_is_flattened_even_on_stale_data(world: dict, tmp_path: Path) -> None:
    bars, art = world["bars"], world["art"]
    r1, pb, clock, feed = _runner(tmp_path, bars, art, start_i=600, sizer={"target_vol": 0.3})
    r1.run(max_cycles=20, install_signal_handlers=False)
    assert net_lots(pb.positions("XAUUSD", r1.config.magic)) != 0.0
    r1.risk.halt("operator kill switch (test)", time=clock.now())
    r2, _, _, _ = _runner(tmp_path, bars, art, start_i=600, broker=pb)
    r2.start()
    # a new bar exists, but the process only looks at it 3 hours later (stuck host/feed)
    clock.advance_to(feed.decision_time(620) + DELAY)
    r = r2.run_once(now=clock.now() + pd.Timedelta(hours=3))
    assert r is not None and r.status == "stale"
    assert net_lots(pb.positions("XAUUSD", r2.config.magic)) == 0.0
    assert r.record["halt_flatten"] is True
    r2.shutdown()


def test_forming_bar_from_a_buggy_adapter_is_never_used(world: dict, tmp_path: Path) -> None:
    bars, art = world["bars"], world["art"]

    class LeakyFeedBroker:
        """Returns the bar still in progress as if it were closed (adapter/clock bug)."""

        def __init__(self, inner: PaperBroker) -> None:
            self.inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def latest_bars(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
            k = self.inner.feed.n_closed(self.inner.clock.now())
            return self.inner.feed.bars.iloc[max(0, k + 1 - n): k + 1]

    clock = SimulatedClock(bars["available_at"].iloc[600] + DELAY)
    inner = PaperBroker(ReplayFeed(bars), costs=COSTS, clock=clock)
    runner = LiveRunner(LiveConfig(state_dir=str(tmp_path / "state"), calendar=None, dry_run=True),
                        broker=LeakyFeedBroker(inner), artifact=load_artifact(art), clock=clock)
    r = runner.run_once()
    assert r is not None and r.bar_time == bars.index[600] and r.time <= clock.now()
    alerts = read_jsonl(tmp_path / "state" / "alerts.jsonl")
    assert any(a["kind"] == "forming_bar" and a["level"] == "critical" for a in alerts)
    runner.shutdown()


def test_deferred_risk_reduction_does_not_expire_over_the_weekend(world: dict, tmp_path: Path) -> None:
    """A flatten decided at the Friday close (here: kill switch) cannot fill until the reopen.
    With the default max_defer (1.5 bars) it used to EXPIRE, leaving the position open past
    the Monday open; only risk-increasing intents may go stale."""
    bars, art = world["gap_bars"], world["gap_art"]
    avail = pd.DatetimeIndex(bars["available_at"])
    weekends = np.flatnonzero(bars.index[1:] - avail[:-1] > pd.Timedelta(hours=24))
    for w in (int(x) for x in weekends[weekends > 440]):
        d = tmp_path / f"w{w}"
        r1, pb, clock, _ = _runner(d, bars, art, start_i=w - 40, sizer={"target_vol": 0.3})
        r1.run(until=avail[w] - pd.Timedelta(minutes=30), install_signal_handlers=False)
        if net_lots(pb.positions("XAUUSD", r1.config.magic)) != 0.0:
            break
    else:
        pytest.skip("no open position into a weekend for this seed")
    r1.risk.halt("kill switch before the weekend (test)", time=clock.now())
    r2, _, _, _ = _runner(d, bars, art, start_i=w - 40, broker=pb, sizer={"target_vol": 0.3})
    r2.run(until=bars.index[w + 1] + pd.Timedelta(minutes=10), install_signal_handlers=False)
    assert net_lots(pb.positions("XAUUSD", r2.config.magic)) == 0.0  # flat right after the reopen
    log = read_jsonl(d / "state" / "decisions.jsonl")
    assert not any(r["type"] == "deferred_expired" for r in log)
    assert any(r["type"] == "deferred_execution" for r in log)
    closes = [x for x in pb.deals(magic=r2.config.magic) if x.time >= bars.index[w + 1]]
    assert closes and closes[0].time < avail[w + 1]  # filled at the reopen, not a bar later
