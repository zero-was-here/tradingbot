"""v2.1 live-runner hardening: desk_config, desk prerequisites, in-sleep heartbeats, logging,
per-decision INFO line and decision attribution (no network, paper broker, fake clocks)."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from test_live_support import EmaCrossStub, ExplodingStub, build_artifact, synthetic_bars

import aurum.live.runner as runner_mod
from aurum.core.types import Side
from aurum.execution.costs import CostModel
from aurum.live.broker import BrokerError, SimulatedClock
from aurum.live.oms import ExecutionReport, LegReport
from aurum.live.oms import Leg as OmsLeg
from aurum.live.paper import PaperBroker, ReplayFeed
from aurum.live.runner import (
    DESK_ON_ERROR,
    LiveConfig,
    LiveDeskDataProvider,
    LiveRunner,
    RunnerConfigError,
    _configure_logging,
    _orders_summary,
    check_desk_prerequisites,
    load_artifact,
    main,
)
from aurum.live.state import read_json, read_jsonl

COSTS = CostModel(commission_per_lot=3.5)
RISK = {"daily_loss_persistent": False, "max_daily_loss": 0.05}
LINE_RE = re.compile(r"^decision \S+ XAUUSD \[[A-Z -]+\]: combined (n/a|[+-]\d+\.\d{3}) desk \S+ "
                     r"final (n/a|[+-]\d+\.\d{3}) target (n/a|[+-]\d+\.\d{2}) approved (n/a|[+-]\d+\.\d{2}) "
                     r"held (n/a|[+-]\d+\.\d{2}) orders .+ status=\S+$")


@pytest.fixture(autouse=True)
def _no_network_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AURUM_ALERT_WEBHOOK_URL", "ANTHROPIC_API_KEY", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER",
                "AURUM_ARTIFACT_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("v21_live_world")
    bars = synthetic_bars(700, weekend_gaps=False)
    return {"bars": bars, "art": build_artifact(root / "art", bars, n_train=350),
            "art_nocomb": build_artifact(root / "art_nocomb", bars, n_train=350, combiner=False)}


class WallClock:
    """A NON-simulated clock whose sleep advances time instantly (wall-clock code paths, fast)."""

    simulated = False

    def __init__(self, start: pd.Timestamp) -> None:
        self._now = pd.Timestamp(start)
        self.sleeps = 0

    def now(self) -> pd.Timestamp:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self._now = self._now + pd.Timedelta(seconds=max(0.0, float(seconds)))


def _runner(tmp: Path, world: dict, *, art_key: str = "art", start_i: int = 600, clock: Any = None,
            artifact: Any = None, runner_kw: dict | None = None, **cfg: Any) -> tuple[LiveRunner, PaperBroker]:
    bars = world["bars"]
    clock = clock if clock is not None else SimulatedClock(bars["available_at"].iloc[start_i])
    broker = PaperBroker(ReplayFeed(bars), costs=COSTS, clock=clock)
    base = {"dry_run": False, "state_dir": str(tmp / "state"), "calendar": None, "risk": dict(RISK)}
    base.update(cfg)
    art = artifact if artifact is not None else load_artifact(world[art_key])
    return LiveRunner(LiveConfig(**base), broker=broker, artifact=art, clock=clock, **(runner_kw or {})), broker


def _decisions(state_dir: Path) -> list[dict]:
    return [r for r in read_jsonl(state_dir / "decisions.jsonl") if r["type"] == "decision"]


def _fake_desk(scale: float, n: int) -> Any:
    from aurum.agents import DecisionPolicy, TradingDesk
    from aurum.agents.testing import FakeAnthropicClient, decision_call, message

    client = FakeAnthropicClient({"chief": [message(decision_call(action="scale", scale=scale, confidence=0.9))
                                            for _ in range(n)]})
    return TradingDesk(LiveDeskDataProvider(), client=client, policy=DecisionPolicy(mode="overlay"), journal_dir=None)


# =============================================================================================
# 1) desk_config + strict desk validation
# =============================================================================================
@pytest.mark.parametrize("value", list(DESK_ON_ERROR))
def test_desk_on_error_accepts_the_three_fallbacks(value: str) -> None:
    assert LiveConfig(desk={"on_error": value}).desk["on_error"] == value


@pytest.mark.parametrize("value", ["flatten", "FLAT", "follow-quant", "", 0, True, ["hold"]])
def test_desk_on_error_is_strict(value: Any) -> None:
    with pytest.raises(ValueError, match="desk.on_error"):
        LiveConfig(desk={"on_error": value})


@pytest.mark.parametrize("value", ["false", "true", 1, 0])
def test_desk_enabled_must_be_a_real_boolean(value: Any) -> None:
    with pytest.raises(ValueError, match="desk.enabled"):
        LiveConfig(desk={"enabled": value})
    assert LiveConfig(desk={"enabled": False}).desk_enabled is False
    assert LiveConfig(desk={"enabled": np.bool_(True)}).desk_enabled is True
    assert LiveConfig(desk={"enabled": None}).desk_enabled is False


def test_desk_config_builds_a_full_desk_config() -> None:
    from aurum.agents.config import DEFAULT_PRICES, AgentModelConfig, DeskConfig, ModelPrice

    cfg = LiveConfig(desk_config={
        "chief": {"effort": "low", "max_turns": 3},
        "specialist": {"effort": "medium", "max_turns": 2},
        "role_models": {"risk_officer": {"effort": "high", "max_turns": 4}},
        "max_cost_usd_per_cycle": 1.5, "cache_ttl": "1h",
        "prices": {"my-model": {"input_per_mtok": 2.0, "output_per_mtok": 8.0}},
        "adhoc_tool_whitelist": ["get_market_snapshot", "get_calendar"],
    })
    dc = cfg.build_desk_config()
    assert isinstance(dc, DeskConfig)
    assert dc.chief == AgentModelConfig(effort="low", max_turns=3)
    assert dc.role_models["risk_officer"].effort == "high"
    assert dc.max_cost_usd_per_cycle == 1.5 and dc.cache_ttl == "1h"
    assert dc.prices["my-model"] == ModelPrice(2.0, 8.0) and set(DEFAULT_PRICES) <= set(dc.prices)
    assert isinstance(dc.adhoc_tool_whitelist, tuple)
    assert set(dc.adhoc_tool_whitelist) == {"get_market_snapshot", "get_calendar"}
    assert cfg.to_dict()["desk_config"]["chief"] == {"effort": "low", "max_turns": 3}  # stays plain data
    # defaults when nothing is given
    assert LiveConfig().build_desk_config() == DeskConfig()


@pytest.mark.parametrize("bad", [{"no_such_knob": 1}, {"chief": {"effort": "ultra"}}, {"cache_ttl": "2h"},
                                 {"adhoc_tool_whitelist": ["rm_rf"]}, {"chief": {"bogus": 1}}])
def test_desk_config_is_validated_at_load_time(bad: dict) -> None:
    with pytest.raises(ValueError):
        LiveConfig(desk_config=bad)
    with pytest.raises(ValueError):  # even while the desk is disabled: a bad config is a bad config
        LiveConfig.from_mapping({"live": {"dry_run": True}, "desk": {"enabled": False}, "desk_config": bad})


def test_desk_config_mapping_forms_and_legacy_merge() -> None:
    a = LiveConfig.from_mapping({"live": {"dry_run": True}, "desk_config": {"max_parallel_agents": 2}})
    b = LiveConfig.from_mapping({"live": {"dry_run": True, "desk_config": {"max_parallel_agents": 2}}})
    assert a.desk_config == b.desk_config == {"max_parallel_agents": 2}
    with pytest.raises(ValueError, match="both"):
        LiveConfig.from_mapping({"live": {"desk_config": {}}, "desk_config": {}})
    with pytest.raises(ValueError, match="mapping"):
        LiveConfig.from_mapping({"desk_config": ["max_parallel_agents"]})
    # legacy flat desk.config keys still work and merge with desk_config ...
    m = LiveConfig(desk={"config": {"max_parallel_agents": 2, "cache_ttl": "1h"}},
                   desk_config={"cache_ttl": "1h", "chief": {"effort": "low"}}).build_desk_config()
    assert m.max_parallel_agents == 2 and m.cache_ttl == "1h" and m.chief.effort == "low"
    # ... but a key set to DIFFERENT values in both places is ambiguous
    with pytest.raises(ValueError, match="different values"):
        LiveConfig(desk={"config": {"cache_ttl": "5m"}}, desk_config={"cache_ttl": "1h"})


def test_runner_builds_the_desk_from_desk_config(world: dict, tmp_path: Path) -> None:
    from aurum.agents.testing import FakeAnthropicClient

    runner, _ = _runner(tmp_path, world, dry_run=True, desk={"enabled": True, "mode": "overlay"},
                        desk_config={"chief": {"effort": "low", "max_turns": 2}, "max_cost_usd_per_cycle": 0.5},
                        runner_kw={"desk_client": FakeAnthropicClient({"chief": []})})
    runner.start()
    try:
        assert runner.desk is not None and runner.desk.config.chief.effort == "low"
        assert runner.desk.config.chief.max_turns == 2 and runner.desk.config.max_cost_usd_per_cycle == 0.5
        assert runner.desk.policy.mode == "overlay"
    finally:
        runner.shutdown()


# =============================================================================================
# 2) desk prerequisites before any broker call
# =============================================================================================
class SpyBroker:
    """Records every method call made on the wrapped venue."""

    def __init__(self, inner: PaperBroker) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def wrapped(*a: Any, **k: Any) -> Any:
            self.calls.append(name)
            return attr(*a, **k)
        return wrapped


def test_desk_without_api_key_is_refused_before_any_broker_call(world: dict, tmp_path: Path) -> None:
    bars = world["bars"]
    clock = SimulatedClock(bars["available_at"].iloc[600])
    spy = SpyBroker(PaperBroker(ReplayFeed(bars), costs=COSTS, clock=clock))
    cfg = LiveConfig(state_dir=str(tmp_path / "state"), calendar=None, desk={"enabled": True})
    runner = LiveRunner(cfg, broker=spy, artifact=load_artifact(world["art"]), clock=clock)
    with pytest.raises(RunnerConfigError, match="ANTHROPIC_API_KEY") as ei:
        runner.start()
    assert spy.calls == []                                   # the venue was never touched
    assert not (tmp_path / "state" / "runner.lock").exists()  # nor the state directory
    assert isinstance(ei.value, ValueError)                   # a configuration error
    assert not runner._started


def test_desk_without_anthropic_package_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)   # import anthropic -> ImportError
    cfg = LiveConfig(desk={"enabled": True})
    with pytest.raises(RunnerConfigError, match="anthropic"):
        check_desk_prerequisites(cfg, env={"ANTHROPIC_API_KEY": "sk-test"})


def test_desk_prerequisite_check_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    on, off = LiveConfig(desk={"enabled": True}), LiveConfig(desk={"enabled": False})
    check_desk_prerequisites(on, env={"ANTHROPIC_API_KEY": "sk-test"})   # all present: passes
    check_desk_prerequisites(off, env={})                                 # desk off: nothing needed
    check_desk_prerequisites(on, env={}, desk=object())                   # injected desk
    check_desk_prerequisites(on, env={}, desk_client=object())            # injected client
    for blank in ("", "   "):
        with pytest.raises(RunnerConfigError) as ei:
            check_desk_prerequisites(on, env={"ANTHROPIC_API_KEY": blank})
        assert "ANTHROPIC_API_KEY" in str(ei.value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value-123")
    check_desk_prerequisites(on)                                          # reads os.environ by default
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RunnerConfigError) as ei:
        check_desk_prerequisites(on)
    assert "sk-secret" not in str(ei.value)


def test_injected_desk_needs_no_key(world: dict, tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path, world, dry_run=True, desk={"enabled": True}, runner_kw={"desk": _fake_desk(1.0, 2)})
    res = runner.run(max_cycles=2, install_signal_handlers=False)
    assert len(res) == 2


def test_main_refuses_desk_without_key_before_connecting(world: dict, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    def no_broker(*a: Any, **k: Any) -> Any:
        raise AssertionError("the broker must not be built when the desk cannot run")

    monkeypatch.setattr(runner_mod, "_build_broker", no_broker)
    doc = {"live": {"artifact_dir": str(world["art"]), "state_dir": str(tmp_path / "state"), "calendar": None},
           "desk": {"enabled": True}}
    (tmp_path / "live.yaml").write_text(yaml.safe_dump(doc))
    assert main(["--config", str(tmp_path / "live.yaml"), "--max-cycles", "1"]) == 2
    err = capsys.readouterr().err
    assert "CONFIG ERROR" in err and "ANTHROPIC_API_KEY" in err


# =============================================================================================
# 3) heartbeat during the inter-bar sleep
# =============================================================================================
def _record_heartbeats(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    seen: list[dict] = []
    real = runner_mod.write_heartbeat

    def spy(path: Any, **fields: Any) -> Any:
        seen.append(dict(fields))
        return real(path, **fields)

    monkeypatch.setattr(runner_mod, "write_heartbeat", spy)
    return seen


def test_heartbeat_is_rewritten_while_sleeping(world: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = world["bars"]["available_at"].iloc[600]
    clock = WallClock(t0)
    runner, _ = _runner(tmp_path, world, clock=clock, dry_run=True, heartbeat_seconds=60)
    seen = _record_heartbeats(monkeypatch)
    wake = t0 + pd.Timedelta(seconds=300)
    runner._sleep_until(wake)
    assert clock.now() == wake
    assert len(seen) == 5                                     # every 60 s of a 300 s nap
    assert all(h["status"] == "running" and h["next_wake"] == wake and h["heartbeat_every_s"] == 60 for h in seen)
    hb = read_json(tmp_path / "state" / "heartbeat.json")
    assert hb["status"] == "running" and pd.Timestamp(hb["next_wake"]) == wake
    # the back-off sleep after venue errors keeps the "degraded" status
    seen.clear()
    runner._sleep_until(clock.now() + pd.Timedelta(seconds=120), status="degraded")
    assert [h["status"] for h in seen] == ["degraded", "degraded"]


def test_heartbeat_period_defaults_to_the_poll(world: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = world["bars"]["available_at"].iloc[600]
    clock = WallClock(t0)
    runner, _ = _runner(tmp_path, world, clock=clock, dry_run=True, retry_poll_seconds=20)
    assert runner.config.heartbeat_every_seconds == 20
    seen = _record_heartbeats(monkeypatch)
    runner._sleep_until(t0 + pd.Timedelta(seconds=100))
    assert len(seen) == 5


def test_simulated_clock_does_not_write_in_sleep_heartbeats(world: dict, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _ = _runner(tmp_path, world, dry_run=True, heartbeat_seconds=1)
    seen = _record_heartbeats(monkeypatch)
    t = runner.clock.now() + pd.Timedelta(hours=5)
    runner._sleep_until(t)
    assert seen == [] and runner.clock.now() == t


def test_full_loop_heartbeats_between_bars(world: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end on a wall-clock (non-simulated) loop: one H1 sleep -> one heartbeat per period."""
    clock = WallClock(world["bars"]["available_at"].iloc[600])
    runner, _ = _runner(tmp_path, world, clock=clock, dry_run=True, heartbeat_seconds=600)
    seen = _record_heartbeats(monkeypatch)
    res = runner.run(max_cycles=2, install_signal_handlers=False)
    assert len(res) == 2
    in_sleep = [h for h in seen if "next_wake" in h]
    assert len(in_sleep) == 6                       # 3605 s to the next bar close / 600 s
    assert all(h["status"] == "running" for h in in_sleep)
    assert read_json(tmp_path / "state" / "heartbeat.json")["status"] == "stopped"
    start = read_jsonl(tmp_path / "state" / "decisions.jsonl")[0]
    assert start["type"] == "start" and start["heartbeat_every_s"] == 600


def test_heartbeat_never_raises_on_venue_trouble(world: dict, tmp_path: Path) -> None:
    runner, broker = _runner(tmp_path, world, dry_run=True)

    def boom() -> pd.Timestamp:
        raise BrokerError("terminal disconnected")

    broker.server_time = boom  # type: ignore[method-assign]
    runner._heartbeat("degraded", force=True)
    hb = read_json(tmp_path / "state" / "heartbeat.json")
    assert hb["status"] == "degraded" and hb["broker_time"] is None


@pytest.mark.parametrize("kw", [{"heartbeat_seconds": 0}, {"heartbeat_seconds": -5}, {"retry_poll_seconds": 0},
                                {"heartbeat_seconds": float("nan")}])
def test_heartbeat_and_poll_validation(kw: dict) -> None:
    with pytest.raises(ValueError):
        LiveConfig(**kw)


# =============================================================================================
# 4) logging: never clobber; one INFO line per decision
# =============================================================================================
def test_configure_logging_respects_an_existing_root_config() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    marker = logging.NullHandler()
    try:
        root.handlers = [marker]
        root.setLevel(logging.ERROR)
        assert _configure_logging("DEBUG") is False
        assert root.handlers == [marker] and root.level == logging.ERROR
        root.handlers = []
        assert _configure_logging("debug") is True
        assert len(root.handlers) == 1 and root.level == logging.DEBUG
        root.handlers = []
        assert _configure_logging("NOT_A_LEVEL") is True and root.level == logging.INFO
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_main_does_not_clobber_the_host_logging(world: dict, tmp_path: Path) -> None:
    from aurum.data.store import save_bars

    save_bars(world["bars"], tmp_path / "bars.parquet")
    cfg = {"live": {"artifact_dir": str(world["art"]), "state_dir": str(tmp_path / "state"), "calendar": None},
           "paper": {"bars_path": str(tmp_path / "bars.parquet"), "warmup_bars": 600}}
    (tmp_path / "live.yaml").write_text(yaml.safe_dump(cfg))
    root = logging.getLogger()
    saved_level = root.level
    marker = logging.NullHandler()
    root.addHandler(marker)
    try:
        root.setLevel(logging.WARNING)
        before = root.handlers[:]
        assert main(["--config", str(tmp_path / "live.yaml"), "--max-cycles", "2", "--log-level", "DEBUG"]) == 0
        assert root.handlers == before and root.level == logging.WARNING
    finally:
        root.removeHandler(marker)
        root.setLevel(saved_level)


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "aurum.live.runner" and r.levelno == logging.INFO and r.getMessage().startswith("decision ")]


def test_one_concise_info_line_per_decision(world: dict, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="aurum.live.runner")
    runner, _ = _runner(tmp_path / "dry", world, dry_run=True)
    res = runner.run(max_cycles=6, install_signal_handlers=False)
    lines = _decision_lines(caplog)
    assert len(lines) == len(res) == 6
    assert all(LINE_RE.match(line) for line in lines), lines
    assert all(" desk off " in line for line in lines)
    for r, line in zip(res, lines, strict=True):
        assert r.time.isoformat() in line and f"status={r.status}" in line
        assert f"combined {r.combined:+.3f}" in line and f"approved {r.approved_lots:+.2f}" in line
        assert f"target {r.requested_lots:+.2f}" in line
    assert any("orders planned open" in line for line in lines)   # dry-run shows what it would send
    # between start and stop the runner logs nothing else at INFO
    msgs = [r.getMessage() for r in caplog.records if r.name == "aurum.live.runner" and r.levelno == logging.INFO]
    i0 = next(i for i, m in enumerate(msgs) if m.startswith("LiveRunner started"))
    i1 = next(i for i, m in enumerate(msgs) if m.startswith("LiveRunner stopped"))
    assert msgs[i0 + 1:i1] == lines
    # live paper orders show their outcome
    caplog.clear()
    runner, _ = _runner(tmp_path / "live", world, sizer={"target_vol": 0.3})
    runner.run(max_cycles=20, install_signal_handlers=False)
    assert any(" filled" in line and "orders open" in line for line in _decision_lines(caplog))


def test_decision_line_shows_the_desk_action(world: dict, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="aurum.live.runner")
    runner, _ = _runner(tmp_path, world, dry_run=True, runner_kw={"desk": _fake_desk(0.5, 4)})
    res = runner.run(max_cycles=4, install_signal_handlers=False)
    lines = _decision_lines(caplog)
    for r, line in zip(res, lines, strict=True):
        assert LINE_RE.match(line)
        assert f" desk {r.desk.get('action') or r.desk.get('status')} " in line
    assert any(" desk scale " in line for line in lines)


def test_orders_summary_formats() -> None:
    t = pd.Timestamp("2026-07-15 10:00", tz="UTC")

    def rep(status: str, *, planned: list | None = None, legs: list | None = None, dry: bool = False) -> ExecutionReport:
        return ExecutionReport(t, "XAUUSD", 7, 0.5, 0.0, 0.0, status, planned=planned or [], legs=legs or [],
                               dry_run=dry)

    buy, sell = OmsLeg("open", Side.BUY, 0.5), OmsLeg("close", Side.SELL, 1.0, ticket=4)
    assert _orders_summary(None) == "none"
    assert _orders_summary(rep("noop")) == "noop"
    assert _orders_summary(rep("dry_run", planned=[buy], dry=True)) == "planned open BUY 0.50"
    lr = [LegReport("c1", sell, True, "filled", 10009, "", 1), LegReport("c2", buy, False, "market_closed", 10018,
                                                                         "", 1)]
    assert _orders_summary(rep("deferred", planned=[sell, buy], legs=lr)) == \
        "close SELL 1.00 filled, open BUY 0.50 market_closed"
    many = [OmsLeg("open", Side.BUY, 0.1)] * 5
    assert _orders_summary(rep("dry_run", planned=many, dry=True)).endswith(", +2 more")


# =============================================================================================
# 7) decision attribution
# =============================================================================================
def _check_attribution(rec: dict) -> dict:
    att = rec["attribution"]
    assert att is not None
    assert set(att) >= {"method", "fdm", "strategies", "raw_combined", "combined", "clipped", "unallocated",
                        "exact", "final_forecast", "desk_delta"}
    assert set(att["strategies"]) == set(rec["forecasts"])
    total = 0.0
    for name, s in att["strategies"].items():
        assert s["forecast"] == pytest.approx(rec["forecasts"][name])
        assert s["contribution"] == pytest.approx(s["weight"] * s["forecast"] * att["fdm"])
        total += s["contribution"]
    assert att["raw_combined"] == pytest.approx(total)
    assert att["combined"] == pytest.approx(rec["combined"])
    assert np.clip(att["raw_combined"], -1, 1) == pytest.approx(rec["combined"], abs=1e-9)
    assert att["exact"] is True
    assert att["final_forecast"] == pytest.approx(rec["final_forecast"])
    assert att["desk_delta"] == pytest.approx(rec["final_forecast"] - rec["combined"])
    return att


def test_attribution_with_a_fitted_combiner(world: dict, tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path, world, dry_run=True)
    res = runner.run(max_cycles=8, install_signal_handlers=False)
    comb = runner.artifact.combiner
    recs = _decisions(tmp_path / "state")              # read back from the JSONL (the analytics format)
    assert len(recs) == 8
    for r, rec in zip(res, recs, strict=True):
        att = _check_attribution(rec)
        assert att["method"] == comb.method and att["fdm"] == pytest.approx(comb.fdm_)
        for name, s in att["strategies"].items():
            assert s["weight"] == pytest.approx(float(comb.weights_[name]))
        assert att["unallocated"] == pytest.approx(max(0.0, 1.0 - float(comb.weights_.sum())))
        assert att["desk_delta"] == 0.0
        assert r.attribution is not None and r.attribution["combined"] == pytest.approx(r.combined)
    json.dumps(recs[0]["attribution"])  # plain JSON


def test_attribution_without_a_combiner_is_the_equal_mean(world: dict, tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path, world, art_key="art_nocomb", dry_run=True)
    runner.run(max_cycles=4, install_signal_handlers=False)
    for rec in _decisions(tmp_path / "state"):
        att = _check_attribution(rec)
        n = len(rec["forecasts"])
        assert att["method"] == "equal_mean" and att["fdm"] == 1.0
        assert all(s["weight"] == pytest.approx(1.0 / n) for s in att["strategies"].values())
        assert att["unallocated"] == pytest.approx(0.0)


def test_attribution_records_the_desk_delta(world: dict, tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path, world, dry_run=True, runner_kw={"desk": _fake_desk(0.5, 6)})
    runner.run(max_cycles=6, install_signal_handlers=False)
    decided = [r for r in _decisions(tmp_path / "state") if r["desk"] and r["desk"].get("status") == "decided"]
    assert decided
    for rec in decided:
        att = _check_attribution(rec)
        assert att["desk_delta"] == pytest.approx(-0.5 * rec["combined"])


class _BigFDM:
    """A combiner whose FDM pushes the sum past 1 (exercises the clip)."""

    method = "custom_big_fdm"

    def __init__(self, names: list[str]) -> None:
        self.weights_ = pd.Series(1.0 / len(names), index=names)
        self.fdm_ = 5.0

    def combine(self, fc: pd.DataFrame) -> pd.Series:
        out = fc[list(self.weights_.index)].to_numpy() @ self.weights_.to_numpy() * self.fdm_
        return pd.Series(np.clip(out, -1, 1), index=fc.index, name="combined")


class _Opaque:
    """A combiner exposing no weights: attribution degrades gracefully."""

    def combine(self, fc: pd.DataFrame) -> pd.Series:
        return fc.mean(axis=1).clip(-1, 1).rename("combined")


def test_attribution_clipping_and_opaque_combiners(world: dict, tmp_path: Path) -> None:
    bars = world["bars"]
    path = build_artifact(tmp_path / "art_const", bars, n_train=350, combiner=False,
                          strategies=[ExplodingStub()])          # constant +0.5 forecast
    art = load_artifact(path)
    art.combiner = _BigFDM(["exploding_stub"])
    runner, _ = _runner(tmp_path / "big", world, artifact=art, dry_run=True)
    runner.run(max_cycles=2, install_signal_handlers=False)
    for rec in _decisions(tmp_path / "big" / "state"):
        att = _check_attribution(rec)
        assert att["method"] == "custom_big_fdm" and att["raw_combined"] == pytest.approx(2.5)
        assert att["clipped"] is True and att["combined"] == 1.0

    path2 = build_artifact(tmp_path / "art_two", bars, n_train=350, combiner=False,
                           strategies=[EmaCrossStub(), ExplodingStub()])
    art2 = load_artifact(path2)
    art2.combiner = _Opaque()
    runner, _ = _runner(tmp_path / "opaque", world, artifact=art2, dry_run=True)
    runner.run(max_cycles=2, install_signal_handlers=False)
    for rec in _decisions(tmp_path / "opaque" / "state"):
        att = rec["attribution"]
        assert att["method"] == "_Opaque" and att["fdm"] is None and att["exact"] is False
        assert att["raw_combined"] is None and att["unallocated"] is None
        assert all(s["weight"] is None and s["contribution"] is None for s in att["strategies"].values())
        assert att["strategies"]["exploding_stub"]["forecast"] == 0.5


def test_attribution_is_null_when_signals_fail_or_data_is_skipped(world: dict, tmp_path: Path) -> None:
    bars = world["bars"]
    path = build_artifact(tmp_path / "art_boom", bars, n_train=350, combiner=False,
                          strategies=[ExplodingStub(always=True)])
    runner, _ = _runner(tmp_path / "boom", world, artifact=load_artifact(path), dry_run=True)
    res = runner.run(max_cycles=2, install_signal_handlers=False)
    assert all(r.status.startswith("error") and r.attribution is None for r in res)
    assert all(rec["attribution"] is None for rec in _decisions(tmp_path / "boom" / "state"))
    # a bar skipped for insufficient history: the key exists, the value is null
    runner, _ = _runner(tmp_path / "hist", world, start_i=60, dry_run=True)
    runner.run(max_cycles=1, install_signal_handlers=False)
    rec = _decisions(tmp_path / "hist" / "state")[0]
    assert rec["status"] == "insufficient_history" and "attribution" in rec and rec["attribution"] is None


# =============================================================================================
# 5) runner side of an unconfirmed venue time zone
# =============================================================================================
class UnconfirmedTzBroker:
    """A paper venue reporting an MT5-style unconfirmed server time zone."""

    tz_confirmed = False
    tz_status = {"server_tz": "NY+7", "source": "persisted", "confirmed": False, "config": "auto"}

    def __init__(self, inner: PaperBroker) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def test_runner_reports_an_unconfirmed_venue_time_zone(world: dict, tmp_path: Path) -> None:
    bars = world["bars"]
    clock = SimulatedClock(bars["available_at"].iloc[600])
    broker = UnconfirmedTzBroker(PaperBroker(ReplayFeed(bars), costs=COSTS, clock=clock))
    cfg = LiveConfig(dry_run=True, state_dir=str(tmp_path / "state"), calendar=None)
    runner = LiveRunner(cfg, broker=broker, artifact=load_artifact(world["art"]), clock=clock)
    res = runner.run(max_cycles=3, install_signal_handlers=False)
    assert len(res) == 3 and all(r.status in ("dry_run", "noop") for r in res)   # dry-run decides anyway
    log = read_jsonl(tmp_path / "state" / "decisions.jsonl")
    assert log[0]["venue_tz"] == UnconfirmedTzBroker.tz_status
    assert all(r["venue_tz_confirmed"] is False for r in log if r["type"] == "decision")
    alerts = read_jsonl(tmp_path / "state" / "alerts.jsonl")
    assert any(a["kind"] == "server_tz_unconfirmed" and "dry-run" in a["message"] for a in alerts)


def test_paper_venue_records_no_tz_fields(world: dict, tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path, world, dry_run=True)
    runner.run(max_cycles=1, install_signal_handlers=False)
    log = read_jsonl(tmp_path / "state" / "decisions.jsonl")
    assert "venue_tz" not in log[0] and "venue_tz_confirmed" not in log[1]


def test_build_broker_gives_mt5_a_persistent_tz_state(tmp_path: Path) -> None:
    cfg = LiveConfig(broker="mt5", state_dir=str(tmp_path / "s"), mt5={"server_tz": "auto"})
    assert runner_mod._mt5_kwargs(cfg)["tz_state_path"] == str(tmp_path / "s" / runner_mod.MT5_TZ_STATE_FILE)
    own = LiveConfig(broker="mt5", state_dir=str(tmp_path / "s"), mt5={"tz_state_path": str(tmp_path / "x.json")})
    assert runner_mod._mt5_kwargs(own)["tz_state_path"] == str(tmp_path / "x.json")
    assert own.mt5 == {"tz_state_path": str(tmp_path / "x.json")}   # the config itself is not mutated


def test_desk_config_clash_check_compares_normalised_values() -> None:
    cfg = LiveConfig(desk={"config": {"adhoc_tool_whitelist": ["get_calendar"]}},
                     desk_config={"adhoc_tool_whitelist": ("get_calendar",), "chief": {"effort": "low"}})
    assert cfg.build_desk_config().adhoc_tool_whitelist == ("get_calendar",)


def test_attribution_treats_unkeyed_weights_as_opaque(world: dict, tmp_path: Path) -> None:
    class ArrayWeights:
        method = "array_weights"
        weights_ = np.array([0.2, 0.3, 0.5])
        fdm_ = 1.2

        def combine(self, fc: pd.DataFrame) -> pd.Series:
            return fc.mean(axis=1).clip(-1, 1).rename("combined")

    art = load_artifact(world["art_nocomb"])
    art.combiner = ArrayWeights()
    runner, _ = _runner(tmp_path, world, artifact=art, dry_run=True)
    res = runner.run(max_cycles=1, install_signal_handlers=False)
    att = res[0].attribution
    assert att is not None and att["method"] == "array_weights" and att["exact"] is False
    assert all(s["weight"] is None for s in att["strategies"].values())
