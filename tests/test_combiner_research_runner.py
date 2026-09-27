"""LiveRunner -> TradingDesk ``previous_forecast``: DecisionPolicy(on_failure="hold") keeps
the standing forecast, also across a restart with a fresh desk (whose own memory is empty)
and after cycles in which the desk raised."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from test_live_support import build_artifact, synthetic_bars

from aurum.agents import DecisionPolicy, TradingDesk
from aurum.agents.testing import FakeAnthropicClient, decision_call, message, refusal
from aurum.execution.costs import CostModel
from aurum.live.broker import SimulatedClock
from aurum.live.paper import PaperBroker, ReplayFeed
from aurum.live.runner import LiveConfig, LiveDeskDataProvider, LiveRunner, load_artifact
from aurum.live.state import read_json, read_jsonl


@pytest.fixture(autouse=True)
def _no_network_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AURUM_ALERT_WEBHOOK_URL", "ANTHROPIC_API_KEY", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER",
                "AURUM_ARTIFACT_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("prev_fc")
    bars = synthetic_bars(760, weekend_gaps=False)
    return {"bars": bars, "art": build_artifact(root / "art", bars, n_train=350)}


def _desk(client: Any, **policy: Any) -> TradingDesk:
    return TradingDesk(LiveDeskDataProvider(), client=client, policy=DecisionPolicy(**policy), journal_dir=None)


def _runner(state: Path, world: dict, *, desk_obj: Any, broker: PaperBroker | None = None, start_i: int = 600,
            **cfg: Any) -> tuple[LiveRunner, PaperBroker]:
    bars = world["bars"]
    if broker is None:
        broker = PaperBroker(ReplayFeed(bars), costs=CostModel(), clock=SimulatedClock(bars["available_at"].iloc[start_i]))
    conf = LiveConfig(dry_run=True, state_dir=str(state), calendar=None,
                      risk={"daily_loss_persistent": False, "max_daily_loss": 0.05}, **cfg)
    return LiveRunner(conf, broker=broker, artifact=load_artifact(world["art"]), clock=broker.clock, desk=desk_obj), broker


def test_hold_keeps_the_standing_forecast_across_a_restart(world: dict, tmp_path: Path) -> None:
    state = tmp_path / "state"
    pol = {"mode": "discretionary", "on_failure": "hold"}
    # cycle 1: the Chief overrides to +0.30
    d1 = _desk(FakeAnthropicClient({"chief": [message(decision_call(action="override", forecast=0.3,
                                                                    confidence=0.9))]}), **pol)
    r1, broker = _runner(state, world, desk_obj=d1)
    (c1,) = r1.run(max_cycles=1, install_signal_handlers=False)
    assert c1.desk["status"] == "decided" and c1.final_forecast == pytest.approx(0.3)
    assert c1.desk["previous_forecast"] is None                         # fresh state: nothing standing
    assert read_json(state / "runner_state.json")["prev_final_forecast"] == pytest.approx(0.3)

    # restart with a NEW desk (empty memory) whose Chief fails: "hold" must keep +0.30, not go flat
    d2 = _desk(FakeAnthropicClient({"chief": [refusal()] * 3}), **pol)
    assert d2.last_final_forecast is None
    r2, _ = _runner(state, world, desk_obj=d2, broker=broker)
    res = r2.run(max_cycles=3, install_signal_handlers=False)
    assert [r.desk["status"] for r in res] == ["failed"] * 3
    assert all(r.final_forecast == pytest.approx(0.3) for r in res)
    assert all(r.desk["previous_forecast"] == pytest.approx(0.3) for r in res)
    recs = [x for x in read_jsonl(state / "decisions.jsonl") if x.get("type") == "decision"]
    assert [x["previous_forecast"] for x in recs][:2] == [None, pytest.approx(0.3)]
    # the sizer sized the held forecast (desk -> sizer -> risk path unchanged)
    assert all(x["sizing"]["forecast"] == pytest.approx(0.3) for x in recs[1:])


def test_previous_forecast_follows_what_was_actually_traded(world: dict, tmp_path: Path) -> None:
    """Overlay + hold: the standing forecast is the policy-gated final of the last cycle, and a
    desk that RAISES (runner fallback on_error="hold") also uses the runner's standing value."""
    state = tmp_path / "state"
    d1 = _desk(FakeAnthropicClient({"chief": [message(decision_call(action="scale", scale=0.5, confidence=0.8))]}),
               mode="overlay", on_failure="hold")
    r1, broker = _runner(state, world, desk_obj=d1)
    (c1,) = r1.run(max_cycles=1, install_signal_handlers=False)
    assert c1.final_forecast == pytest.approx(0.5 * c1.combined) and c1.combined != 0.0

    class Broken:
        last_final_forecast = None          # a desk with no memory of its own
        calls: list = []

        def run_cycle(self, now: Any, q: float, **kw: Any) -> Any:
            self.calls.append(kw.get("previous_forecast"))
            raise RuntimeError("desk exploded")

    b = Broken()
    r2, _ = _runner(state, world, desk_obj=b, broker=broker, desk={"on_error": "hold"})
    (c2,) = r2.run(max_cycles=1, install_signal_handlers=False)
    assert b.calls == [pytest.approx(c1.final_forecast)]              # the runner passed it
    assert c2.desk["status"] == "error"
    # overlay semantics of the fallback: hold, but never beyond / against the current quant forecast
    q, prev = c2.combined, c1.final_forecast
    if prev * q > 0:
        assert c2.final_forecast == pytest.approx(prev if abs(prev) <= abs(q) else q)
    else:
        assert c2.final_forecast == 0.0
    st = read_json(state / "runner_state.json")
    assert st["prev_final_forecast"] == pytest.approx(c2.final_forecast)
    assert pd.Timestamp(st["last_bar"]) == c2.bar_time
