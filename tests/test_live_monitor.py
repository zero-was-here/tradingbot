"""Monitoring: PSI drift, slippage vs cost model, PnL band, alert sinks (no network)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import Fill, Side
from aurum.execution.costs import CostModel
from aurum.features.pipeline import FeaturePipeline
from aurum.live.monitor import (
    Alert,
    AlertManager,
    DriftMonitor,
    FeatureReference,
    JsonlAlertSink,
    LiveMonitor,
    PnLBand,
    SlippageTracker,
    WebhookAlertSink,
    check_heartbeat,
    psi,
    redact,
)
from aurum.live.state import read_jsonl, write_heartbeat

T0 = pd.Timestamp("2026-09-01 10:00", tz="UTC")


def test_psi_basics() -> None:
    e = np.full(10, 0.1)
    assert psi(e, e) == pytest.approx(0.0, abs=1e-12)
    a = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
    assert psi(e, a) > 0.25
    assert psi(e, a * 1000) == pytest.approx(psi(e, a))  # counts or proportions
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [1.0])


def test_drift_monitor_flags_shifted_feature(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000),
                          "flag": rng.choice([-1.0, 0.0, 1.0], size=5000)})
    ref = FeatureReference.from_frame(train)
    assert set(ref.columns) == {"a", "b", "flag"} and len(ref.edges["flag"]) == 2
    ref2 = FeatureReference.load(ref.save(tmp_path / "ref.json"))
    assert ref2.edges == ref.edges and ref2.source == "train"
    live = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(loc=1.0, size=600),
                         "flag": rng.choice([-1.0, 0.0, 1.0], size=600, p=[0.1, 0.1, 0.8])})
    rep = DriftMonitor(ref).evaluate(live, time=T0)
    assert rep is not None
    assert rep.psi["a"] < 0.1 and rep.psi["b"] > 0.25 and rep.psi["flag"] > 0.25
    assert rep.alert == ["b", "flag"] and rep.max_psi == max(rep.psi.values())
    assert DriftMonitor(ref).evaluate(live.iloc[:10]) is None  # too few rows
    rep = DriftMonitor(ref).evaluate(live.drop(columns=["a"]))
    assert rep.missing == ["a"]


def test_reference_from_pipeline_stats() -> None:
    from aurum.core.types import MarketData
    from aurum.data.synthetic import make_synthetic_bars

    bars = make_synthetic_bars(800, "H1", seed=1)
    pipe = FeaturePipeline(groups=["returns"])
    raw = pipe.compute(MarketData(bars))
    pipe.fit(raw.iloc[100:])
    ref = FeatureReference.from_pipeline(pipe)
    assert ref.source == "gaussian_approx" and ref.columns
    X = pipe.transform(raw).iloc[100:]
    rep = DriftMonitor(ref).evaluate(X)
    assert rep is not None and np.median(list(rep.psi.values())) < 0.25  # same data: mostly stable


def test_slippage_tracker() -> None:
    costs = CostModel()
    tr = SlippageTracker(costs, min_fills=5, alert_ratio=2.0)
    for i in range(5):
        f = Fill(client_id=f"c{i}", symbol="XAUUSD", side=Side.BUY, lots=1.0, price=2000.0 + 0.2, time=T0)
        tr.record(f, reference_mid=2000.0, spread=0.3, bar_range=1.0)
    s = tr.summary()
    exp = 0.15 + 0.02 + 0.02 * 1.0
    assert s["mean_expected_per_oz"] == pytest.approx(exp)
    assert s["mean_realized_per_oz"] == pytest.approx(0.2)
    assert tr.check(T0) == []
    for i in range(5):  # sells filled 1.5 below mid: far worse than modelled
        f = Fill(client_id=f"s{i}", symbol="XAUUSD", side=Side.SELL, lots=1.0, price=1998.5, time=T0)
        tr.record(f, reference_mid=2000.0, spread=0.3, bar_range=1.0)
    alerts = tr.check(T0)
    assert len(alerts) == 1 and alerts[0].kind == "slippage"


def test_pnl_band() -> None:
    band = PnLBand(daily_mean=0.0005, daily_std=0.005, min_days=5)
    eq = 100_000.0
    for d in range(10):
        band.update(T0 + pd.offsets.BDay(d), eq)  # ten trading days
        eq *= 0.99  # -1%/day, i.e. -2 sigma per day
    s = band.status()
    assert s["n_days"] == 9 and s["z"] < -3
    alerts = band.check(T0)
    assert alerts and alerts[0].level == "critical"
    band2 = PnLBand(0.0, 0.01)
    band2.load_state(band.to_state())
    assert band2.status()["n_days"] == 9
    assert PnLBand.from_stats({"sharpe": 1.0, "ann_vol": 0.1}).sigma == pytest.approx(0.1 / np.sqrt(252))
    assert PnLBand.from_stats({}) is None


def test_webhook_payload_redacted_and_url_never_logged(monkeypatch: pytest.MonkeyPatch,
                                                       caplog: pytest.LogCaptureFixture) -> None:
    url = "https://hooks.example.test/services/T000/B000/SECRETTOKEN123"
    monkeypatch.setenv("MT5_PASSWORD", "hunter2-password")
    sent: list[tuple[str, dict]] = []

    def post(u: str, body: bytes, timeout: float) -> None:
        sent.append((u, json.loads(body)))

    sink = WebhookAlertSink(url, post=post)
    sink.send(Alert("info", "x", "below min level"))
    assert sent == []
    sink.send(Alert("critical", "risk_halt", "halted; password hunter2-password leaked?", time=T0,
                    data={"password": "p", "api_key": "k", "equity": 1.5, "nested": {"token": "t", "ok": 1}}))
    assert len(sent) == 1
    u, payload = sent[0]
    blob = json.dumps(payload)
    assert u == url and "hunter2-password" not in blob and "SECRETTOKEN123" not in blob
    assert payload["data"]["password"] == "***" and payload["data"]["nested"] == {"token": "***", "ok": 1}
    assert payload["text"] == payload["content"] and "risk_halt" in payload["text"]

    def failing(u: str, body: bytes, timeout: float) -> None:
        raise OSError(f"connection refused for {u}")

    caplog.set_level(logging.DEBUG)
    bad = WebhookAlertSink(url, post=failing)
    bad.send(Alert("critical", "k", "m"))  # must not raise
    assert bad.n_failed == 1 and "SECRETTOKEN123" not in caplog.text
    assert not WebhookAlertSink(None, env_var="AURUM_TEST_UNSET_VAR").enabled
    tg = WebhookAlertSink(url, post=post, style="telegram")
    monkeypatch.setenv("AURUM_ALERT_TELEGRAM_CHAT_ID", "12345")
    tg.send(Alert("warning", "k", "hello"))
    assert set(sent[-1][1]) == {"chat_id", "text"}
    assert redact({"login": 1, "x": "y"}) == {"login": "***", "x": "y"}


def test_alert_manager_cooldown_and_jsonl(tmp_path: Path) -> None:
    mgr = AlertManager([JsonlAlertSink(tmp_path / "alerts.jsonl")], cooldown_seconds=600)
    assert mgr.alert("warning", "drift", "one", time=T0)
    assert not mgr.alert("warning", "drift", "two", time=T0 + pd.Timedelta(minutes=5))
    assert mgr.alert("critical", "drift", "escalated", time=T0 + pd.Timedelta(minutes=5))
    assert mgr.alert("warning", "drift", "three", time=T0 + pd.Timedelta(minutes=11))
    rows = read_jsonl(tmp_path / "alerts.jsonl")
    assert [r["message"] for r in rows] == ["one", "escalated", "three"]


def test_check_heartbeat(tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    assert check_heartbeat(p, max_age_seconds=60).kind == "heartbeat"
    write_heartbeat(p, status="running")
    assert check_heartbeat(p, max_age_seconds=60) is None
    late = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5)
    a = check_heartbeat(p, max_age_seconds=60, now=late)
    assert a is not None and a.level == "critical" and a.data["status"] == "running"


def test_live_monitor_facade() -> None:
    rng = np.random.default_rng(1)
    ref = FeatureReference.from_frame(pd.DataFrame({"a": rng.normal(size=2000)}))
    sent: list[Alert] = []

    class Sink:
        def send(self, alert: Alert) -> None:
            sent.append(alert)

    mon = LiveMonitor(alerts=AlertManager([Sink()]), drift=DriftMonitor(ref), slippage=SlippageTracker(),
                      pnl=PnLBand(0.0, 0.01), drift_every=2)
    feats = pd.DataFrame({"a": rng.normal(loc=2.0, size=300)})
    mon.on_decision(T0, equity=1e5, features=feats)
    assert sent == []
    mon.on_decision(T0 + pd.Timedelta(hours=1), equity=1e5, features=feats)
    assert [a.kind for a in sent] == ["feature_drift"]
    snap = mon.snapshot()
    assert snap["drift"]["n_alert"] == 1 and snap["n_decisions"] == 2


# ------------------------------------------------------------------------------------------------
# adversarial review
# ------------------------------------------------------------------------------------------------
def test_pnl_band_folds_weekend_sessions_into_monday() -> None:
    """Gold trades Sunday evening; the backtest's daily stats fold weekends into Monday
    (aurum.backtest.metrics.daily_returns). Counting Sunday as its own day inflated n by ~20%."""
    band = PnLBand(daily_mean=0.0, daily_std=0.01, min_days=1)
    fri = pd.Timestamp("2026-09-04 20:00", tz="UTC")  # a Friday
    band.update(fri, 100_000.0)
    band.update(fri + pd.Timedelta(days=2, hours=3), 100_100.0)   # Sunday 23:00 session
    band.update(fri + pd.Timedelta(days=3, hours=1), 100_200.0)   # Monday 21:00
    assert sorted(band.day_close) == ["2026-09-04", "2026-09-07"]
    assert band.status()["n_days"] == 1 and band.day_close["2026-09-07"] == 100_200.0
    raw = PnLBand(daily_mean=0.0, daily_std=0.01, fold_weekends=False)
    for t in (fri, fri + pd.Timedelta(days=2, hours=3), fri + pd.Timedelta(days=3, hours=1)):
        raw.update(t, 100_000.0)
    assert raw.status()["n_days"] == 2


def test_check_heartbeat_tolerates_naive_time(tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    p.write_text(json.dumps({"time": "2026-09-01T10:00:00", "status": "running"}))
    a = check_heartbeat(p, max_age_seconds=60, now=pd.Timestamp("2026-09-01 10:05", tz="UTC"))
    assert a is not None and a.level == "critical" and "300" in a.message
    p.write_text(json.dumps({"time": None}))
    assert check_heartbeat(p, max_age_seconds=60).message == "heartbeat has no valid time"


def test_pnl_band_from_annualised_stats_has_the_right_drift() -> None:
    """SR = mean_d/sd_d*sqrt(252): the daily mean is SR*ann_vol/252, not SR*ann_vol/sqrt(252)
    (which expected ~159%/yr for SR 1 at 10% vol and fired critical alerts on normal PnL)."""
    b = PnLBand.from_stats({"sharpe": 1.0, "ann_vol": 0.10})
    sd = 0.10 / np.sqrt(252)
    assert b.sigma == pytest.approx(sd)
    assert b.mu == pytest.approx(0.10 / 252 - 0.5 * sd**2)
    assert b.mu * 252 == pytest.approx(0.10, abs=0.01)
    nested = PnLBand.from_stats({"source_run": "runs/wf", "combined": {"sharpe": 0.5, "ann_vol": 0.08}})
    assert nested is not None and nested.mu == pytest.approx(0.5 * 0.08 / 252 - 0.5 * (0.08 / np.sqrt(252)) ** 2)
    assert PnLBand.from_stats({"combined": {"sharpe": None, "ann_vol": 0.1}}) is None
    # a strategy performing exactly as expected for a year is NOT flagged
    band = PnLBand.from_stats({"sharpe": 1.0, "ann_vol": 0.10}, min_days=5)
    eq, t = 100_000.0, pd.Timestamp("2026-01-05", tz="UTC")
    for d in range(252):
        band.update(t + pd.offsets.BDay(d), eq)
        eq *= float(np.exp(band.mu))
    assert band.check() == [] and abs(band.status()["z"]) < 0.5
