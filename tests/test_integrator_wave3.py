"""Wave-3 integration: the pieces that connect the financing, cost-aware combiner/research and
stability changes across module boundaries.

* RL training/evaluation/rollouts read the SAME point-in-time benchmark rates as the engine
  (``md.macro``), so the engine-vs-rollout parity diagnostic stays at zero under rate financing.
* The live runner hands its macro data to the paper broker's rate financing (at construction
  and on every macro reload), so paper equity matches ``run_backtest`` on the same macro.
* ``combiner.allow_unallocated`` / ``combiner.cost_multiplier`` are real config fields that reach
  every combiner the research code builds.
* Package-level exports of the new public names.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.engine import run_backtest
from aurum.core.config import load_config
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.execution.costs import CostModel

DELAY = pd.Timedelta(seconds=5)


def fedfunds_for(bars: pd.DataFrame, *, seed: int = 0, level: float = 5.5) -> pd.DataFrame:
    """Daily FRED-style series (percent, available the next day 21:30 UTC) that moves every day
    and sits far from the 3% fallback, so a missing or ignored series changes the numbers."""
    days = pd.date_range(bars.index[0].normalize() - pd.Timedelta(days=10), bars.index[-1].normalize(),
                         freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    vals = level + np.cumsum(rng.normal(0.0, 0.1, len(days)))
    return pd.DataFrame({"value": vals, "available_at": days + pd.Timedelta(hours=45, minutes=30)},
                        index=days.rename("date"))


# ---- RL: training / evaluation / rollouts use md.macro rates ------------------------------------
def test_rl_evaluation_env_reads_macro_rates_and_matches_the_engine() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    from aurum.rl.env import EnvConfig
    from aurum.rl.train import RLTrainConfig, _make_env, evaluate_policy, prepare_data

    bars = make_synthetic_bars(2300, "H1", seed=4, model="trend", regime_params={"phi": 0.2})
    macro = {"fedfunds": fedfunds_for(bars, seed=3)}
    n_tr = 1700
    md_tr = MarketData(bars=bars.iloc[:n_tr], macro=macro)
    md_va = MarketData(bars=bars.iloc[n_tr:], macro=macro)
    cfg = RLTrainConfig(feature_groups=("returns", "volatility"),
                        env=EnvConfig(window=2, episode_length=256, episode_anchor=None, risk=None))
    data = prepare_data(md_tr, md_va, cfg)
    for train in (True, False):
        env = _make_env(data, cfg.env, train=train)
        assert env.sim.rate_curve is not None and len(env.sim.rate_curve) > 0

    k = {"i": 0}

    def predict(obs: np.ndarray) -> int:   # deterministic, trades often and holds overnight
        k["i"] += 1
        return (k["i"] // 30) % 5

    ev = evaluate_policy(predict, data, cfg.env)
    assert ev.backtest.costs["swap"].abs().sum() > 0
    # env (training reward / rollout) and engine now read the same benchmark: exact parity
    assert ev.metrics["engine_rollout_max_abs_diff"] <= 1e-6


def test_rollout_artifact_passes_macro_rates_to_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    import aurum.rl.train as train
    from aurum.features.pipeline import FeaturePipeline
    from aurum.rl.env import EnvConfig, GoldTradingEnv

    bars = make_synthetic_bars(1200, "H1", seed=8)
    md = MarketData(bars=bars, macro={"fedfunds": fedfunds_for(bars, seed=1)})
    pipe = FeaturePipeline(groups=["returns", "volatility"])
    raw = pipe.compute(md)
    pipe.fit(raw.iloc[pipe.max_lookback:800])
    seen: dict[str, Any] = {}

    class SpyEnv(GoldTradingEnv):
        def __init__(self, *a: Any, **kw: Any) -> None:
            seen["rates"] = kw.get("rates")
            super().__init__(*a, **kw)

    monkeypatch.setattr(train, "GoldTradingEnv", SpyEnv)
    art = SimpleNamespace(pipeline=pipe, config=SimpleNamespace(env=EnvConfig(window=2, risk=None)),
                          predict=lambda obs: 4)   # always max long: holds across rollovers
    ro = train.rollout_artifact(art, md, start=900)
    assert seen["rates"] is md.macro
    assert ro.result.meta["financing"]["rate_first_available"] is not None
    assert ro.result.costs["swap"].sum() < 0


# ---- live runner -> paper broker rates ------------------------------------------------------------
def test_build_broker_passes_macro_dir_rates(tmp_path: Path) -> None:
    from aurum.core.instrument import XAUUSD
    from aurum.data.macro import save_macro_dir
    from aurum.data.store import save_bars
    from aurum.live.runner import LiveConfig, _build_broker

    bars = make_synthetic_bars(600, "H1", seed=2)
    save_bars(bars, tmp_path / "bars.parquet")
    save_macro_dir({"fedfunds": fedfunds_for(bars)}, tmp_path / "macro")
    base = {"broker": "paper", "state_dir": str(tmp_path / "s"), "calendar": None,
            "paper": {"bars_path": str(tmp_path / "bars.parquet"), "warmup_bars": 300}}
    pb, _, _ = _build_broker(LiveConfig(**base, macro_dir=str(tmp_path / "macro")), XAUUSD)
    assert pb.rate_curve is not None and pb.rate_curve.name == "fedfunds" and len(pb.rate_curve) > 30
    pb2, _, _ = _build_broker(LiveConfig(**{**base, "state_dir": str(tmp_path / "s2")}), XAUUSD)
    assert pb2.rate_curve is None                        # no macro: fallback rate (warned)
    fixed = {**base, "state_dir": str(tmp_path / "s3"), "costs": {"financing": {"mode": "fixed"}}}
    pb3, _, _ = _build_broker(LiveConfig(**fixed, macro_dir=str(tmp_path / "macro")), XAUUSD)
    assert pb3.rate_curve is None                        # fixed mode reads no rates


def test_paper_run_uses_macro_dir_rates_and_matches_the_engine(tmp_path: Path) -> None:
    from test_live_support import build_artifact, synthetic_bars

    from aurum.data.macro import save_macro_dir
    from aurum.live.broker import SimulatedClock
    from aurum.live.paper import PaperBroker, ReplayFeed
    from aurum.live.runner import LiveConfig, LiveRunner, load_artifact
    from aurum.live.state import read_jsonl
    from aurum.portfolio.sizing import VolTargetSizer
    from aurum.risk.manager import RiskLimits, StandardRiskManager

    bars = synthetic_bars(760, weekend_gaps=False)   # (market-closed deferrals are tested elsewhere)
    art_path = build_artifact(tmp_path / "art", bars, n_train=350)
    macro = {"fedfunds": fedfunds_for(bars, seed=5)}
    save_macro_dir(macro, tmp_path / "macro")
    costs = CostModel(commission_per_lot=3.5)            # default: rate financing
    risk = {"daily_loss_persistent": False, "max_daily_loss": 0.05}
    start = 450
    feed = ReplayFeed(bars)
    clock = SimulatedClock(bars["available_at"].iloc[start])
    broker = PaperBroker(feed, costs=costs, clock=clock)  # built WITHOUT rates, like a custom setup
    cfg = LiveConfig(dry_run=False, state_dir=str(tmp_path / "state"), calendar=None, risk=dict(risk),
                     history_bars=10_000, sizer={"target_vol": 0.2}, macro_dir=str(tmp_path / "macro"),
                     macro_max_age_days=30.0)
    runner = LiveRunner(cfg, broker=broker, artifact=load_artifact(art_path), clock=clock)
    runner.run(until=feed.end + DELAY + pd.Timedelta(seconds=1), install_signal_handlers=False)
    assert broker.rate_curve is not None                # the runner handed its macro data over
    recs = [r for r in read_jsonl(tmp_path / "state" / "decisions.jsonl") if r["type"] == "decision"]
    live = pd.Series({pd.Timestamp(r["bar_time"]): r["equity"] for r in recs})

    art = load_artifact(art_path)
    md = MarketData(bars, macro=macro)
    X = art.features(md)
    comb = art.combine(pd.DataFrame({n: s.generate(md, X) for n, s in art.strategies.items()}))
    kw = dict(sizer=VolTargetSizer(**{**art.sizer_config, "target_vol": 0.2}), costs=costs,
              start=bars.index[start], compute_metrics=False)
    bt = run_backtest(md, comb, risk=StandardRiskManager(RiskLimits(**risk)), **kw)
    assert bt.costs["swap"].abs().sum() > 0
    diff = np.abs(live.to_numpy() - bt.equity.reindex(live.index).to_numpy())
    assert np.max(diff) < 1e-6
    # ... and the series mattered: the fallback rate gives a different engine path
    fb = run_backtest(md, comb, risk=StandardRiskManager(RiskLimits(**risk)), rates={}, **kw)
    assert abs(fb.equity.iloc[-1] - bt.equity.iloc[-1]) > 1.0


def test_refresh_broker_rates_is_safe_for_any_broker() -> None:
    from aurum.live.runner import LiveRunner

    calls: list[Any] = []
    LiveRunner._refresh_broker_rates(SimpleNamespace(broker=SimpleNamespace(set_rates=calls.append)), {"x": 1})
    assert calls == [{"x": 1}]
    LiveRunner._refresh_broker_rates(SimpleNamespace(broker=object()), {"x": 1})   # no set_rates: no-op

    def bad(frames: Any) -> None:
        raise ValueError("no available_at")

    LiveRunner._refresh_broker_rates(SimpleNamespace(broker=SimpleNamespace(set_rates=bad)), {"x": 1})


# ---- combiner config knobs ------------------------------------------------------------------------
def test_combiner_config_fields_reach_every_research_combiner() -> None:
    from aurum.research.walkforward import _combiner_options, _make_combiner

    cfg = load_config(None, env={})
    assert cfg.combiner.allow_unallocated is True and cfg.combiner.cost_multiplier == 1.0
    c = cfg.combiner.build()
    assert c.allow_unallocated is True and c.cost_multiplier == 1.0
    legacy = load_config({"combiner": {"allow_unallocated": False, "cost_multiplier": 2.0}}, env={})
    assert legacy.config_hash() != cfg.config_hash()
    comb = _make_combiner(legacy, ["a", "b"])
    assert comb.allow_unallocated is False and comb.cost_multiplier == 2.0
    assert _combiner_options(legacy) == {"allow_unallocated": False, "cost_multiplier": 2.0}
    fixed = load_config({"combiner": {"method": "fixed", "allow_unallocated": False},
                         "strategies": [{"name": "tsmom", "weight": 1.0}, {"name": "ema_cross", "weight": 2.0}]},
                        env={})
    fc = _make_combiner(fixed, ["tsmom", "ema_cross"])
    assert fc.allow_unallocated is False
    with pytest.raises(ValueError, match="cost_multiplier"):
        load_config({"combiner": {"cost_multiplier": -1.0}}, env={}).validate()
    assert load_config("configs/default.yaml", env={}).combiner.allow_unallocated is True


# ---- exports ------------------------------------------------------------------------------------------
def test_new_public_names_are_exported() -> None:
    import aurum.execution as ex
    import aurum.rl as rl
    from aurum.execution.costs import FinancingModel, RateCurve
    from aurum.execution.simulator import intrabar_exit

    assert ex.FinancingModel is FinancingModel and ex.RateCurve is RateCurve
    assert ex.intrabar_exit is intrabar_exit
    assert {"FinancingModel", "RateCurve", "intrabar_exit"} <= set(ex.__all__)
    assert {"load_artifact_bytes", "read_artifact_bytes", "POLICY_FILES"} <= set(rl.__all__)
    pytest.importorskip("stable_baselines3")
    assert callable(rl.load_artifact_bytes) and rl.POLICY_FILES[0] == "policy.zip"


# ---- CLI holdout ledger --------------------------------------------------------------------------------
def test_cli_records_holdout_looks_even_without_writing_results(tmp_path: Path) -> None:
    from aurum.cli import _cli_holdout_ledger

    cfg = load_config({"output": {"dir": str(tmp_path / "runs")}}, env={})
    assert _cli_holdout_ledger(cfg) is None                 # written runs: ledger next to the run dirs
    cfg.output.save_results = False                         # --no-write still SHOWS the holdout
    assert _cli_holdout_ledger(cfg) == tmp_path / "runs" / "holdout_ledger.jsonl"
