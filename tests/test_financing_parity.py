"""Rate-based financing is identical everywhere the SAME simulator semantics are used:
simulator vs PaperBroker, GoldTradingEnv vs run_backtest, the YAML config, and the shared
intrabar exit rule (live reviewer requests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.engine import run_backtest
from aurum.core.config import load_config
from aurum.core.types import MarketData, Side
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events
from aurum.execution.costs import CostModel, FinancingModel
from aurum.execution.simulator import ExecutionSimulator, intrabar_exit
from aurum.live.broker import OrderRequest, SimulatedClock, net_lots
from aurum.live.paper import PaperBroker, ReplayFeed

DELAY = pd.Timedelta(seconds=5)
MAGIC = 77


def fedfunds_for(bars: pd.DataFrame, *, seed: int = 0, level: float = 4.0) -> pd.DataFrame:
    """A daily FRED-style series (percent, available the next day 21:30 UTC) that moves every
    day, so any off-by-one-rollover or look-ahead in the rate lookup changes the result."""
    days = pd.date_range(bars.index[0].normalize() - pd.Timedelta(days=10), bars.index[-1].normalize(),
                         freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    vals = level + np.cumsum(rng.normal(0.0, 0.15, len(days)))
    return pd.DataFrame({"value": vals, "available_at": days + pd.Timedelta(hours=45, minutes=30)},
                        index=days.rename("date"))


def _order(cid: str, side: int, lots: float, clock: SimulatedClock, **kw) -> OrderRequest:
    return OrderRequest(client_id=cid, symbol="XAUUSD", side=Side(side), lots=lots, time=clock.now(),
                        magic=MAGIC, **kw)


def _reconcile(pb: PaperBroker, clock: SimulatedClock, target: float, cur: float, tag: str) -> None:
    d = round(target - cur, 8)
    if abs(d) < 1e-9:
        return
    if cur != 0 and (target == 0 or np.sign(target) != np.sign(cur)):
        pos = pb.positions("XAUUSD", MAGIC)[0]
        assert pb.place_order(_order(f"c{tag}", -int(np.sign(cur)), abs(cur), clock,
                                     position_ticket=pos.ticket)).ok
        if target != 0:
            assert pb.place_order(_order(f"o{tag}", int(np.sign(target)), abs(target), clock)).ok
    else:
        assert pb.place_order(_order(f"o{tag}", int(np.sign(d)), abs(d), clock)).ok


@pytest.mark.parametrize("markups", [(0.025, 0.025), (0.01, 0.04)])
def test_paper_broker_equals_simulator_under_rate_financing(markups) -> None:
    bars = make_synthetic_bars(24 * 25, "H1", seed=8, weekend_gaps=False)
    rates = {"fedfunds": fedfunds_for(bars, seed=2)}
    fin = FinancingModel(markup_long=markups[0], markup_short=markups[1], lease_rate=0.003)
    costs = CostModel(commission_per_lot=3.5, spread_multiplier=1.2, slippage_range_frac=0.05, financing=fin)
    sim = ExecutionSimulator(bars, costs=costs, rates=rates)
    clock = SimulatedClock(bars["available_at"].iloc[0] + DELAY)
    pb = PaperBroker(ReplayFeed(bars), costs=costs, clock=clock, rates=rates)
    rng = np.random.default_rng(4)
    tgt, max_diff = 0.0, 0.0
    for t in range(len(bars) - 1):
        clock.advance_to(bars["available_at"].iloc[t] + DELAY)
        max_diff = max(max_diff, abs(pb.account().equity - sim.equity))
        cur = net_lots(pb.positions("XAUUSD", MAGIC))
        assert cur == pytest.approx(sim.position, abs=1e-9)
        if rng.random() < 0.05:
            tgt = float(rng.choice([-2.0, -1.0, 0.0, 1.0, 1.5, 3.0]))
        _reconcile(pb, clock, tgt, cur, str(t))
        sim.step(tgt)
    clock.advance_to(bars["available_at"].iloc[-1] + DELAY)
    res = sim.result(compute_metrics=False)
    assert res.costs["swap"].abs().sum() > 100.0           # financing really was exercised
    if markups[1] < 0.03:                                   # ~4% benchmark: shorts are paid
        assert (res.costs["swap"] > 0).any() and (res.costs["swap"] < 0).any()
    assert max_diff < 1e-6
    assert abs(pb.account().equity - sim.equity) < 1e-6
    # booked swap = accrued (open) + realised (deals) on the paper side
    paper_swap = sum(p.swap for p in pb.positions()) + pb.deals_frame()["swap"].sum()
    assert paper_swap == pytest.approx(res.costs["swap"].sum(), rel=1e-9)


def test_paper_broker_rate_financing_over_gaps_and_without_series() -> None:
    """Rollovers that fall in a gap between bars (valued at the previous close) and the
    fallback rate (no series) agree too."""
    bars = make_synthetic_bars(24 * 20, "H1", seed=3, weekend_gaps=True)
    bars = bars.loc[bars.index.hour != 20]                 # the rollover (21:00) is in a gap
    for rates in ({"fedfunds": fedfunds_for(bars, seed=5)}, None):
        costs = CostModel(financing=FinancingModel(fallback_rate=0.045))
        sim = ExecutionSimulator(bars, costs=costs, rates=rates)
        clock = SimulatedClock(bars["available_at"].iloc[0] + DELAY)
        pb = PaperBroker(ReplayFeed(bars), costs=costs, clock=clock, rates=rates)
        assert pb.place_order(_order("open", 1, 2.0, clock)).ok
        for t in range(len(bars) - 1):
            clock.advance_to(bars["available_at"].iloc[t] + DELAY)
            if t == 200:
                assert abs(pb.account().equity - sim.equity) < 1e-6
            sim.step(2.0)
        clock.advance_to(bars["available_at"].iloc[-1] + DELAY)
        swap_sim = sim.result(compute_metrics=False).costs["swap"].sum()
        assert swap_sim < 0
        assert pb.positions("XAUUSD", MAGIC)[0].swap == pytest.approx(swap_sim, rel=1e-12)
        assert pb.account().equity == pytest.approx(sim.equity, abs=1e-6)


def test_paper_broker_set_rates_refreshes_the_curve() -> None:
    bars = make_synthetic_bars(100, "H1", seed=1, weekend_gaps=False)
    clock = SimulatedClock(bars["available_at"].iloc[0] + DELAY)
    pb = PaperBroker(ReplayFeed(bars), clock=clock)
    assert pb.costs.financing.mode == "rate" and pb.rate_curve is None
    pb.set_rates({"fedfunds": fedfunds_for(bars)})
    assert pb.rate_curve is not None and len(pb.rate_curve) > 5
    fixed = PaperBroker(ReplayFeed(bars), costs=CostModel(financing="fixed"), clock=clock,
                        rates={"fedfunds": fedfunds_for(bars)})
    assert fixed.rate_curve is None


def test_rl_env_equity_matches_engine_with_rate_financing() -> None:
    pytest.importorskip("gymnasium")
    from aurum.features.pipeline import FeaturePipeline
    from aurum.portfolio.sizing import VolTargetSizer
    from aurum.risk.manager import RiskLimits, StandardRiskManager
    from aurum.rl.env import EnvConfig, GoldTradingEnv, rollout

    bars = make_synthetic_bars(1500, "H1", seed=3, model="regime")
    events = make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=7))
    md = MarketData(bars=bars, events=events, macro={"fedfunds": fedfunds_for(bars, seed=9, level=2.0)})
    pipe = FeaturePipeline(groups=["returns", "volatility"])
    raw = pipe.compute(md)
    feats = pipe.fit(raw.iloc[pipe.max_lookback:1000]).transform(raw)
    cfg = EnvConfig(action_mode="discrete", costs={"financing": {"mode": "rate", "markup_long": 0.03}})
    env = GoldTradingEnv(md.bars, feats, cfg, events=md.events, start=1000, random_start=False,
                         rates=md.macro)
    rng = np.random.default_rng(11)
    ro = rollout(env, lambda obs: int(rng.integers(0, 5)))
    risk = StandardRiskManager(RiskLimits(**cfg.risk), events=md.events)
    bt = run_backtest(md, ro.forecast, sizer=VolTargetSizer(**cfg.sizer), risk=risk,
                      costs=CostModel(**cfg.costs), start=ro.start, end=ro.end,
                      vol=pd.Series(env._vol, index=md.bars.index))
    assert bt.costs["swap"].abs().sum() > 0
    np.testing.assert_allclose(bt.equity.to_numpy(), ro.result.equity.to_numpy(), rtol=0, atol=1e-8)
    # the engine really used the series: the fallback rate gives a different path
    fb = run_backtest(md, ro.forecast, sizer=VolTargetSizer(**cfg.sizer),
                      risk=StandardRiskManager(RiskLimits(**cfg.risk), events=md.events),
                      costs=CostModel(**cfg.costs), start=ro.start, end=ro.end, rates={},
                      vol=pd.Series(env._vol, index=md.bars.index))
    assert abs(fb.equity.iloc[-1] - bt.equity.iloc[-1]) > 1.0


# ---- config ---------------------------------------------------------------------------------------
def test_config_financing_section_and_defaults() -> None:
    cfg = load_config(None, env={})
    fin = cfg.costs.build().financing
    assert fin == FinancingModel()                       # code default = rate
    assert fin.mode == "rate" and fin.rate_series == "fedfunds" and fin.markup_long == 0.025
    for name in ("default", "fast", "live_paper", "desk_overlay"):
        assert load_config(f"configs/{name}.yaml", env={}).costs.build().financing.mode == "rate"
    fixed = load_config({"costs": {"financing": {"mode": "fixed"}}}, env={})
    assert fixed.costs.build().financing.mode == "fixed"
    assert fixed.config_hash() != cfg.config_hash()
    with pytest.raises(ValueError, match="mode"):
        load_config({"costs": {"financing": {"mode": "libor"}}}, env={})
    with pytest.raises(ValueError, match="unknown key"):
        load_config({"costs": {"financing": {"markup": 0.01}}}, env={})
    # the live runner receives the same financing (CostModel(**costs) accepts the nested dict)
    m = cfg.live_runner_mapping()
    assert CostModel(**m["costs"]) == cfg.costs.build()


def test_live_runner_mapping_passes_stop_cooldown_bars() -> None:
    from aurum.live.runner import LiveConfig

    cfg = load_config({"backtest": {"stop_atr_mult": 2.0, "stop_cooldown_bars": 6}}, env={})
    m = cfg.live_runner_mapping()
    assert m["stop_cooldown_bars"] == 6 and m["stop_atr_mult"] == 2.0
    assert LiveConfig.from_mapping(m).stop_cooldown_bars == 6
    assert load_config(None, env={}).live_runner_mapping()["stop_cooldown_bars"] == 0


# ---- shared intrabar exit ---------------------------------------------------------------------------
def test_intrabar_exit_is_public_and_shared() -> None:
    import aurum.live.paper as paper

    assert paper._protective_exit is intrabar_exit                 # no mirror copy left
    assert ExecutionSimulator._intrabar_exit is intrabar_exit
    assert intrabar_exit(1.0, 100.0, 101.0, 95.0, 96.0, 104.0) == (96.0, "stop", False)
    assert intrabar_exit(1.0, 94.0, 101.0, 93.0, 96.0, None) == (94.0, "stop", False)      # gap
    assert intrabar_exit(1.0, 100.0, 105.0, 97.0, 96.0, 104.0) == (104.0, "take_profit", True)
    assert intrabar_exit(-1.0, 100.0, 105.0, 95.0, 104.0, 96.0) == (104.0, "stop", False)  # stop first
    assert intrabar_exit(-1.0, 100.0, 101.0, 99.0, 104.0, 96.0) is None
