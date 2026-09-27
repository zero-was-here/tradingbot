"""GoldTradingEnv (aurum.rl.env): gymnasium contract, point-in-time observations, reward vs
equity separation, episode sampling, termination and parity with the backtest engine."""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from aurum.backtest.engine import run_backtest  # noqa: E402
from aurum.core.instrument import XAUUSD  # noqa: E402
from aurum.core.types import MarketData  # noqa: E402
from aurum.data.schema import validate_bars  # noqa: E402
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events  # noqa: E402
from aurum.execution.costs import CostModel  # noqa: E402
from aurum.execution.simulator import ExecutionSimulator  # noqa: E402
from aurum.features.pipeline import FeaturePipeline  # noqa: E402
from aurum.portfolio.sizing import VolTargetSizer  # noqa: E402
from aurum.risk.manager import RiskLimits, StandardRiskManager  # noqa: E402
from aurum.rl.env import STATE_NAMES, EnvConfig, GoldTradingEnv, rollout  # noqa: E402

N = 2400
TRAIN_END = 1700
GROUPS = ["returns", "trend", "volatility", "session", "mtf"]


def _market(seed: int = 3, model: str = "regime") -> MarketData:
    bars = make_synthetic_bars(N, "H1", seed=seed, model=model)
    events = make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=7))
    return MarketData(bars=bars, events=events)


@pytest.fixture(scope="module")
def md() -> MarketData:
    return _market()


@pytest.fixture(scope="module")
def pipe(md: MarketData) -> FeaturePipeline:
    p = FeaturePipeline(groups=GROUPS)
    raw = p.compute(md)
    return p.fit(raw.iloc[p.max_lookback:TRAIN_END])


@pytest.fixture(scope="module")
def feats(md: MarketData, pipe: FeaturePipeline) -> pd.DataFrame:
    return pipe.transform(pipe.compute(md))


def _env(md: MarketData, feats: pd.DataFrame, **kw) -> GoldTradingEnv:
    cfg = kw.pop("config", EnvConfig(episode_length=128))
    return GoldTradingEnv(md.bars, feats, cfg, events=md.events, **kw)


def _perturbed(bars: pd.DataFrame, t: int, seed: int) -> pd.DataFrame:
    """Same bars up to ``t`` (inclusive), an unrelated path afterwards (same timestamps)."""
    alt = make_synthetic_bars(len(bars), "H1", seed=1000 + seed, model="jump",
                              start_price=float(bars["close"].iloc[t]) * 1.37, annual_vol=0.45,
                              spread=0.9)
    assert alt.index.equals(bars.index)
    new = pd.concat([bars.iloc[: t + 1], alt.iloc[t + 1:]])
    new.attrs = dict(bars.attrs)
    validate_bars(new)
    return new


# --------------------------------------------------------------------------------------------
# gymnasium contract
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["discrete", "continuous"])
def test_check_env_passes(md: MarketData, feats: pd.DataFrame, mode: str) -> None:
    env = _env(md, feats, config=EnvConfig(action_mode=mode, episode_length=64), end=TRAIN_END - 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*render.*")
        check_env(env, skip_render_check=True)
    assert env.observation_space.dtype == np.float32


def test_observation_layout(md: MarketData, feats: pd.DataFrame) -> None:
    w = 3
    env = _env(md, feats, config=EnvConfig(window=w, episode_length=64))
    obs, info = env.reset(seed=1)
    t = info["bar"]
    assert obs.shape == (w * feats.shape[1] + len(STATE_NAMES),) == env.observation_space.shape
    assert obs.dtype == np.float32
    expected = feats.iloc[t - w + 1: t + 1].to_numpy(np.float32).reshape(-1)
    np.testing.assert_array_equal(obs[: expected.size], np.clip(expected, -10, 10))
    # flat at the start: position, upnl, time in trade and drawdown are all zero
    np.testing.assert_array_equal(obs[-4:], np.zeros(4, np.float32))
    assert env.observation_space.contains(obs)


def test_warmup_rows_are_never_observed(md: MarketData, feats: pd.DataFrame) -> None:
    env = _env(md, feats, config=EnvConfig(window=5))
    first_finite = int(np.argmax(np.isfinite(feats.to_numpy()).all(axis=1)))
    assert env.first_valid_row == first_finite > 0
    assert env.first_obs_index == first_finite + 4
    assert env.range_start == env.first_obs_index
    with pytest.raises(IndexError):
        env.observation_at(env.first_obs_index - 1)


def test_misaligned_features_raise(md: MarketData, feats: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="indexed exactly like bars"):
        GoldTradingEnv(md.bars, feats.iloc[1:], EnvConfig())
    with pytest.raises(ValueError):
        EnvConfig(action_levels=(0.0, 2.0))


def test_action_mapping(md: MarketData, feats: pd.DataFrame) -> None:
    env = _env(md, feats)
    assert [env.action_to_forecast(a) for a in range(5)] == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert env.forecast_to_action(0.4) == 3
    envc = _env(md, feats, config=EnvConfig(action_mode="continuous"))
    assert envc.action_to_forecast(np.array([0.3], np.float32)) == pytest.approx(0.3)
    assert envc.action_to_forecast(np.array([7.0])) == 1.0


# --------------------------------------------------------------------------------------------
# point in time
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cut_offset", [40, 250])
def test_observation_at_t_ignores_future_bars(md: MarketData, pipe: FeaturePipeline,
                                              feats: pd.DataFrame, cut_offset: int) -> None:
    """Perturbing every bar after t must leave all observations up to t unchanged, while the
    reward of the decision taken at t (realised over bar t+1) does change."""
    env_a = _env(md, feats, config=EnvConfig(episode_length=10_000), random_start=False)
    s = env_a.range_start
    t = s + cut_offset
    bars_b = _perturbed(md.bars, t, seed=cut_offset)
    md_b = MarketData(bars=bars_b, events=md.events)
    feats_b = pipe.transform(pipe.compute(md_b))
    env_b = _env(md_b, feats_b, config=EnvConfig(episode_length=10_000), random_start=False)
    assert env_b.range_start == s

    rng = np.random.default_rng(0)
    actions = rng.integers(0, 5, size=cut_offset + 1)
    actions[-1] = 4  # long into the perturbed bar
    oa, _ = env_a.reset()
    ob, _ = env_b.reset()
    np.testing.assert_array_equal(oa, ob)
    for k, a in enumerate(actions):
        oa, ra, *_ , ia = env_a.step(a)
        ob, rb, *_ , ib = env_b.step(a)
        if k < cut_offset:  # decision bars s..t-1: settle bars <= t, identical histories
            np.testing.assert_array_equal(oa, ob)
            assert ra == rb and ia["equity"] == ib["equity"]
        else:  # decision at t is realised over bar t+1, which differs
            assert ia["decision_bar"] == t
            assert ra != rb
            assert not np.array_equal(oa, ob)


def test_reward_is_realised_over_the_next_bar(md: MarketData, feats: pd.DataFrame) -> None:
    """Entering from flat at the close of t: equity moves by lots*100*(close[t+1]-open[t+1])
    (zero costs/swap), and the reward is exactly scale*log(E[t+1]/E[t])."""
    inst = dataclasses.replace(XAUUSD, swap_long_per_lot=0.0, swap_short_per_lot=0.0)
    cfg = EnvConfig(risk=None, reward_scale=50.0, episode_length=100)
    env = GoldTradingEnv(md.bars, feats, cfg, costs=CostModel.zero(), instrument=inst,
                         random_start=False)
    _, info0 = env.reset()
    t = info0["bar"]
    e0 = info0["equity"]
    _, r, term, trunc, info = env.step(4)
    lots = info["approved_lots"]
    assert lots > 0 and info["bar"] == t + 1 and info["decision_bar"] == t
    b = md.bars
    move = lots * 100.0 * (b["close"].iloc[t + 1] - b["open"].iloc[t + 1])
    assert info["equity"] - e0 == pytest.approx(move, abs=1e-6)
    assert r == pytest.approx(50.0 * math.log(info["equity"] / e0), rel=1e-12)
    assert sum(info["costs"][k] for k in ("spread", "slippage", "commission")) == 0.0


def test_reward_shaping_never_touches_equity(md: MarketData, feats: pd.DataFrame) -> None:
    base = EnvConfig(episode_length=10_000)
    shaped = dataclasses.replace(base, dd_penalty=5.0, turnover_penalty=0.7)
    rng = np.random.default_rng(7)
    actions = rng.integers(0, 5, size=300)
    runs = {}
    for name, cfg in (("base", base), ("shaped", shaped)):
        env = _env(md, feats, config=cfg, random_start=False)
        _, info = env.reset()
        eq, rew, lots, growth = [info["equity"]], [], [], []
        for a in actions:
            _, r, term, trunc, info = env.step(a)
            eq.append(info["equity"])
            rew.append(r)
            lots.append(info["approved_lots"])
            growth.append(info["reward_components"]["growth"])
            assert not (term or trunc)
        runs[name] = (np.array(eq), np.array(rew), np.array(lots), np.array(growth), env)
    eq_b, rew_b, lots_b, g_b, env_b = runs["base"]
    eq_s, rew_s, lots_s, g_s, env_s = runs["shaped"]
    np.testing.assert_array_equal(eq_b, eq_s)          # identical money path
    np.testing.assert_array_equal(lots_b, lots_s)
    assert np.all(rew_s <= rew_b + 1e-12) and np.any(rew_s < rew_b - 1e-9)
    # growth terms telescope to the log equity growth of the account
    assert g_s.sum() / 100.0 == pytest.approx(math.log(eq_s[-1] / eq_s[0]), rel=1e-9, abs=1e-12)
    # the env's equity is exactly what a bare simulator makes of the approved lots
    sim = ExecutionSimulator(md.bars.iloc[: env_s.range_end + 1], XAUUSD, CostModel())
    sim.reset(start=env_s.episode_start)
    replay = [sim.equity] + [sim.step(x).equity for x in lots_s]
    np.testing.assert_allclose(replay, eq_s, rtol=0, atol=1e-9)
    # and the simulator's own records agree with the info dicts
    res = env_s.sim.result(compute_metrics=False)
    np.testing.assert_allclose(res.equity.to_numpy(), eq_s, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------------------------
def test_random_starts_cover_the_whole_training_range(md: MarketData, feats: pd.DataFrame) -> None:
    length = 200
    env = _env(md, feats, config=EnvConfig(episode_length=length), end=TRAIN_END - 1)
    lo, hi = env.range_start, env.max_start
    assert hi == TRAIN_END - 1 - length
    starts = []
    env.reset(seed=123)
    starts.append(env.episode_start)
    for _ in range(599):
        env.reset()
        starts.append(env.episode_start)
    s = np.array(starts)
    assert s.min() >= lo and s.max() <= hi
    assert s.min() - lo < 0.02 * (hi - lo) and hi - s.max() < 0.02 * (hi - lo)
    counts = np.histogram(s, bins=10, range=(lo, hi + 1))[0]
    assert counts.min() >= 30  # roughly uniform (expected 60 per decile)
    # seeded -> reproducible
    env2 = _env(md, feats, config=EnvConfig(episode_length=length), end=TRAIN_END - 1)
    env2.reset(seed=123)
    again = [env2.episode_start] + [env2.reset()[1]["episode_start"] for _ in range(599)]
    assert again == starts


def test_fixed_episode_length_and_range_end(md: MarketData, feats: pd.DataFrame) -> None:
    env = _env(md, feats, config=EnvConfig(episode_length=50), end=TRAIN_END - 1)
    env.reset(seed=0)
    for k in range(50):
        _, _, term, trunc, info = env.step(2)
        assert not term
        assert trunc == (k == 49)
    with pytest.raises(RuntimeError):
        env.step(2)
    # evaluation mode runs to the range end regardless of episode_length
    ev = _env(md, feats, config=EnvConfig(episode_length=5), start=TRAIN_END, random_start=False)
    ev.reset()
    n = 0
    while True:
        _, _, term, trunc, info = ev.step(2)
        n += 1
        if term or trunc:
            break
    assert trunc and info["bar"] == ev.range_end == N - 1 and n == N - 1 - TRAIN_END


def test_terminates_on_risk_kill_switch() -> None:
    bars = make_synthetic_bars(900, "H1", seed=1, model="gbm", drift=-0.002, annual_vol=0.10)
    md = MarketData(bars=bars)
    p = FeaturePipeline(groups=["returns"])
    raw = p.compute(md)
    x = p.fit(raw.iloc[60:]).transform(raw)
    cfg = EnvConfig(risk={"max_drawdown": 0.02, "max_daily_loss": None}, episode_length=800,
                    sizer={"target_vol": 0.5, "max_leverage": 5.0, "drawdown_derisk": None})
    env = GoldTradingEnv(bars, x, cfg, random_start=False)
    env.reset()
    for _ in range(800):
        _, _, term, trunc, info = env.step(4)  # stay max long into a falling market
        if term or trunc:
            break
    assert term and info["hard_halt"] and not trunc
    assert info["drawdown"] >= 0.02
    # without the kill switch the same account keeps going (ruin threshold far away)
    env2 = GoldTradingEnv(bars, x, dataclasses.replace(cfg, risk=None), random_start=False)
    env2.reset()
    _, _, term2, _, _ = env2.step(4)
    assert not term2


def test_transient_daily_loss_halt_does_not_end_episode() -> None:
    bars = make_synthetic_bars(600, "H1", seed=2, model="gbm", drift=-0.003, annual_vol=0.10)
    p = FeaturePipeline(groups=["returns"])
    raw = p.compute(MarketData(bars=bars))
    x = p.fit(raw.iloc[60:]).transform(raw)
    cfg = EnvConfig(risk={"max_daily_loss": 0.005, "max_drawdown": None,
                          "daily_loss_persistent": False},
                    sizer={"target_vol": 0.5, "max_leverage": 5.0, "drawdown_derisk": None},
                    min_equity_frac=0.0)
    env = GoldTradingEnv(bars, x, cfg, random_start=False)
    env.reset()
    halts = 0
    while True:
        _, _, term, trunc, info = env.step(4)
        halts += int(info["halted"])
        if term or trunc:
            break
    assert halts > 0 and not term and trunc


# --------------------------------------------------------------------------------------------
# parity with the backtest engine & rollouts
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["discrete", "continuous"])
def test_rollout_equity_matches_run_backtest(md: MarketData, feats: pd.DataFrame, mode: str) -> None:
    cfg = EnvConfig(action_mode=mode)
    env = _env(md, feats, config=cfg, start=TRAIN_END, random_start=False)
    rng = np.random.default_rng(11)
    if mode == "discrete":
        def policy(obs: np.ndarray) -> int:
            return int(rng.integers(0, 5))
    else:
        def policy(obs: np.ndarray) -> np.ndarray:
            return rng.uniform(-1, 1, size=1).astype(np.float32)
    ro = rollout(env, policy)
    assert ro.forecast.index.equals(md.bars.index)
    assert (ro.forecast.iloc[: ro.start] == 0).all()
    assert ro.end == env.range_end == N - 1 and not ro.terminated
    assert np.isnan(ro.approved_lots.iloc[-1]) and not np.isnan(ro.approved_lots.iloc[-2])
    risk = StandardRiskManager(RiskLimits(**cfg.risk), events=md.events)
    bt = run_backtest(md, ro.forecast, sizer=VolTargetSizer(**cfg.sizer), risk=risk,
                      start=ro.start, end=ro.end, vol=pd.Series(env._vol, index=md.bars.index))
    np.testing.assert_allclose(bt.equity.to_numpy(), ro.result.equity.to_numpy(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(bt.positions.to_numpy(), ro.result.positions.to_numpy(), atol=1e-12)
    assert bt.metrics["n_trades"] == ro.result.metrics["n_trades"] > 10


def test_rollout_requires_evaluation_env(md: MarketData, feats: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="random_start=False"):
        rollout(_env(md, feats), lambda o: 2)


def test_agent_state_tracks_position(md: MarketData, feats: pd.DataFrame) -> None:
    env = _env(md, feats, config=EnvConfig(episode_length=10_000), random_start=False)
    env.reset()
    for _ in range(5):
        obs, _, _, _, info = env.step(4)
    pos, upnl, tit, dd = obs[-4:]
    assert info["position"] > 0
    assert 0.8 <= pos <= 1.2          # holding (about) a full +1 forecast
    assert tit == pytest.approx(math.log1p(5) / math.log1p(240), rel=1e-5)
    assert dd >= 0.0
    obs, *_ = env.step(2)             # flatten
    np.testing.assert_array_equal(obs[-4:-1], np.zeros(3, np.float32))


# --------------------------------------------------------------------------------------------
# reviewer: adversarial tests (engine parity through kill switches, in-place episode re-base)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["discrete", "continuous"])
def test_rollout_matches_engine_through_kill_switch(md: MarketData, feats: pd.DataFrame,
                                                    mode: str) -> None:
    """Daily-loss halts and a max-drawdown kill fire inside the rollout: the engine replay of
    the forecasts must still reproduce the simulated account bar for bar."""
    cfg = EnvConfig(action_mode=mode,
                    risk={"max_drawdown": 0.01, "max_daily_loss": 0.004,
                          "daily_loss_persistent": False},
                    sizer={"target_vol": 0.4, "max_leverage": 4.0, "rebalance_band": 0.1})
    env = _env(md, feats, config=cfg, start=TRAIN_END, random_start=False)
    rng = np.random.default_rng(1)
    if mode == "discrete":
        def policy(obs: np.ndarray) -> int:
            return int(rng.integers(0, 5))
    else:
        def policy(obs: np.ndarray) -> np.ndarray:
            return rng.uniform(-1, 1, size=1).astype(np.float32)
    ro = rollout(env, policy, compute_metrics=False)
    assert ro.terminated and ro.end < env.range_end and ro.n_halts > 0
    risk = StandardRiskManager(RiskLimits(**cfg.risk), events=md.events)
    bt = run_backtest(md, ro.forecast, sizer=VolTargetSizer(**cfg.sizer), risk=risk,
                      start=ro.start, vol=pd.Series(env._vol, index=md.bars.index))
    common = ro.result.equity.index
    np.testing.assert_allclose(bt.equity.reindex(common).to_numpy(),
                               ro.result.equity.to_numpy(), rtol=0, atol=1e-8)
    assert bt.risk_events["halted"].any()


def test_rebase_episode_restarts_relative_state_in_place(md: MarketData, feats: pd.DataFrame) -> None:
    cfg = EnvConfig(risk={"max_drawdown": 0.005, "max_daily_loss": None},
                    sizer={"target_vol": 0.5, "max_leverage": 5.0, "drawdown_derisk": None})
    env = _env(md, feats, config=cfg, start=TRAIN_END, random_start=False)
    env.reset()
    for _ in range(N):
        _, _, term, trunc, info = env.step(0)  # max short until the kill switch fires
        if term or trunc:
            break
    assert term and info["hard_halt"] and info["position"] == 0.0
    with pytest.raises(RuntimeError):
        env.step(2)
    eq, t = env.sim.equity, env.index
    obs = env.rebase_episode()
    assert env.sim.equity == eq and env.index == t and env.sim.drawdown == 0.0
    assert env.risk is not None and not env.risk.halted
    np.testing.assert_array_equal(obs[-4:], np.zeros(4, np.float32))  # flat, fresh episode
    _, _, term2, _, _ = env.step(2)  # can trade again
    assert not term2


def test_episode_anchor_mask() -> None:
    from aurum.rl import episode_anchor_mask, episode_anchor_span_bars

    idx = pd.date_range("2021-01-28", "2021-04-03", freq="h", tz="UTC")
    idx = idx[idx.dayofweek < 5]
    m = episode_anchor_mask(idx, "M")
    assert list(idx[m]) == [pd.Timestamp("2021-02-01", tz="UTC"),
                            pd.Timestamp("2021-03-01", tz="UTC"),
                            pd.Timestamp("2021-04-01", tz="UTC")]
    w = episode_anchor_mask(idx, "W")
    assert w.sum() > 0 and (idx[w].dayofweek == 0).all() and (idx[w].hour == 0).all()
    np.testing.assert_array_equal(episode_anchor_mask(idx[:500], "M"), m[:500])  # prefix-stable
    assert not episode_anchor_mask(idx, None).any()
    assert episode_anchor_span_bars("M", 60.0) == 34 * 24 and episode_anchor_span_bars(None, 60) == 0
    with pytest.raises(ValueError):
        episode_anchor_mask(idx, "Y")


def _stateful_policy(obs: np.ndarray) -> int:
    """Deterministic, path-dependent toy policy (hysteresis on its own position/drawdown)."""
    f, pos, dd = float(obs[0]), float(obs[-4]), float(obs[-1])
    if dd > 0.15:
        return 2
    if pos > 0.3:
        return 4 if f > -0.8 else 2
    if pos < -0.3:
        return 0 if f < 0.8 else 2
    return 4 if f > 0.6 else (0 if f < -0.6 else 2)


def test_episode_starts_make_forecasts_independent_of_the_rollout_start(
        md: MarketData, feats: pd.DataFrame) -> None:
    from aurum.rl import episode_anchor_mask

    env = _env(md, feats, config=EnvConfig(), random_start=False)
    starts = episode_anchor_mask(md.bars.index, "W")
    a, b = env.range_start, env.range_start + 157
    ra = rollout(env, _stateful_policy, start=a, compute_metrics=False, episode_starts=starts)
    rb = rollout(env, _stateful_policy, start=b, compute_metrics=False, episode_starts=starts)
    first = int(np.flatnonzero(starts[b + 1:])[0]) + b + 1  # first anchor after both starts
    assert ra.info["episode_starts"] and first in rb.info["episode_starts"]
    pd.testing.assert_series_equal(ra.forecast.iloc[first:], rb.forecast.iloc[first:])
    assert (ra.forecast.iloc[first:] != 0).any()
    # the account of the LAST episode is identical too (fresh at the same anchor)
    last = ra.info["episode_starts"][-1]
    assert rb.info["episode_starts"][-1] == last
    np.testing.assert_array_equal(ra.result.equity.to_numpy(), rb.result.equity.to_numpy())
    # without anchors the forecast depends on where the account started
    ca = rollout(env, _stateful_policy, start=a, compute_metrics=False)
    cb = rollout(env, _stateful_policy, start=b, compute_metrics=False)
    assert not ca.forecast.iloc[first:].equals(cb.forecast.iloc[first:])


def test_simulated_risk_manager_does_not_log_like_the_real_account(
        md: MarketData, feats: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    """The env's risk manager is a simulation (re-rolled on every live bar by rl_ppo): its
    'RISK KILL SWITCH' / intervention records must not look like the real account's."""
    import logging

    cfg = EnvConfig(risk={"max_drawdown": 0.005, "max_daily_loss": 0.002,
                          "daily_loss_persistent": False},
                    sizer={"target_vol": 0.5, "max_leverage": 5.0, "drawdown_derisk": None})
    env = _env(md, feats, config=cfg, start=TRAIN_END, random_start=False)
    with caplog.at_level(logging.DEBUG, logger="aurum.risk.manager"):
        ro = rollout(env, lambda o: 0, compute_metrics=False)
        assert ro.terminated and ro.n_halts > 0          # kills and halts did happen
        assert not [r for r in caplog.records if r.name == "aurum.risk.manager"]
        real = StandardRiskManager(RiskLimits(max_drawdown=0.01))  # a REAL manager still logs
        t0 = md.bars.index[0]
        real.on_bar(t0, 100_000.0)
        real.on_bar(t0 + pd.Timedelta(hours=1), 98_000.0)
    kills = [r for r in caplog.records
             if r.name == "aurum.risk.manager" and r.levelno == logging.CRITICAL]
    assert real.halted and len(kills) == 1
