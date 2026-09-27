"""Engine forecast hook (LLM-desk replay): bit-identical without a hook, cannot bypass risk.

Inputs are built from a local numpy RNG with ``aurum.data.schema.make_bars`` (not the
synthetic-data module) and the sizer / risk manager are local classes, so the pinned
digests below depend only on the backtest engine, the execution simulator and the cost
model. They were produced by the engine BEFORE the hook was added (wave 1); if the
simulator or cost model is changed on purpose, regenerate them with the previous engine.
The scenarios pin ``financing="fixed"`` (the per-lot swaps of wave 1, reproduced
bit-identically) since rate-based financing became the ``CostModel`` default.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.engine import run_backtest
from aurum.core.instrument import Instrument
from aurum.core.interfaces import RiskContext, RiskDecision
from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.store import frame_hash
from aurum.execution.costs import CostModel
from aurum.portfolio.sizing import VolTargetSizer
from aurum.risk.manager import RiskLimits, StandardRiskManager


def make_md(n: int = 2500, seed: int = 3) -> tuple[MarketData, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n * 7 // 5 + 100, freq="h", tz="UTC")
    idx = idx[idx.dayofweek < 5][: n]
    r = 0.002 * rng.standard_normal(len(idx)) + 0.0002 * np.sin(np.arange(len(idx)) / 50.0)
    close = 1800.0 * np.exp(np.cumsum(r))
    open_ = np.r_[1800.0, close[:-1]] * np.exp(0.0002 * rng.standard_normal(len(idx)))
    high = np.maximum(open_, close) * (1 + 0.001 * rng.random(len(idx)))
    low = np.minimum(open_, close) * (1 - 0.001 * rng.random(len(idx)))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": 100.0, "spread": 0.25 + 0.3 * rng.random(len(idx))}, index=idx)
    bars = make_bars(df, "H1")
    events = pd.DataFrame({"time": idx[::97] + pd.Timedelta(minutes=30), "name": "NFP", "currency": "USD",
                           "importance": 3})
    c = bars["close"]
    fc = ((c.ewm(span=10, adjust=False).mean() - c.ewm(span=40, adjust=False).mean()) / c * 500).clip(-1, 1)
    return MarketData(bars=bars, events=events), fc.rename("fc")


class LinearSizer:
    """lots = forecast * k (rounded) - deterministic."""

    def __init__(self, k: float = 3.0) -> None:
        self.k = k

    def target_lots(self, forecast: float, vol_ann: float, equity: float, price: float,
                    instrument: Instrument, *, current_lots: float = 0.0, drawdown: float = 0.0) -> float:
        return instrument.round_lots(forecast * self.k * (1.0 - drawdown))


class CapHaltRisk:
    """Caps |lots| and halts permanently from ``halt_from`` (decision time) on."""

    def __init__(self, cap: float = 2.0, halt_from: pd.Timestamp | None = None) -> None:
        self.cap = cap
        self.halt_from = halt_from
        self.halted = False

    def on_bar(self, time: pd.Timestamp, equity: float) -> None:
        if self.halt_from is not None and time >= self.halt_from:
            self.halted = True

    def evaluate(self, ctx: RiskContext) -> RiskDecision:
        if self.halted:
            return RiskDecision(0.0, halted=True, reasons=["test halt"])
        t = ctx.target_lots
        if abs(t) > self.cap:
            return RiskDecision(float(np.sign(t) * self.cap), reasons=["cap"])
        return RiskDecision(t)


def digest(res) -> str:
    frame = pd.DataFrame({"equity": res.equity, "positions": res.positions, "target": res.target,
                          "forecast": res.forecast})
    return f"{frame_hash(frame)}|{frame_hash(res.costs)}|{len(res.risk_events)}|{len(res.trades)}"


def scenarios():
    """(name, kwargs) of engine runs whose digests are pinned in the golden test."""
    return [
        ("cap_risk_stops", dict(risk_factory=lambda md: CapHaltRisk(2.0), costs=CostModel(financing="fixed"),
                                stop_atr_mult=2.0, take_profit_atr_mult=3.0, stop_cooldown_bars=4)),
        ("window_no_risk", dict(risk_factory=lambda md: None,
                                costs=CostModel(slippage_fixed=0.05, financing="fixed"),
                                start=300, end=2000)),
        ("halt_midway", dict(risk_factory=lambda md: CapHaltRisk(5.0, md.bars["available_at"].iloc[1200]),
                             costs=CostModel(financing="fixed"))),
    ]


def run_scenario(run_backtest, md: MarketData, fc: pd.Series, spec: dict, **extra):
    spec = dict(spec)
    risk = spec.pop("risk_factory")(md)
    return run_backtest(md, fc, sizer=LinearSizer(), risk=risk, **spec, **extra)


#: digests of the pre-hook (wave 1) engine for ``scenarios()`` (see module docstring)
GOLDEN = {
    "cap_risk_stops": "9018a7abbb00bd1837e8a8a4208bcb2f1520c044de9518fe175f9c3ffe0b832d|693608b213dcb768184747b4c517c52770a531a650127ca048b53ed0852cb7db|1878|212",
    "window_no_risk": "4fb5d18dbfb6fe3a82462f439224837a6d08239c2ab9be34690e300c89d4e558|18338e9b14d9555fc216c6b754d1d7f38971e981f85d6143c7b3a4c0a81a69db|0|56",
    "halt_midway": "051a0ac9361b618d1c80d718df0c28a0de461e31dd27c6181d7668953cbcccaf|20b2a07988c897c0ff407a84b7e2e69156a95c84575897a4a7572b17455ed7cc|1299|42",
}


@pytest.fixture(scope="module")
def data() -> tuple[MarketData, pd.Series]:
    return make_md()


@pytest.mark.parametrize("name", [s[0] for s in scenarios()])
def test_no_hook_is_bit_identical_to_pre_hook_engine(data, name):
    md, fc = data
    spec = dict(scenarios())[name]
    res = run_scenario(run_backtest, md, fc, spec)
    assert digest(res) == GOLDEN[name]
    assert "hook" not in res.meta


@pytest.mark.parametrize("name", [s[0] for s in scenarios()])
def test_identity_hook_matches_no_hook_exactly(data, name):
    md, fc = data
    spec = dict(scenarios())[name]
    base = run_scenario(run_backtest, md, fc, spec)
    calls = []

    def identity(bar, now, f, state):
        calls.append(bar)
        return f

    hooked = run_scenario(run_backtest, md, fc, spec, forecast_hook=identity)
    assert digest(hooked) == digest(base)
    np.testing.assert_array_equal(hooked.equity.to_numpy(), base.equity.to_numpy())
    assert hooked.meta["hook"]["calls"] == len(calls) == len(base.equity) - 1
    assert hooked.meta["hook"]["overrides"] == 0


def test_hook_receives_point_in_time_state(data):
    md, fc = data
    seen = []

    def hook(bar, now, f, state):
        seen.append((bar, now, f, dict(state)))
        return f

    res = run_backtest(md, fc, sizer=LinearSizer(), risk=CapHaltRisk(2.0), start=100, end=400,
                       forecast_hook=hook)
    avail = md.bars["available_at"]
    for k, (bar, now, f, st) in enumerate(seen):
        assert bar == 100 + k                               # absolute position in the bars frame
        assert now == avail.iloc[bar]                        # decision time = bar close
        assert f == pytest.approx(float(np.clip(fc.iloc[bar], -1, 1)))
        assert st["bar_time"] == md.bars.index[bar]
        assert st["price"] == md.bars["close"].iloc[bar]
        assert st["window_bar"] == k
        assert set(st) >= {"equity", "position", "drawdown", "peak_equity", "halted", "spread", "vol"}
    # the state is the book BEFORE the decision: equity marked at the close of that bar
    for (bar, _, _, st) in seen[1:]:
        assert st["equity"] == pytest.approx(res.equity.loc[md.bars.index[bar]])
        assert st["position"] == pytest.approx(res.position_close.loc[md.bars.index[bar]])
        assert 0.0 <= st["drawdown"] < 1.0
        assert st["peak_equity"] >= st["equity"] - 1e-9


def test_hook_cannot_bypass_risk_halt(data):
    md, fc = data
    halt_t = md.bars["available_at"].iloc[800]
    halted_flags = []

    def max_long(bar, now, f, state):
        halted_flags.append((bar, state["halted"]))
        return 1.0  # the desk always asks for maximum long

    res = run_backtest(md, fc, sizer=LinearSizer(k=10.0), risk=CapHaltRisk(2.0, halt_from=halt_t),
                       forecast_hook=max_long)
    pos = res.positions
    after = pos.index > md.bars.index[800]
    assert (pos[after] == 0.0).all()                      # halted -> flat, whatever the hook says
    assert pos[~after].abs().max() <= 2.0 + 1e-9          # the cap binds before the halt
    assert (pos[~after].iloc[1:] > 0).all()
    assert all(h for b, h in halted_flags if b >= 800)    # the hook is told the book is halted
    assert not any(h for b, h in halted_flags if b < 800)
    assert (res.forecast.iloc[:-1] == 1.0).all()          # reported = used (last bar: no decision)


def test_hook_cannot_bypass_standard_risk_manager_kill_switch():
    md, fc = make_md(1500, seed=5)
    # A tight daily-loss kill (persistent) on a leveraged always-long book: once it fires,
    # the approved position must be 0 for the rest of the run even though the hook insists.
    rm = StandardRiskManager(RiskLimits(max_daily_loss=0.005, daily_loss_persistent=True, max_leverage=None,
                                        max_margin_utilisation=None))
    res = run_backtest(md, fc, sizer=VolTargetSizer(target_vol=0.6, max_leverage=5.0), risk=rm,
                       forecast_hook=lambda b, t, f, s: 1.0)
    ev = res.risk_events
    assert rm.halted and ev["halted"].any()
    first = int(ev.loc[ev["halted"], "bar"].iloc[0])
    assert (res.positions.iloc[first + 1:] == 0.0).all()


def test_hook_every_holds_last_value(data):
    md, fc = data
    calls = []

    def hook(bar, now, f, state):
        calls.append(bar)
        return 0.5 if len(calls) % 2 else -0.5

    res = run_backtest(md, fc, sizer=LinearSizer(k=2.0), start=10, end=110, forecast_hook=hook, hook_every=7)
    n_decisions = 100  # window of 101 bars -> 100 decisions
    assert calls == list(range(10, 110, 7))
    assert res.meta["hook"]["calls"] == math.ceil(n_decisions / 7)
    used = res.forecast.to_numpy()[:-1]
    expected = np.array([0.5 if (i // 7) % 2 == 0 else -0.5 for i in range(n_decisions)])
    np.testing.assert_array_equal(used, expected)


def test_hook_output_clipped_and_invalid_goes_flat(data):
    md, fc = data
    outs = iter([3.0, float("nan"), "junk", -7.0] + [0.0] * 1000)
    res = run_backtest(md, fc, sizer=LinearSizer(k=2.0), start=0, end=20, forecast_hook=lambda *a: next(outs))
    f = res.forecast.to_numpy()
    assert f[0] == 1.0 and f[1] == 0.0 and f[2] == 0.0 and f[3] == -1.0
    assert res.meta["hook"]["invalid"] == 2


def test_hook_arguments_validated(data):
    md, fc = data
    with pytest.raises(ValueError, match="hook_every"):
        run_backtest(md, fc, sizer=LinearSizer(), forecast_hook=lambda *a: 0.0, hook_every=0)
    with pytest.raises(TypeError, match="callable"):
        run_backtest(md, fc, sizer=LinearSizer(), forecast_hook=1.0)  # type: ignore[arg-type]


def test_hook_exceptions_propagate(data):
    md, fc = data

    def boom(*a):
        raise RuntimeError("desk exploded")

    with pytest.raises(RuntimeError, match="desk exploded"):
        run_backtest(md, fc, sizer=LinearSizer(), forecast_hook=boom)
