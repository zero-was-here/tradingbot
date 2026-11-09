"""``regime_switch`` (aurum/strategies/regime.py): HMM fit on training returns, causal
filtered regime probabilities, scaling of the base trend forecast, abstention, parameters."""

from __future__ import annotations

import json
import logging
import pickle

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.research.walkforward import label_horizon
from aurum.strategies.base import get_strategy, list_strategies
from aurum.strategies.regime import RegimeSwitch, regime_scale

BASE = {"horizons": (24, 96, 240), "vol_halflife": 48, "vol_min_periods": 24}


@pytest.fixture(scope="module")
def h1() -> MarketData:
    return MarketData(bars=make_synthetic_bars(9000, "H1", seed=21, model="regime"))


@pytest.fixture(scope="module")
def fitted(h1: MarketData) -> RegimeSwitch:
    s = RegimeSwitch(base_params=BASE)
    return s.fit(MarketData(bars=h1.bars.iloc[:5000]))


def test_registered_trainable_with_defaults() -> None:
    assert "regime_switch" in list_strategies()
    s = get_strategy("regime_switch")
    assert s.trainable and not s.is_fitted and s.description
    assert s.params["base"] == "tsmom" and s.params["n_states"] == 2
    assert s.warmup_bars == get_strategy("tsmom").warmup_bars
    assert RegimeSwitch(warmup=7).warmup_bars == 7


@pytest.mark.parametrize("bad, exc", [
    ({"base": "regime_switch"}, ValueError), ({"base": "no_such_strategy"}, KeyError),
    ({"returns": "weekly"}, ValueError), ({"p_flat": 0.9, "p_full": 0.5}, ValueError),
    ({"p_full": 1.5}, ValueError), ({"n_states": 1}, ValueError), ({"min_train_obs": 3}, ValueError),
    ({"daily_anchor_hour_utc": 24}, ValueError), ({"warmup": -1}, ValueError), ({"lenght": 2}, ValueError),
    ({"base_params": {"lenght": 3}}, ValueError),
])
def test_rejects_invalid_parameters(bad: dict, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        RegimeSwitch(**bad)


def test_generate_before_fit_raises(h1: MarketData) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        RegimeSwitch().generate(h1)


def test_regime_scale_piecewise_linear() -> None:
    p = np.array([0.0, 0.5, 0.6, 0.7, 0.9, 1.0, np.nan])
    out = regime_scale(p, 0.5, 0.9)
    np.testing.assert_allclose(out[:6], [0.0, 0.0, 0.25, 0.5, 1.0, 1.0])
    assert np.isnan(out[6])


def test_fit_finds_a_calm_and_a_chaotic_state(fitted: RegimeSwitch) -> None:
    summ = fitted.summary()
    json.dumps(summ)                                   # JSON-friendly for reports / agents
    v = np.asarray(summ["hmm"]["vars"]).ravel()
    assert v[1] > 2.0 * v[0], "state 0 must be the calm (low-variance) regime"
    assert "daily returns" in summ["fit_note"]
    assert min(summ["hmm"]["expected_durations"]) > 2.0   # regimes persist for days


def test_forecast_is_base_times_scale_and_flat_when_chaotic(fitted: RegimeSwitch, h1: MarketData) -> None:
    f = fitted.generate(h1)
    base = get_strategy("tsmom", **BASE).generate(h1)
    probs = fitted.regime_probabilities(h1)
    w = fitted.warmup_bars
    expected = (base * probs["scale"].fillna(0.0)).clip(-1, 1)
    np.testing.assert_allclose(f.iloc[w:], expected.iloc[w:], atol=1e-12)
    assert (f.iloc[:w] == 0).all() and np.isfinite(f).all() and f.abs().max() <= 1.0
    chaotic = probs["p_ok"] <= fitted.params["p_flat"]
    assert chaotic.iloc[w:].any() and (f.iloc[w:][chaotic.iloc[w:]] == 0.0).all()
    calm = probs["p_calm"] >= fitted.params["p_full"]
    np.testing.assert_allclose(f.iloc[w:][calm.iloc[w:]], base.iloc[w:][calm.iloc[w:]])
    # the filter does separate the regimes on this synthetic market
    assert 0.1 < (probs["p_calm"].dropna() > 0.5).mean() < 0.9


def test_probability_columns_are_consistent(fitted: RegimeSwitch, h1: MarketData) -> None:
    obs = fitted.filtered_probabilities(h1)
    assert {"p_state_0", "p_state_1", "p_calm", "p_chaotic", "p_ok", "scale", "available_at"} <= set(obs.columns)
    body = obs.dropna(subset=["p_calm"])
    np.testing.assert_allclose(body["p_state_0"] + body["p_state_1"], 1.0, atol=1e-9)
    np.testing.assert_allclose(body["p_ok"], 1.0 - body["p_chaotic"])
    # daily mode on H1: one observation per completed day, available at the day's end
    assert fitted.return_mode(h1.bars) == "daily"
    assert len(obs) < len(h1.bars) / 15
    assert (pd.DatetimeIndex(obs["available_at"]) > obs.index).all()


def test_intraday_bars_only_see_completed_days(fitted: RegimeSwitch, h1: MarketData) -> None:
    probs = fitted.regime_probabilities(h1)["p_calm"].dropna()
    assert len(probs) > 0.9 * len(h1.bars)
    changes = probs.ne(probs.shift()).to_numpy()[1:]
    n_days = fitted.filtered_probabilities(h1).shape[0]
    assert changes.sum() <= n_days + 1, "the probability may only change when a day completes"


@pytest.mark.parametrize("t", [1500, 4321, 6007, 8999])
def test_point_in_time_under_truncation(fitted: RegimeSwitch, h1: MarketData, t: int) -> None:
    full_f = fitted.generate(h1).to_numpy()
    full_p = fitted.regime_probabilities(h1).to_numpy()
    md_t = MarketData(bars=h1.bars.iloc[: t + 1])
    np.testing.assert_array_equal(full_f[: t + 1], fitted.generate(md_t).to_numpy())
    np.testing.assert_array_equal(full_p[: t + 1], fitted.regime_probabilities(md_t).to_numpy())


def test_daily_bars_use_bar_returns() -> None:
    bars = make_synthetic_bars(2500, "D1", seed=5, model="regime")
    md = MarketData(bars=bars)
    s = RegimeSwitch(base_params={"horizons": (21, 63), "vol_halflife": 30, "vol_min_periods": 20})
    assert s.return_mode(bars) == "bar"
    s.fit(MarketData(bars=bars.iloc[:1200]))
    obs = s.filtered_probabilities(md)
    assert len(obs) == len(bars)
    f = s.generate(md)
    assert f.abs().max() <= 1.0 and (f != 0).any()
    explicit = RegimeSwitch(returns="daily").return_mode(bars)
    assert explicit == "daily"


def test_abstains_without_enough_training_returns(h1: MarketData, caplog: pytest.LogCaptureFixture) -> None:
    s = RegimeSwitch(base_params=BASE)
    with caplog.at_level(logging.WARNING, logger="aurum.strategies.regime"):
        s.fit(MarketData(bars=h1.bars.iloc[:400]))           # ~17 daily returns < 60
    assert s.is_fitted and s.hmm_ is None and "abstaining" in s.fit_note_
    assert any("abstaining" in r.getMessage() for r in caplog.records)
    f = s.generate(h1)
    assert (f == 0).all()
    assert s.regime_probabilities(h1)["p_calm"].isna().all()


def test_trainable_base_is_fitted_and_purge_horizon_propagates(h1: MarketData) -> None:
    s = RegimeSwitch(base="intraday_seasonality",
                     base_params={"significance": None, "cost_multiplier": 0.0, "min_obs": 5})
    s.fit(MarketData(bars=h1.bars.iloc[:4000]))
    assert s.base_.is_fitted
    assert s.generate(h1).abs().max() <= 1.0
    assert label_horizon(RegimeSwitch()) == 0                      # tsmom learns nothing
    meta = get_strategy("meta_label")
    assert label_horizon(RegimeSwitch(base="meta_label")) == label_horizon(meta) > 0


def test_clone_and_pickle_keep_the_fitted_model(fitted: RegimeSwitch, h1: MarketData) -> None:
    f = fitted.generate(h1).to_numpy()
    for other in (fitted.clone(), pickle.loads(pickle.dumps(fitted))):
        assert other.is_fitted
        np.testing.assert_array_equal(f, other.generate(h1).to_numpy())


def test_fit_is_deterministic(h1: MarketData) -> None:
    a = RegimeSwitch(base_params=BASE).fit(MarketData(bars=h1.bars.iloc[:5000])).generate(h1)
    b = RegimeSwitch(base_params=BASE).fit(MarketData(bars=h1.bars.iloc[:5000])).generate(h1)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
