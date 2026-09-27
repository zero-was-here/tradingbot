"""Tests for aurum.portfolio.sizing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.interfaces import PositionSizer
from aurum.portfolio.sizing import FixedFractionalSizer, VolTargetSizer, drawdown_multiplier

EQ = 100_000.0
PX = 2_000.0


def _sizer(**kw) -> VolTargetSizer:
    base = dict(target_vol=0.10, max_leverage=2.0, rebalance_band=0.0, drawdown_derisk=())
    base.update(kw)
    return VolTargetSizer(**base)


def test_implements_protocol():
    assert isinstance(VolTargetSizer(), PositionSizer)
    assert isinstance(FixedFractionalSizer(), PositionSizer)


def test_vol_targeting_math():
    s = _sizer()
    # notional = 1 * 0.10 / 0.20 * 100k = 50k -> 50k / (100 oz * 2000) = 0.25 lots
    assert s.target_lots(1.0, 0.20, EQ, PX, XAUUSD) == pytest.approx(0.25)
    assert s.target_lots(-0.5, 0.20, EQ, PX, XAUUSD) == pytest.approx(-0.12)  # -0.125 -> toward 0
    assert s.target_lots(1.0, 0.10, EQ, PX, XAUUSD) == pytest.approx(0.50)
    # linear in equity and inverse in vol
    assert s.target_lots(1.0, 0.20, 2 * EQ, PX, XAUUSD) == pytest.approx(0.50)
    # forecasts beyond +/-1 are clipped
    assert s.target_lots(3.0, 0.20, EQ, PX, XAUUSD) == pytest.approx(0.25)


def test_leverage_cap():
    s = _sizer(max_leverage=2.0)
    # vol 1% (floored at min_vol 2%) -> uncapped notional 500k = 5x; capped at 2x = 200k -> 1.0 lot
    assert s.target_lots(1.0, 0.01, EQ, PX, XAUUSD) == pytest.approx(1.0)
    # vol 6% -> 166.7k notional (1.67x) is below the cap -> 0.83 lots
    assert s.target_lots(1.0, 0.06, EQ, PX, XAUUSD) == pytest.approx(0.83)
    s2 = _sizer(max_leverage=2.0, min_vol=0.001)
    assert s2.target_lots(1.0, 0.01, EQ, PX, XAUUSD) == pytest.approx(1.0)
    assert s2.target_lots(-1.0, 0.01, EQ, PX, XAUUSD) == pytest.approx(-1.0)
    bd = s2.breakdown(1.0, 0.01, EQ, PX, XAUUSD)
    assert any("max_leverage" in c for c in bd.caps_hit)
    # max_lots cap
    s3 = _sizer(max_lots=0.1)
    assert s3.target_lots(1.0, 0.2, EQ, PX, XAUUSD) == pytest.approx(0.1)


def test_kelly_cap():
    s = _sizer(target_vol=0.40, max_leverage=10.0, kelly_cap=0.10)
    # target 40% vol but Kelly cap limits position vol to 10% of equity -> same as 10% target
    assert s.target_lots(1.0, 0.20, EQ, PX, XAUUSD) == pytest.approx(0.25)
    bd = s.breakdown(1.0, 0.20, EQ, PX, XAUUSD)
    assert any("kelly" in c for c in bd.caps_hit)


def test_drawdown_derisk_steps():
    s = _sizer(drawdown_derisk=((0.10, 0.5), (0.15, 0.25)))
    assert s.target_lots(1.0, 0.10, EQ, PX, XAUUSD, drawdown=0.05) == pytest.approx(0.50)
    assert s.target_lots(1.0, 0.10, EQ, PX, XAUUSD, drawdown=0.10) == pytest.approx(0.25)
    assert s.target_lots(1.0, 0.10, EQ, PX, XAUUSD, drawdown=0.12) == pytest.approx(0.25)
    assert s.target_lots(1.0, 0.10, EQ, PX, XAUUSD, drawdown=0.16) == pytest.approx(0.12)  # 0.125 -> 0.12
    # negative sign convention is accepted (magnitude)
    assert s.target_lots(1.0, 0.10, EQ, PX, XAUUSD, drawdown=-0.16) == pytest.approx(0.12)
    assert drawdown_multiplier(0.3, ((0.15, 0.25), (0.10, 0.5))) == 0.25
    assert drawdown_multiplier(0.0, ((0.10, 0.5),)) == 1.0
    with pytest.raises(ValueError):
        VolTargetSizer(drawdown_derisk=((1.5, 0.5),))


def test_rebalance_band_hysteresis():
    s = _sizer(rebalance_band=0.10)
    # fresh target 0.25; holding 0.24 -> |0.01| < 0.1*0.25 -> keep 0.24
    assert s.target_lots(1.0, 0.20, EQ, PX, XAUUSD, current_lots=0.24) == pytest.approx(0.24)
    # holding 0.20 -> |0.05| >= 0.025 -> trade to 0.25
    assert s.target_lots(1.0, 0.20, EQ, PX, XAUUSD, current_lots=0.20) == pytest.approx(0.25)
    # hysteresis path: forecast drifts slowly; position only moves when the gap exceeds the band
    path, cur = [], 0.0
    for f in np.linspace(0.8, 1.0, 21):
        cur = s.target_lots(float(f), 0.20, EQ, PX, XAUUSD, current_lots=cur)
        path.append(cur)
    assert len(set(path)) < 6          # far fewer changes than 21 steps
    assert path[-1] == pytest.approx(0.25, abs=0.03)
    # reversals and exits always trade
    assert s.target_lots(-1.0, 0.20, EQ, PX, XAUUSD, current_lots=0.25) == pytest.approx(-0.25)
    assert s.target_lots(0.0, 0.20, EQ, PX, XAUUSD, current_lots=0.25) == 0.0
    # a current position above a hard cap is not kept by the band
    s_cap = _sizer(rebalance_band=0.5, max_lots=0.2)
    assert s_cap.target_lots(1.0, 0.20, EQ, PX, XAUUSD, current_lots=0.25) == pytest.approx(0.2)


def test_rounding_toward_zero():
    s = _sizer()
    # raw lots = 0.2599... must round DOWN to 0.25 and -0.2599 UP to -0.25 (toward zero)
    vol = 0.10 * EQ / (0.2599 * 100 * PX)
    assert s.target_lots(1.0, vol, EQ, PX, XAUUSD) == pytest.approx(0.25)
    assert s.target_lots(-1.0, vol, EQ, PX, XAUUSD) == pytest.approx(-0.25)
    # below min_lot -> 0, never rounded up into risk
    assert s.target_lots(0.01, 0.20, EQ, PX, XAUUSD) == 0.0
    coarse = Instrument(lot_step=0.1, min_lot=0.1)
    assert s.target_lots(1.0, 0.20, EQ, PX, coarse) == pytest.approx(0.2)


def test_zero_or_invalid_inputs_give_zero():
    s = VolTargetSizer()
    assert s.target_lots(0.0, 0.2, EQ, PX, XAUUSD) == 0.0
    assert s.target_lots(0.0, 0.2, EQ, PX, XAUUSD, current_lots=0.5) == 0.0
    assert s.target_lots(math.nan, 0.2, EQ, PX, XAUUSD) == 0.0
    assert s.target_lots(1.0, math.nan, EQ, PX, XAUUSD) == 0.0
    assert s.target_lots(1.0, 0.0, EQ, PX, XAUUSD) == 0.0
    assert s.target_lots(1.0, 0.2, 0.0, PX, XAUUSD) == 0.0
    assert s.target_lots(1.0, 0.2, EQ, -1.0, XAUUSD) == 0.0


def test_breakdown_consistent_with_target():
    s = VolTargetSizer()
    bd = s.breakdown(0.7, 0.18, EQ, PX, XAUUSD, current_lots=0.1, drawdown=0.11)
    assert bd.final_lots == s.target_lots(0.7, 0.18, EQ, PX, XAUUSD, current_lots=0.1, drawdown=0.11)
    assert bd.drawdown_mult == 0.5
    assert isinstance(bd.to_dict(), dict)


def test_fixed_fractional_sizer():
    s = FixedFractionalSizer(risk_per_trade=0.005, stop_atr=2.0, rebalance_band=0.0, drawdown_derisk=())
    # explicit ATR 10$: stop 20$ -> risk 500$ -> 500 / (20 * 100) = 0.25 lots
    assert s.target_lots(1.0, 0.2, EQ, PX, XAUUSD, atr=10.0) == pytest.approx(0.25)
    assert s.target_lots(-0.5, 0.2, EQ, PX, XAUUSD, atr=10.0) == pytest.approx(-0.12)
    assert s.stop_distance(0.2, PX, atr=10.0) == pytest.approx(20.0)
    # ATR proxy from vol: daily sigma = 2000 * 0.2 / sqrt(252)
    d = s.stop_distance(0.2, PX)
    assert d == pytest.approx(2.0 * PX * 0.2 / math.sqrt(252))
    lots = s.target_lots(1.0, 0.2, EQ, PX, XAUUSD)
    assert lots == XAUUSD.round_lots(500.0 / (d * 100))
    # leverage cap
    assert s.target_lots(1.0, 0.2, EQ, PX, XAUUSD, atr=0.01) == pytest.approx(1.0)
    assert s.target_lots(0.0, 0.2, EQ, PX, XAUUSD, atr=10.0) == 0.0


# ---------------------------------------------------------------------------------------
# Review regressions (adversarial)
# ---------------------------------------------------------------------------------------
def test_randomised_sizing_invariants():
    """Sign follows the forecast, leverage/Kelly/max_lots caps hold (band included), and the
    result sits on the lot grid — over random parameters, positions and drawdowns."""
    rng = np.random.default_rng(0)
    for _ in range(3000):
        s = VolTargetSizer(
            target_vol=rng.uniform(0.02, 0.5), max_leverage=rng.uniform(0.5, 5.0),
            rebalance_band=rng.uniform(0.0, 0.5),
            max_lots=None if rng.random() < 0.5 else rng.uniform(0.05, 5.0),
            kelly_cap=None if rng.random() < 0.5 else rng.uniform(0.05, 0.5),
        )
        f = float(np.clip(rng.normal(0, 0.7), -1, 1))
        vol, eq, px = rng.uniform(0.001, 1.0), rng.uniform(1e3, 1e7), rng.uniform(500, 4000)
        cur = XAUUSD.round_lots(float(rng.normal(0, 3)))
        lots = s.target_lots(f, vol, eq, px, XAUUSD, current_lots=cur, drawdown=rng.uniform(0, 0.3))
        assert lots == 0.0 or math.copysign(1.0, lots) == math.copysign(1.0, f)
        cap = s.max_leverage if s.kelly_cap is None else min(s.max_leverage, s.kelly_cap / max(vol, s.min_vol))
        assert abs(lots) * XAUUSD.contract_size * px / eq <= cap + 1e-9
        assert s.max_lots is None or abs(lots) <= s.max_lots + 1e-12
        assert abs(lots) <= XAUUSD.max_lot
        assert abs(round(lots / XAUUSD.lot_step) * XAUUSD.lot_step - lots) < 1e-9
