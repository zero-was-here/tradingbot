"""Hand-computed tests for aurum.execution.costs."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from aurum.core.instrument import XAUUSD
from aurum.core.types import Side
from aurum.execution.costs import CostModel, FinancingModel, rollover_nights, rollover_nights_ns


def test_fill_price_half_spread_no_slippage() -> None:
    cm = CostModel(min_spread=0.0, slippage_fixed=0.0, slippage_range_frac=0.0)
    buy = cm.fill_price(Side.BUY, 2000.0, 0.30, 5.0, 1.0)
    assert buy.price == pytest.approx(2000.15)
    assert buy.spread_cost == pytest.approx(15.0)       # 0.15 $/oz * 100 oz
    assert buy.slippage_cost == 0.0
    sell = cm.fill_price(-1, 2000.0, 0.30, 5.0, 2.0)
    assert sell.price == pytest.approx(1999.85)
    assert sell.spread_cost == pytest.approx(30.0)


def test_fill_price_slippage_components_and_min_spread() -> None:
    cm = CostModel(spread_multiplier=1.0, min_spread=0.40, slippage_fixed=0.02,
                   slippage_range_frac=0.01, impact_coef=0.05)
    # effective spread floored at 0.40 -> half 0.20; slip = 0.02 + 0.01*3 + 0.05*sqrt(4) = 0.15
    fp = cm.fill_price(1, 1800.0, 0.25, 3.0, 4.0)
    assert fp.price == pytest.approx(1800.0 + 0.20 + 0.15)
    assert fp.spread_cost == pytest.approx(0.20 * 4 * 100)
    assert fp.slippage_cost == pytest.approx(0.15 * 4 * 100)
    # limit orders: no slippage
    lim = cm.fill_price(-1, 1800.0, 0.25, 3.0, 4.0, limit=True)
    assert lim.price == pytest.approx(1800.0 - 0.20)
    assert lim.slippage_cost == 0.0


def test_spread_multiplier_scales() -> None:
    cm = CostModel(spread_multiplier=2.0, min_spread=0.0, slippage_fixed=0.0, slippage_range_frac=0.0)
    assert cm.fill_price(1, 100.0, 0.3, 0.0, 1.0).spread_cost == pytest.approx(30.0)


def test_commission_defaults_to_instrument() -> None:
    inst = dataclasses.replace(XAUUSD, commission_per_lot=3.5)
    assert CostModel().commission(2.0, instrument=inst) == pytest.approx(7.0)
    assert CostModel(commission_per_lot=5.0).commission(-2.0, instrument=inst) == pytest.approx(10.0)
    assert CostModel().commission(1.0) == 0.0  # XAUUSD default: no commission


def test_swap_sign_and_rates() -> None:
    cm = CostModel(financing=FinancingModel.fixed())  # broker-quoted per-lot swaps
    assert cm.swap(1.0, 1) == pytest.approx(-45.0)
    assert cm.swap(2.0, 3) == pytest.approx(-270.0)
    assert cm.swap(-1.0, 3) == pytest.approx(45.0)
    assert cm.swap(0.0, 3) == 0.0


def test_rollover_nights_triple_wednesday_and_weekend() -> None:
    # 2024-01-10 is a Wednesday; rollover at 21:00 UTC
    assert rollover_nights("2024-01-10 20:00Z", "2024-01-10 21:00Z") == 3
    assert rollover_nights("2024-01-09 20:00Z", "2024-01-09 21:00Z") == 1  # Tuesday
    # interval is (start, end]: a position opened exactly at the rollover is not charged
    assert rollover_nights("2024-01-10 21:00Z", "2024-01-10 22:00Z") == 0
    # Friday close to Sunday reopen: Friday roll (1), no Saturday/Sunday roll
    assert rollover_nights("2024-01-12 20:00Z", "2024-01-14 22:00Z") == 1
    # a full week Mon..Mon: 1+1+3+1+1 = 7
    assert rollover_nights("2024-01-08 00:00Z", "2024-01-15 00:00Z") == 7
    assert rollover_nights("2024-01-10 22:00Z", "2024-01-10 20:00Z") == 0  # negative interval


def test_rollover_nights_matches_brute_force() -> None:
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2023-12-25", tz="UTC")
    t0s, t1s, expected = [], [], []
    for _ in range(500):
        a = base + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 40)))
        b = a + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 12)))
        cnt = 0
        d = a.normalize() - pd.Timedelta(days=1)
        while d <= b:
            r = d + pd.Timedelta(hours=21)
            if a < r <= b and r.weekday() < 5:
                cnt += 3 if r.weekday() == 2 else 1
            d += pd.Timedelta(days=1)
        t0s.append(a.as_unit("ns").value)
        t1s.append(b.as_unit("ns").value)
        expected.append(cnt)
    got = rollover_nights_ns(np.array(t0s), np.array(t1s), XAUUSD)
    np.testing.assert_array_equal(got, np.array(expected, dtype=float))


def test_rollover_custom_instrument() -> None:
    inst = dataclasses.replace(XAUUSD, rollover_hour_utc=22, triple_swap_weekday=4)
    assert rollover_nights("2024-01-12 21:00Z", "2024-01-12 22:00Z", inst) == 3  # Friday triple
    assert rollover_nights("2024-01-12 20:00Z", "2024-01-12 21:00Z", inst) == 0


def test_round_trip_cost() -> None:
    cm = CostModel(min_spread=0.0, slippage_fixed=0.02, slippage_range_frac=0.0, commission_per_lot=3.0)
    # per side: (0.15 + 0.02) * 100 + 3 = 20 ; round trip 40
    assert cm.round_trip_cost(1.0, 0.30) == pytest.approx(40.0)


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        CostModel(min_spread=-1.0)
    with pytest.raises(ValueError):
        CostModel(commission_per_lot=float("nan"))
    with pytest.raises(ValueError):
        CostModel().fill_price(0, 100.0, 0.3, 1.0, 1.0)
