"""DecisionPolicy semantics (SPEC §10): overlay / advisory / discretionary + failure paths."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aurum.agents.policy import DecisionPolicy
from aurum.agents.records import Decision


def dec(action: str, *, scale: float = 1.0, forecast: float = 0.0, confidence: float = 0.7) -> Decision:
    return Decision(action=action, scale=scale, forecast=forecast, confidence=confidence, horizon_bars=12,
                    rationale="test", key_risks=(), dissent="")


# ----------------------------------------------------------------------------- overlay
@pytest.mark.parametrize(
    "action,kwargs,q,prev,expected",
    [
        ("follow_quant", {}, 0.6, None, 0.6),
        ("follow_quant", {}, -0.4, None, -0.4),
        ("scale", {"scale": 0.5}, 0.6, None, 0.3),
        ("scale", {"scale": 0.25}, -0.8, None, -0.2),
        ("scale", {"scale": 0.0}, 0.6, None, 0.0),
        ("veto", {}, 0.6, None, 0.0),
        ("veto", {}, -0.6, None, 0.0),
        # override may only reduce toward zero
        ("override", {"forecast": 0.2}, 0.6, None, 0.2),
        ("override", {"forecast": 0.9}, 0.6, None, 0.6),     # cannot increase
        ("override", {"forecast": -0.5}, 0.6, None, 0.0),    # cannot flip
        ("override", {"forecast": 0.5}, -0.6, None, 0.0),    # cannot flip (short book)
        ("override", {"forecast": -0.3}, -0.6, None, -0.3),
        ("override", {"forecast": 0.4}, 0.0, None, 0.0),     # flat quant -> flat
        # hold keeps the previous forecast, projected into [0, q]
        ("hold", {}, 0.8, 0.5, 0.5),
        ("hold", {}, 0.3, 0.5, 0.3),
        ("hold", {}, 0.3, -0.5, 0.0),
        ("hold", {}, 0.3, None, 0.0),
    ],
)
def test_overlay_semantics(action, kwargs, q, prev, expected):
    out = DecisionPolicy(mode="overlay").evaluate(q, dec(action, **kwargs), previous_forecast=prev)
    assert out.final_forecast == pytest.approx(expected)
    assert out.mode == "overlay"


def test_overlay_never_flips_or_increases_randomised():
    rng = np.random.default_rng(0)
    pol = DecisionPolicy(mode="overlay")
    actions = ["follow_quant", "scale", "veto", "override", "hold"]
    for _ in range(2000):
        q = float(rng.uniform(-1, 1))
        d = dec(str(rng.choice(actions)), scale=float(rng.uniform(0, 1)), forecast=float(rng.uniform(-1, 1)))
        prev = float(rng.uniform(-1, 1)) if rng.random() < 0.7 else None
        f = pol.apply(q, d, previous_forecast=prev)
        assert abs(f) <= abs(q) + 1e-12
        assert f == 0.0 or math.copysign(1.0, f) == math.copysign(1.0, q)


def test_overlay_reports_constraint():
    out = DecisionPolicy().evaluate(0.5, dec("override", forecast=-1.0))
    assert out.constrained and out.final_forecast == 0.0 and out.requested_forecast == -1.0
    assert any("overlay" in n for n in out.notes)


# ----------------------------------------------------------------------------- advisory
@pytest.mark.parametrize("action", ["follow_quant", "scale", "veto", "override", "hold"])
def test_advisory_returns_quant_unchanged(action):
    pol = DecisionPolicy(mode="advisory")
    assert pol.apply(0.37, dec(action, scale=0.1, forecast=-0.9), previous_forecast=-0.2) == pytest.approx(0.37)
    assert pol.apply(-0.5, None) == pytest.approx(-0.5)


def test_advisory_ignores_on_failure_veto():
    assert DecisionPolicy(mode="advisory", on_failure="veto").apply(0.4, None) == pytest.approx(0.4)


# ----------------------------------------------------------------------------- discretionary
@pytest.mark.parametrize(
    "action,kwargs,q,max_abs,expected",
    [
        ("override", {"forecast": -0.7}, 0.6, 1.0, -0.7),   # may flip
        ("override", {"forecast": 0.9}, 0.2, 1.0, 0.9),     # may increase
        ("override", {"forecast": 0.9}, 0.2, 0.5, 0.5),     # clipped
        ("override", {"forecast": -0.9}, 0.2, 0.5, -0.5),
        ("follow_quant", {}, 0.8, 0.5, 0.5),
        ("scale", {"scale": 0.5}, -0.8, 1.0, -0.4),
        ("veto", {}, 0.8, 1.0, 0.0),
    ],
)
def test_discretionary_semantics(action, kwargs, q, max_abs, expected):
    pol = DecisionPolicy(mode="discretionary", max_abs_forecast=max_abs)
    assert pol.apply(q, dec(action, **kwargs)) == pytest.approx(expected)


def test_discretionary_hold_uses_previous():
    pol = DecisionPolicy(mode="discretionary", max_abs_forecast=0.6)
    assert pol.apply(0.1, dec("hold"), previous_forecast=-0.9) == pytest.approx(-0.6)


# ----------------------------------------------------------------------------- failures
@pytest.mark.parametrize(
    "mode,on_failure,prev,expected",
    [
        ("overlay", "follow_quant", None, 0.5),
        ("overlay", "veto", None, 0.0),
        ("overlay", "hold", 0.2, 0.2),
        ("overlay", "hold", None, 0.0),
        ("discretionary", "follow_quant", None, 0.5),
        ("discretionary", "veto", None, 0.0),
    ],
)
def test_on_failure(mode, on_failure, prev, expected):
    out = DecisionPolicy(mode=mode, on_failure=on_failure).evaluate(0.5, None, previous_forecast=prev)
    assert out.final_forecast == pytest.approx(expected)
    assert out.used_fallback and out.action == on_failure


def test_low_confidence_uses_fallback():
    pol = DecisionPolicy(min_confidence=0.6, on_failure="veto")
    out = pol.evaluate(0.5, dec("follow_quant", confidence=0.4))
    assert out.used_fallback and out.final_forecast == 0.0
    assert pol.apply(0.5, dec("follow_quant", confidence=0.8)) == pytest.approx(0.5)


def test_non_finite_quant_is_flat():
    assert DecisionPolicy().apply(float("nan"), dec("follow_quant")) == 0.0
    assert DecisionPolicy(mode="advisory").apply(float("inf"), None) == 0.0


def test_quant_clipped_to_unit_interval():
    assert DecisionPolicy(mode="advisory").apply(3.0, None) == pytest.approx(1.0)


def test_negative_zero_normalised():
    f = DecisionPolicy().apply(-0.5, dec("veto"))
    assert f == 0.0 and math.copysign(1.0, f) == 1.0


@pytest.mark.parametrize("kwargs", [{"mode": "yolo"}, {"on_failure": "buy"}, {"max_abs_forecast": 0.0},
                                    {"max_abs_forecast": 1.5}, {"min_confidence": 2.0}])
def test_invalid_policy_config(kwargs):
    with pytest.raises(ValueError):
        DecisionPolicy(**kwargs)
