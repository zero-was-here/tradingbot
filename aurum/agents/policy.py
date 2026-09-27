"""DecisionPolicy: the deterministic gate between the LLM desk and the trading stack.

The Chief's :class:`~aurum.agents.records.Decision` is advice. This module turns it into a
*forecast* in [-1, 1] using fixed rules that do not depend on the model's cooperation.
That forecast is then handed by the caller to the SAME sizer (``VolTargetSizer``) and risk
manager (``StandardRiskManager``) as the quant book, so the LLM can never bypass risk
limits, kill switches, event blackouts or leverage caps (SPEC §0.5, §10): at worst it can
change the forecast within the bounds set here, and the risk manager can still only reduce
the resulting position.

Modes
-----
``overlay`` (default, SPEC §10)
    The final forecast is the projection of the requested forecast onto the segment between
    0 and the quant forecast ``q``: ``final = clip(requested, min(0, q), max(0, q))``.
    Hence ``|final| <= |q|`` and ``sign(final) in {0, sign(q)}`` — the desk can scale toward
    zero or veto but can never flip the direction or add risk. An ``override`` or ``hold``
    that would do either is clipped (and the clipping is reported).
``advisory``
    The decision is recorded only; the final forecast is ``q`` unchanged. Useful to collect
    an out-of-sample track record of the desk's calls before letting it act.
``discretionary``
    The requested forecast is used, clipped to ``[-max_abs_forecast, max_abs_forecast]``.

Action semantics (requested forecast before the mode's constraint)
------------------------------------------------------------------
follow_quant -> ``q``; scale -> ``q * scale`` (scale in [0, 1]); veto -> ``0``;
override -> ``decision.forecast``; hold -> the previous final forecast (``0`` if unknown —
without a known prior position "hold" cannot add risk).

Failures (no decision: refusal, max turns, budget, API error) and decisions whose
confidence is below ``min_confidence`` are replaced by ``on_failure``
(``follow_quant`` | ``veto`` | ``hold``) and then go through the same mode constraint.
"""

from __future__ import annotations

import logging
import math
import numbers
from dataclasses import asdict, dataclass, field
from typing import Any

from aurum.agents.records import ACTIONS, Decision

logger = logging.getLogger(__name__)

__all__ = ["MODES", "FAILURE_ACTIONS", "PolicyOutcome", "DecisionPolicy"]

MODES = ("overlay", "advisory", "discretionary")
FAILURE_ACTIONS = ("follow_quant", "veto", "hold")


@dataclass(frozen=True)
class PolicyOutcome:
    final_forecast: float
    requested_forecast: float
    quant_forecast: float
    mode: str
    action: str  # action actually applied (on_failure action when the decision was unusable)
    used_fallback: bool
    constrained: bool  # the mode constraint changed the requested forecast
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


def _finite_or_zero(x: float | None, what: str) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not numeric; using 0.0", what, x)
        return 0.0
    if not math.isfinite(v):
        logger.warning("%s=%r is not finite; using 0.0", what, x)
        return 0.0
    return v


def _is_confidence(x: object) -> bool:
    """A usable confidence: a real (non-bool, non-string) number in [0, 1]."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return False
    v = float(x)
    return math.isfinite(v) and 0.0 <= v <= 1.0


@dataclass(frozen=True)
class DecisionPolicy:
    mode: str = "overlay"
    max_abs_forecast: float = 1.0
    on_failure: str = "follow_quant"
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.on_failure not in FAILURE_ACTIONS:
            raise ValueError(f"on_failure must be one of {FAILURE_ACTIONS}, got {self.on_failure!r}")
        if not (0.0 < self.max_abs_forecast <= 1.0):
            raise ValueError("max_abs_forecast must be in (0, 1]")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")

    # ------------------------------------------------------------------------------
    def apply(self, quant_forecast: float, decision: Decision | None, *,
              previous_forecast: float | None = None) -> float:
        """Final forecast in [-1, 1] for the sizer (see module docstring)."""
        return self.evaluate(quant_forecast, decision, previous_forecast=previous_forecast).final_forecast

    def evaluate(self, quant_forecast: float, decision: Decision | None, *,
                 previous_forecast: float | None = None) -> PolicyOutcome:
        q = max(-1.0, min(1.0, _finite_or_zero(quant_forecast, "quant_forecast")))
        prev = None if previous_forecast is None else max(
            -1.0, min(1.0, _finite_or_zero(previous_forecast, "previous_forecast"))
        )
        notes: list[str] = []
        used_fallback = False
        if decision is None:
            action = self.on_failure
            used_fallback = True
            notes.append(f"no usable decision; on_failure={self.on_failure}")
        elif decision.action not in ACTIONS:  # defensive: Decision is validated upstream
            action = self.on_failure
            used_fallback = True
            notes.append(f"unknown action {decision.action!r}; on_failure={self.on_failure}")
        elif not _is_confidence(getattr(decision, "confidence", None)):
            # NaN compares False with everything: ``nan < min_confidence`` would ACCEPT it
            action = self.on_failure
            used_fallback = True
            notes.append(f"invalid confidence {getattr(decision, 'confidence', None)!r}; on_failure={self.on_failure}")
        elif decision.confidence < self.min_confidence:
            action = self.on_failure
            used_fallback = True
            notes.append(
                f"confidence {decision.confidence:.2f} < min_confidence {self.min_confidence:.2f}; "
                f"on_failure={self.on_failure}"
            )
        else:
            action = decision.action

        requested = self._requested(action, q, decision, prev, notes)

        if self.mode == "advisory":
            final = q
            constrained = requested != q
            if constrained:
                notes.append("advisory mode: decision recorded, quant forecast executed unchanged")
        elif self.mode == "overlay":
            lo, hi = min(0.0, q), max(0.0, q)
            final = min(hi, max(lo, requested))
            constrained = final != requested
            if constrained:
                notes.append(
                    f"overlay mode: requested {requested:+.4f} clipped to {final:+.4f} "
                    "(may only scale toward zero or veto)"
                )
        else:  # discretionary
            m = self.max_abs_forecast
            final = min(m, max(-m, requested))
            constrained = final != requested
            if constrained:
                notes.append(f"discretionary mode: requested {requested:+.4f} clipped to +/-{m:.4f}")
        final = 0.0 if final == 0 else float(final)  # normalise -0.0
        return PolicyOutcome(
            final_forecast=final,
            requested_forecast=float(requested),
            quant_forecast=q,
            mode=self.mode,
            action=action,
            used_fallback=used_fallback,
            constrained=constrained,
            notes=tuple(notes),
        )

    @staticmethod
    def _requested(action: str, q: float, decision: Decision | None, prev: float | None,
                   notes: list[str]) -> float:
        if action == "follow_quant":
            return q
        if action == "veto":
            return 0.0
        if action == "scale":
            s = 1.0 if decision is None else max(0.0, min(1.0, _finite_or_zero(decision.scale, "scale")))
            return q * s
        if action == "override":
            return 0.0 if decision is None else max(-1.0, min(1.0, _finite_or_zero(decision.forecast, "forecast")))
        if action == "hold":
            if prev is None:
                notes.append("hold without a known previous forecast -> flat")
                return 0.0
            return prev
        raise ValueError(f"unknown action {action!r}")  # pragma: no cover - guarded above
