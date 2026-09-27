"""Validated records produced by the agents: the Chief's :class:`Decision` and specialist
:class:`Memo` s.

Tool schemas are sent with ``strict: true``, which guarantees *types* and required fields,
but strict JSON schemas cannot express numeric ranges or string lengths (``minimum``,
``maxLength`` ... are unsupported). Ranges are therefore enforced here, client side; a
violation is returned to the model as an ``is_error`` tool result so it can correct itself
instead of the harness silently clipping a value the model did not intend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "ACTIONS",
    "STANCES",
    "RecordValidationError",
    "Decision",
    "Memo",
]

#: Chief actions (see ``prompts.CHIEF_SYSTEM_PROMPT`` and ``policy.DecisionPolicy``).
ACTIONS = ("follow_quant", "scale", "veto", "override", "hold")
STANCES = ("bullish", "bearish", "neutral")

MAX_LIST_ITEMS = 12
MAX_ITEM_CHARS = 600
MAX_HORIZON_BARS = 10_000
MAX_ERROR_CHARS = 1_000


class RecordValidationError(ValueError):
    """Tool input that is schema-valid but semantically invalid; lists every problem."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _num(data: Mapping[str, Any], key: str, lo: float, hi: float, problems: list[str]) -> float:
    v = data.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        problems.append(f"'{key}' must be a number in [{lo}, {hi}], got {v!r}")
        return float("nan")
    x = float(v)
    if not math.isfinite(x) or x < lo or x > hi:
        problems.append(f"'{key}' must be a finite number in [{lo}, {hi}], got {v!r}")
    return x


def _text(data: Mapping[str, Any], key: str, problems: list[str], *, max_chars: int,
          allow_empty: bool = False) -> str:
    v = data.get(key)
    if not isinstance(v, str):
        problems.append(f"'{key}' must be a string")
        return ""
    s = v.strip()
    if not s and not allow_empty:
        problems.append(f"'{key}' must not be empty")
    if len(s) > max_chars:
        problems.append(f"'{key}' is too long ({len(s)} chars; max {max_chars})")
    return s


def _str_list(data: Mapping[str, Any], key: str, problems: list[str]) -> tuple[str, ...]:
    v = data.get(key)
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        problems.append(f"'{key}' must be an array of strings")
        return ()
    items = tuple(x.strip() for x in v if x.strip())
    if len(items) > MAX_LIST_ITEMS:
        problems.append(f"'{key}' has {len(items)} items; max {MAX_LIST_ITEMS}")
    too_long = [i for i, x in enumerate(items) if len(x) > MAX_ITEM_CHARS]
    if too_long:
        problems.append(f"'{key}' items {too_long} exceed {MAX_ITEM_CHARS} chars")
    return items


def _enum(data: Mapping[str, Any], key: str, allowed: Sequence[str], problems: list[str]) -> str:
    v = data.get(key)
    if v not in allowed:
        problems.append(f"'{key}' must be one of {list(allowed)}, got {v!r}")
    return str(v)


@dataclass(frozen=True)
class Decision:
    """The Chief's final decision for one cycle (validated ``submit_decision`` input).

    action       : follow_quant | scale | veto | override | hold (policy semantics in
                   :class:`aurum.agents.policy.DecisionPolicy`).
    scale        : multiplier in [0, 1] applied to the quant forecast when action == scale.
    forecast     : the Chief's own view in [-1, 1]; acted upon only for ``override`` (and
                   then only within what the policy mode allows) but always recorded so the
                   desk's discretionary skill can be evaluated against the quant signal.
    confidence   : subjective probability-like confidence in [0, 1].
    horizon_bars : bars over which the view is expected to play out (>= 1).
    """

    action: str
    scale: float
    forecast: float
    confidence: float
    horizon_bars: int
    rationale: str
    key_risks: tuple[str, ...] = ()
    dissent: str = ""

    @classmethod
    def from_tool_input(cls, data: Any, *, max_text_chars: int = 2000) -> Decision:
        if not isinstance(data, Mapping):
            raise RecordValidationError(["input must be a JSON object"])
        problems: list[str] = []
        action = _enum(data, "action", ACTIONS, problems)
        scale = _num(data, "scale", 0.0, 1.0, problems)
        forecast = _num(data, "forecast", -1.0, 1.0, problems)
        confidence = _num(data, "confidence", 0.0, 1.0, problems)
        hb = data.get("horizon_bars")
        if isinstance(hb, bool) or not isinstance(hb, (int, float)) or (
            isinstance(hb, float) and not hb.is_integer()
        ):
            problems.append(f"'horizon_bars' must be an integer in [1, {MAX_HORIZON_BARS}], got {hb!r}")
            horizon = 0
        else:
            horizon = int(hb)
            if not 1 <= horizon <= MAX_HORIZON_BARS:
                problems.append(f"'horizon_bars' must be in [1, {MAX_HORIZON_BARS}], got {hb!r}")
        rationale = _text(data, "rationale", problems, max_chars=max_text_chars)
        key_risks = _str_list(data, "key_risks", problems)
        dissent = _text(data, "dissent", problems, max_chars=max_text_chars, allow_empty=True)
        unknown = sorted(set(data) - {
            "action", "scale", "forecast", "confidence", "horizon_bars", "rationale", "key_risks", "dissent"
        })
        if unknown:
            problems.append(f"unexpected fields {unknown}")
        if problems:
            raise RecordValidationError(problems)
        return cls(
            action=action,
            scale=scale,
            forecast=forecast,
            confidence=confidence,
            horizon_bars=horizon,
            rationale=rationale,
            key_risks=key_risks,
            dissent=dissent,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key_risks"] = list(self.key_risks)
        return d


@dataclass(frozen=True)
class Memo:
    """A specialist's memo (validated ``submit_memo`` input) plus run metadata.

    ``status`` is ``"ok"`` when the specialist submitted a valid memo; otherwise the body
    fields are ``None`` and ``error`` explains why (refusal, max turns, budget, error).
    ``suggested_exposure`` in [-1, 1] must agree in sign with ``stance``.
    """

    agent_id: str
    role: str
    question: str
    status: str = "ok"
    stance: str | None = None
    confidence: float | None = None
    key_points: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    suggested_exposure: float | None = None
    error: str | None = None
    turns: int = 0
    tools: tuple[str, ...] = field(default_factory=tuple)
    mandate: str | None = None

    @classmethod
    def from_tool_input(cls, data: Any, *, agent_id: str, role: str, question: str,
                        tools: Sequence[str] = (), mandate: str | None = None) -> Memo:
        if not isinstance(data, Mapping):
            raise RecordValidationError(["input must be a JSON object"])
        problems: list[str] = []
        stance = _enum(data, "stance", STANCES, problems)
        confidence = _num(data, "confidence", 0.0, 1.0, problems)
        exposure = _num(data, "suggested_exposure", -1.0, 1.0, problems)
        key_points = _str_list(data, "key_points", problems)
        risks = _str_list(data, "risks", problems)
        if not key_points:
            problems.append("'key_points' must contain at least one point")
        if math.isfinite(exposure):
            if stance == "bullish" and exposure < 0:
                problems.append("stance 'bullish' is inconsistent with negative suggested_exposure")
            if stance == "bearish" and exposure > 0:
                problems.append("stance 'bearish' is inconsistent with positive suggested_exposure")
        unknown = sorted(set(data) - {"stance", "confidence", "key_points", "risks", "suggested_exposure"})
        if unknown:
            problems.append(f"unexpected fields {unknown}")
        if problems:
            raise RecordValidationError(problems)
        return cls(
            agent_id=agent_id,
            role=role,
            question=question,
            status="ok",
            stance=stance,
            confidence=confidence,
            key_points=key_points,
            risks=risks,
            suggested_exposure=exposure,
            tools=tuple(tools),
            mandate=mandate,
        )

    @classmethod
    def failed(cls, *, agent_id: str, role: str, question: str, status: str, error: str,
               turns: int = 0, tools: Sequence[str] = (), mandate: str | None = None) -> Memo:
        return cls(agent_id=agent_id, role=role, question=question, status=status, error=error,
                   turns=turns, tools=tuple(tools), mandate=mandate)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("key_points", "risks", "tools"):
            d[k] = list(d[k])
        return d

    def for_chief(self) -> dict[str, Any]:
        """The view returned to the Chief as a tool result (no internal bookkeeping)."""
        if not self.ok:
            # bounded, so the memo can be delivered whole (never cut mid-JSON)
            err = self.error if self.error is None or len(self.error) <= MAX_ERROR_CHARS else (
                self.error[:MAX_ERROR_CHARS] + "...[truncated]")
            return {"agent": self.agent_id, "role": self.role, "status": self.status, "error": err}
        return {
            "agent": self.agent_id,
            "role": self.role,
            "status": "ok",
            "stance": self.stance,
            "confidence": self.confidence,
            "suggested_exposure": self.suggested_exposure,
            "key_points": list(self.key_points),
            "risks": list(self.risks),
        }
