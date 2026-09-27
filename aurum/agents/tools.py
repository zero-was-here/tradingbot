"""Tool definitions (JSON schemas) for the desk's agents.

All tools are sent with ``strict: true`` and ``additionalProperties: false`` on every
object, so the API guarantees schema-valid arguments. Strict schemas do not support
numeric/string constraints (``minimum``, ``maxLength`` ...), so ranges are stated in the
descriptions and enforced client side (:mod:`aurum.agents.records`).

Tool lists are built deterministically (sorted by name) because tools render at the very
front of the prompt: any reordering would invalidate the prompt cache.

Descriptions say *when* to call a tool, not only what it returns — recent Opus models
reach for tools conservatively and trigger conditions measurably improve call rates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "DATA_TOOLS",
    "DATA_TOOL_NAMES",
    "SPECIALIST_ROLES",
    "CONSULT_SPECIALIST",
    "CREATE_SPECIALIST",
    "SUBMIT_DECISION",
    "SUBMIT_MEMO",
    "tool_definition",
    "data_tool_definitions",
    "chief_tool_definitions",
    "specialist_tool_definitions",
    "validate_against_schema",
]

_EMPTY_OBJECT = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

#: Read-only data tools: name -> (provider method, description).
DATA_TOOLS: dict[str, tuple[str, str]] = {
    "get_market_snapshot": (
        "market_snapshot",
        "Point-in-time XAUUSD market state at the decision time: last completed bar, "
        "multi-horizon returns, realised volatility, ATR, distance to moving averages in ATR "
        "units, recent range, spread and session. Call this first in almost every analysis; "
        "it is the ground truth for price-based claims.",
    ),
    "get_quant_signals": (
        "quant_signals",
        "Current outputs of the systematic strategies (per-strategy forecasts in [-1, 1], "
        "the combined forecast under review, agreement/dispersion across strategies and "
        "recent history). Call when you need to understand what the quant book wants and why.",
    ),
    "get_risk_status": (
        "risk_status",
        "Risk-manager state: equity, drawdown from peak, daily P&L versus limits, leverage, "
        "halt status and active risk rules (event blackout, max spread). Call before any "
        "recommendation that keeps or adds exposure.",
    ),
    "get_macro_snapshot": (
        "macro_snapshot",
        "Point-in-time macro drivers of gold: US dollar index, nominal and real 10-year "
        "yields, breakevens, equity volatility, equities — levels (unless anonymised), "
        "recent changes and z-scores. Call when the question involves macro drivers.",
    ),
    "get_calendar": (
        "calendar",
        "Scheduled economic events (NFP, CPI, FOMC, ...) around the decision time: upcoming "
        "events with hours until release and importance, and recently released events. Call "
        "whenever event risk could matter for the next few hours to days.",
    ),
    "get_backtest_stats": (
        "backtest_stats",
        "Out-of-sample performance statistics of the quant strategies and the combined book "
        "(Sharpe, deflated Sharpe, drawdown, hit rate, recent performance). Call when judging "
        "how much to trust the quant signal.",
    ),
    "get_positions": (
        "positions",
        "Current open position(s) in lots, entry price, unrealised P&L and holding time. Call "
        "when the decision depends on what is already held.",
    ),
}
DATA_TOOL_NAMES: tuple[str, ...] = tuple(sorted(DATA_TOOLS))

#: Predefined specialist roles the Chief can consult.
SPECIALIST_ROLES: tuple[str, ...] = ("execution_trader", "macro_strategist", "quant_analyst", "risk_officer")

CONSULT_SPECIALIST = "consult_specialist"
CREATE_SPECIALIST = "create_specialist"
SUBMIT_DECISION = "submit_decision"
SUBMIT_MEMO = "submit_memo"


def tool_definition(name: str, description: str, input_schema: Mapping[str, Any]) -> dict[str, Any]:
    """A strict custom-tool definition in Messages API format."""
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": dict(input_schema),
    }


def data_tool_definitions(names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    selected = DATA_TOOL_NAMES if names is None else tuple(sorted(set(names)))
    unknown = [n for n in selected if n not in DATA_TOOLS]
    if unknown:
        raise KeyError(f"unknown data tools {unknown}")
    return [tool_definition(n, DATA_TOOLS[n][1], _EMPTY_OBJECT) for n in selected]


def _consult_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "enum": list(SPECIALIST_ROLES),
                "description": "Which predefined specialist to consult.",
            },
            "question": {
                "type": "string",
                "description": "A precise, self-contained question (max 2000 characters). The "
                "specialist sees only this question and its own data tools.",
            },
        },
        "required": ["role", "question"],
        "additionalProperties": False,
    }


def _create_schema(whitelist: Iterable[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short snake_case name for the new agent, e.g. 'event_risk_analyst' "
                "(letters, digits, underscores; max 40 characters).",
            },
            "mandate": {
                "type": "string",
                "description": "The agent's charter: what it is responsible for analysing and "
                "which lens to apply (max 2000 characters). It cannot override the desk's "
                "standing rules.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(set(whitelist))},
                "description": "Read-only data tools the agent may use (at least one; grant only "
                "what the mandate needs).",
            },
            "question": {
                "type": "string",
                "description": "The specific question the agent must answer in its memo "
                "(max 2000 characters).",
            },
        },
        "required": ["name", "mandate", "tools", "question"],
        "additionalProperties": False,
    }


def _decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["follow_quant", "scale", "veto", "override", "hold"],
                "description": "follow_quant: execute the quant forecast as is. scale: multiply it "
                "by 'scale'. veto: go flat. override: use your own 'forecast' (subject to the "
                "policy mode). hold: keep the previous final forecast unchanged.",
            },
            "scale": {
                "type": "number",
                "description": "Multiplier in [0, 1] applied to the quant forecast when "
                "action='scale'. Set 1.0 for other actions.",
            },
            "forecast": {
                "type": "number",
                "description": "Your own exposure view in [-1, 1] (-1 max short, +1 max long). "
                "Acted upon only for action='override' within the policy mode's limits; always "
                "recorded for evaluation.",
            },
            "confidence": {"type": "number", "description": "Confidence in [0, 1]."},
            "horizon_bars": {
                "type": "integer",
                "description": "Bars over which the view should play out (integer >= 1).",
            },
            "rationale": {
                "type": "string",
                "description": "Concise, evidence-based rationale citing the data and memos "
                "relied upon (max 2000 characters).",
            },
            "key_risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The main risks to this decision (at most 12 short items).",
            },
            "dissent": {
                "type": "string",
                "description": "Material disagreements among specialists (or with the quant "
                "signal) and how you weighed them; empty string if none.",
            },
        },
        "required": [
            "action", "scale", "forecast", "confidence", "horizon_bars", "rationale", "key_risks", "dissent",
        ],
        "additionalProperties": False,
    }


def _memo_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "stance": {
                "type": "string",
                "enum": ["bullish", "bearish", "neutral"],
                "description": "Directional stance on gold over the question's horizon.",
            },
            "confidence": {"type": "number", "description": "Confidence in [0, 1]."},
            "key_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Evidence-based findings, each quantified and tied to tool data "
                "(1 to 12 short items).",
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Risks, caveats and data gaps (at most 12 short items).",
            },
            "suggested_exposure": {
                "type": "number",
                "description": "Suggested exposure in [-1, 1]; its sign must agree with the "
                "stance (0 is always allowed).",
            },
        },
        "required": ["stance", "confidence", "key_points", "risks", "suggested_exposure"],
        "additionalProperties": False,
    }


def chief_tool_definitions(adhoc_whitelist: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Chief tools: data tools + consult/create specialist + submit_decision (sorted)."""
    whitelist = DATA_TOOL_NAMES if adhoc_whitelist is None else tuple(adhoc_whitelist)
    tools = data_tool_definitions()
    tools.append(tool_definition(
        CONSULT_SPECIALIST,
        "Ask one of the desk's predefined specialists (macro_strategist, quant_analyst, "
        "risk_officer, execution_trader) a question; returns their memo (stance, confidence, "
        "key points, risks, suggested exposure). Call when a specialist's domain analysis would "
        "change your decision. To consult several, issue the calls in the same turn so they "
        "run in parallel.",
        _consult_schema(),
    ))
    tools.append(tool_definition(
        CREATE_SPECIALIST,
        "Create a new ad-hoc specialist agent for this cycle with a custom mandate and a subset "
        "of read-only data tools, and ask it a question; returns its memo. Call only when a "
        "question falls outside every predefined specialist's remit. Created agents cannot "
        "create further agents.",
        _create_schema(whitelist),
    ))
    tools.append(tool_definition(
        SUBMIT_DECISION,
        "Submit the final decision for this cycle. Call exactly once, alone in its turn, after "
        "reviewing the data and memos you requested. This ends the cycle.",
        _decision_schema(),
    ))
    return sorted(tools, key=lambda t: t["name"])


def specialist_tool_definitions(data_tools: Iterable[str]) -> list[dict[str, Any]]:
    """Specialist tools: its whitelisted data tools + submit_memo (sorted). No agent spawning."""
    tools = data_tool_definitions(data_tools)
    tools.append(tool_definition(
        SUBMIT_MEMO,
        "Submit your memo to the Chief Investment Officer. Call exactly once, alone in its turn, "
        "after gathering the data you need. This ends your task.",
        _memo_schema(),
    ))
    return sorted(tools, key=lambda t: t["name"])


# --------------------------------------------------------------------------------------
# minimal JSON-schema validation (defence in depth; the subset used above)
# --------------------------------------------------------------------------------------
def validate_against_schema(schema: Mapping[str, Any], value: Any, path: str = "input") -> list[str]:
    """Validate ``value`` against the JSON-schema subset used by the desk's tools.

    Supports object (properties/required/additionalProperties=false), array (items),
    string, number, integer, boolean and enum. Returns a list of problems (empty = valid).
    """
    problems: list[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, Mapping):
            return [f"{path} must be an object"]
        props: Mapping[str, Any] = schema.get("properties", {}) or {}
        for req in schema.get("required", []) or []:
            if req not in value:
                problems.append(f"{path}.{req} is required")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                problems.append(f"{path} has unexpected fields {extra}")
        for k, sub in props.items():
            if k in value:
                problems.extend(validate_against_schema(sub, value[k], f"{path}.{k}"))
    elif t == "array":
        if not isinstance(value, list):
            return [f"{path} must be an array"]
        items = schema.get("items")
        if items:
            for i, v in enumerate(value):
                problems.extend(validate_against_schema(items, v, f"{path}[{i}]"))
    elif t == "string":
        if not isinstance(value, str):
            problems.append(f"{path} must be a string")
    elif t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path} must be a number")
    elif t == "integer":
        if isinstance(value, bool) or not (
            isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        ):
            problems.append(f"{path} must be an integer")
    elif t == "boolean":
        if not isinstance(value, bool):
            problems.append(f"{path} must be a boolean")
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path} must be one of {list(schema['enum'])}")
    return problems
