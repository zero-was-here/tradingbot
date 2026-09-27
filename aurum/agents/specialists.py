"""Specialist agents: predefined desk roles and Chief-created ad-hoc agents.

Each specialist is an independent Claude tool-use loop with its own frozen system prompt
(:mod:`aurum.agents.prompts`), a whitelisted set of read-only data tools and one terminal
tool, ``submit_memo``. Specialists have NO agent-spawning tools: the desk's delegation depth
is exactly one (Chief -> specialist). If a specialist nevertheless emits a call to
``create_specialist`` or ``consult_specialist`` it receives an "unknown tool" error result.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aurum.agents.config import AgentModelConfig
from aurum.agents.loop import AgentRuntime, AgentSpec, ToolHandler, ToolOutcome, run_agent_loop
from aurum.agents.prompts import adhoc_brief, specialist_brief, specialist_system_prompt
from aurum.agents.records import Memo, RecordValidationError
from aurum.agents.tools import DATA_TOOLS, SUBMIT_MEMO, specialist_tool_definitions

__all__ = [
    "PREDEFINED_SPECIALISTS",
    "SpecialistTask",
    "slugify",
    "build_specialist_spec",
    "run_specialist",
    "predefined_task",
    "adhoc_task",
]

#: role -> data tools it may call (read-only). Sorted for prompt-cache stability.
PREDEFINED_SPECIALISTS: dict[str, tuple[str, ...]] = {
    "macro_strategist": ("get_calendar", "get_macro_snapshot", "get_market_snapshot"),
    "quant_analyst": ("get_backtest_stats", "get_market_snapshot", "get_quant_signals"),
    "risk_officer": ("get_calendar", "get_market_snapshot", "get_positions", "get_quant_signals",
                     "get_risk_status"),
    "execution_trader": ("get_calendar", "get_market_snapshot", "get_positions"),
}

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(name: str, max_len: int = 40) -> str:
    """Lower-case snake_case identifier for an ad-hoc agent name ('' if nothing usable)."""
    s = _SLUG_RE.sub("_", name.strip().lower()).strip("_")
    s = re.sub(r"_+", "_", s)
    return s[:max_len].rstrip("_")


@dataclass(frozen=True)
class SpecialistTask:
    """What the Chief asked for: a predefined role or an ad-hoc agent."""

    agent_id: str
    role: str  # predefined role name, or "adhoc"
    question: str
    data_tools: tuple[str, ...]
    name: str | None = None  # ad-hoc display name
    mandate: str | None = None  # ad-hoc charter


def build_specialist_spec(task: SpecialistTask, *, cfg: AgentModelConfig,
                          data_handlers: Mapping[str, ToolHandler], budget_share: float = 1.0) -> AgentSpec:
    """Assemble the loop spec: data tools (whitelisted) + ``submit_memo`` only.

    ``budget_share`` caps the fraction of the cycle budget at which this specialist stops
    making API calls (the desk keeps the rest for the Chief's decision).
    """
    unknown = [t for t in task.data_tools if t not in DATA_TOOLS]
    if unknown:
        raise KeyError(f"unknown data tools {unknown}")
    handlers: dict[str, ToolHandler] = {t: data_handlers[t] for t in task.data_tools}

    def submit_memo(tool_input: Mapping[str, object]) -> ToolOutcome:
        try:
            memo = Memo.from_tool_input(
                tool_input, agent_id=task.agent_id, role=task.role, question=task.question,
                tools=task.data_tools, mandate=task.mandate,
            )
        except RecordValidationError as exc:
            return ToolOutcome.error(f"Memo rejected, fix and resubmit: {exc}")
        return ToolOutcome(content="memo accepted", terminal=memo)

    handlers[SUBMIT_MEMO] = submit_memo
    prompt_role = "adhoc" if task.role == "adhoc" else task.role
    return AgentSpec(
        agent_id=task.agent_id,
        role=task.role,
        system_prompt=specialist_system_prompt(prompt_role),
        tools=specialist_tool_definitions(task.data_tools),
        handlers=handlers,
        terminal_tool=SUBMIT_MEMO,
        cfg=cfg,
        concurrent_tools=False,  # never nest work on the Chief's pool (deadlock-free)
        budget_share=budget_share,
    )


def run_specialist(task: SpecialistTask, *, cfg: AgentModelConfig, runtime: AgentRuntime,
                   data_handlers: Mapping[str, ToolHandler], as_of: str, quant_forecast: float,
                   budget_share: float = 1.0) -> Memo:
    """Run one specialist loop to completion and return its memo (or a failed memo)."""
    spec = build_specialist_spec(task, cfg=cfg, data_handlers=data_handlers, budget_share=budget_share)
    if task.role == "adhoc":
        brief = adhoc_brief(agent_id=task.agent_id, name=task.name or task.agent_id, mandate=task.mandate or "",
                            tools=task.data_tools, as_of=as_of, quant_forecast=quant_forecast,
                            question=task.question)
    else:
        brief = specialist_brief(agent_id=task.agent_id, as_of=as_of, quant_forecast=quant_forecast,
                                 question=task.question)
    result = run_agent_loop(spec, brief, runtime)
    if result.completed and isinstance(result.terminal, Memo):
        memo = dataclasses.replace(result.terminal, turns=result.turns)
    else:
        memo = Memo.failed(agent_id=task.agent_id, role=task.role, question=task.question,
                           status=result.status, error=result.error or result.status, turns=result.turns,
                           tools=task.data_tools, mandate=task.mandate)
    runtime.journal.log("memo", agent=task.agent_id, memo=memo.to_dict())
    return memo


def predefined_task(role: str, question: str, *, agent_id: str | None = None) -> SpecialistTask:
    if role not in PREDEFINED_SPECIALISTS:
        raise KeyError(f"unknown specialist role {role!r}")
    return SpecialistTask(agent_id=agent_id or role, role=role, question=question,
                          data_tools=PREDEFINED_SPECIALISTS[role])


def adhoc_task(*, agent_id: str, name: str, mandate: str, tools: Sequence[str], question: str) -> SpecialistTask:
    return SpecialistTask(agent_id=agent_id, role="adhoc", question=question,
                          data_tools=tuple(sorted(set(tools))), name=name, mandate=mandate)
