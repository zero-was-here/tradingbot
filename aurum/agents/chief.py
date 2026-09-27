"""The Chief Investment Officer agent and the per-cycle tool handlers it drives.

:class:`DeskCycle` owns everything that lives for exactly one decision cycle: the decision
time, the quant forecast under review, a cache of data-tool results (every agent in the
cycle sees identical, point-in-time snapshots and the provider is queried once per tool),
the specialist counter enforcing ``max_specialists_per_cycle``, and the memos collected.

Chief tools
-----------
* data tools (``get_*``) — read-only snapshots from the :class:`DeskDataProvider`;
* ``consult_specialist(role, question)`` — runs a predefined specialist loop;
* ``create_specialist(name, mandate, tools, question)`` — instantiates a NEW agent at
  runtime with a Chief-written mandate and a whitelisted subset of data tools, runs it and
  returns its memo. Created agents get only data tools + ``submit_memo`` (depth 1);
* ``submit_decision`` — terminal; validated into a :class:`Decision`.

Several consult/create calls in one Chief turn run concurrently on the cycle's thread pool,
and all results go back to the Chief in one user message. Admission — validation, the
specialist-cap slot and the agent id (``role``, ``role#2`` ...) — happens sequentially in
the order the Chief issued the calls (:class:`~aurum.agents.loop.PreparedHandler`), so which
call gets a slot or an id never depends on thread scheduling.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from aurum.agents._json import dumps
from aurum.agents.config import DeskConfig
from aurum.agents.loop import (
    AgentRuntime,
    AgentSpec,
    LoopResult,
    PreparedHandler,
    ToolHandler,
    ToolOutcome,
    run_agent_loop,
)
from aurum.agents.prompts import CHIEF_SYSTEM_PROMPT
from aurum.agents.providers import DeskDataProvider
from aurum.agents.records import Decision, Memo, RecordValidationError
from aurum.agents.specialists import (
    PREDEFINED_SPECIALISTS,
    SpecialistTask,
    adhoc_task,
    predefined_task,
    run_specialist,
    slugify,
)
from aurum.agents.tools import (
    CONSULT_SPECIALIST,
    CREATE_SPECIALIST,
    DATA_TOOL_NAMES,
    DATA_TOOLS,
    SUBMIT_DECISION,
    chief_tool_definitions,
)

logger = logging.getLogger(__name__)

__all__ = ["DeskCycle"]


class DeskCycle:
    """State + tool handlers for one decision cycle."""

    def __init__(
        self,
        *,
        provider: DeskDataProvider,
        now: pd.Timestamp,
        as_of: str,
        quant_forecast: float,
        config: DeskConfig,
        runtime: AgentRuntime,
    ) -> None:
        self.provider = provider
        self.now = now
        self.as_of = as_of
        self.quant_forecast = quant_forecast
        self.config = config
        self.runtime = runtime
        self.memos: list[Memo] = []
        self._data_cache: dict[str, str] = {}
        self._data_lock = threading.Lock()
        self._slot_lock = threading.Lock()
        self._slots_used = 0
        self._id_counts: dict[str, int] = {}
        whitelist = config.adhoc_tool_whitelist
        self.adhoc_whitelist: tuple[str, ...] = tuple(sorted(DATA_TOOL_NAMES if whitelist is None else whitelist))
        unknown = [t for t in self.adhoc_whitelist if t not in DATA_TOOLS]
        if unknown:
            raise KeyError(f"adhoc_tool_whitelist contains unknown tools {unknown}")

    # ------------------------------------------------------------------ data tools
    def _fetch(self, tool: str) -> str:
        with self._data_lock:
            cached = self._data_cache.get(tool)
            if cached is not None:
                return cached
            method = getattr(self.provider, DATA_TOOLS[tool][0])
            snapshot = method(self.now)
            envelope: dict[str, Any] = {"tool": tool, "as_of": self.as_of, "data": snapshot}
            if tool == "get_quant_signals":
                envelope["quant_forecast_under_review"] = self.quant_forecast
            text = dumps(envelope, float_sig=6)
            self._data_cache[tool] = text
            return text

    def data_handler(self, tool: str) -> ToolHandler:
        def handler(_tool_input: Mapping[str, Any]) -> ToolOutcome:
            try:
                return ToolOutcome.ok(self._fetch(tool))
            except Exception as exc:
                logger.warning("data tool %s failed: %s: %s", tool, type(exc).__name__, exc)
                return ToolOutcome.error(f"data unavailable: {type(exc).__name__}: {exc}")

        return handler

    def data_handlers(self) -> dict[str, ToolHandler]:
        return {t: self.data_handler(t) for t in DATA_TOOL_NAMES}

    # ------------------------------------------------------------------ specialists
    def _reserve_slot(self) -> bool:
        with self._slot_lock:
            if self._slots_used >= self.config.max_specialists_per_cycle:
                return False
            self._slots_used += 1
            return True

    def _new_agent_id(self, base: str) -> str:
        with self._slot_lock:
            n = self._id_counts.get(base, 0) + 1
            self._id_counts[base] = n
        return base if n == 1 else f"{base}#{n}"

    def _check_text(self, value: Any, field: str, problems: list[str]) -> str:
        s = value.strip() if isinstance(value, str) else ""
        if not s:
            problems.append(f"'{field}' must be a non-empty string")
        elif len(s) > self.config.max_text_field_chars:
            problems.append(f"'{field}' is too long ({len(s)} chars; max {self.config.max_text_field_chars})")
        return s

    def _limit_error(self) -> ToolOutcome:
        return ToolOutcome.error(
            f"Specialist limit reached ({self.config.max_specialists_per_cycle} per cycle). "
            "Decide with the information you have."
        )

    def _record_memo(self, memo: Memo) -> ToolOutcome:
        with self._slot_lock:
            self.memos.append(memo)
        payload = memo.for_chief()
        # A memo is size-bounded by record validation (<= 2 x MAX_LIST_ITEMS x MAX_ITEM_CHARS)
        # and its sorted-key JSON ends with stance/status/suggested_exposure: truncating it
        # would deliver the Chief an unparseable memo without its recommendation.
        return ToolOutcome(content=dumps(payload, float_sig=6), is_error=not memo.ok, truncatable=False)

    def _run_task(self, task: SpecialistTask, role_key: str) -> ToolOutcome:
        memo = run_specialist(
            task, cfg=self.config.model_for_role(role_key), runtime=self.runtime,
            data_handlers=self.data_handlers(), as_of=self.as_of, quant_forecast=self.quant_forecast,
            budget_share=self.config.specialist_budget_share,
        )
        return self._record_memo(memo)

    def prepare_consult(self, tool_input: Mapping[str, Any]) -> Callable[[], ToolOutcome]:
        """Admission for ``consult_specialist`` (sequential, in tool_use order): validate,
        reserve a slot and assign the agent id; the returned callable runs the specialist."""
        problems: list[str] = []
        role = tool_input.get("role")
        if role not in PREDEFINED_SPECIALISTS:
            problems.append(f"'role' must be one of {sorted(PREDEFINED_SPECIALISTS)}")
        question = self._check_text(tool_input.get("question"), "question", problems)
        if problems:
            error = ToolOutcome.error("Invalid input: " + "; ".join(problems))
            return lambda: error
        if not self._reserve_slot():
            limit = self._limit_error()
            return lambda: limit
        task = predefined_task(str(role), question, agent_id=self._new_agent_id(str(role)))
        return lambda: self._run_task(task, task.role)

    def prepare_create(self, tool_input: Mapping[str, Any]) -> Callable[[], ToolOutcome]:
        """Admission for ``create_specialist``: validate the mandate and tool grant, reserve a
        slot, assign the id and journal the creation; the returned callable runs the agent."""
        problems: list[str] = []
        raw_name = tool_input.get("name")
        name = slugify(raw_name) if isinstance(raw_name, str) else ""
        if not name:
            problems.append("'name' must contain letters or digits")
        mandate = self._check_text(tool_input.get("mandate"), "mandate", problems)
        question = self._check_text(tool_input.get("question"), "question", problems)
        tools = tool_input.get("tools")
        if not isinstance(tools, list) or not tools:
            problems.append("'tools' must be a non-empty array of data tool names")
            tools = []
        else:
            if len(set(tools)) != len(tools):
                problems.append("'tools' must not contain duplicates")
            not_allowed = sorted({t for t in tools if t not in self.adhoc_whitelist})
            if not_allowed:
                problems.append(f"tools {not_allowed} are not grantable; allowed: {list(self.adhoc_whitelist)}")
        if problems:
            error = ToolOutcome.error("Invalid input: " + "; ".join(problems))
            return lambda: error
        if not self._reserve_slot():
            limit = self._limit_error()
            return lambda: limit
        agent_id = self._new_agent_id(f"adhoc:{name}")
        task = adhoc_task(agent_id=agent_id, name=name, mandate=mandate, tools=tools, question=question)
        self.runtime.journal.log("agent_created", agent=agent_id, name=name, mandate=mandate,
                                 tools=list(task.data_tools), question=question)
        return lambda: self._run_task(task, "adhoc")

    def consult_specialist(self, tool_input: Mapping[str, Any]) -> ToolOutcome:
        return self.prepare_consult(tool_input)()

    def create_specialist(self, tool_input: Mapping[str, Any]) -> ToolOutcome:
        return self.prepare_create(tool_input)()

    # ------------------------------------------------------------------ chief
    def submit_decision(self, tool_input: Mapping[str, Any]) -> ToolOutcome:
        try:
            decision = Decision.from_tool_input(tool_input, max_text_chars=self.config.max_text_field_chars)
        except RecordValidationError as exc:
            return ToolOutcome.error(f"Decision rejected, fix and resubmit: {exc}")
        return ToolOutcome(content="decision accepted", terminal=decision)

    def chief_spec(self) -> AgentSpec:
        tools = chief_tool_definitions(self.adhoc_whitelist)
        handlers: dict[str, ToolHandler] = dict(self.data_handlers())
        hidden: set[str] = set()
        if self.config.max_specialists_per_cycle > 0:
            handlers[CONSULT_SPECIALIST] = PreparedHandler(self.prepare_consult)
            if self.adhoc_whitelist:
                handlers[CREATE_SPECIALIST] = PreparedHandler(self.prepare_create)
            else:
                hidden.add(CREATE_SPECIALIST)
        else:
            hidden |= {CONSULT_SPECIALIST, CREATE_SPECIALIST}
        tools = [t for t in tools if t["name"] not in hidden]
        handlers[SUBMIT_DECISION] = self.submit_decision
        return AgentSpec(
            agent_id="chief",
            role="chief",
            system_prompt=CHIEF_SYSTEM_PROMPT,
            tools=tools,
            handlers=handlers,
            terminal_tool=SUBMIT_DECISION,
            cfg=self.config.model_for_role("chief"),
            concurrent_tools=True,
        )

    def run_chief(self, brief: str) -> LoopResult:
        return run_agent_loop(self.chief_spec(), brief, self.runtime)
