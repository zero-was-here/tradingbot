"""Generic manual tool-use loop shared by the Chief and the specialists.

Why a manual loop rather than the SDK's beta tool runner: the desk needs control the
runner does not expose in one place — a terminal tool that ends the loop without a round
trip, per-cycle budget and deadline checks before every call, concurrent execution of slow
tools (specialist consultations) with all results returned in ONE user message, explicit
``pause_turn`` / ``max_tokens`` / ``refusal`` handling, and journaling of every step.

Loop invariants (Messages API rules):

* the full assistant ``content`` (including thinking blocks, unmodified) is appended before
  its tool results; after a server-side fallback it is first sanitised
  (:func:`~aurum.agents.client.sanitize_assistant_content`);
* every ``tool_use`` gets exactly one ``tool_result`` with the matching ``tool_use_id``, all
  in the next user message, placed before any text block;
* ``stop_reason`` is checked before content is used: ``refusal`` ends the loop (partial
  output discarded), ``max_tokens`` discards the truncated turn (a cut-off ``tool_use``
  input must never be executed), ``pause_turn`` re-sends the paused turn as-is only when it
  ends in a server-tool block (the documented resume); otherwise it continues with a user
  nudge, because a request ending on an assistant turn is a prefill (400 on Opus 4.6+);
* specialists run with ``budget_share < 1`` so they cannot exhaust the cycle budget the Chief
  needs to submit its decision.

Final turn: the results of tools called on the last allowed turn could never be read by
the model (the loop ends right after), so they are NOT executed — a ``consult_specialist``
there would otherwise spend a whole specialist loop's budget on a memo nobody reads. A
single terminal call on the final turn is honoured even when batched with other calls (the
"alone" rule exists to buy a review round-trip, which no longer exists on the last turn);
rejecting it would silently replace, e.g., a veto with ``on_failure``.

Deadline: when the runtime has a deadline, every request carries a per-request ``timeout``
of ``min(request_timeout_s, time left)`` so a single slow call cannot run past it (SDK
retries of a timed-out call can still extend it by up to ``max_retries`` more timeouts).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from aurum.agents._json import dumps, truncate
from aurum.agents.client import (
    LLMClient,
    block_get,
    block_to_dict,
    fallback_events,
    sanitize_assistant_content,
)
from aurum.agents.config import AgentModelConfig
from aurum.agents.journal import CycleJournal
from aurum.agents.tools import validate_against_schema
from aurum.agents.usage import UsageLedger

logger = logging.getLogger(__name__)

__all__ = ["ToolOutcome", "ToolHandler", "PreparedHandler", "AgentSpec", "LoopResult", "AgentRuntime",
           "run_agent_loop"]

MAX_PAUSE_CONTINUATIONS = 3
#: On ``stop_reason == "max_tokens"`` the turn is retried with a larger ``max_tokens`` (x2) up
#: to this ceiling (``max_tokens`` caps thinking + output; it is not part of the cached
#: prefix). The desk uses non-streaming requests, and the Python SDK refuses those when
#: ``max_tokens`` implies > 10 minutes of generation (about 21.3k tokens) on a client with the
#: default timeout — the ceiling stays below that limit.
MAX_TOKENS_RETRY_CEILING = 21_000
HARNESS = "[desk harness]"
#: Content-block types whose presence at the end of a paused turn means the API can resume a
#: server-side sampling loop from it (``pause_turn`` semantics).
SERVER_TOOL_BLOCKS = frozenset({"server_tool_use"})


@dataclass
class ToolOutcome:
    """Result of one tool call. ``terminal`` is set only by an accepted terminal tool.

    ``truncatable=False`` exempts the content from ``tool_result_max_chars``: use it only for
    payloads that are already size-bounded and must arrive whole — e.g. specialist memos,
    where a cut would drop the trailing JSON keys (stance, suggested exposure) and hand the
    Chief an unparseable memo without its recommendation.
    """

    content: str
    is_error: bool = False
    terminal: Any = None
    truncatable: bool = True

    @classmethod
    def error(cls, message: str) -> ToolOutcome:
        return cls(content=dumps({"error": message}), is_error=True)

    @classmethod
    def ok(cls, payload: Any) -> ToolOutcome:
        return cls(content=payload if isinstance(payload, str) else dumps(payload))


ToolHandler = Callable[[Mapping[str, Any]], ToolOutcome]


class PreparedHandler:
    """A tool handler split into a sequential *admission* step and deferred *work*.

    ``prepare(tool_input)`` runs on the loop's thread, in the order the model issued the
    calls, and returns a zero-argument callable that does the (possibly slow, possibly
    concurrent) work. Use it when a call claims shared resources — e.g. the per-cycle
    specialist cap or agent-id numbering — so the outcome is reproducible.
    """

    def __init__(self, prepare: Callable[[Mapping[str, Any]], Callable[[], ToolOutcome]]) -> None:
        self.prepare = prepare

    def __call__(self, tool_input: Mapping[str, Any]) -> ToolOutcome:
        return self.prepare(tool_input)()


@dataclass
class AgentSpec:
    agent_id: str
    role: str
    system_prompt: str
    tools: list[dict[str, Any]]
    handlers: Mapping[str, ToolHandler]
    terminal_tool: str
    cfg: AgentModelConfig
    concurrent_tools: bool = False  # run a turn's tool calls on the runtime's thread pool
    #: fraction of the cycle's hard budget (cost and tokens) this agent may use up to; the
    #: Chief has 1.0, specialists less, so they cannot starve the Chief of its final decision
    budget_share: float = 1.0


@dataclass
class LoopResult:
    agent_id: str
    status: str  # completed | refusal | max_turns | budget_exceeded | deadline | error
    terminal: Any = None
    turns: int = 0
    error: str | None = None
    refusal_category: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == "completed"


@dataclass
class AgentRuntime:
    """Per-cycle services shared by every agent loop in the cycle."""

    llm: LLMClient
    ledger: UsageLedger
    journal: CycleJournal
    executor: ThreadPoolExecutor | None = None
    deadline: float | None = None  # time.monotonic() value
    soft_budget_fraction: float = 0.8
    tool_result_max_chars: int = 12_000
    journal_max_chars: int = 4_000
    request_timeout_s: float = 600.0
    extra: dict[str, Any] = field(default_factory=dict)

    def deadline_passed(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def blocked(self, budget_share: float = 1.0) -> tuple[str, str] | None:
        """(status, reason) if an agent entitled to ``budget_share`` of the cycle budget may
        make no further API call."""
        reason = self.ledger.exceeded(budget_share)
        if reason:
            return "budget_exceeded", reason
        if self.deadline_passed():
            return "deadline", "cycle deadline reached"
        return None

    def request_timeout(self) -> float | None:
        """Per-request timeout bounding an in-flight call by the cycle deadline.

        ``None`` (no deadline) leaves the client's own timeout configuration untouched.
        """
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        return max(0.001, min(float(self.request_timeout_s), remaining))

    def notes(self, *, turns_left: int, terminal: str) -> list[str]:
        out = []
        frac = self.ledger.fraction_used()
        if frac >= self.soft_budget_fraction:
            out.append(f"{HARNESS} {frac:.0%} of the cycle budget is used; conclude and call {terminal} promptly.")
        if turns_left == 1:
            out.append(f"{HARNESS} One turn left: call {terminal} now, alone in its turn, with your best "
                       "assessment. Other tool calls made in the final turn will not be executed.")
        return out


def _append_user_text(messages: list[dict[str, Any]], text: str, *, runtime: AgentRuntime,
                      agent_id: str) -> None:
    """Append a harness note to the conversation (after any tool_results) and journal it.

    Notes are appended, never removed later — deleting an injected note would be a history
    edit that invalidates the prompt cache from that point on.
    """
    block = {"type": "text", "text": text}
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"].append(block)
    else:
        messages.append({"role": "user", "content": [block]})
    runtime.journal.log("harness_note", agent=agent_id, text=text)


def _has_visible_content(blocks: Sequence[Any]) -> bool:
    return any(
        block_get(b, "type") == "text" and str(block_get(b, "text", "")).strip() for b in blocks
    )


def _admit_one(spec: AgentSpec, block: Any, schemas: Mapping[str, Any], runtime: AgentRuntime,
               forced_error: str | None) -> Callable[[], ToolOutcome]:
    """Phase 1 (sequential, in tool_use order): journal the call, validate it and let a
    :class:`PreparedHandler` claim shared resources. Returns the work to run in phase 2."""
    name = str(block_get(block, "name"))
    tool_input = block_get(block, "input")
    runtime.journal.log("tool_call", agent=spec.agent_id, tool=name, tool_use_id=block_get(block, "id"),
                        input=tool_input)
    if forced_error is not None:
        outcome = ToolOutcome.error(forced_error)
    elif name not in schemas or name not in spec.handlers:
        outcome = ToolOutcome.error(f"Unknown or unavailable tool '{name}'. Available tools: {sorted(schemas)}.")
    else:
        problems = validate_against_schema(schemas[name], tool_input)
        if problems:
            outcome = ToolOutcome.error("Invalid input: " + "; ".join(problems))
        else:
            handler = spec.handlers[name]
            if isinstance(handler, PreparedHandler):
                try:
                    return handler.prepare(tool_input)
                except Exception as exc:
                    logger.exception("tool %s failed for agent %s", name, spec.agent_id)
                    outcome = ToolOutcome.error(f"Tool failed: {type(exc).__name__}: {exc}")
            else:
                return lambda: handler(tool_input)
    return lambda: outcome


def _run_one(spec: AgentSpec, block: Any, work: Callable[[], ToolOutcome], runtime: AgentRuntime) -> ToolOutcome:
    """Phase 2 (possibly concurrent): run the admitted work and journal its result."""
    name = str(block_get(block, "name"))
    try:
        outcome = work()
    except Exception as exc:  # a tool failure is reported to the model, not raised
        logger.exception("tool %s failed for agent %s", name, spec.agent_id)
        outcome = ToolOutcome.error(f"Tool failed: {type(exc).__name__}: {exc}")
    runtime.journal.log("tool_result", agent=spec.agent_id, tool=name, tool_use_id=block_get(block, "id"),
                        is_error=outcome.is_error, accepted_terminal=outcome.terminal is not None,
                        content=outcome.content)
    return outcome


def _execute_tools(spec: AgentSpec, tool_uses: Sequence[Any], schemas: Mapping[str, Any],
                   runtime: AgentRuntime) -> list[ToolOutcome]:
    """Execute one turn's tool calls; results are returned in tool_use order.

    Admission (validation, slot reservation, agent-id assignment) is sequential in the order
    the model issued the calls, so the outcome never depends on thread scheduling; only the
    admitted work runs concurrently.
    """
    n_terminal = sum(1 for b in tool_uses if block_get(b, "name") == spec.terminal_tool)
    alone_error = None
    if n_terminal and len(tool_uses) > 1:
        alone_error = (
            f"{spec.terminal_tool} must be called alone in its turn. The results of your other "
            f"calls are included in this message; review them, then call {spec.terminal_tool} again."
        )
    work = [
        _admit_one(spec, b, schemas, runtime, alone_error if block_get(b, "name") == spec.terminal_tool else None)
        for b in tool_uses
    ]
    if spec.concurrent_tools and runtime.executor is not None and len(tool_uses) > 1:
        futures = [runtime.executor.submit(_run_one, spec, b, w, runtime)
                   for b, w in zip(tool_uses, work, strict=True)]
        return [f.result() for f in futures]
    return [_run_one(spec, b, w, runtime) for b, w in zip(tool_uses, work, strict=True)]


def run_agent_loop(spec: AgentSpec, first_message: str, runtime: AgentRuntime) -> LoopResult:
    """Run one agent until its terminal tool is accepted or a stop condition is hit."""
    schemas = {t["name"]: t["input_schema"] for t in spec.tools}
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": first_message}]}]
    runtime.journal.log(
        "agent_start", agent=spec.agent_id, full=True, role=spec.role, model=spec.cfg.model,
        effort=spec.cfg.effort, max_turns=spec.cfg.max_turns, tools=sorted(schemas),
        system_prompt=spec.system_prompt, first_message=first_message,
    )
    turns = 0
    pauses = 0
    cfg = spec.cfg  # may be replaced by a copy with a larger max_tokens after truncation

    def finish(status: str, **kw: Any) -> LoopResult:
        res = LoopResult(agent_id=spec.agent_id, status=status, turns=turns, **kw)
        runtime.journal.log("agent_end", agent=spec.agent_id, status=status, turns=turns,
                            error=res.error, refusal_category=res.refusal_category)
        if status != "completed":
            logger.info("agent %s ended with status=%s (%s)", spec.agent_id, status, res.error)
        return res

    max_turns_error = f"no {spec.terminal_tool} within {spec.cfg.max_turns} turns"

    def nudge(text: str) -> None:
        _append_user_text(messages, text, runtime=runtime, agent_id=spec.agent_id)

    while turns < spec.cfg.max_turns:
        blocked = runtime.blocked(spec.budget_share)
        if blocked is not None:
            return finish(blocked[0], error=blocked[1])
        turns += 1
        last_turn = turns >= spec.cfg.max_turns
        try:
            response = runtime.llm.create(cfg=cfg, system_prompt=spec.system_prompt,
                                          tools=spec.tools, messages=messages,
                                          timeout=runtime.request_timeout())
        except Exception as exc:  # API/network errors end this agent; the desk falls back safely
            logger.warning("API call failed for agent %s: %s: %s", spec.agent_id, type(exc).__name__, exc)
            if runtime.deadline_passed():  # e.g. the deadline-bounded request timeout fired
                return finish("deadline", error=f"cycle deadline reached during an API call "
                                                f"({type(exc).__name__}: {exc})")
            return finish("error", error=f"API call failed: {type(exc).__name__}: {exc}")

        usage = runtime.ledger.record(spec.agent_id, response, requested_model=spec.cfg.model)
        stop = getattr(response, "stop_reason", None)
        content = list(getattr(response, "content", None) or [])
        stop_details = getattr(response, "stop_details", None)
        runtime.journal.log(
            "llm_response", agent=spec.agent_id, turn=turns, stop_reason=stop,
            model=getattr(response, "model", None), request_id=getattr(response, "_request_id", None),
            content=[block_to_dict(b, max_chars=runtime.journal_max_chars) for b in content],
            usage=usage.to_dict(), fallback=fallback_events(response),
            stop_details=block_to_dict(stop_details) if stop_details is not None else None,
        )

        if stop == "refusal":
            category = block_get(stop_details, "category") if stop_details is not None else None
            return finish("refusal", error=f"model declined the request (category={category})",
                          refusal_category=category)
        if stop == "model_context_window_exceeded":
            return finish("error", error="context window exceeded")
        if stop == "max_tokens":
            if last_turn:
                return finish("max_turns", error=f"{max_turns_error} (last turn hit max_tokens)")
            if cfg.max_tokens < MAX_TOKENS_RETRY_CEILING:
                cfg = cfg.replace(max_tokens=min(2 * cfg.max_tokens, MAX_TOKENS_RETRY_CEILING))
                runtime.journal.log("max_tokens_retry", agent=spec.agent_id, new_max_tokens=cfg.max_tokens)
            nudge(f"{HARNESS} Your previous response hit the output token limit and was discarded. "
                  "Be more concise and make the tool call you need directly.")
            continue

        echo = sanitize_assistant_content(content)
        if stop == "pause_turn":
            if pauses >= MAX_PAUSE_CONTINUATIONS:
                return finish("error", error="too many pause_turn continuations")
            pauses += 1
            if echo and block_get(echo[-1], "type") in SERVER_TOOL_BLOCKS:
                # Documented resume: re-send the paused turn as-is; the API detects the trailing
                # server-tool block and continues the server-side loop.
                messages.append({"role": "assistant", "content": echo})
                continue
            # Nothing server-side to resume (the desk declares client tools only). Re-sending
            # the turn would end the request on an assistant message — a prefill, which Opus
            # 4.6+ (incl. Opus 5) rejects with a 400 — so keep any visible text (never a client
            # tool_use, which would need results) and ask the model to continue.
            if _has_visible_content(echo) and not any(block_get(b, "type") == "tool_use" for b in echo):
                messages.append({"role": "assistant", "content": echo})
            nudge(f"{HARNESS} Continue the task and finish by calling {spec.terminal_tool}, alone in its turn.")
            continue

        tool_uses = [b for b in echo if block_get(b, "type") == "tool_use"]
        if last_turn:
            # Results of calls made now could never be read: run only a lone terminal call.
            terminal_calls = [b for b in tool_uses if block_get(b, "name") == spec.terminal_tool]
            skipped = [str(block_get(b, "name")) for b in tool_uses
                       if len(terminal_calls) != 1 or block_get(b, "name") != spec.terminal_tool]
            if skipped:
                runtime.journal.log("tools_skipped", agent=spec.agent_id, tools=skipped,
                                    reason="final turn: results could never be read by the model")
            if len(terminal_calls) == 1:
                outcome = _execute_tools(spec, terminal_calls, schemas, runtime)[0]
                if outcome.terminal is not None:
                    return finish("completed", terminal=outcome.terminal)
            return finish("max_turns", error=max_turns_error)

        if not tool_uses:
            if _has_visible_content(echo):
                messages.append({"role": "assistant", "content": echo})
            nudge(f"{HARNESS} No tool was called. Continue the task and finish by calling "
                  f"{spec.terminal_tool}, alone in its turn.")
            for note in runtime.notes(turns_left=spec.cfg.max_turns - turns, terminal=spec.terminal_tool):
                nudge(note)
            continue

        messages.append({"role": "assistant", "content": echo})
        outcomes = _execute_tools(spec, tool_uses, schemas, runtime)
        for outcome in outcomes:
            if outcome.terminal is not None:
                return finish("completed", terminal=outcome.terminal)
        user_content: list[dict[str, Any]] = []
        for block, outcome in zip(tool_uses, outcomes, strict=True):
            result_text = outcome.content
            if outcome.truncatable:
                result_text = truncate(result_text, runtime.tool_result_max_chars)
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block_get(block, "id"),
                "content": result_text,
            }
            if outcome.is_error:
                result["is_error"] = True
            user_content.append(result)
        messages.append({"role": "user", "content": user_content})
        for note in runtime.notes(turns_left=spec.cfg.max_turns - turns, terminal=spec.terminal_tool):
            nudge(note)  # text after the tool_result blocks of the same user message

    return finish("max_turns", error=max_turns_error)
