"""Offline test double for the Claude Messages API.

:class:`FakeAnthropicClient` exposes the exact call path the desk uses
(``client.beta.messages.create(**kwargs)``; ``client.messages.create`` too) and returns
SDK-shaped response objects: ``.content`` blocks with ``type``/``id``/``name``/``input``/
``text``, ``.stop_reason``, ``.stop_details`` and ``.usage`` with ``input_tokens``,
``output_tokens``, ``cache_creation_input_tokens``, ``cache_read_input_tokens`` (and optional
``iterations``). ``FakeMessage.to_api_dict()`` produces the wire JSON so tests can check the
shapes against the real SDK models.

Scripts are routed per agent, because specialists run concurrently and a single global
queue would be nondeterministic. Every agent's first user message starts with
``"Agent: <agent_id>"``; :func:`default_router` reads it. A script for ``"macro_strategist#2"``
falls back to ``"macro_strategist"`` when absent. Script entries are :class:`FakeMessage`
objects or callables ``request_kwargs -> FakeMessage`` (useful to assert on a request or to
synchronise threads).

With ``validate=True`` (default) each request is checked against the Messages API rules the
desk must respect (tool_result pairing in ONE following user message, tool_results before
text, no last-assistant-turn prefill, strict tool schemas, auto tool choice, fallback beta
header, ...). Violations raise
:class:`FakeRequestError` and are also collected in ``client.errors``.
"""

from __future__ import annotations

import copy
import itertools
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aurum.agents.config import BETA_SERVER_SIDE_FALLBACK, EFFORT_LEVELS

__all__ = [
    "FakeTextBlock",
    "FakeThinkingBlock",
    "FakeToolUseBlock",
    "FakeFallbackBlock",
    "FakeUsage",
    "FakeStopDetails",
    "FakeMessage",
    "FakeRequestError",
    "FakeScriptExhausted",
    "FakeAnthropicClient",
    "default_router",
    "text",
    "thinking",
    "tool_use",
    "message",
    "refusal",
    "truncated",
    "decision_call",
    "memo_call",
]

_ids = itertools.count(1)
_ids_lock = threading.Lock()


def _next_id(prefix: str) -> str:
    with _ids_lock:
        return f"{prefix}_fake_{next(_ids):06d}"


# ------------------------------------------------------------------------ SDK-shaped types
@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"

    def to_api_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass
class FakeThinkingBlock:
    thinking: str = ""
    signature: str = "fake-signature"
    type: str = "thinking"

    def to_api_dict(self) -> dict[str, Any]:
        return {"type": "thinking", "thinking": self.thinking, "signature": self.signature}


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _next_id("toolu"))
    type: str = "tool_use"

    def to_api_dict(self) -> dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": copy.deepcopy(self.input)}


@dataclass
class FakeFallbackInfo:
    model: str


@dataclass
class FakeFallbackBlock:
    from_: FakeFallbackInfo
    to: FakeFallbackInfo
    type: str = "fallback"

    def to_api_dict(self) -> dict[str, Any]:
        return {"type": "fallback", "from": {"model": self.from_.model}, "to": {"model": self.to.model},
                "trigger": {"type": "refusal"}}


@dataclass
class FakeIterationUsage:
    type: str  # "message" | "fallback_message"
    input_tokens: int
    output_tokens: int
    model: str | None = None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeUsage:
    input_tokens: int = 1000
    output_tokens: int = 200
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    iterations: list[FakeIterationUsage] | None = None

    def to_api_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }
        if self.iterations is not None:
            d["iterations"] = [
                {k: v for k, v in it.__dict__.items() if v is not None} for it in self.iterations
            ]
        return d


@dataclass
class FakeStopDetails:
    category: str | None = None
    explanation: str | None = None
    type: str = "refusal"


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "claude-opus-5"
    stop_details: FakeStopDetails | None = None
    id: str = field(default_factory=lambda: _next_id("msg"))
    role: str = "assistant"
    type: str = "message"
    stop_sequence: str | None = None

    def to_api_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [b.to_api_dict() for b in self.content],
            "stop_reason": self.stop_reason,
            "stop_sequence": self.stop_sequence,
            "usage": self.usage.to_api_dict(),
        }
        if self.stop_details is not None:
            d["stop_details"] = {"type": "refusal", "category": self.stop_details.category,
                                 "explanation": self.stop_details.explanation}
        return d


# ------------------------------------------------------------------------ builders
def text(s: str) -> FakeTextBlock:
    return FakeTextBlock(text=s)


def thinking(s: str = "") -> FakeThinkingBlock:
    return FakeThinkingBlock(thinking=s)


def tool_use(name: str, input: Mapping[str, Any] | None = None, *, id: str | None = None) -> FakeToolUseBlock:
    block = FakeToolUseBlock(name=name, input=dict(input or {}))
    if id is not None:
        block.id = id
    return block


def message(*blocks: Any, stop_reason: str | None = None, usage: FakeUsage | None = None,
            model: str = "claude-opus-5", with_thinking: bool = True) -> FakeMessage:
    """Assistant message; ``stop_reason`` inferred (tool_use if any tool call else end_turn).

    A leading (empty, signed) thinking block is added by default, as Opus 5 returns with
    adaptive thinking and ``display`` omitted — it exercises pass-through of thinking blocks.
    """
    content = list(blocks)
    if with_thinking:
        content.insert(0, FakeThinkingBlock())
    if stop_reason is None:
        stop_reason = "tool_use" if any(getattr(b, "type", None) == "tool_use" for b in content) else "end_turn"
    return FakeMessage(content=content, stop_reason=stop_reason, usage=usage or FakeUsage(), model=model)


def refusal(category: str | None = "cyber", explanation: str | None = None, *,
            usage: FakeUsage | None = None) -> FakeMessage:
    """A classifier decline before any output (empty content)."""
    return FakeMessage(content=[], stop_reason="refusal",
                       usage=usage or FakeUsage(input_tokens=0, output_tokens=0),
                       stop_details=FakeStopDetails(category=category, explanation=explanation))


def truncated(*blocks: Any, usage: FakeUsage | None = None) -> FakeMessage:
    """A response cut off at ``max_tokens``."""
    return FakeMessage(content=list(blocks), stop_reason="max_tokens", usage=usage or FakeUsage())


def decision_call(*, action: str = "follow_quant", scale: float = 1.0, forecast: float = 0.0,
                  confidence: float = 0.6, horizon_bars: int = 24,
                  rationale: str = "Evidence supports the quant forecast.",
                  key_risks: Sequence[str] = ("event risk",), dissent: str = "",
                  **overrides: Any) -> FakeToolUseBlock:
    payload = {"action": action, "scale": scale, "forecast": forecast, "confidence": confidence,
               "horizon_bars": horizon_bars, "rationale": rationale, "key_risks": list(key_risks),
               "dissent": dissent}
    payload.update(overrides)
    return tool_use("submit_decision", payload)


def memo_call(*, stance: str = "neutral", confidence: float = 0.5,
              key_points: Sequence[str] = ("finding backed by tool data",),
              risks: Sequence[str] = ("data gap",), suggested_exposure: float = 0.0,
              **overrides: Any) -> FakeToolUseBlock:
    payload = {"stance": stance, "confidence": confidence, "key_points": list(key_points),
               "risks": list(risks), "suggested_exposure": suggested_exposure}
    payload.update(overrides)
    return tool_use("submit_memo", payload)


# ------------------------------------------------------------------------ the client
class FakeRequestError(ValueError):
    """The desk sent a request the real API would reject."""


class FakeScriptExhausted(RuntimeError):
    """No scripted response left for an agent."""


def _first_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    for b in content or []:
        t = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        if t == "text":
            return b.get("text") if isinstance(b, dict) else b.text
    return ""


def default_router(kwargs: Mapping[str, Any]) -> str:
    """Agent key from the ``Agent: <id>`` header of the first user message."""
    msgs = kwargs.get("messages") or []
    if not msgs:
        return "unknown"
    first = _first_text(msgs[0].get("content"))
    line = first.splitlines()[0] if first else ""
    return line.split(":", 1)[1].strip() if line.startswith("Agent:") else "unknown"


def _bget(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _check_schema_strict(schema: Any, path: str, problems: list[str]) -> None:
    if not isinstance(schema, Mapping):
        return
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: objects must set additionalProperties=false in strict mode")
        for k, sub in (schema.get("properties") or {}).items():
            _check_schema_strict(sub, f"{path}.{k}", problems)
    if schema.get("type") == "array":
        _check_schema_strict(schema.get("items"), f"{path}[]", problems)
    for bad in ("minimum", "maximum", "minLength", "maxLength", "multipleOf", "minItems", "maxItems"):
        if bad in schema:
            problems.append(f"{path}: '{bad}' is not supported in strict schemas")


def validate_request(kwargs: Mapping[str, Any]) -> list[str]:
    """Check a request against the Messages API rules the desk relies on."""
    problems: list[str] = []
    for key in ("model", "max_tokens", "messages"):
        if key not in kwargs:
            problems.append(f"missing '{key}'")
    msgs = kwargs.get("messages") or []
    if msgs and msgs[0].get("role") != "user":
        problems.append("first message must be from the user")
    if msgs and msgs[-1].get("role") == "assistant":
        # Last-assistant-turn prefill: 400 on Opus 4.6+ (incl. Opus 5). The one exception is
        # resuming a pause_turn, where the turn ends in a server-tool block.
        last = msgs[-1].get("content")
        last_type = _bget(last[-1], "type") if isinstance(last, list) and last else None
        if last_type != "server_tool_use":
            problems.append("request ends with an assistant turn (prefill is not supported: 400)")
    for i, m in enumerate(msgs):
        if m.get("role") not in ("user", "assistant"):
            problems.append(f"messages[{i}] has invalid role {m.get('role')!r}")
        content = m.get("content")
        if isinstance(content, list) and not content:
            problems.append(f"messages[{i}] has empty content")
        if m.get("role") != "assistant" or not isinstance(content, list):
            continue
        ids = [_bget(b, "id") for b in content if _bget(b, "type") == "tool_use"]
        if not ids:
            continue
        if i + 1 >= len(msgs) or msgs[i + 1].get("role") != "user":
            problems.append(f"messages[{i}] tool_use must be followed by a user message with tool_results")
            continue
        nxt = msgs[i + 1].get("content")
        nxt = nxt if isinstance(nxt, list) else []
        types = [_bget(b, "type") for b in nxt]
        n_results = 0
        while n_results < len(types) and types[n_results] == "tool_result":
            n_results += 1
        if "tool_result" in types[n_results:]:
            problems.append(f"messages[{i + 1}]: tool_result blocks must come before other content")
        result_ids = [_bget(b, "tool_use_id") for b in nxt[:n_results]]
        if sorted(result_ids) != sorted(ids):
            problems.append(f"messages[{i + 1}]: tool_result ids {result_ids} do not match tool_use ids {ids}")
    tools = kwargs.get("tools") or []
    names = [t.get("name") for t in tools]
    if len(set(names)) != len(names):
        problems.append("duplicate tool names")
    for t in tools:
        if t.get("strict") is not True:
            problems.append(f"tool {t.get('name')}: expected strict=true")
        _check_schema_strict(t.get("input_schema"), f"tool {t.get('name')}", problems)
    tc = kwargs.get("tool_choice")
    if tc is not None and tc.get("type") != "auto":
        problems.append(f"tool_choice must be auto, got {tc}")
    if "fallbacks" in kwargs:
        if kwargs["fallbacks"] == "default" and BETA_SERVER_SIDE_FALLBACK not in (kwargs.get("betas") or []):
            problems.append(f"fallbacks='default' requires beta {BETA_SERVER_SIDE_FALLBACK}")
    oc = kwargs.get("output_config") or {}
    if "effort" in oc and oc["effort"] not in EFFORT_LEVELS:
        problems.append(f"invalid effort {oc['effort']!r}")
    th = kwargs.get("thinking")
    if th is not None and th.get("type") == "disabled" and oc.get("effort") in ("xhigh", "max"):
        problems.append("thinking disabled is not allowed above effort 'high'")
    return problems


@dataclass
class FakeCall:
    agent: str
    kwargs: dict[str, Any]
    thread: str


class _FakeMessages:
    def __init__(self, owner: FakeAnthropicClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(**kwargs)


class _FakeBeta:
    def __init__(self, owner: FakeAnthropicClient) -> None:
        self.messages = _FakeMessages(owner)


class FakeAnthropicClient:
    """Scripted, thread-safe stand-in for ``anthropic.Anthropic``.

    Parameters
    ----------
    scripts  : ``{agent_key: [FakeMessage | callable(kwargs) -> FakeMessage, ...]}``.
    router   : maps request kwargs to an agent key (default: :func:`default_router`).
    default  : optional factory ``kwargs -> FakeMessage`` used when a script is exhausted.
    validate : check each request with :func:`validate_request`.
    """

    def __init__(
        self,
        scripts: Mapping[str, Sequence[FakeMessage | Callable[[dict[str, Any]], FakeMessage]]] | None = None,
        *,
        router: Callable[[Mapping[str, Any]], str] = default_router,
        default: Callable[[dict[str, Any]], FakeMessage] | None = None,
        validate: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._queues: dict[str, deque[Any]] = {k: deque(v) for k, v in (scripts or {}).items()}
        self.router = router
        self.default = default
        self.validate = validate
        self.calls: list[FakeCall] = []
        self.errors: list[str] = []
        self.beta = _FakeBeta(self)
        self.messages = _FakeMessages(self)

    def add(self, key: str, *responses: Any) -> None:
        with self._lock:
            self._queues.setdefault(key, deque()).extend(responses)

    def calls_for(self, key: str) -> list[FakeCall]:
        with self._lock:
            return [c for c in self.calls if c.agent == key]

    def remaining(self, key: str) -> int:
        with self._lock:
            return len(self._queues.get(key, ()))

    def _create(self, **kwargs: Any) -> Any:
        snapshot = copy.deepcopy(kwargs)  # the caller keeps mutating its message list
        key = self.router(snapshot)
        with self._lock:
            self.calls.append(FakeCall(agent=key, kwargs=snapshot, thread=threading.current_thread().name))
        if self.validate:
            problems = validate_request(snapshot)
            if problems:
                with self._lock:
                    self.errors.extend(f"{key}: {p}" for p in problems)
                raise FakeRequestError(f"{key}: " + "; ".join(problems))
        with self._lock:
            queue = self._queues.get(key)
            if not queue and "#" in key:
                queue = self._queues.get(key.split("#", 1)[0])
            item = queue.popleft() if queue else None
        if item is None:
            if self.default is not None:
                return self.default(snapshot)
            raise FakeScriptExhausted(f"no scripted response left for agent {key!r}")
        return item(snapshot) if callable(item) else item
