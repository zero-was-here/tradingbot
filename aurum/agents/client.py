"""Thin wrapper around the Claude Messages API (``anthropic`` Python SDK, v1.x).

Every agent call goes through :meth:`LLMClient.create`, which issues
``client.beta.messages.create(...)`` — the beta surface is required for server-side
refusal fallbacks (``betas=["server-side-fallback-2026-07-01"], fallbacks="default"``).
The same call path is exposed by :class:`aurum.agents.testing.FakeAnthropicClient`, so the
whole desk runs offline in tests.

Request shape (see the Claude API docs on tool use and prompt caching):

* ``system`` is a single text block carrying an explicit ``cache_control`` breakpoint. The
  system prompt and tool list are byte-stable across cycles (no timestamps, sorted tool
  list), so ``tools + system`` is served from cache after the first call.
* Top-level ``cache_control`` enables automatic caching of the growing conversation tail
  (the recommended "explicit static prefix + automatic tail" combination for agent loops).
* ``tool_choice={"type": "auto"}`` — never forced; the prompts instruct the agent to finish
  with its terminal tool and the loop nudges when it does not (``auto`` does not guarantee
  a call).
* ``thinking`` / ``output_config.effort`` come from :class:`AgentModelConfig` and are pinned
  per role (changing them per request would invalidate the messages cache).

``anthropic`` is an optional dependency: it is imported lazily by
:func:`make_default_client` only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from aurum.agents.config import BETA_SERVER_SIDE_FALLBACK, AgentModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "LLMClient",
    "make_default_client",
    "build_request",
    "block_get",
    "block_to_dict",
    "sanitize_assistant_content",
    "fallback_events",
]


def make_default_client(*, timeout: float | None = 600.0, max_retries: int = 2) -> Any:
    """Construct ``anthropic.Anthropic()`` (credentials resolved from the environment).

    The SDK already retries connection errors, 408/409/429 and 5xx with exponential
    backoff; ``max_retries`` tunes that. Raises ``ImportError`` with an actionable message
    if the optional dependency is missing.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The LLM desk needs the 'anthropic' package: pip install 'aurum[agents]'"
        ) from exc
    kwargs: dict[str, Any] = {"max_retries": max_retries}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return anthropic.Anthropic(**kwargs)


# --------------------------------------------------------------------------------------
# content-block helpers (work on SDK pydantic objects, dicts and the test doubles alike)
# --------------------------------------------------------------------------------------
def block_get(block: Any, name: str, default: Any = None) -> Any:
    """Read attribute ``name`` from an SDK block object or a plain dict."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def block_to_dict(block: Any, *, max_chars: int | None = None) -> dict[str, Any]:
    """JSON-friendly view of a content block for the journal (never sent to the API).

    Thinking signatures are dropped (opaque, large); long strings are truncated.
    """
    from aurum.agents._json import to_jsonable, truncate

    if isinstance(block, dict):
        raw = dict(block)
    elif hasattr(block, "model_dump"):
        raw = block.model_dump(mode="json", by_alias=True, exclude_none=True)
    elif hasattr(block, "__dict__"):
        raw = {k: v for k, v in vars(block).items() if not k.startswith("_") and v is not None}
    else:
        raw = {"repr": repr(block)}
    if "from_" in raw:
        raw["from"] = raw.pop("from_")
    raw.pop("signature", None)
    raw.pop("data", None)  # redacted_thinking payload
    out = to_jsonable(raw)
    for key in ("text", "thinking"):
        if isinstance(out.get(key), str):
            out[key] = truncate(out[key], max_chars)
    return out


def sanitize_assistant_content(content: Sequence[Any]) -> list[Any]:
    """Prepare a response's content for echoing back as the next assistant turn.

    Server-side fallback semantics: after a mid-output fallback, ``thinking``,
    ``redacted_thinking``, ``tool_use`` and any other model-internal blocks that appear
    *before* the final ``fallback`` block must be omitted; ``text`` blocks before it and
    everything after it echo normally. The ``fallback`` marker itself is an audit block and
    is dropped here. Without a fallback block the content is returned unchanged (same
    objects — thinking blocks must be passed back unmodified).
    """
    blocks = list(content or [])
    idx = [i for i, b in enumerate(blocks) if block_get(b, "type") == "fallback"]
    if not idx:
        return blocks
    boundary = idx[-1]
    before = [b for b in blocks[:boundary] if block_get(b, "type") == "text"]
    after = [b for b in blocks[boundary + 1:] if block_get(b, "type") != "fallback"]
    return before + after


def fallback_events(response: Any) -> dict[str, Any] | None:
    """Summarise server-side fallback activity on a response (``None`` if none ran).

    ``fallback`` content blocks mark switch points; the served-by signal is a
    ``fallback_message`` entry in ``usage.iterations`` (sticky-routed turns carry no block).
    """
    hops = []
    for b in getattr(response, "content", None) or []:
        if block_get(b, "type") == "fallback":
            src = block_get(b, "from_") or block_get(b, "from")
            dst = block_get(b, "to")
            hops.append({"from": block_get(src, "model"), "to": block_get(dst, "model")})
    usage = getattr(response, "usage", None)
    iterations = block_get(usage, "iterations") if usage is not None else None
    served_by_fallback = any(block_get(it, "type") == "fallback_message" for it in iterations or [])
    if not hops and not served_by_fallback:
        return None
    return {
        "hops": hops,
        "served_by_fallback": served_by_fallback,
        "served_by": getattr(response, "model", None),
    }


# --------------------------------------------------------------------------------------
# request construction
# --------------------------------------------------------------------------------------
def build_request(
    *,
    cfg: AgentModelConfig,
    system_prompt: str,
    tools: Sequence[dict[str, Any]],
    messages: list[dict[str, Any]],
    prompt_caching: bool = True,
    cache_ttl: str = "5m",
) -> dict[str, Any]:
    """Keyword arguments for ``client.beta.messages.create``."""
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        # Longer TTLs must precede shorter ones: use 1h for both the explicit marker and
        # the automatic tail breakpoint.
        cache_control["ttl"] = "1h"
    system_block: dict[str, Any] = {"type": "text", "text": system_prompt}
    if prompt_caching:
        system_block["cache_control"] = dict(cache_control)
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": [system_block],
        "tools": list(tools),
        "tool_choice": {"type": "auto"},
        "messages": messages,
    }
    if prompt_caching:
        kwargs["cache_control"] = dict(cache_control)
    if cfg.thinking is not None:
        kwargs["thinking"] = dict(cfg.thinking)
    if cfg.effort is not None:
        kwargs["output_config"] = {"effort": cfg.effort}
    if cfg.fallbacks:
        kwargs["betas"] = [BETA_SERVER_SIDE_FALLBACK]
        kwargs["fallbacks"] = "default"
    return kwargs


class LLMClient:
    """Issues Messages API calls for the desk's agents.

    ``client`` is anything exposing ``.beta.messages.create(**kwargs)`` — an
    ``anthropic.Anthropic`` instance (thread-safe; shared by concurrent specialists) or the
    offline :class:`~aurum.agents.testing.FakeAnthropicClient`.
    """

    def __init__(self, client: Any | None = None, *, prompt_caching: bool = True, cache_ttl: str = "5m",
                 timeout: float | None = 600.0) -> None:
        self._client = client if client is not None else make_default_client(timeout=timeout)
        self.prompt_caching = prompt_caching
        self.cache_ttl = cache_ttl

    @property
    def raw(self) -> Any:
        return self._client

    def create(
        self,
        *,
        cfg: AgentModelConfig,
        system_prompt: str,
        tools: Sequence[dict[str, Any]],
        messages: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> Any:
        """Issue one request.

        ``timeout`` (seconds) is the SDK's per-request transport timeout, used to bound an
        in-flight call by the cycle deadline. It is not part of the request body (and an
        explicit per-request timeout also lifts the SDK's non-streaming duration guard,
        which is fine because it is never above the desk's ``request_timeout_s``).
        """
        kwargs = build_request(
            cfg=cfg,
            system_prompt=system_prompt,
            tools=tools,
            messages=messages,
            prompt_caching=self.prompt_caching,
            cache_ttl=self.cache_ttl,
        )
        if timeout is not None:
            kwargs["timeout"] = float(timeout)
        return self._client.beta.messages.create(**kwargs)
