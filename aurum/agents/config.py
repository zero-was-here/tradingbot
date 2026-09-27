"""Configuration for the LLM trading desk: per-role model settings, caps and prices.

Defaults follow the Claude API guidance for Claude Opus 5 (the model this desk targets):

* ``thinking={"type": "adaptive"}`` — the model decides how much to reason per turn and
  interleaves reasoning between tool calls. On Opus 5 this is also the API default.
* ``output_config={"effort": ...}`` — the thoroughness/cost lever. ``high`` for the Chief
  (it makes the final call), ``medium`` for specialists (narrow, data-driven memos;
  ``low``/``medium`` are unusually strong on Opus 5 and are the primary cost lever).
* ``max_tokens=16000`` — a hard cap on thinking *plus* visible output per response. Requests
  are non-streaming; the Python SDK rejects non-streaming calls whose ``max_tokens`` implies
  more than 10 minutes (about 21k tokens) unless the client has a non-default timeout, so
  keep ``max_tokens`` below that or construct the client with an explicit ``timeout``.
* Server-side refusal fallbacks (``fallbacks="default"`` behind the
  ``server-side-fallback-2026-07-01`` beta): if Opus 5's safety classifiers decline a
  request, the API re-runs it on Anthropic's recommended fallback model inside the same
  call instead of returning a refusal. A final ``stop_reason == "refusal"`` then means the
  whole chain declined.

Costs: :data:`DEFAULT_PRICES` is a plain dict (USD per million tokens) so it can be
overridden from config. Cache writes bill at 1.25x input (5-minute TTL) or 2x (1-hour
TTL); cache reads at ``cache_read_multiplier`` x input.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BETA_SERVER_SIDE_FALLBACK",
    "DEFAULT_MODEL",
    "EFFORT_LEVELS",
    "ModelPrice",
    "DEFAULT_PRICES",
    "AgentModelConfig",
    "DeskConfig",
]

#: Beta header that gates the scalar ``fallbacks="default"`` form (the array form uses a
#: different, older header — pairing a header with the other form is a 400).
BETA_SERVER_SIDE_FALLBACK = "server-side-fallback-2026-07-01"

DEFAULT_MODEL = "claude-opus-5"

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Cache-write multipliers relative to the base input price, by TTL.
CACHE_WRITE_MULTIPLIER = {"5m": 1.25, "1h": 2.0}


@dataclass(frozen=True)
class ModelPrice:
    """List prices in USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float = 0.1

    def cost(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_write_multiplier: float = 1.25,
    ) -> float:
        """USD cost of one sampling iteration.

        ``input_tokens`` is the *uncached* remainder only; total prompt size is
        ``input + cache_creation + cache_read`` (Anthropic usage semantics).
        """
        p_in = self.input_per_mtok / 1e6
        p_out = self.output_per_mtok / 1e6
        return (
            input_tokens * p_in
            + cache_creation_input_tokens * p_in * cache_write_multiplier
            + cache_read_input_tokens * p_in * self.cache_read_multiplier
            + output_tokens * p_out
        )


#: USD per MTok (input, output). Opus 5 is $5/$25; Opus 4.8 (the default refusal fallback
#: for cyber-category declines) has the same list price. Override via ``DeskConfig.prices``.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(5.0, 25.0),
    "claude-opus-4-8": ModelPrice(5.0, 25.0),
    "claude-opus-5-5": ModelPrice(4.0, 20.0, cache_read_multiplier=0.05),
    "claude-fable-5-1": ModelPrice(10.0, 50.0, cache_read_multiplier=0.025),
    "claude-fable-5": ModelPrice(10.0, 50.0),
    "claude-sonnet-5": ModelPrice(2.0, 10.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
}


@dataclass(frozen=True)
class AgentModelConfig:
    """Model settings for one agent role.

    ``thinking=None`` omits the parameter (on Opus 5 that still means adaptive thinking).
    ``effort=None`` omits ``output_config`` (API default, ``high`` on Opus 5).
    """

    model: str = DEFAULT_MODEL
    effort: str | None = "medium"
    max_tokens: int = 16000
    thinking: Mapping[str, Any] | None = field(default_factory=lambda: {"type": "adaptive"})
    fallbacks: bool = True
    max_turns: int = 6

    def __post_init__(self) -> None:
        if self.effort is not None and self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS} or None, got {self.effort!r}")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.thinking is not None:
            kind = dict(self.thinking).get("type")
            if kind not in ("adaptive", "disabled", "enabled"):
                raise ValueError(f"unsupported thinking config {self.thinking!r}")
            # Claude Opus 5 rejects disabled thinking above `high` effort (400).
            if kind == "disabled" and self.effort in ("xhigh", "max"):
                raise ValueError("thinking disabled is only valid with effort <= 'high'")

    def replace(self, **changes: Any) -> AgentModelConfig:
        return dataclasses.replace(self, **changes)


def _chief_default() -> AgentModelConfig:
    return AgentModelConfig(effort="high", max_turns=8)


def _specialist_default() -> AgentModelConfig:
    return AgentModelConfig(effort="medium", max_turns=5)


@dataclass(frozen=True)
class DeskConfig:
    """Caps, budgets and model settings for one :class:`~aurum.agents.desk.TradingDesk`.

    role_models          : per-role overrides keyed by role name (``"macro_strategist"``,
                           ``"quant_analyst"``, ``"risk_officer"``, ``"execution_trader"``
                           or ``"adhoc"`` for Chief-created agents).
    max_specialists_per_cycle : deterministic cap on consulted + created agents per cycle.
                           Opus 5 delegates readily; a hard ceiling is the reliable lever.
    max_parallel_agents  : thread-pool width for concurrent specialist loops.
    max_cost_usd_per_cycle / max_tokens_per_cycle : budget checked before every API call
                           (best effort under concurrency: in-flight calls can overshoot
                           by at most their own cost). ``None`` disables.
    soft_budget_fraction : when crossed, agents are told to conclude.
    chief_budget_reserve : share of the hard budget (cost and tokens) reserved for the Chief:
                           specialists may not start an API call once usage reaches
                           ``1 - chief_budget_reserve`` of the cap. Without it, a few parallel
                           specialists can exhaust the budget and the Chief can never submit
                           its decision — the cycle then falls back to ``on_failure`` exactly
                           when the desk had the most to say. ``0`` disables the reserve.
    max_cycle_seconds    : wall-clock deadline for a cycle (live trading must decide before
                           the next bar). ``None`` disables.
    cache_ttl            : ``"5m"`` (cheapest when calls are < 5 min apart — true within a
                           cycle) or ``"1h"`` (keeps the static prefix warm across H1 cycles).
    tool_result_max_chars: truncation of data returned to the model.
    journal_max_chars    : truncation of long strings in the journal.
    adhoc_tool_whitelist : data tools the Chief may grant to agents it creates.
    skip_llm_when_outcome_fixed : in ``overlay`` mode with a flat quant forecast (q == 0) the
                           final forecast is 0 whatever the desk decides (the overlay interval
                           is {0}); the cycle is then skipped without any API call
                           (``DeskResult.status == "skipped"``). Advisory cycles always run,
                           since recording the desk's calls is their purpose.
    """

    chief: AgentModelConfig = field(default_factory=_chief_default)
    specialist: AgentModelConfig = field(default_factory=_specialist_default)
    role_models: Mapping[str, AgentModelConfig] = field(default_factory=dict)
    max_specialists_per_cycle: int = 6
    max_parallel_agents: int = 4
    max_cost_usd_per_cycle: float | None = 3.0
    max_tokens_per_cycle: int | None = None
    soft_budget_fraction: float = 0.8
    chief_budget_reserve: float = 0.2
    max_cycle_seconds: float | None = None
    prompt_caching: bool = True
    cache_ttl: str = "5m"
    request_timeout_s: float = 600.0
    tool_result_max_chars: int = 12_000
    journal_max_chars: int = 4_000
    max_text_field_chars: int = 2_000
    adhoc_tool_whitelist: tuple[str, ...] | None = None  # None -> all data tools
    prices: Mapping[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    skip_llm_when_outcome_fixed: bool = True

    def __post_init__(self) -> None:
        if self.max_specialists_per_cycle < 0:
            raise ValueError("max_specialists_per_cycle must be >= 0")
        if self.max_parallel_agents < 1:
            raise ValueError("max_parallel_agents must be >= 1")
        if self.cache_ttl not in CACHE_WRITE_MULTIPLIER:
            raise ValueError(f"cache_ttl must be one of {sorted(CACHE_WRITE_MULTIPLIER)}")
        if not 0.0 < self.soft_budget_fraction <= 1.0:
            raise ValueError("soft_budget_fraction must be in (0, 1]")
        if not 0.0 <= self.chief_budget_reserve < 1.0:
            raise ValueError("chief_budget_reserve must be in [0, 1)")
        if not self.prices:
            raise ValueError("prices must not be empty (cost estimates and budgets need a price table)")
        if self.max_cost_usd_per_cycle is not None and self.max_cost_usd_per_cycle <= 0:
            raise ValueError("max_cost_usd_per_cycle must be positive or None")
        if self.max_tokens_per_cycle is not None and self.max_tokens_per_cycle <= 0:
            raise ValueError("max_tokens_per_cycle must be positive or None")
        if self.adhoc_tool_whitelist is not None:
            from aurum.agents.tools import DATA_TOOL_NAMES

            unknown = sorted(set(self.adhoc_tool_whitelist) - set(DATA_TOOL_NAMES))
            if unknown:
                raise ValueError(f"adhoc_tool_whitelist has unknown data tools {unknown}")

    def model_for_role(self, role: str) -> AgentModelConfig:
        """Model config for a specialist role (falls back to the specialist default)."""
        if role == "chief":
            return self.role_models.get("chief", self.chief)
        return self.role_models.get(role, self.specialist)

    @property
    def specialist_budget_share(self) -> float:
        """Fraction of the cycle budget specialists may consume (see ``chief_budget_reserve``)."""
        return 1.0 - self.chief_budget_reserve

    @property
    def cache_write_multiplier(self) -> float:
        return CACHE_WRITE_MULTIPLIER[self.cache_ttl]
