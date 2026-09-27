"""Token usage accounting and per-cycle budget enforcement.

Anthropic ``usage`` semantics (per response):

* ``input_tokens`` is the *uncached* remainder; ``cache_creation_input_tokens`` were written
  to cache (1.25x input price at 5-minute TTL, 2x at 1-hour) and ``cache_read_input_tokens``
  were served from cache (~0.1x). Total prompt = the sum of the three.
* With server-side fallbacks, top-level ``usage`` covers only the attempt that produced the
  returned message, while ``usage.iterations`` lists every attempt (``message`` /
  ``fallback_message`` entries, each with its own ``model``). When present, iterations are
  the source of truth and each entry is priced at its own model's rates.
* Declined-before-output attempts are reported but NOT billed (docs: model migration guide,
  "refusal stop reason" / server-side fallbacks). An attempt is treated as declined before
  output when it produced zero output tokens and either a later attempt followed it (the
  fallback ran) or the response's final ``stop_reason`` is ``refusal`` (the whole chain
  declined before output). Such tokens are reported in ``unbilled_input_tokens`` and are
  excluded from cost and from the token budget. Pricing them would double-count the full
  prompt — typically the largest term, the Chief's whole context — on every fallback.
  Mid-output declines (output tokens > 0) stay billed in full, which errs on the high side.

The ledger is shared by concurrently running agents, hence the lock. Budget checks happen
*before* each API call, so the cap can be overshot by at most the calls already in flight.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from aurum.agents.client import block_get
from aurum.agents.config import CACHE_WRITE_MULTIPLIER, DEFAULT_PRICES, ModelPrice

logger = logging.getLogger(__name__)

__all__ = ["UsageTotals", "UsageLedger", "estimate_response_cost"]

_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


@dataclass
class UsageTotals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    unbilled_input_tokens: int = 0  # prompt tokens of attempts declined before output

    @property
    def total_tokens(self) -> int:
        """Billed tokens (the quantity the token budget caps)."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def add(self, other: UsageTotals) -> None:
        self.calls += other.calls
        for f in _TOKEN_FIELDS:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.cost_usd += other.cost_usd
        self.unbilled_input_tokens += other.unbilled_input_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["cost_usd"] = round(self.cost_usd, 6)
        return d


def _int(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


_WARNED_MODELS: set[str] = set()


def _price_for(model: str | None, prices: Mapping[str, ModelPrice]) -> ModelPrice:
    if model and model in prices:
        return prices[model]
    # Unknown model: price conservatively at the most expensive known rate.
    worst = max(prices.values(), key=lambda p: (p.output_per_mtok, p.input_per_mtok))
    key = str(model)
    if key not in _WARNED_MODELS:
        _WARNED_MODELS.add(key)
        logger.warning("no price for model %r; using the most expensive known rate for the estimate", model)
    return worst


def _write_multiplier(usage_like: Any, default_ttl: str) -> float:
    """Blend 5m/1h write multipliers from ``cache_creation`` breakdown when available."""
    breakdown = block_get(usage_like, "cache_creation")
    if breakdown is not None:
        t5 = _int(block_get(breakdown, "ephemeral_5m_input_tokens"))
        t1 = _int(block_get(breakdown, "ephemeral_1h_input_tokens"))
        if t5 + t1 > 0:
            return (t5 * CACHE_WRITE_MULTIPLIER["5m"] + t1 * CACHE_WRITE_MULTIPLIER["1h"]) / (t5 + t1)
    return CACHE_WRITE_MULTIPLIER[default_ttl]


def estimate_response_cost(
    response: Any,
    *,
    requested_model: str,
    prices: Mapping[str, ModelPrice] = DEFAULT_PRICES,
    cache_ttl: str = "5m",
) -> UsageTotals:
    """Token counts and estimated USD cost of one API response."""
    usage = getattr(response, "usage", None)
    out = UsageTotals(calls=1)
    if usage is None:
        return out
    iterations = [
        it for it in (block_get(usage, "iterations") or [])
        if block_get(it, "input_tokens") is not None
    ]
    entries: list[tuple[Any, str | None]]
    if iterations:
        entries = [(it, block_get(it, "model") or getattr(response, "model", None) or requested_model)
                   for it in iterations]
    else:
        entries = [(usage, getattr(response, "model", None) or requested_model)]
    final_refusal = getattr(response, "stop_reason", None) == "refusal"
    for i, (entry, model) in enumerate(entries):
        counts = {f: _int(block_get(entry, f)) for f in _TOKEN_FIELDS}
        followed_by_retry = i < len(entries) - 1
        if counts["output_tokens"] == 0 and (followed_by_retry or final_refusal):
            # declined before output: reported, not billed
            out.unbilled_input_tokens += (counts["input_tokens"] + counts["cache_creation_input_tokens"]
                                          + counts["cache_read_input_tokens"])
            continue
        price = _price_for(str(model) if model else None, prices)
        out.cost_usd += price.cost(cache_write_multiplier=_write_multiplier(entry, cache_ttl), **counts)
        for f, v in counts.items():
            setattr(out, f, getattr(out, f) + v)
    return out


class UsageLedger:
    """Thread-safe accumulation of usage per agent with budget checks."""

    def __init__(
        self,
        *,
        prices: Mapping[str, ModelPrice] = DEFAULT_PRICES,
        max_cost_usd: float | None = None,
        max_tokens: int | None = None,
        cache_ttl: str = "5m",
    ) -> None:
        self._prices = dict(prices)
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._total = UsageTotals()
        self._by_agent: dict[str, UsageTotals] = {}

    def record(self, agent_id: str, response: Any, *, requested_model: str) -> UsageTotals:
        u = estimate_response_cost(
            response, requested_model=requested_model, prices=self._prices, cache_ttl=self.cache_ttl
        )
        with self._lock:
            self._total.add(u)
            self._by_agent.setdefault(agent_id, UsageTotals()).add(u)
        return u

    def totals(self) -> UsageTotals:
        with self._lock:
            t = UsageTotals()
            t.add(self._total)
            return t

    def exceeded(self, share: float = 1.0) -> str | None:
        """Reason string if ``share`` of a hard budget is exhausted, else ``None``.

        ``share < 1`` lets a caller stop some agents early so the rest of the budget stays
        available to others (the desk reserves the tail of each cycle's budget for the Chief).
        """
        t = self.totals()
        if self.max_cost_usd is not None and t.cost_usd >= share * self.max_cost_usd:
            cap = share * self.max_cost_usd
            scope = "cost budget" if share >= 1.0 else f"{share:.0%} share of the cost budget"
            return f"{scope} exhausted (${t.cost_usd:.4f} >= ${cap:.4f})"
        if self.max_tokens is not None and t.total_tokens >= share * self.max_tokens:
            cap_t = share * self.max_tokens
            scope = "token budget" if share >= 1.0 else f"{share:.0%} share of the token budget"
            return f"{scope} exhausted ({t.total_tokens} >= {cap_t:.0f})"
        return None

    def fraction_used(self) -> float:
        t = self.totals()
        fracs = [0.0]
        if self.max_cost_usd:
            fracs.append(t.cost_usd / self.max_cost_usd)
        if self.max_tokens:
            fracs.append(t.total_tokens / self.max_tokens)
        return max(fracs)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_agent = {k: v.to_dict() for k, v in sorted(self._by_agent.items())}
            total = self._total.to_dict()
        return {
            "total": total,
            "by_agent": by_agent,
            "budget": {"max_cost_usd": self.max_cost_usd, "max_tokens": self.max_tokens},
        }
