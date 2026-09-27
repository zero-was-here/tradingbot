"""TradingDesk: runs one LLM decision cycle per bar close and returns a policy-gated forecast.

Flow of one cycle (:meth:`TradingDesk.run_cycle`)::

    quant forecast q ──► Chief (CIO) loop ──► data tools / specialists (parallel) ──► submit_decision
                                                                                          │
    final forecast ◄── DecisionPolicy(mode) ◄──────────────────────────────────────────────┘
          │
          └─► caller: VolTargetSizer.target_lots(...) ─► StandardRiskManager.evaluate(...) ─► orders

The desk returns a FORECAST, never lots or orders. The caller must pass it through the same
sizer and risk manager as the quant book (backtest engine, paper and live runners do), so
the LLM cannot bypass risk limits (SPEC §0.5). In the default ``overlay`` mode the desk can
only scale the quant forecast toward zero or veto it.

A cycle never raises for model-side failures (refusal, max turns, budget, deadline, API or
network errors): it returns ``DecisionPolicy.on_failure`` (default: follow the quant
forecast) with ``status="failed"`` and the reason, and journals everything.

Demo (no API key needed — uses the scripted offline client)::

    from aurum.agents.desk import demo
    result = demo()                       # synthetic data, anonymised, scripted agents
    print(result.status, result.final_forecast, result.decision.rationale)
    print(result.usage["total"]["cost_usd"], result.journal_path)

Live model (requires ``pip install 'aurum[agents]'`` and ``ANTHROPIC_API_KEY``)::

    import anthropic, pandas as pd
    from aurum.agents import TradingDesk, StaticDeskDataProvider, DecisionPolicy
    provider = StaticDeskDataProvider({"market": {...}, "quant_signals": {...}, "risk": {...}})
    desk = TradingDesk(provider, client=anthropic.Anthropic(), policy=DecisionPolicy("overlay"),
                       journal_dir="runs/desk_journal")
    res = desk.run_cycle(pd.Timestamp("2026-09-25 14:00", tz="UTC"), quant_forecast=0.42)
    lots = sizer.target_lots(res.final_forecast, vol, equity, price, XAUUSD, current_lots=pos)
    approved = risk.evaluate(RiskContext(..., target_lots=lots, ...)).approved_lots
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from aurum.agents.chief import DeskCycle
from aurum.agents.client import LLMClient
from aurum.agents.config import DeskConfig
from aurum.agents.journal import CycleJournal
from aurum.agents.loop import AgentRuntime, LoopResult
from aurum.agents.policy import DecisionPolicy, PolicyOutcome
from aurum.agents.prompts import chief_brief
from aurum.agents.providers import DeskDataProvider
from aurum.agents.records import Decision, Memo
from aurum.agents.usage import UsageLedger

logger = logging.getLogger(__name__)

__all__ = ["DeskResult", "TradingDesk", "demo"]


def _finite_or_none(x: float | None) -> float | None:
    """A previous forecast that is missing or not a finite number is *unknown* (so a ``hold``
    decision means flat rather than an arbitrary or NaN exposure)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        logger.warning("previous_forecast=%r is not numeric; treating it as unknown", x)
        return None
    if not math.isfinite(v):
        logger.warning("previous_forecast=%r is not finite; treating it as unknown", x)
        return None
    return max(-1.0, min(1.0, v))


@dataclass
class DeskResult:
    """Outcome of one desk cycle.

    status         : ``"decided"`` (the Chief submitted a valid decision), ``"failed"`` (no
                     usable decision: ``DecisionPolicy.on_failure`` applied) or ``"skipped"``
                     (overlay mode with a flat quant forecast: the outcome is 0 whatever the
                     decision, so no API call is made — see ``DeskConfig.skip_llm_when_outcome_fixed``).
    final_forecast : policy-gated forecast in [-1, 1] for the sizer.
    usage          : ``{"total": {...tokens, cost_usd}, "by_agent": {...}, "budget": {...}}``.
    """

    cycle_id: str
    now: pd.Timestamp
    quant_forecast: float
    decision: Decision | None
    final_forecast: float
    memos: list[Memo]
    usage: dict[str, Any]
    journal_path: Path | None
    status: str
    failure_reason: str | None
    policy: PolicyOutcome
    chief_turns: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return float(self.usage.get("total", {}).get("cost_usd", 0.0))


class TradingDesk:
    """Multi-agent LLM desk (SPEC §10).

    Parameters
    ----------
    provider    : :class:`DeskDataProvider` supplying point-in-time snapshots.
    client      : object exposing ``.beta.messages.create`` (``anthropic.Anthropic()`` or
                  :class:`~aurum.agents.testing.FakeAnthropicClient`). ``None`` builds
                  ``anthropic.Anthropic()`` lazily from the environment.
    config      : :class:`DeskConfig` (models, efforts, caps, budget, caching).
    policy      : :class:`DecisionPolicy` (default overlay / follow_quant on failure).
    journal_dir : directory for per-cycle JSONL journals; ``None`` keeps them in memory.
    """

    def __init__(
        self,
        provider: DeskDataProvider,
        *,
        client: Any | None = None,
        config: DeskConfig | None = None,
        policy: DecisionPolicy | None = None,
        journal_dir: str | Path | None = "runs/desk_journal",
    ) -> None:
        self.provider = provider
        self.config = config or DeskConfig()
        self.policy = policy or DecisionPolicy()
        self.journal_dir = Path(journal_dir) if journal_dir is not None else None
        self._client_arg = client
        self._llm: LLMClient | None = None
        self._last_final: float | None = None
        self.last_journal: CycleJournal | None = None

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self._client_arg, prompt_caching=self.config.prompt_caching,
                                  cache_ttl=self.config.cache_ttl, timeout=self.config.request_timeout_s)
        return self._llm

    @property
    def last_final_forecast(self) -> float | None:
        return self._last_final

    def run_cycle(
        self,
        now: pd.Timestamp | str,
        quant_forecast: float,
        *,
        previous_forecast: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> DeskResult:
        """Run one decision cycle at decision time ``now`` (tz-aware; the bar close).

        ``previous_forecast`` defaults to the final forecast of this desk's previous cycle
        (used by the ``hold`` action). ``context`` is optional operator context (JSON) shown
        to the Chief, e.g. ``{"account": "paper"}``.
        """
        ts = pd.Timestamp(now)
        if ts.tz is None:
            raise ValueError("now must be tz-aware (UTC)")
        ts = ts.tz_convert("UTC")
        # A missing quant forecast (None, NaN, pd.NA from a nullable-dtype series) is data, not
        # a programming error: review it as flat instead of raising out of the trading loop.
        try:
            q = float(quant_forecast)
        except (TypeError, ValueError):
            q = float("nan")
        if not math.isfinite(q):
            logger.warning("quant_forecast=%r is not finite; reviewing 0.0", quant_forecast)
            q = 0.0
        q = max(-1.0, min(1.0, q))
        prev = _finite_or_none(self._last_final if previous_forecast is None else previous_forecast)
        cfg = self.config
        cycle_id = uuid.uuid4().hex[:12]
        path = None
        if self.journal_dir is not None:
            path = self.journal_dir / f"{ts:%Y%m%dT%H%M%SZ}_{cycle_id}.jsonl"
        journal = CycleJournal(path, cycle_id=cycle_id, max_chars=cfg.journal_max_chars)
        path = journal.path  # None if the journal directory could not be created
        self.last_journal = journal
        ledger = UsageLedger(prices=cfg.prices, max_cost_usd=cfg.max_cost_usd_per_cycle,
                             max_tokens=cfg.max_tokens_per_cycle, cache_ttl=cfg.cache_ttl)
        started = time.monotonic()
        try:
            as_of = self.provider.as_of(ts)
        except Exception:  # presentation only; never block a cycle on it
            logger.exception("provider.as_of failed; using the raw decision time")
            as_of = ts.isoformat()
        journal.log(
            "cycle_start", now=ts, as_of=as_of, quant_forecast=q, previous_forecast=prev,
            policy={"mode": self.policy.mode, "max_abs_forecast": self.policy.max_abs_forecast,
                    "on_failure": self.policy.on_failure, "min_confidence": self.policy.min_confidence},
            limits={"max_specialists": cfg.max_specialists_per_cycle,
                    "max_cost_usd": cfg.max_cost_usd_per_cycle, "max_tokens": cfg.max_tokens_per_cycle,
                    "chief_max_turns": cfg.model_for_role("chief").max_turns},
            context=context or {},
        )

        if cfg.skip_llm_when_outcome_fixed and self.policy.mode == "overlay" and q == 0.0:
            return self._skip_cycle(cycle_id, ts, q, journal, ledger, started)

        result: LoopResult | None = None
        memos: list[Memo] = []
        failure: str | None = None
        with ThreadPoolExecutor(max_workers=cfg.max_parallel_agents, thread_name_prefix="aurum-desk") as pool:
            runtime = AgentRuntime(
                llm=self.llm, ledger=ledger, journal=journal, executor=pool,
                deadline=None if cfg.max_cycle_seconds is None else started + cfg.max_cycle_seconds,
                soft_budget_fraction=cfg.soft_budget_fraction,
                tool_result_max_chars=cfg.tool_result_max_chars,
                journal_max_chars=cfg.journal_max_chars,
                request_timeout_s=cfg.request_timeout_s,
            )
            cycle = DeskCycle(provider=self.provider, now=ts, as_of=as_of, quant_forecast=q,
                              config=cfg, runtime=runtime)
            brief = chief_brief(
                as_of=as_of, quant_forecast=q, previous_forecast=prev, mode=self.policy.mode,
                max_abs_forecast=self.policy.max_abs_forecast,
                max_specialists=cfg.max_specialists_per_cycle,
                max_turns=cfg.model_for_role("chief").max_turns,
                max_cost_usd=cfg.max_cost_usd_per_cycle, context=context,
            )
            try:
                result = cycle.run_chief(brief)
            except Exception as exc:  # defensive: a cycle must never crash the trading loop
                logger.exception("desk cycle %s crashed", cycle_id)
                failure = f"desk error: {type(exc).__name__}: {exc}"
            memos = sorted(cycle.memos, key=lambda m: m.agent_id)

        decision: Decision | None = None
        if result is not None:
            if result.completed and isinstance(result.terminal, Decision):
                decision = result.terminal
            else:
                failure = f"{result.status}: {result.error}"
        outcome = self.policy.evaluate(q, decision, previous_forecast=prev)
        usage = ledger.summary()
        status = "decided" if decision is not None else "failed"
        journal.log("decision", decision=decision.to_dict() if decision else None, failure_reason=failure)
        journal.log("policy", outcome=outcome.to_dict())
        journal.log("cycle_end", status=status, final_forecast=outcome.final_forecast,
                    quant_forecast=q, usage=usage, duration_s=round(time.monotonic() - started, 3),
                    n_memos=len(memos))
        self._last_final = outcome.final_forecast
        return DeskResult(
            cycle_id=cycle_id, now=ts, quant_forecast=q, decision=decision,
            final_forecast=outcome.final_forecast, memos=memos, usage=usage, journal_path=path,
            status=status, failure_reason=failure, policy=outcome,
            chief_turns=result.turns if result is not None else 0,
        )


    def _skip_cycle(self, cycle_id: str, ts: pd.Timestamp, q: float, journal: CycleJournal,
                    ledger: UsageLedger, started: float) -> DeskResult:
        """Overlay with a flat quant forecast: the outcome is 0 whatever the desk decides."""
        note = ("overlay mode with a flat quant forecast: the final forecast is 0 whatever the "
                "decision (overlay interval is {0}); LLM cycle skipped")
        outcome = PolicyOutcome(final_forecast=0.0, requested_forecast=0.0, quant_forecast=q,
                                mode=self.policy.mode, action="none", used_fallback=False,
                                constrained=False, notes=(note,))
        usage = ledger.summary()
        journal.log("cycle_skipped", reason=note)
        journal.log("policy", outcome=outcome.to_dict())
        journal.log("cycle_end", status="skipped", final_forecast=0.0, quant_forecast=q, usage=usage,
                    duration_s=round(time.monotonic() - started, 3), n_memos=0)
        self._last_final = 0.0
        return DeskResult(cycle_id=cycle_id, now=ts, quant_forecast=q, decision=None, final_forecast=0.0,
                          memos=[], usage=usage, journal_path=journal.path, status="skipped",
                          failure_reason=None, policy=outcome)


def demo(client: Any | None = None, *, journal_dir: str | Path | None = None, n_bars: int = 600,
         seed: int = 7) -> DeskResult:
    """End-to-end desk cycle on synthetic, anonymised data.

    With ``client=None`` a scripted :class:`~aurum.agents.testing.FakeAnthropicClient` plays
    the agents (Chief consults two specialists in parallel, creates an ad-hoc event-risk
    agent, then scales the quant forecast by 0.5) — fully offline. Pass
    ``client=anthropic.Anthropic()`` to run the same cycle against the live model.
    """
    from aurum.agents.providers import HistoricalDeskDataProvider
    from aurum.agents.testing import FakeAnthropicClient, decision_call, memo_call, message, tool_use
    from aurum.core.types import MarketData
    from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro
    from aurum.models.volatility import ewma_volatility

    bars = make_synthetic_bars(n_bars, "H1", seed=seed, model="trend")
    md = MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=seed),
                    events=make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=7)))
    close = bars["close"]
    fast = close.ewm(span=24, adjust=False).mean()
    slow = close.ewm(span=96, adjust=False).mean()
    vol = ewma_volatility(close, bars_per_year=252 * 23.0)  # fixed constant: no full-sample inference
    signals = pd.DataFrame({
        "ema_cross": ((fast - slow) / (close * vol / 20)).clip(-1, 1).fillna(0.0),
        "tsmom_24": (close.pct_change(24) / (vol / 8)).clip(-1, 1).fillna(0.0),
    }, index=bars.index)
    combined = signals.mean(axis=1).clip(-1, 1)
    provider = HistoricalDeskDataProvider(
        md, signals=signals, combined=combined, vol=vol, anonymise=True,
        backtest_stats={"note": "illustrative only", "oos_sharpe_combined": 0.6, "oos_max_drawdown": -0.12},
        risk_status_fn=lambda now: {"drawdown_from_peak": -0.03, "daily_pnl_pct": 0.1, "halted": False},
        positions_fn=lambda now: {"lots": 0.0},
    )
    now = bars["available_at"].iloc[-1]
    q = float(combined.iloc[-1])
    if client is None:
        client = FakeAnthropicClient({
            "chief": [
                message(tool_use("get_market_snapshot"), tool_use("get_quant_signals"), tool_use("get_calendar")),
                message(
                    tool_use("consult_specialist", {"role": "macro_strategist",
                                                    "question": "Do macro drivers support the quant direction?"}),
                    tool_use("consult_specialist", {"role": "risk_officer",
                                                    "question": "What is the prudent maximum exposure now?"}),
                    tool_use("create_specialist", {
                        "name": "event_risk_analyst",
                        "mandate": "Assess scheduled-event risk over the next 24 bars.",
                        "tools": ["get_calendar", "get_market_snapshot"],
                        "question": "Is there event risk that argues for reducing exposure?"}),
                ),
                message(decision_call(action="scale", scale=0.5, forecast=q * 0.5, confidence=0.55,
                                      rationale="Mixed specialist views and upcoming event risk: halve exposure.",
                                      dissent="Macro aligned with the quant direction; risk officer cautious.")),
            ],
            "macro_strategist": [message(tool_use("get_macro_snapshot")),
                                 message(memo_call(stance="bullish" if q >= 0 else "bearish",
                                                   suggested_exposure=0.3 if q >= 0 else -0.3))],
            "risk_officer": [message(tool_use("get_risk_status")),
                             message(memo_call(stance="neutral", suggested_exposure=0.0, confidence=0.6))],
            "adhoc:event_risk_analyst": [message(tool_use("get_calendar")),
                                         message(memo_call(stance="neutral", suggested_exposure=0.0))],
        })
    desk = TradingDesk(provider, client=client, policy=DecisionPolicy(mode="overlay"), journal_dir=journal_dir)
    return desk.run_cycle(now, q)
