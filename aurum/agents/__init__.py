"""LLM multi-agent trading desk on the Claude API (SPEC §10).

A Chief Investment Officer agent reviews the quant forecast each cycle, consults predefined
specialists (macro strategist, quant analyst, risk officer, execution trader) and can create
new ad-hoc specialist agents at runtime, then submits a decision. A deterministic
:class:`DecisionPolicy` converts the decision into a forecast that the caller feeds through
the SAME sizer and risk manager as the quant book — the LLM cannot bypass risk.

Importing this package does not require the optional ``anthropic`` dependency; it is loaded
only when a :class:`TradingDesk` is created without an explicit client.
"""

from aurum.agents.config import DEFAULT_PRICES, AgentModelConfig, DeskConfig, ModelPrice
from aurum.agents.desk import DeskResult, TradingDesk, demo
from aurum.agents.journal import CycleJournal, read_journal
from aurum.agents.policy import DecisionPolicy, PolicyOutcome
from aurum.agents.providers import DeskDataProvider, HistoricalDeskDataProvider, StaticDeskDataProvider
from aurum.agents.records import Decision, Memo, RecordValidationError
from aurum.agents.usage import UsageLedger

__all__ = [
    "AgentModelConfig",
    "CycleJournal",
    "DEFAULT_PRICES",
    "Decision",
    "DecisionPolicy",
    "DeskConfig",
    "DeskDataProvider",
    "DeskResult",
    "HistoricalDeskDataProvider",
    "Memo",
    "ModelPrice",
    "PolicyOutcome",
    "RecordValidationError",
    "StaticDeskDataProvider",
    "TradingDesk",
    "UsageLedger",
    "demo",
    "read_journal",
]
