"""Append-only JSONL journal: one file per desk cycle, one JSON object per line.

Every line carries ``ts`` (wall-clock UTC when the event was logged), ``cycle_id``,
``event`` and, for agent events, ``agent``. Events written by the desk:

``cycle_start`` (decision time, quant forecast, policy, limits) → ``agent_start`` (system
prompt, tool names, first message) → ``llm_response`` (stop reason, content blocks, usage,
estimated cost, fallback info) → ``tool_call`` / ``tool_result`` (inputs and outputs,
truncated) → ``memo`` → ``agent_end`` → ``decision`` → ``policy`` → ``cycle_end`` (usage &
cost summary). Also: ``agent_created`` (Chief-created agents: mandate and tool grant),
``harness_note`` (every text the harness injects into a conversation), ``max_tokens_retry``
and ``tools_skipped`` (calls on an agent's final turn whose results could never be read).
Together they reconstruct every conversation for audit and post-trade review.

Writes are serialised with a lock (specialists log concurrently) and flushed per line so a
crash mid-cycle leaves a readable partial journal. If the journal directory cannot be
created the journal degrades to memory-only (``path`` becomes ``None``) instead of failing
the cycle.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from aurum.agents._json import to_jsonable, truncate

logger = logging.getLogger(__name__)

__all__ = ["CycleJournal", "read_journal"]


def _truncate_strings(obj: Any, max_chars: int | None) -> Any:
    if isinstance(obj, str):
        return truncate(obj, max_chars)
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings(v, max_chars) for v in obj]
    return obj


class CycleJournal:
    """Journal for one cycle. ``path=None`` keeps events in memory only."""

    def __init__(self, path: str | Path | None, *, cycle_id: str, max_chars: int | None = 4000,
                 keep_in_memory: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self.cycle_id = cycle_id
        self.max_chars = max_chars
        self.keep_in_memory = keep_in_memory
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:  # journaling must never break a trading cycle: keep it in memory
                logger.exception("cannot create desk journal directory %s; journaling in memory only",
                                 self.path.parent)
                self.path = None

    def log(self, event: str, *, agent: str | None = None, full: bool = False, **payload: Any) -> None:
        """Record an event. ``full=True`` skips string truncation (e.g. system prompts)."""
        record: dict[str, Any] = {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "cycle_id": self.cycle_id,
            "event": event,
        }
        if agent is not None:
            record["agent"] = agent
        body = to_jsonable(payload)
        if not full:
            body = _truncate_strings(body, self.max_chars)
        record.update(body)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self._lock:
            if self.keep_in_memory:
                self.events.append(record)
            if self.path is not None:
                try:
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                        fh.flush()
                except OSError:  # journaling must never break a trading cycle
                    logger.exception("failed to write desk journal %s", self.path)

    def of_type(self, event: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e["event"] == event]


def read_journal(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL journal written by :class:`CycleJournal`."""
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
