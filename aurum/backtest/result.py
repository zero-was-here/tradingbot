"""Container for a completed backtest (shared by engine, reports, research and agents)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class BacktestResult:
    """All series are indexed by bar OPEN time (the bars index) unless stated otherwise.

    equity    : account equity (USD) marked-to-market at each bar's close.
    returns   : simple per-bar returns of ``equity`` (first bar = 0).
    positions : signed lots held DURING each bar (after the fill at that bar's open).
    target    : signed lots requested at each bar's close (before rounding/risk if available).
    forecast  : combined forecast in [-1, 1] at each bar's close (NaN if not applicable).
    costs     : DataFrame per bar with columns ``spread``, ``slippage``, ``commission``, ``swap``
                (USD; costs positive, swap signed with + = received).
    trades    : DataFrame of round trips, columns = fields of ``aurum.core.types.Trade``.
    fills     : DataFrame of individual fills (time, side, lots, price, costs...).
    metrics   : dict produced by ``aurum.backtest.metrics.compute_metrics``.
    risk_events : DataFrame of risk-manager interventions (time, reasons, requested, approved).
    """

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    costs: pd.DataFrame
    trades: pd.DataFrame
    fills: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    target: pd.Series | None = None
    forecast: pd.Series | None = None
    risk_events: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        keys = [
            "total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
            "n_trades", "win_rate", "profit_factor", "exposure", "total_costs",
        ]
        lines = []
        for k in keys:
            if k in self.metrics:
                v = self.metrics[k]
                lines.append(f"{k:>16}: {v:,.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
        return "\n".join(lines)

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {"equity": self.equity, "returns": self.returns, "positions": self.positions}
        )
        if self.target is not None:
            frame["target"] = self.target
        if self.forecast is not None:
            frame["forecast"] = self.forecast
        frame = frame.join(self.costs.add_prefix("cost_"), how="left")
        frame.to_parquet(d / "timeseries.parquet")
        self.trades.to_csv(d / "trades.csv", index=False)
        self.fills.to_csv(d / "fills.csv", index=False)
        if self.risk_events is not None:
            self.risk_events.to_csv(d / "risk_events.csv", index=False)
        (d / "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str))
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))
        return d
