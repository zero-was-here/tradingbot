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
    pnl       : optional DataFrame per bar with columns ``price`` (mark-to-market PnL at MID
                prices), ``costs`` (spread + slippage + commission, positive), ``swap``
                (signed) and ``net`` = price - costs + swap. ``equity.diff() == net``.
    position_close : optional signed lots held at each bar's CLOSE (after intrabar stop /
                take-profit exits). Equals ``positions`` unless a protective exit fired.
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
    pnl: pd.DataFrame | None = None
    position_close: pd.Series | None = None

    def reconcile(self) -> dict[str, float]:
        """PnL identity check: equity change vs. sum of price PnL, costs and swap.

        Returns the components and ``residual`` (should be ~0, e.g. < 1e-6 USD). Uses
        ``pnl`` when present, otherwise ``costs`` only (price PnL then = residual).
        """
        change = float(self.equity.iloc[-1] - self.equity.iloc[0]) if len(self.equity) else 0.0
        cost_cols = [c for c in ("spread", "slippage", "commission") if c in self.costs.columns]
        total_costs = float(self.costs[cost_cols].to_numpy().sum()) if cost_cols else 0.0
        swap = float(self.costs["swap"].sum()) if "swap" in self.costs.columns else 0.0
        price = float(self.pnl["price"].sum()) if self.pnl is not None else change + total_costs - swap
        trade_pnl = float(self.trades["pnl"].sum()) if "pnl" in self.trades.columns else float("nan")
        return {
            "equity_change": change,
            "price_pnl": price,
            "costs": total_costs,
            "swap": swap,
            "residual": change - (price - total_costs + swap),
            "trade_pnl": trade_pnl,
        }

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
        if self.position_close is not None:
            frame["position_close"] = self.position_close
        frame = frame.join(self.costs.add_prefix("cost_"), how="left")
        if self.pnl is not None:
            frame = frame.join(self.pnl.add_prefix("pnl_"), how="left")
        frame.to_parquet(d / "timeseries.parquet")
        self.trades.to_csv(d / "trades.csv", index=False)
        self.fills.to_csv(d / "fills.csv", index=False)
        if self.risk_events is not None:
            self.risk_events.to_csv(d / "risk_events.csv", index=False)
        (d / "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str))
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))
        return d
