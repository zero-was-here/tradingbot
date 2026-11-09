# Research protocol (pre-registered)

Written **before** the first full walk-forward of Aurum v2 on real data, so that the choice
of what to report cannot be influenced by the results. Any deviation from this protocol must
be recorded in the results section with the reason.

## Data

- XAUUSD mid prices and bid/ask spreads from Dukascopy (M1 aggregated to H1),
  2013-01-01 → 2026-09-25. 2012 is excluded (zero spreads and flat bars in the source).
- Daily macro series from Yahoo Finance and FRED, joined point-in-time. Publication lags are
  documented in `aurum/data/macro.py`.
- A rule-based NFP/FOMC calendar. Only scheduled times are used; release outcomes are never
  used before they happen.

## Periods

| Period | Dates | Use |
|---|---|---|
| Development | 2013 → 2019 | Module authors sanity-checked strategy behaviour here. No parameter was tuned on later data. |
| Walk-forward OOS | first test fold → 2024-12-31 | Rolling 3Y train / 6M test, purge = max label horizon, 24-bar embargo. Every trainable component (feature scaler, ML models, seasonality table, combiner) is refit per fold on training data only. |
| Final holdout | 2025-01-01 → 2026-09-25 | Excluded from every fold. Evaluated **once**, for the configuration selected on walk-forward OOS results only. |

Known contamination, disclosed: during integration testing a wave-2 reviewer ran the
`fast.yaml` smoke configuration (a subset of strategies) through a period that included 2021–2025
and the 2026 holdout. No parameters were changed as a result.

## Configurations evaluated (and only these)

1. `configs/default.yaml`: 14 strategies (12 rule-based, `ml_gbm`, `meta_label`) on H1.
2. `configs/trend_core.yaml`: the literature-backed low-turnover subset on H1: `tsmom`,
   `ema_cross`, `donchian`, `kalman_trend`, `vol_squeeze`, `macro_factor`, `risk_off`.

Both configurations use:

- a `sharpe_shrink` combiner scored net of costs;
- 10% annualised vol targeting;
- rate-based financing (Fed funds + 2.5% markup);
- recorded bid/ask spreads plus slippage;
- research risk limits.

## Statistics reported

For each strategy standalone and for each combined book, on stitched OOS:

- CAGR, annualised volatility, daily Sharpe, Sortino, maximum drawdown, Calmar;
- trade count, turnover, and cost and financing attribution;
- PSR and DSR, with `n_trials = 23`: 21 strategy configurations plus 2 combined books;
- a stationary-bootstrap 95% confidence interval for Sharpe;
- PBO via CSCV across the strategy set;
- a buy-and-hold benchmark with the same financing.

## Selection rule

The configuration with the higher walk-forward OOS **DSR** becomes the recommended default.
Ties are broken by lower maximum drawdown. Only that configuration is then run on the
holdout, and its result is reported whatever it is.

A strategy is recommended for live use only if **all** of the following hold:

- DSR > 0.95 on walk-forward OOS;
- the bootstrap Sharpe CI lower bound is > 0;
- holdout Sharpe > 0.

If nothing clears this bar, the honest conclusion is "no demonstrated edge". The system is then
shipped as a research platform, with paper trading as the only recommended mode.

## Addendum (v2.1, unreleased)

*Added 2026-11-09, after the results in [RESULTS.md](RESULTS.md) were published. It records
changes to the tools around this protocol and the trials run since. Nothing above this
addendum has been modified: the protocol stays frozen as pre-registered.*

### What changed in the holdout ledger

The protocol relies on the holdout ledger to make every look at the holdout visible
([research.md](research.md#the-holdout-and-the-ledger)). In v2.1:

- **More writers.** Besides the walk-forward's own holdout evaluation, three more actions
  append an entry: a single-split backtest that evaluates holdout bars
  (`aurum backtest --include-holdout`, kind `backtest`), an LLM-desk replay whose window
  reaches the holdout (kind `desk_replay`, under the config hash plus `:desk`, or
  `:desk-fake` for the offline client, so it counts as a separate trial), and the public
  `record_holdout_look`. In v2.0 the first two evaluated holdout bars without leaving an
  entry.
- **Counting by data source.** Entries record `data_source` (`real` or `synthetic`) and
  `data_hash`. When a run counts earlier looks, entries with a different `data_source` are
  ignored, so synthetic smoke tests no longer raise `n_trials` for real research, and the
  other way round. Entries without the field still count for both.
- **Benchmark excluded.** `metrics.strategy_sharpe` now lists strategies only; the
  buy-and-hold Sharpe is recorded separately as `metrics.benchmark_sharpe`.

Effect on this protocol: none. The protocol's holdout evaluation (`trend_core`, written
2026-09-27) is the only entry in `runs/protocol/holdout_ledger.jsonl`, and the `fast.yaml`
smoke look disclosed above is the only entry in `runs/holdout_ledger.jsonl`. Both were
written by v2.0 (no `data_source`, benchmark inside `strategy_sharpe`), both are unchanged,
and each ledger still holds exactly that one entry: no v2.1 run evaluated the holdout.

### The exploratory v2.1 trials are not part of this protocol

`configs/d1_trend.yaml`, `configs/managed_gold.yaml` and an `ema_cross` parameter sweep
were run after the results were published. They are exploratory additional trials:

- they were not pre-registered, and no selection or recommendation is based on them;
- their data ends 2024-12-31 and their holdout is disabled, so they did not evaluate the
  2025-01-01 → 2026-09-25 holdout and wrote no ledger entry;
- they reuse the 2016–2024 walk-forward span this protocol already evaluated, so each new
  configuration is a further trial on the same data.

Their results are reported separately in
[RESULTS.md](RESULTS.md#additional-trials-exploratory-v21).

### n_trials is now 30, and at least 38 for anything new

The exploratory D1 configs set `walkforward.n_trials: 30`: the 23 configurations counted
above, the five new strategy configurations (daily `tsmom`, `ema_cross` and `donchian`,
`managed_gold` and `regime_switch`) and their two combined books. `configs/default.yaml`
and `configs/trend_core.yaml` keep the pre-registered 23, so re-running the protocol
reproduces its numbers. The sweep counted only its own eight valid grid points
(`n_trials` = 8); counting the 30 earlier trials as well gives 38. A future study on
2016–2024 data should start its count at no less than 38, or use data that arrives after
its own protocol is written.
