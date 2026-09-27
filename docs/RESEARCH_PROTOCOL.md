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
