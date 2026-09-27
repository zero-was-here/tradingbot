# Research results: pre-registered walk-forward, 2026-09-27

This document follows [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md), which was written before
any full run. Every number here comes from `aurum walkforward` on real Dukascopy XAUUSD H1
data and can be reproduced (see "Reproduce" below).

**Provenance.** Git `02eace1`, data hash `8f57cab74e3e` (through 2024) and `b05d27bf5688`
(through 2026-09-25), pandas 3.0.6, numpy 2.5.3.

## Setup

| | |
|---|---|
| Data | XAUUSD H1 mid prices with recorded bid/ask spreads, 2013-01 → 2026-09 (81,587 bars) |
| Walk-forward | Rolling 3-year train / 6-month test, 18 folds, embargo 24 bars. OOS runs 2015-12 → 2024-12 (53k bars). |
| Holdout | 2025-01-01 → 2026-09-25. Excluded from all folds and evaluated **once**, for the selected configuration only. |
| Costs | Recorded spread (min $0.10), $0.02 plus 2% of bar range slippage, and financing at Fed funds (point-in-time) + 2.5% |
| Sizing | 10% annualised vol target, 2x max leverage, drawdown de-risking |
| Combiner | `sharpe_shrink`, scored **net of costs** on earlier folds' OOS forecasts only |
| Statistics | Daily Sharpe; stationary-bootstrap 95% CI; PSR; DSR with n_trials = 23; PBO via CSCV |

## Walk-forward OOS, 2016 → 2024

### Configuration `default` (14 strategies)

| Book | Sharpe | 95% CI | DSR | CAGR | Vol | Max DD | Trades |
|---|---|---|---|---|---|---|---|
| **Combined** | **−0.26** | [−0.95, 0.38] | 0.003 | −1.1% | 3.8% | −13.9% | 956 |
| tsmom | 0.01 | [−0.64, 0.63] | 0.027 | −0.1% | 5.0% | −15.0% | 1229 |
| ema_cross | 0.11 | [−0.52, 0.73] | 0.052 | 0.5% | 5.4% | −12.7% | 489 |
| donchian | −0.16 | [−0.80, 0.44] | 0.007 | −0.9% | 4.6% | −15.8% | 379 |
| kalman_trend | 0.01 | [−0.68, 0.61] | 0.026 | −0.1% | 5.1% | −13.6% | 1165 |
| zscore_fade | −1.51 | [−2.08, −0.87] | 0.000 | −2.5% | 1.6% | −20.5% | 1341 |
| rsi2 | −1.06 | [−1.60, −0.45] | 0.000 | −2.1% | 1.9% | −17.6% | 2334 |
| bollinger_revert | −1.52 | [−2.06, −0.97] | 0.000 | −3.3% | 2.2% | −27.3% | 1301 |
| vol_squeeze | 0.09 | [−0.60, 0.73] | 0.044 | 0.2% | 2.9% | −9.9% | 636 |
| orb | −1.58 | [−2.25, −0.91] | 0.000 | −2.7% | 1.7% | −22.3% | 3496 |
| macro_factor | −0.35 | [−1.03, 0.31] | 0.001 | −1.5% | 3.9% | −17.1% | 299 |
| risk_off | −0.13 | [−0.69, 0.47] | 0.009 | −0.3% | 2.2% | −8.8% | 37 |
| intraday_seasonality | −0.81 | [−1.42, −0.19] | 0.000 | −0.1% | 0.2% | −1.2% | 26 |
| ml_gbm | n/a | | | 0.0% | 0.0% | 0.0% | 0 |
| meta_label | −0.28 | [−1.02, 0.40] | 0.002 | −0.8% | 2.6% | −10.9% | 434 |
| *Buy & hold (with financing)* | *0.48* | *[−0.17, 1.15]* | *0.926* | *6.8%* | *16.2%* | *−31.4%* | |

`ml_gbm` failed its validation skill gate in every fold (validation AUC 0.43–0.57, never
significantly above 0.5), so it made no trades. The whole-set PBO is 0.41. The probability
that the in-sample best strategy loses OOS is 0.81.

### Configuration `trend_core` (7 strategies)

| Book | Sharpe | 95% CI | DSR | CAGR | Vol | Max DD | Trades |
|---|---|---|---|---|---|---|---|
| **Combined** | **−0.17** | [−0.82, 0.48] | 0.007 | −0.8% | 4.0% | −12.3% | 875 |
| tsmom | 0.00 | [−0.65, 0.66] | 0.026 | −0.1% | 5.0% | −15.1% | 1230 |
| ema_cross | 0.09 | [−0.55, 0.77] | 0.047 | 0.4% | 5.3% | −13.2% | 491 |
| donchian | −0.08 | [−0.73, 0.57] | 0.014 | −0.6% | 5.6% | −13.6% | 379 |
| kalman_trend | 0.01 | [−0.65, 0.64] | 0.027 | −0.1% | 5.1% | −13.7% | 1165 |
| vol_squeeze | 0.08 | [−0.56, 0.70] | 0.042 | 0.2% | 2.9% | −9.8% | 637 |
| macro_factor | −0.33 | [−0.93, 0.30] | 0.002 | −1.4% | 3.9% | −17.1% | 301 |
| risk_off | −0.13 | [−0.72, 0.48] | 0.009 | −0.3% | 2.2% | −8.8% | 38 |
| *Buy & hold (with financing)* | *0.46* | *[−0.15, 1.13]* | *0.921* | *6.6%* | *16.2%* | *−31.4%* | |

The same strategies score slightly differently here than in the `default` table, although
their forecasts are identical. Without the ML strategies' 24-bar label horizon there is no
purge, so the stitched OOS backtest starts 24 bars earlier (2015-12-16 instead of
2015-12-17). The equity-dependent sizer (rebalance band, drawdown de-risking) then follows a
different path; donchian moves most. PBO is 0.90, and the probability that the in-sample best
strategy loses OOS is 0.84.

## Selection and holdout

By the pre-registered rule (higher walk-forward DSR), **`trend_core` was selected**
(0.007 vs 0.003). Only `trend_core` was run on the holdout.

### Final holdout, 2025-01-01 → 2026-09-25, evaluated once

| Book | Sharpe | 95% CI | CAGR | Vol | Max DD |
|---|---|---|---|---|---|
| **Combined (trend_core)** | **1.34** | [−0.08, 2.78] | 9.5% | 6.8% | −8.4% |
| tsmom | 1.43 | [0.07, 2.88] | 10.8% | 7.1% | −8.6% |
| ema_cross | 1.16 | [−0.21, 2.58] | 7.9% | 6.5% | −7.3% |
| donchian | 0.98 | [−0.47, 2.51] | 7.0% | 6.9% | −9.1% |
| kalman_trend | 1.50 | [0.12, 3.01] | 11.1% | 6.9% | −7.4% |
| vol_squeeze | 0.78 | [−0.67, 1.96] | 2.2% | 2.8% | −2.5% |
| macro_factor | 0.00 | [−1.35, 1.45] | −0.2% | 5.6% | −6.8% |
| risk_off | 0.37 | [−1.24, 1.56] | 0.9% | 2.4% | −4.8% |
| *Buy & hold* | *0.94* | *[−0.50, 2.51]* | *24.6%* | *26.4%* | *−32.5%* |

The holdout run read its own ledger (`runs/protocol/holdout_ledger.jsonl`), so the earlier
`fast.yaml` smoke look at 2026-01 → 2026-09 disclosed in
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) was not added to `n_trials` automatically.
Counting it (`n_trials` = 24) lowers the combined book's holdout DSR from 0.43 to 0.42 and
changes nothing else in this document.

## Verdict under the protocol

| Criterion (all required for live use) | Result | |
|---|---|---|
| Walk-forward DSR > 0.95 | 0.007 | ❌ |
| Bootstrap Sharpe CI lower bound > 0 | −0.82 | ❌ |
| Holdout Sharpe > 0 | 1.34 | ✅ |

**No strategy or book has a demonstrated edge.** Aurum v2 therefore ships as a research and
paper-trading platform. Real-money trading is not recommended on this evidence.

## What the evidence says

1. **Hourly mean reversion and opening-range breakouts lose reliably after costs.** z-score
   fade, RSI(2), Bollinger reversion and the London/New York opening-range breakout all have
   95% CIs entirely below zero (so does intraday seasonality, at tiny size). Flipping the sign
   does not help: z-score fade, Bollinger reversion and the ORB lost money even before costs,
   but by less than the spread and slippage a flipped version would pay again.
2. **Trend following is regime-dependent.** It was flat to slightly negative through the
   choppy 2016–2024 sample and strongly positive in the 2025–26 trend. That matches the
   literature: trend following is a *crisis-alpha / convexity* return stream, not a steady
   one. In the holdout the combined book ran at about a quarter of buy-and-hold's volatility
   and drawdown, and earned well under half its return (9.5% vs 24.6% a year). About 21
   months is still not enough to separate skill from regime.
3. **Costs decide everything.** Before spread, slippage and financing the four trend
   strategies each made money in the 2016–2024 walk-forward (tsmom +$22.9k gross on $100k);
   after them three of the four lost money (tsmom −$0.7k). The net-of-cost combiner holds
   risk back when nothing clears costs: in fold 15 the whole book stood flat.
4. **ML earns its keep by abstaining.** The skill-gated GBM refused to trade on
   coin-flip validation AUC. That is the right behaviour, and it is why it shows zero rather
   than a loss.
5. **Gold's own drift dominated this period.** Buy-and-hold beat every active book on raw
   return (6.8%/yr walk-forward, 24.6%/yr holdout), at about 4x the combined books'
   volatility and 2–4x their drawdown.

## Research directions (each is a new trial and must be counted in DSR)

- Daily-bar (NY-close) trend following with the classic monthly TSMOM horizons, which is
  lower turnover.
- A **long-biased**, vol-targeted gold allocation with a trend overlay, aiming to keep the
  drift while cutting drawdowns. This is the institutional "managed gold exposure" product,
  not an alpha claim.
- Better macro data: consensus and surprise economic calendars, TIPS real yields at
  intraday frequency, CFTC positioning, ETF flows.
- Broker-specific cost calibration from your own fills (TCA), fed back into `CostModel`.

## Reproduce

```bash
aurum data download --end 2026-09-25                 # ~130 MB Dukascopy + macro
aurum walkforward -c configs/default.yaml    --set data.end=2024-12-31T23:59:59Z --set walkforward.holdout_start=null --out runs/protocol/default_selection
aurum walkforward -c configs/trend_core.yaml --set data.end=2024-12-31T23:59:59Z --set walkforward.holdout_start=null --out runs/protocol/trend_core_selection
aurum walkforward -c configs/trend_core.yaml --out runs/protocol/trend_core_holdout   # includes the one-time holdout
```

The runs are deterministic: re-running the last command on the same data reproduces the
tables above exactly. Compare the data hashes in each run's `provenance.json` with the ones
at the top, since Dukascopy and FRED can revise history.

Every holdout evaluation is appended to `holdout_ledger.jsonl` next to the run directories.
When a *different* configuration later evaluates an overlapping holdout window, the run
warns and adds it to `n_trials`. `runs/` is git-ignored, so a fresh clone starts with an
empty ledger and cannot know about the looks recorded here. Anyone re-running this should
treat the 2025–26 holdout as **already used**.
