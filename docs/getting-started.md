# Getting started

This page takes you from a fresh clone to four working results: an install you have checked,
a local copy of 14 years of XAUUSD bid/ask history, a first walk-forward research run with its
HTML tearsheet, and a dry-run paper-trading session. It also shows the free, offline LLM-desk
demo. Every output block below comes from running the command against the current code;
`aurum data download` itself was not re-run for this page, so its section is based on
`--help`, the code and the existing data store. Aurum is a research and paper-trading
platform. Under its pre-registered protocol no strategy has
shown a statistically demonstrated edge (see [RESULTS.md](RESULTS.md)), so none of these
steps is a reason to trade real money.

**On this page**

- [Prerequisites](#prerequisites)
- [Install](#install)
- [Try it without downloading data](#try-it-without-downloading-data)
- [Download the market data](#download-the-market-data)
- [Your first walk-forward](#your-first-walk-forward)
- [The offline LLM-desk demo](#the-offline-llm-desk-demo)
- [Paper-trading quickstart](#paper-trading-quickstart)
- [When something goes wrong](#when-something-goes-wrong)
- [Where to go next](#where-to-go-next)

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 or newer | `pyproject.toml` sets `requires-python = ">=3.10"`. CI tests 3.10 and 3.12. |
| git and a shell | Run every `aurum` command from the repository root: data and output paths in the configs (`data_store/`, `runs/`, `artifacts/`) are relative to the working directory. |
| Disk | About 130 MB for the raw Dukascopy cache (`cache/dukascopy/`) and about 20 MB for the built data store (`data_store/`). Each walk-forward run directory takes tens of MB (39 MB for `trend_core` with its holdout, 56 MB for the 14-strategy `live_paper` walk-forward). |
| Network | Only `aurum data download` and the paid desk commands (`aurum desk run`, `aurum desk replay`) use the network. Research, the tests and the desk demo run offline. |

**Operating systems.** Research, backtesting, RL and paper trading work on macOS, Linux and
Windows. The MetaTrader 5 adapter is Windows-only: the official `MetaTrader5` Python package
exists only for Windows, because it drives a local MT5 terminal. On macOS or Linux, use the
paper broker, which is the default.

**Linux and PyTorch.** The `[rl]` extra pulls in `torch`. On Linux, the default PyTorch wheel
downloads several GB of CUDA libraries. Unless you want GPU training, install the CPU-only
wheel *before* the extras, as the CI does:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Install

```bash
git clone https://github.com/zero-was-here/tradingbot.git && cd tradingbot
git checkout aurum-v2                                   # if your clone's default branch does not have Aurum v2 yet
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                 # core + test tools
# or
pip install -e ".[all]"                                 # core + data, rl, agents and dev extras
```

The core dependencies are numpy, pandas, scipy, scikit-learn, PyYAML, pyarrow, matplotlib and
requests. The optional extras, as defined in `pyproject.toml`:

| Extra | Installs | What it enables | Without it |
|---|---|---|---|
| `data` | `yfinance>=0.2.40` | Yahoo macro series (dxy, us10y, vix, spx, silver, gold_fut, oil) in `aurum data download` | The Yahoo series are skipped with a warning. FRED series still download. `macro_factor` and `risk_off` need the Yahoo series. |
| `rl` | `torch>=2.2`, `gymnasium>=0.29`, `stable-baselines3>=2.3` | `aurum rl train` and running a trained `rl_ppo` policy | `aurum rl train` exits with code 3 and an install hint. Everything else works. |
| `agents` | `anthropic>=1.8` | Live Claude calls: `aurum desk run`, `aurum desk replay`, `live.use_desk` | `aurum desk demo` still works: it uses a scripted fake client. |
| `mt5` | `MetaTrader5>=5.0.45` (Windows only) | `live.broker: mt5` | Paper broker only. **Not included in `[all]`**. Install it separately on Windows with `pip install -e ".[mt5]"`. |
| `dev` | `pytest>=8`, `ruff>=0.13` | Tests and linting (see [development.md](development.md)) | |
| `docs` | `mkdocs>=1.6`, `mkdocs-material>=9.5` | Browsing these pages as a local site with `mkdocs serve` | Read the Markdown files directly, for example on GitHub |
| `all` | `data`, `rl`, `agents`, `dev` | Everything except `mt5` and `docs` | |

Importing `aurum` never loads the optional packages eagerly. CI checks that the CLI, the
strategy registry and the agents package import on a core-only install.

Check the install. Both commands are offline:

```bash
aurum --help            # or: python -m aurum --help
aurum strategies list   # 15 registered strategies, with warm-up bars and a one-line description
```

## Try it without downloading data

Any config can run on generated data instead of files. Setting a `data.synthetic` key replaces
the data store with `aurum.data.synthetic.make_synthetic_bars` bars, plus synthetic macro
series and events. The default model is a random walk (`gbm`), which has no edge by
construction, so this doubles as a sanity check: expect Sharpe confidence intervals that
straddle zero.

```bash
aurum walkforward -c configs/trend_core.yaml \
  --set data.synthetic.n=30000 --set walkforward.holdout_start=null
```

This runs the full walk-forward protocol (4 folds, 7 strategies, tearsheet); it took 6 to 19
seconds of wall time on the machines used for this page, depending on load. Rate-based financing logs a warning (once per worker process): the synthetic macro
set has no `fedfunds` series, so every rollover uses `costs.financing.fallback_rate` (3%).
The rest of the output has the same shape as the real-data run described
[below](#your-first-walk-forward).

## Download the market data

```bash
aurum data download      # Dukascopy XAUUSD bid/ask 2012 -> yesterday (UTC) + Yahoo/FRED macro
aurum data info          # verify: row counts, ranges, median spreads, hashes
```

`aurum data download` has two passes:

1. **Bars.** A prefetch pass fills `cache/dukascopy/` with raw LZMA-compressed candle files
   from Dukascopy's free datafeed. It fetches the newest days first. Files already in the cache
   are skipped, so the pass is **resumable**. Individual failures are counted, not raised. An
   adaptive limiter reduces concurrency when the feed throttles (HTTP 429/503). A build pass
   then decodes the cache into mid-price bars with the recorded bid/ask spread. It writes
   `data_store/xauusd_{M15,H1,H4,D1,D1_nyclose}.parquet` and `data_store/manifest.json`.
2. **Macro.** Yahoo series (needs the `[data]` extra) and FRED series (`real10y`,
   `breakeven10y`, `fedfunds`) go to `data_store/macro/`. Each row is stamped with an
   `available_at` publication time, so the series can be joined point-in-time.

**Time and size.** The free feed is heavily rate-limited. The first full download takes about
an hour or more, and the full 2012–2026 minute cache is about 130 MB. If it stops early
(for example with `--time-budget`, or after Ctrl-C), run the same command again: it resumes
from the cache. Rebuilding the data store from a complete cache with `--offline` took about 11
seconds (peak memory about 1.8 GB) when the published dataset was built
([INTERFACES.md](INTERFACES.md)).

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `data_store` | Output directory for the parquet files, `manifest.json` and `macro/` |
| `--symbol` | `XAUUSD` | Dukascopy symbol |
| `--start` | `2012-01-01` | First day to download |
| `--end` | yesterday (UTC) | Inclusive last day |
| `--timeframes` | `all` | `all` builds M15, H1, H4, D1 and D1_nyclose with a manifest. A list such as `H1,H4` downloads just those timeframes directly, with no prefetch pass and no manifest. |
| `--cache` | `cache/dukascopy` | Raw file cache (resumable) |
| `--workers` | `4` | Download workers. The prefetch pass caps them at 8. |
| `--time-budget` | none | Seconds for the prefetch pass. It stops cleanly when the budget runs out; re-run to resume. |
| `--offline` | off | Build the bars only from the local cache. Macro is still fetched unless you also pass `--no-macro`. |
| `--no-bars` | off | Skip the Dukascopy bars |
| `--no-macro` | off | Skip the Yahoo/FRED macro series |
| `--macro-start` | `2011-01-01` | First day of the macro series |
| `--macro-cache` | `cache/macro` | Macro download cache |

**The manifest.** `data_store/manifest.json` records, for every file, the row count, first and
last bar, a SHA-256 content hash and a data-quality report. The report covers gaps, the
zero-spread fraction, spread quantiles and median spread by year, flat bars and return outliers.
`build_path` says how the bars were built. `"M1"` means every file was aggregated from minute
candles, which is the specification's path and the one behind [RESULTS.md](RESULTS.md). If the
minute cache did not yet cover the whole range, the build falls back to monthly hour files for
H1/H4/D1 (`"build_path": "H1"`). In that case M15 covers only the contiguous cached range.
Re-run the download until `aurum data info` reports `via M1`.

The published manifest is committed to git (every other file under `data_store/` is ignored).
After a download, `git diff data_store/manifest.json` shows whether your files hash to the
same values as the ones behind the published results. With a later `--end`, the files
legitimately differ.

`aurum data info` on the data store used for this documentation:

```text
file                       timeframe    rows                      first                       last  median_spread          hash  check
--------------------------------------------------------------------------------------------------------------------------------------
xauusd_D1.parquet                 D1    4585  2012-01-01 00:00:00+00:00  2026-09-25 00:00:00+00:00          0.343  b9e5cd6df941     ok
xauusd_D1_nyclose.parquet         D1    3841  2012-01-01 22:00:00+00:00  2026-09-24 22:00:00+00:00          0.337  131df1e6ed86     ok
xauusd_H1.parquet                 H1   87829  2012-01-01 22:00:00+00:00  2026-09-25 20:00:00+00:00          0.333  a9d73169411a     ok
xauusd_H4.parquet                 H4   23581  2012-01-01 20:00:00+00:00  2026-09-25 20:00:00+00:00          0.337  2e00965f1bd5     ok
xauusd_M15.parquet               M15  349660  2012-01-01 22:45:00+00:00  2026-09-25 20:45:00+00:00          0.332  58144724438a     ok

macro:
series        rows       first        last          last_available_at
---------------------------------------------------------------------
breakeven10y  3935  2011-01-03  2026-09-25  2026-09-28 21:30:00+00:00
dxy           3957  2011-01-03  2026-09-25  2026-09-25 22:30:00+00:00
fedfunds      5746  2011-01-01  2026-09-24  2026-09-25 21:30:00+00:00
...
vix           3958  2011-01-03  2026-09-25  2026-09-25 21:30:00+00:00

manifest: built 2026-09-27T00:43:45.881797+00:00 via M1 from dukascopy datafeed
```

`check` re-verifies each file's stored hash on load. `last_available_at` is when the last row
was published, which can be later than its observation date: FRED prints become usable at
21:30 UTC on the next US business day. Timestamps are bar **open** times in UTC. Details are in [data.md](data.md).

## Your first walk-forward

`configs/trend_core.yaml` is the pre-registered low-turnover book: `tsmom`, `ema_cross`,
`donchian`, `kalman_trend`, `vol_squeeze`, `macro_factor` and `risk_off` on H1 bars, with
everything else inherited from `configs/default.yaml`.

```bash
aurum walkforward -c configs/trend_core.yaml
```

This loads H1 bars from 2013 (`data.start`), splits them into rolling 3-year train and 6-month
test folds, and refits every trainable piece per fold on past data only. It stitches the
out-of-sample (OOS) forecasts, runs **one** continuous backtest per book through the shared
sizer, risk manager and cost model, and computes deflated statistics. The
[research page](research.md) explains the protocol.

> **The holdout.** Both shipped research configs set `walkforward.holdout_start: "2025-01-01"`.
> The plain command above also evaluates the 2025–26 holdout and appends that look to
> `runs/holdout_ledger.jsonl`. The published protocol has already used this holdout (see
> [RESULTS.md](RESULTS.md)). Re-running the *same* deterministic config is not counted as a new
> trial, but a changed config is. For your own experiments, stay out of the holdout:
>
> ```bash
> aurum walkforward -c configs/trend_core.yaml \
>   --set data.end=2024-12-31T23:59:59Z --set walkforward.holdout_start=null
> ```

Output of the plain command on the data store above, abbreviated with `...`. The numbers are the
ones published in [RESULTS.md](RESULTS.md): the run is deterministic.

```text
loaded 81587 H1 bars 2013-01-01 23:00:00+00:00 -> 2026-09-25 20:00:00+00:00 (2.3s); 7 strategies
2026-09-28 10:18:42,699 WARNING aurum.portfolio.combiner: ForecastCombiner: no strategy with a positive net Sharpe in train: all weights 0, the combined book stays flat
== walkforward: trend_core  config 06a2fca027fc  data b05d27bf5688
OOS 2015-12-16 20:00:00+00:00 -> 2024-12-31 21:00:00+00:00  (53462 bars, 18 fold(s), train 17834 / test 2972 bars, purge 0, embargo 24, n_trials 23)

book          sharpe  ci_lower  ci_upper    psr    dsr    cagr  ann_vol  max_drawdown  n_trades  total_costs  n_halt_episodes
-----------------------------------------------------------------------------------------------------------------------------
combined      -0.170    -0.821     0.482  0.304  0.007  -0.008    0.040        -0.123       875       11,749                0
tsmom          0.003    -0.651     0.656  0.504  0.026  -0.001    0.050        -0.151      1230       14,958                0
...
benchmark      0.463    -0.154     1.131  0.921  0.921   0.066    0.162        -0.314         1       26.236                0

PBO (CSCV, 16 splits, 7 strategies): 0.898  prob OOS loss of IS-best 0.838

== FINAL HOLDOUT (evaluated once) 2025-01-01 23:00:00+00:00 -> 2026-09-25 20:00:00+00:00
book          sharpe  ci_lower  ci_upper    psr    dsr    cagr  ann_vol  max_drawdown  n_trades  total_costs  n_halt_episodes
...

notes:
  - fold 13 combiner: 60.0% of the risk budget left UNALLOCATED: max_weight=0.400 cannot be met by the 1 positively scored strategies; ...
  ...
  - 1 fold(s) used equal combiner weights (not enough earlier OOS history)

timing: features 0.0s, strategies 64.1s, combine 2.3s, backtests 112.1s, holdout 19.2s, stats 6.8s, write 23.7s, wall 233.0s

report dir: runs/walkforward_<utc>_06a2fca0
tearsheet:  runs/walkforward_<utc>_06a2fca0/tearsheet.html
```

How long it takes depends on your cores and how busy the machine is. The timing line above
comes from a 10-core machine under a very high load average (about 77 seconds of CPU, 233
seconds of wall time). On the same machine when idle, the command took about 12 seconds of
wall time, in line with the README's figure of about 15 seconds. Your data will also differ if you downloaded it
later: `--end` defaults to yesterday, which lengthens the holdout window. For byte-identical
data, use `aurum data download --end 2026-09-25` and compare the data hash in
`provenance.json` with the hashes at the top of [RESULTS.md](RESULTS.md).

**Reading the table.** Each row is a book: the combined portfolio, each strategy traded on its
own, and `benchmark` (buy-and-hold with the same financing costs).

| Column | Meaning |
|---|---|
| `sharpe` | Annualised Sharpe ratio of **daily** returns (UTC dates, 252 days per year) |
| `ci_lower`, `ci_upper` | 95% stationary-bootstrap confidence interval for that Sharpe |
| `psr` | Probabilistic Sharpe ratio: probability that the true Sharpe is above 0 |
| `dsr` | Deflated Sharpe ratio: the PSR against the best Sharpe expected from `n_trials` skill-less trials. The protocol's bar is 0.95. |
| `cagr`, `ann_vol`, `max_drawdown` | Fractions (`-0.123` = −12.3%) |
| `n_trades` | Round trips |
| `total_costs` | USD of spread, slippage and commission. Financing is reported separately as `swap` in `costs.csv`. |
| `n_halt_episodes` | Number of separate periods in which the risk manager kept the book halted (with the research limits, these are daily-loss halts) |

`PBO` is the probability of backtest overfitting across the strategy set (CSCV). The `notes`
flag folds where the net-of-cost combiner left risk unallocated or kept the book flat. The
[research page](research.md) covers each statistic.

**The run directory.** The default is `runs/walkforward_<UTC timestamp>_<first 8 hex of the config hash>`.
Use `--out DIR` to choose the path.

| Path | Contents |
|---|---|
| `summary.json` | Everything the terminal shows and more (settings, folds, weights, costs, PBO, notes, timing, provenance). `aurum report --run DIR` reprints it; add `--json` for raw JSON. |
| `config.yaml` | The fully resolved config (secrets are never written) |
| `provenance.json` | Config hash, bar and macro data hashes, calendar hash, git SHA and dirty flag, Python and package versions, the command line |
| `stats.csv`, `costs.csv` | Per-book statistics and cost attribution (spread, slippage, commission, swap, gross vs net PnL) |
| `folds.csv`, `weights.csv`, `fold_sharpe.csv` | Fold windows and per-fold OOS Sharpe, combiner weights and FDM per fold, per-strategy Sharpe per fold |
| `oos_forecasts.parquet` | Stitched OOS forecast of every strategy and of `combined`, plus the `fold` id (input to `train-final --from-run`) |
| `books/<book>/` | For `combined`, each strategy and `benchmark`: `timeseries.parquet` (equity, returns, positions, target, forecast, per-bar costs and PnL), `trades.csv`, `fills.csv`, `risk_events.csv`, `metrics.json`, `meta.json` |
| `tearsheet.html` | Self-contained HTML report of the combined OOS book |
| `holdout/` | Only when a holdout was evaluated: `stats.csv`, `costs.csv`, `holdout.json`, `oos_forecasts.parquet`, `books/{combined,benchmark}/`, `tearsheet.html` |
| `../holdout_ledger.jsonl` | Next to the run directories: one line per holdout evaluation |

**The tearsheet.** Open `tearsheet.html` in a browser. It is a single file with inline charts
(light and dark variants) and no external assets. Sections, top to bottom:

| Section | What to look at |
|---|---|
| Header tiles | Total return, CAGR, Sharpe (daily), max drawdown, annual vol, PSR |
| Equity | Equity versus the rebased benchmark on a log scale. Also benchmark return, Sharpe and drawdown, plus the book's daily correlation and beta to it. |
| Drawdown | Underwater curve |
| Rolling Sharpe | 6-month (126 trading-day) window |
| Monthly returns | Month-by-year return grid with yearly totals |
| Return distribution | Histogram of daily returns |
| Position & exposure | Signed lots over time with the combined forecast, time in market, long/short share |
| Cost attribution | Spread, slippage, commission and net financing, cost drag per year, PnL before and after costs |
| Trade statistics | Round-trip statistics, also broken down by exit reason |
| Statistical confidence | Sharpe with its standard error, bootstrap CI, PSR, DSR with the number of trials, minimum track record, skew and kurtosis, PBO |
| Walk-forward folds | OOS Sharpe per fold (chart and table) |
| Strategy weights | Combiner weights per fold |
| Risk-manager interventions | Bars where the risk manager changed or blocked the requested position (shown only when there were any) |
| All metrics | Every metric of the book |
| Notes & provenance | Run metadata, the full config, notes, per-strategy stats, per-fold Sharpe by strategy, costs by book |

For a larger run, `configs/default.yaml` evaluates all 14 default strategies, including the
ML models. The README quotes about 3 minutes on 8 cores for it.

## The offline LLM-desk demo

```bash
aurum desk demo
```

This runs one complete desk cycle offline. A scripted fake Claude client plays every agent on
synthetic, anonymised data. The Chief consults two standing specialists in parallel, creates
an ad-hoc event-risk analyst, and then scales the quant forecast by 0.5 under the default
`overlay` policy. It needs no API key and no network. Real output:

```text
LLM desk demo (offline scripted client, synthetic anonymised data)
status:          decided
decision time:   2020-02-10 05:00:00+00:00
quant forecast:  -0.300
decision:        action=scale scale=0.5 forecast=-0.1497952633636686 confidence=0.55 horizon=24 bars
rationale:       Mixed specialist views and upcoming event risk: halve exposure.
key risks:       event risk
final forecast:  -0.150  (policy overlay: scale)
memos:           3 adhoc:event_risk_analyst=neutral, macro_strategist=bearish, risk_officer=neutral
cost:            $0.0900  tokens in=9000 out=1800 cache_read=0
```

The `cost` line is the desk's budget ledger at list prices, applied to the token counts the
scripted client *reports*. No API call is made and nothing is charged. Without
`--journal-dir`, the cycle journal stays in memory. Pass `--journal-dir DIR` to get the JSONL
journal of prompts, tool calls, memos, the decision and usage.

`aurum desk run` (one cycle) and `aurum desk replay` call the **paid** Claude API and need
`ANTHROPIC_API_KEY` in the environment, never in YAML. `desk replay` prints a cost estimate and
refuses to start without `--yes`. Read [llm-desk.md](llm-desk.md) before using either.

## Paper-trading quickstart

Two steps: fit a production artifact, then run the bar-close loop against the paper broker.

```bash
aurum train-final -c configs/live_paper.yaml --out artifacts/live_paper
aurum live run    -c configs/live_paper.yaml --max-cycles 24
```

**1. `train-final`** fits everything on the data up to `--cutoff` (default: the last bar):

- the feature scaler and the trainable strategies, on the training window that ends at the
  cutoff;
- the combiner weights, on stitched **out-of-sample** walk-forward forecasts.

Without `--from-run <walk-forward dir>`, it first runs that walk-forward itself: all 14
strategies of `live_paper.yaml`, written like `aurum walkforward`, including a holdout
evaluation that goes into the ledger. Pass `--from-run` with a run made from the same config
(same config hash) to skip it. On the 10-core machine used for this page, the whole command
took about 3.5 minutes when idle and 518 seconds under heavy load; the README quotes about
3–7 minutes. It ends with:

```text
artifact: artifacts/live_paper
  fit window 2023-09-21 02:00:00+00:00 -> 2026-09-25 20:00:00+00:00; combiner oos_history(17827 bars) [net of estimated costs c_t=(spread_eff/2+slippage+commission)/(close*vol)]
  weights: tsmom=0.09, ema_cross=0.10, donchian=0.10, kalman_trend=0.10, zscore_fade=0.00, rsi2=0.00, bollinger_revert=0.00, vol_squeeze=0.17, orb=0.00, macro_factor=0.05, risk_off=0.18, intraday_seasonality=0.00, ml_gbm=0.00, meta_label=0.20  (sum 1.00, fdm 1.78)
  feature reference: 0 columns; OOS source: runs/walkforward_<utc>_<hash8>
```

These weights are what the combiner learned from past OOS forecasts. They are not evidence of
an edge (see [RESULTS.md](RESULTS.md)). The artifact directory holds `manifest.json` (a
SHA-256 for every file, and an HMAC when `AURUM_ARTIFACT_KEY` is set), `strategies.pkl`,
`strategies.json`, `combiner.pkl` and `backtest.json`. An artifact whose strategies use the
shared feature pipeline also gets `pipeline.json` and `feature_reference.json`, which is used
for drift monitoring. `live_paper.yaml`'s strategies do not, hence `0 columns`. The artifact
contains pickles, so only load artifacts you produced yourself.

**2. `live run`** starts the runner. With `live_paper.yaml`, it is in dry-run mode on the paper
broker. The paper broker replays `data_store/xauusd_H1.parquet` on a simulated clock,
starting after `paper.warmup_bars` (6,000 bars, i.e. December 2012). `--max-cycles 24` stops
after 24 bar closes; `--until <UTC time>` stops at a time instead. Output:

```text
live: broker=paper dry_run=True allow_live_real=False magic=20260926 artifact=artifacts/live_paper
resolved runner config: runs/live/paper/aurum_live_config.yaml
aurum live runner [DRY-RUN PAPER] XAUUSD H1 magic=20260926 state=runs/live/paper
```

Everything else goes to the state directory (`live.state_dir`, here `runs/live/paper/`):

| File | Contents |
|---|---|
| `decisions.jsonl` | A `start` record, then one record per bar: forecasts per strategy, combined and final forecast, sizing breakdown, requested and approved lots, risk decision, planned orders and fills, equity. Ends with a `stop` record. |
| `heartbeat.json` | Status, mode, last bar, number of decisions, halted flag |
| `runner_state.json` | Last processed bar, previous final forecast, cooldown state, PnL-band data |
| `risk_state.json` | The risk manager's persistent state, including the kill switch |
| `runner.lock` | Single-runner lock for this state directory |
| `aurum_live_config.yaml` | The resolved runner config (no secrets) |
| `oms_state.json`, `paper_broker.json` | Appear only when orders are actually sent (`live.dry_run: false`) |
| `alerts.jsonl` | Appears when an alert fires |

In dry-run mode, orders are planned and logged (`"status": "dry_run"` with the `planned` legs)
but never sent, so the paper account stays at its initial equity. Set `live.dry_run: false`,
or pass `--set live.dry_run=false`, to fill orders on the paper broker.

Before you read anything into a paper run:

- **The default replay is in-sample.** The artifact was fitted on data up to the last bar, and
  the replay starts in December 2012, so it only tests the plumbing, not the strategy. For
  an out-of-sample replay, fit with `train-final --cutoff <date>` and set
  `live.options.paper.start` to a time after that date.
- **The runner resumes.** A second `live run` with the same `state_dir` continues after the last
  processed bar. Use a fresh `live.state_dir` to start over. Deleting `risk_state.json` does
  not reset a halted book: if it is missing while the directory still holds earlier runner or
  OMS state, the runner starts halted. Only `reset_halt(confirm="RESET")` clears a halt.
- **Real-time paper trading** on MT5 quotes uses `live.options.paper.data: mt5` (Windows,
  MT5 terminal). A real-money account is refused unless `live.allow_live_real: true` **and**
  `--i-understand-real-money` are both given. Given [RESULTS.md](RESULTS.md), we recommend
  against it. See [live-trading.md](live-trading.md).

## When something goes wrong

| Symptom | Cause and fix |
|---|---|
| ``aurum: bars file data_store/xauusd_H1.parquet not found (run `aurum data download` or set data.bars_path)`` | No data yet, or you are not in the repository root |
| `aurum: configuration error: ...` (exit code 2) | Unknown key, bad value or missing file. The message lists every problem, with "did you mean" hints. `aurum config validate -c FILE [--set ...]` checks a config without running it. |
| Exit code 3 | An optional component is missing, for example `aurum rl train` without the `[rl]` extra |
| Exit code 4 | `aurum desk replay` without `--yes` (confirmation required) |
| Terse one-line error | Re-run with `--traceback`, or `-v`/`-vv` for INFO/DEBUG logging |
| `macro directory data_store/macro not found: running without macro data` | Download macro (drop `--no-macro`). `macro_factor` and `risk_off` then have no input. |

## Where to go next

- [architecture.md](architecture.md): how the pieces fit together, the timing model, and
  how research, RL, paper and live share one execution path
- [configuration.md](configuration.md): every config section and the `--set` syntax
- [cli.md](cli.md): every command and flag
- [data.md](data.md), [features.md](features.md), [strategies.md](strategies.md),
  [ml-and-rl.md](ml-and-rl.md): the research building blocks
- [portfolio-and-risk.md](portfolio-and-risk.md) and
  [execution-and-costs.md](execution-and-costs.md): sizing, limits, fills and financing
- [research.md](research.md), [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) and
  [RESULTS.md](RESULTS.md): the methodology and what it found
- [llm-desk.md](llm-desk.md) and [live-trading.md](live-trading.md): the LLM desk and the
  production path
- [development.md](development.md): tests, linting and contributing
