# Aurum v2 documentation

Aurum is a systematic research and trading stack for spot gold (XAUUSD), written as one
Python package, `aurum`, with one command-line tool, `aurum`. It takes point-in-time market
data (Dukascopy bid/ask history, Yahoo and FRED macro series, a rule-based NFP/FOMC
calendar) through a leakage-tested feature library and 15 registered strategies (rule-based,
machine learning and reinforcement learning). A net-of-cost combiner, a volatility-targeting
sizer and a reduce-only risk manager with a persistent kill switch sit downstream, and one
bar-level simulator charges spread, slippage, commission and rate-based financing. The same
classes run in walk-forward research, the RL environment, the LLM-desk replay and the
live/paper runner. An optional **LLM trading desk** puts a Claude-powered Chief Investment
Officer, standing specialists and ad-hoc agents on top of the quant book. It can only return
a bounded forecast that still goes through the same sizer and risk manager.

**Honest status.** Aurum was evaluated under a protocol written before the first full run
([RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)). On 14 years of real bid/ask data, **no
strategy or combined book showed a statistically demonstrated edge**: none came close to the
protocol's walk-forward deflated-Sharpe bar of 0.95. The numbers, the one-time holdout and
what they do and do not mean are in [RESULTS.md](RESULTS.md), which the other pages link to
rather than reinterpret. Aurum therefore ships as a research and paper-trading platform.
Live trading is dry-run by default, and real money needs two separate opt-ins. The LLM desk
has not been evaluated under the protocol at all. Nothing in this documentation is
investment advice.

**On this page**

- [Pages](#pages)
- [Reading paths](#reading-paths)
- [Glossary](#glossary)
- [Conventions used in these pages](#conventions-used-in-these-pages)

## Pages

### Start here

| Page | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install, a no-download synthetic run, the data download, a first walk-forward with its tearsheet, the offline desk demo and a dry-run paper session |
| [Architecture](architecture.md) | Design principles, the end-to-end data flow, decision timing, the module map, parity guarantees and their tests, extension points |

### Concepts

| Page | What it covers |
|---|---|
| [Research](research.md) | Walk-forward protocol, per-fold refits, daily-return statistics (bootstrap CI, PSR, DSR, MinTRL, PBO), the holdout ledger, run directories, tearsheets, splitters |
| [Research protocol](RESEARCH_PROTOCOL.md) | The pre-registered protocol: periods, configurations, statistics and the selection rule |
| [Results](RESULTS.md) | The published walk-forward and holdout results and the verdict under the protocol |

### Components

| Page | What it covers |
|---|---|
| [Data](data.md) | The canonical bars schema and `available_at`, point-in-time joins, resampling, Dukascopy, MT5/CSV imports, macro series and their lags, the calendar, the parquet store and manifest, synthetic data |
| [Features](features.md) | The 12 causal feature groups, `FeaturePipeline` (fit on train only, transform, persistence), warm-ups, adding a group, leakage tests, live parity |
| [Strategies](strategies.md) | The forecast contract, Carver scaling, the 11 rule strategies and `intraday_seasonality`, configuration, writing your own |
| [ML and RL](ml-and-rl.md) | Triple-barrier labels, `ml_gbm`, `meta_label`, the Gymnasium environment, PPO training and the `rl_ppo` adapter |
| [Portfolio and risk](portfolio-and-risk.md) | The net-of-cost forecast combiner, volatility-targeting sizing, the risk manager and kill switch, VaR/ES, volatility and regime models |
| [Execution and costs](execution-and-costs.md) | The execution simulator, the cost model, overnight financing, the backtest engine, `BacktestResult` and metrics |
| [LLM desk](llm-desk.md) | The committee process, tools, decision schema, `DecisionPolicy`, failure handling, budgets, journals and prompt-injection stance |

### Operations

| Page | What it covers |
|---|---|
| [Live and paper trading](live-trading.md) | Artifacts, the bar-close runner, paper and MT5 brokers, the OMS, real-money guards, state files, the kill switch, monitoring and an operations runbook |
| [CLI reference](cli.md) | Every `aurum` command and option, exit codes, environment variables and where each command writes |

### Reference

| Page | What it covers |
|---|---|
| [Configuration](configuration.md) | Loading, `extends` and `--set`, validation, secrets, the config hash, the shipped configs and every key |
| [Module interfaces](INTERFACES.md) | The per-module API map and integration notes recorded while v2 was built |
| [SPEC.md](../SPEC.md) | The binding engineering contract between modules |

### Project

| Page | What it covers |
|---|---|
| [Development](development.md) | Repository layout, tests and their guards, lint, CI, coding rules, adding strategies, feature groups and brokers |
| [CHANGELOG.md](../CHANGELOG.md) | What changed in each release, including the v1 to v2 rebuild |
| [SECURITY.md](../SECURITY.md) | Secrets handling, live-trading safety and how to report a vulnerability |

## Reading paths

**Researcher** (you want to test an idea without fooling yourself):

1. [Getting started](getting-started.md): install, run the synthetic walk-forward, then the
   real one outside the holdout.
2. [Architecture](architecture.md#decision-timing): the timing model every component follows.
3. [Data](data.md) and [Features](features.md): what a decision may see and when.
4. [Strategies](strategies.md) and, for learned models, [ML and RL](ml-and-rl.md).
5. [Research](research.md): how folds, DSR, PBO and the holdout ledger work, and the
   checklist for [your own pre-registered study](research.md#running-your-own-pre-registered-study).
6. [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) and [RESULTS.md](RESULTS.md): what has
   already been tested and spent.
7. [Configuration](configuration.md) when you start writing your own YAML.

**Live operator** (you want to run the paper broker, or MT5, safely):

1. [Getting started](getting-started.md#paper-trading-quickstart): build an artifact and run a
   dry-run paper session.
2. [RESULTS.md](RESULTS.md): the evidence, before you consider anything beyond paper.
3. [Portfolio and risk](portfolio-and-risk.md#risk-manager): limits, blackouts and the kill
   switch.
4. [Execution and costs](execution-and-costs.md): how fills and financing are modelled, and
   what is not modelled.
5. [Live and paper trading](live-trading.md): start with
   [Read this first](live-trading.md#read-this-first), then the real-money guards, the state
   directory and the [operations runbook](live-trading.md#operations-runbook).
6. [LLM desk](llm-desk.md), only if you set `live.use_desk: true`: failure handling and costs.
7. [CLI reference](cli.md#live), [configuration](configuration.md#live) and
   [SECURITY.md](../SECURITY.md).

**Contributor** (you want to change the code):

1. [Architecture](architecture.md): the pipeline, the shared components and the parity tests.
2. [Development](development.md): setup, the test suite, lint, CI and the coding rules.
3. [SPEC.md](../SPEC.md) and [INTERFACES.md](INTERFACES.md): the contracts between modules.
4. The extension guides:
   [adding a strategy](development.md#adding-a-strategy),
   [adding a feature group](development.md#adding-a-feature-group) and
   [adding a broker adapter](development.md#adding-a-broker-adapter).
5. [Features: leakage testing](features.md#leakage-testing) and
   [the safety net](development.md#the-safety-net-leakage-random-walk-and-live-safety-harnesses):
   the harnesses a change must pass.

## Glossary

| Term | Meaning |
|---|---|
| `available_at` | The UTC instant from which a row may influence a decision. For a bar it is `open + timeframe` (the bar's close); for a macro print it is its publication time. Every join is an as-of join on it. See [data.md](data.md#the-bars-contract). |
| Forecast | A strategy's output for bar `t`: one number in `[-1, 1]`, decided at the close of `t` and filled at the open of `t+1`. It expresses conviction, not a position size. See [strategies.md](strategies.md#the-forecast-contract). |
| OOS (out of sample) | Data a model was not fitted or selected on. In a walk-forward each test fold is OOS for the models fitted on its training window, and the stitched test folds form the OOS track record. See [research.md](research.md#the-walk-forward-protocol). |
| Purge and embargo | Bars left out between a training window and its test block. The purge covers the longest label horizon of a trainable strategy, so no training label reaches into the test block; the embargo adds a further gap (24 bars in `default.yaml`). See [research.md](research.md#fold-geometry). |
| PSR | Probabilistic Sharpe Ratio: the probability that the true Sharpe ratio is above zero, given the track-record length, skew and kurtosis. See [research.md](research.md#probabilistic-sharpe-ratio-psr). |
| DSR | Deflated Sharpe Ratio: the PSR measured against the best Sharpe expected from `n_trials` skill-less trials, which corrects for how many configurations were tried. The protocol's bar is 0.95. See [research.md](research.md#deflated-sharpe-ratio-dsr-and-n_trials). |
| PBO | Probability of Backtest Overfitting, estimated with combinatorially symmetric cross-validation (CSCV): how often the strategy that looked best in-sample is no better than the median out of sample. It describes selection among the strategies of one run. See [research.md](research.md#probability-of-backtest-overfitting-pbo-cscv). |
| FDM | Forecast diversification multiplier, `1 / sqrt(w' C w)`: rescales a weighted average of imperfectly correlated forecasts back to the average magnitude of its inputs (Carver). Used inside some strategies and by the combiner, with a cap. See [portfolio-and-risk.md](portfolio-and-risk.md#forecast-diversification-multiplier). |
| Holdout | The final period, from `walkforward.holdout_start` on, excluded from every fold and evaluated once (2025-01-01 in `default.yaml` and the configs that inherit it; `fast.yaml` uses 2026-01-01). The published protocol has already used the 2025–26 holdout. See [research.md](research.md#the-holdout-and-the-ledger). |
| Holdout ledger | `holdout_ledger.jsonl` next to the run directories: one line per holdout evaluation. When a different config looks at an overlapping window, the run warns and raises the DSR `n_trials`. See [research.md](research.md#the-holdout-and-the-ledger). |
| Artifact | The directory `train-final` or `live artifact` writes for the live runner: fitted strategies, combiner, optional feature pipeline and OOS statistics, with a hash manifest. It contains pickles, so load only your own. See [live-trading.md](live-trading.md#trading-artifacts). |
| Dry run | The live runner's default mode: every decision, including the planned orders, is logged, but no order is sent. `aurum live run` has no flag that turns it off: set `live.dry_run: false` in the config (or `--set live.dry_run=false`). See [live-trading.md](live-trading.md#dry-run). |
| Kill switch | The risk manager's halt on a daily-loss, max-drawdown or zero-equity breach (or a manual or state-file halt). While halted, every decision approves 0 lots, so the book goes flat. With the live limits the halt is persisted in `<state_dir>/risk_state.json`, survives restarts and needs a human `reset_halt(confirm="RESET")`; research limits let a daily-loss halt clear the next day. See [portfolio-and-risk.md](portfolio-and-risk.md#kill-switch) and [live-trading.md](live-trading.md#kill-switch). |
| Magic number | The MetaTrader 5 tag (`live.magic`, default `20260926`) that marks this runner's orders. The broker adapter and OMS only read, modify and close positions with their own magic on their own symbol. See [live-trading.md](live-trading.md#metatrader-5-adapter). |
| Real-money double opt-in | A non-demo account is refused unless `live.allow_live_real: true` is in the config **and** `--i-understand-real-money` is on the command line. See [live-trading.md](live-trading.md#real-money-double-opt-in-and-guards). |

## Conventions used in these pages

- Every command, flag, default and output shown was checked against the code and, where
  possible, produced by running it. Examples on synthetic data say so; synthetic numbers
  are noise by construction and are never performance claims.
- Paths are relative to the repository root, which is also where the CLI should be run:
  `data_store/`, `runs/` and `artifacts/` resolve against the working directory.
- Times are UTC and bars are labelled by their **open** time.
- Links into `aurum/`, `tests/` and the files at the repository root point at the source,
  so they work when you browse the repository on GitHub.

To browse these pages locally as a site, install the `docs` extra and run MkDocs from the
repository root:

```bash
pip install -e ".[docs]"
mkdocs serve            # then open http://127.0.0.1:8000
```
