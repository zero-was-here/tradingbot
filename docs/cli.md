# CLI reference

Everything Aurum does from the shell goes through one command, `aurum`, with eleven
subcommands. They cover downloading and inspecting data, validating configs, research runs
(`backtest`, `walkforward`, `report`), production fits (`train-final`, `live artifact`), the
LLM desk (`desk demo`, `desk run`, `desk replay`), RL training and the live runner. This
page lists every option with its type, default and meaning, what each command writes, and
how it exits. Every option was checked against `aurum/cli.py` and `aurum <command> --help`.
The example output was captured from real runs on synthetic data.

**On this page**

- [Invocation and global options](#invocation-and-global-options)
- [Command map](#command-map)
- [Exit codes and error output](#exit-codes-and-error-output)
- [`--config` and `--set`](#--config-and---set)
- [data](#data): [download](#aurum-data-download), [info](#aurum-data-info)
- [config](#aurum-config)
- [backtest](#aurum-backtest), [walkforward](#aurum-walkforward), [report](#aurum-report)
- [train-final](#aurum-train-final)
- [strategies list](#aurum-strategies-list), [features list](#aurum-features-list)
- [desk](#desk): [demo](#aurum-desk-demo), [run](#aurum-desk-run), [replay](#aurum-desk-replay)
- [rl train](#aurum-rl-train)
- [live](#live): [artifact](#aurum-live-artifact), [run](#aurum-live-run)
- [Environment variables](#environment-variables)
- [Where the CLI writes](#where-the-cli-writes)
- [A typical session](#a-typical-session)

## Invocation and global options

The `aurum` console script (`[project.scripts] aurum = "aurum.cli:main"` in
`pyproject.toml`) and `python -m aurum` are the same entry point:

```bash
aurum --help
python -m aurum --help          # identical; the program name is still "aurum"
```

Relative paths in configs and defaults (`data_store/`, `cache/`, `runs/`, `artifacts/`)
resolve against the **current working directory**. Run the CLI from the repository root
unless you have pointed those paths somewhere else.

Global options go **before** the command. `aurum strategies list -v` fails with
`unrecognized arguments: -v`; `aurum -v strategies list` works.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `-h`, `--help` | flag | | Show help and exit 0. Every command and subcommand has its own `--help`. |
| `-v`, `--verbose` | counter | 0 | Logging level: none = WARNING, `-v` = INFO, `-vv` = DEBUG. `-vv` also shows full tracebacks, like `--traceback`. |
| `--traceback` | flag | off | Re-raise unexpected errors so Python prints the full traceback. |

Log records go to stderr as `YYYY-MM-DD HH:MM:SS,mmm LEVEL logger: message`. Command
results are printed to stdout. Some long-running parts only report at INFO level, so use
`-v` to see them. For example, `aurum -v live run ...` logs one line per bar decision:

```text
2026-09-28 10:35:03,710 INFO aurum.live.runner: [DRY-RUN PAPER] XAUUSD decision 2020-10-14 05:00:00+00:00: combined -0.002 final -0.002 -> requested +0.00 approved +0.00 (current +0.00) status=dry_run
```

The same `main(argv)` function can be called from Python. It returns the exit code instead
of calling `sys.exit`, which is how `tests/test_cli.py` drives the CLI:

```python
from aurum.cli import main

code = main(["config", "validate", "-c", "configs/fast.yaml", "--set", "walkforward.test=3M"])
print("exit code:", code)
```

```text
# config OK: configs/fast.yaml  hash 7a6a811479f59e42e4b38a7f540b0b222ea5e0974dfabc03c6b00012e1824e85
exit code: 0
```

## Command map

| Command | What it does | Network | Paid API | Writes |
|---|---|---|---|---|
| `data download` | Dukascopy bid/ask bars and Yahoo/FRED macro into the data store | yes | no | `data_store/`, `cache/` |
| `data info` | Summarise bar files and macro series, verify stored hashes | no | no | nothing |
| `config show` / `config validate` | Print or check the resolved config and its hash | no | no | nothing |
| `backtest` | One train/test split (fit before `--start`, evaluate after) | no | no | run directory |
| `walkforward` | Walk-forward research run with DSR/PBO and an optional holdout | no | no | run directory, holdout ledger |
| `report` | Print the summary of a run directory | no | no | nothing |
| `train-final` | Production artifact fitted on all data up to a cutoff | no | no | artifact (+ walk-forward run) |
| `strategies list` / `features list` | Registry listings | no | no | nothing |
| `desk demo` | One offline desk cycle with a scripted fake client | no | no | optional journal |
| `desk run` | **One** real Claude desk cycle on the latest bar | yes | **yes** | desk journal |
| `desk replay` | Desk over a historical window through the same sizer/risk | yes (unless `--fake`) | **yes** (unless `--fake`) | replay directory, journal |
| `rl train` | PPO training (needs the `rl` extra) | no | no | `rl.out_dir` |
| `live artifact` | Fit strategies and combiner on the latest data into an artifact | no | no | artifact |
| `live run` | The bar-close live loop, **dry-run by default** | broker-dependent | only with `live.use_desk` | `live.state_dir` |

The table shows what the CLI itself does. `live run` with `live.broker: mt5` talks to a
MetaTrader 5 terminal. With `live.use_desk: true` it calls Claude at every decision.

## Exit codes and error output

`aurum/cli.py` defines the exit codes:

| Code | Name in code | When |
|---|---|---|
| 0 | `EXIT_OK` | Success, including `--help`. |
| 1 | `EXIT_ERROR` | Runtime error: a missing input file (`FileNotFoundError`, for example no bars file) or any other unexpected exception. |
| 2 | `EXIT_USAGE` | Usage or configuration error: argparse errors, any `ConfigError` (unknown key, bad value, secret in YAML or `--set`), a missing run or data directory, a refused holdout or budget argument, a missing `ANTHROPIC_API_KEY`. It is also returned by `live run` when the runner refuses a real-money account. |
| 3 | `EXIT_UNAVAILABLE` | An optional component is missing: `aurum.live.runner` cannot be imported, or the RL extras are not installed. |
| 4 | `EXIT_CONFIRM` | Confirmation required: `desk replay` without `--yes`. |
| 130 | | Interrupted with Ctrl-C (`KeyboardInterrupt`). The `live run` loop is the exception: while it runs, the runner traps SIGINT/SIGTERM, finishes the current cycle and exits normally. |

Errors are printed to stderr with an `aurum:` prefix:

```text
aurum: configuration error: walkforward.tset: unknown key (did you mean 'test'?); valid keys: ['anchored', 'combiner_fit', ...]
aurum: bars file data_store/xauusd_H1.parquet not found (run `aurum data download` or set data.bars_path)
aurum: error: DateParseError: Unknown datetime string format, unable to parse: garbage (use --traceback for details)
```

With `--traceback` or `-vv`, an unexpected exception (the last kind above) is re-raised.
Python then prints the traceback and exits with status 1. Configuration errors, `CLIError`s
and missing files are always reported as one line.

## `--config` and `--set`

Every command that reads a configuration takes these two options. They are required for
`backtest`, `walkforward`, `train-final`, `desk run`, `desk replay`, `rl train`,
`live artifact` and `live run`, and optional for `config` and `data info`.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--config`, `-c` | path | required (optional for `config`, `data info`) | YAML config file (see `configs/`). Without it, `config` uses the built-in defaults. |
| `--set KEY=VALUE` | string, repeatable | none | Override one value by dotted path. Applied after `extends:` inheritance and before validation. |

How `--set` works (`aurum.core.config.apply_overrides`):

- The value is parsed as YAML, so `3M` is a string, `7` an integer, `null` (or an empty
  value) is `None`, and `[...]`/`{...}` are lists and mappings.
- Missing intermediate mappings are created: `--set live.options.paper.warmup_bars=11000`
  works even if `live.options` is empty.
- A list is replaced wholesale.
- Secrets cannot be set this way. `--set agents.api_key=...` fails with
  `secrets cannot be overridden; set $ANTHROPIC_API_KEY`.
- The resulting config goes through the same strict validation as a file. Unknown keys get
  a "did you mean" hint, and every problem is reported at once (exit 2).

```bash
# shorter test windows
aurum config validate -c configs/fast.yaml --set walkforward.test=3M --set seed=7

# replace the strategy list
aurum config show -c configs/fast.yaml \
  --set 'strategies=[{name: tsmom}, {name: ema_cross, params: {fast: 16, slow: 64}}]'

# stay out of the 2025-26 holdout, as the protocol's selection runs did
aurum walkforward -c configs/trend_core.yaml \
  --set data.end=2024-12-31T23:59:59Z --set walkforward.holdout_start=null
```

Timestamps on the command line (`--start`, `--end`, `--at`, `--cutoff`, `--until`) are read
by pandas. A value without a time zone is taken as UTC. For `backtest --end` and
`desk replay --end`, a date-only value (10 characters or fewer) includes that whole day.

The full list of keys is in [configuration.md](configuration.md).

## data

### `aurum data download`

Downloads Dukascopy bid/ask history and the macro series into the data store. **This
command uses the network.** The free Dukascopy feed is heavily rate-limited, so a first
full download takes an hour or more. Re-running it resumes from the cache. See
[data.md](data.md) for the data layer itself.

```text
aurum data download [--out OUT] [--symbol SYMBOL] [--start START] [--end END]
                    [--timeframes TIMEFRAMES] [--cache CACHE] [--workers WORKERS]
                    [--time-budget TIME_BUDGET] [--offline] [--no-bars] [--no-macro]
                    [--macro-start MACRO_START] [--macro-cache MACRO_CACHE]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--out` | path | `data_store` | Data store directory, created if missing. |
| `--symbol` | string | `XAUUSD` | Dukascopy symbol. Bar files are named `{symbol.lower()}_{TF}.parquet`. |
| `--start` | date | `2012-01-01` | First day of bar data. |
| `--end` | date | yesterday (UTC) | Last day, inclusive. Also the macro end date. |
| `--timeframes` | string | `all` | `all` builds M15, H1, H4, D1 and D1_nyclose with a `manifest.json` (hashes and a quality report). A comma list (for example `H1,H4`) downloads each timeframe separately, with no manifest. Valid names: M1, M5, M15, M30, H1, H4, D1. |
| `--cache` | path | `cache/dukascopy` | Raw Dukascopy file cache. |
| `--workers` | int | `4` | Download concurrency. |
| `--time-budget` | float (seconds) | none | Time limit for the prefetch pass (only with `--timeframes all`). When it is reached, the command says so; re-run to resume. |
| `--offline` | flag | off | Build the bars from the local cache only (no prefetch, no Dukascopy requests). Macro is still fetched unless you also pass `--no-macro` (see below). |
| `--no-bars` | flag | off | Skip the bar data. |
| `--no-macro` | flag | off | Skip the Yahoo/FRED macro series. |
| `--macro-start` | date | `2011-01-01` | First day of macro data. |
| `--macro-cache` | path | `cache/macro` | Macro download cache. |

Behaviour:

- With `--timeframes all` and no `--offline`, a newest-first prefetch pass fills the cache
  first and prints `prefetch: N fetched, N cached, N failed, N not attempted in Ns`. The
  dataset is then built, one line per file with its row count, range and short sha256.
- The build aggregates every file from the minute data only when the minute cache covers
  the whole range. Otherwise, for example after a time-budgeted prefetch, H1/H4/D1 come
  from Dukascopy's monthly hour files and M15 covers only the contiguous cached minute
  span. `manifest.json` records which path was used (`build_path`). See [data.md](data.md).
- Macro series go to `<out>/macro/`. The Yahoo series (dxy, us10y, vix, spx, silver,
  gold_fut, oil) need the `data` extra (`pip install 'aurum[data]'`). Without it the
  command prints a warning and skips them unless they are already cached. `macro_factor`
  and `risk_off` need those series. The FRED series (real10y, breakeven10y, fedfunds) do
  not need the extra.
- **`--offline` covers only the bars.** The macro step still reads from its cache when
  that exact date range was fetched before, and downloads otherwise. For a fully offline
  rebuild, add `--no-macro`.

```bash
aurum data download                                   # full store, 2012 -> yesterday
aurum data download --time-budget 1200                # prefetch for 20 min, then build; re-run to resume
aurum data download --offline --no-macro              # rebuild parquet files from cache/dukascopy only
aurum data download --timeframes H1 --start 2024-01-01 --no-macro --out /tmp/aurum_h1
```

### `aurum data info`

Summarises the bar files and macro series in a data store. It checks each file's stored
hash and prints the manifest line if `manifest.json` exists. It never uses the network.

```text
aurum data info [--dir DIR] [--config CONFIG] [--set KEY=VALUE]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--dir` | path | `data_store` | Data store to inspect. Macro is read from `<dir>/macro`. Ignored when `--config` is given. |
| `--config`, `-c` | path | none | Use the config's `data.dir` and macro directory (`data.macro_dir` or `<data.dir>/macro`) instead. |
| `--set` | `KEY=VALUE` | none | Config overrides (see above). |

A file whose content does not match its stored hash is still listed, with
`HASH MISMATCH (...)` in the `check` column. A missing directory exits with code 2. On a
synthetic store:

```text
file               timeframe   rows                      first                       last  median_spread          hash  check
-----------------------------------------------------------------------------------------------------------------------------
xauusd_H1.parquet         H1  12000  2019-01-07 00:00:00+00:00  2020-12-11 03:00:00+00:00          0.299  6408134e4b0d     ok

macro:
series   rows       first        last          last_available_at
----------------------------------------------------------------
dxy       505  2019-01-07  2020-12-11  2020-12-11 21:30:00+00:00
...
```

## `aurum config`

Loads a config, applies `--set`, validates it, and prints the resolved YAML (`show`) or only
the status line (`validate`).

```text
aurum config [--config CONFIG] [--set KEY=VALUE] {show,validate}
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `show` / `validate` | positional | required | `show` prints the full resolved config as YAML, then the status line. `validate` prints only the status line. |
| `--config`, `-c` | path | built-in defaults | Config file. |
| `--set` | `KEY=VALUE` | none | Overrides. |

```bash
aurum config validate -c configs/fast.yaml
```

```text
# config OK: configs/fast.yaml  hash c78e62604b529b4f10a1feb90144ae559f5cee92b65bb2c7ef7ce56f22fb1b6b
```

The hash is the SHA-256 of every field that can change results. Secrets, the `output`
section, `name` and pure execution settings such as `walkforward.n_jobs` and
`walkforward.executor` are excluded. `show` never prints secrets: fields read from the
environment are left out of the YAML.

## Research commands

These commands need a data store (or a config with `data.synthetic`). See
[research.md](research.md) for the methodology and [RESULTS.md](RESULTS.md) for the
pre-registered results. **Do not read performance into the example output below: it was
produced on a synthetic random walk, where every Sharpe ratio is noise.**

The examples in this section, and in `train-final`, `desk replay --fake` and `live` below,
ran in an empty working directory holding a synthetic store built by this script. The
commands were then run from that directory, with `configs/...` pointing at the checkout's
config files.

```python
from aurum.data.macro import save_macro_dir
from aurum.data.store import save_bars
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro

bars = make_synthetic_bars(12_000, "H1", seed=1, model="gbm", start="2019-01-07")   # random walk
save_bars(bars, "data_store/xauusd_H1.parquet")
save_macro_dir(make_synthetic_macro(bars, seed=1), "data_store/macro")
print(len(bars), bars.index[0], bars.index[-1])
```

```text
12000 2019-01-07 00:00:00+00:00 2020-12-11 03:00:00+00:00
```

### `aurum backtest`

A single train/test split. Strategies and combiner are fitted on all bars before `--start`
(minus purge and embargo), then evaluated on `[--start, --end]`. Without `--start` (and
without `backtest.start` in the config) everything is fitted and evaluated **in sample**.
The command logs a warning and the report adds a note. Use `walkforward` for honest
estimates.

```text
aurum backtest [-h] --config CONFIG [--set KEY=VALUE] [--strategy ID] [--start START]
               [--end END] [--out OUT] [--jobs JOBS] [--include-holdout] [--no-write]
               [--no-tearsheet]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--strategy ID` | string, repeatable | all enabled | Restrict the run to these strategy ids or names. They must be in the config's `strategies` list (exit 2 otherwise). |
| `--start` | timestamp | `backtest.start` | Evaluation start. Fitting uses earlier data and needs at least `walkforward.min_train_bars` bars. |
| `--end` | timestamp | `backtest.end` | Evaluation end. A date-only value includes the whole day. |
| `--out` | path | `output.dir/backtest_<UTC>_<hash8>` | Report directory. |
| `--jobs` | int | `walkforward.n_jobs` | Parallel workers (0 = auto). |
| `--include-holdout` | flag | off | Also evaluate bars from `walkforward.holdout_start` on. This spends the holdout. |
| `--no-write` | flag | off | Do not write the report directory (`output.save_results: false`). |
| `--no-tearsheet` | flag | off | Skip `tearsheet.html` (`output.tearsheet: false`). |

Bars from `walkforward.holdout_start` on are dropped unless you pass `--include-holdout`. A
`--start` inside the holdout is refused with exit 2.

```bash
aurum backtest -c configs/fast.yaml --strategy tsmom --strategy macro_factor --start 2020-06-01 --no-write
```

```text
loaded 12000 H1 bars 2019-01-07 00:00:00+00:00 -> 2020-12-11 03:00:00+00:00 (0.2s)
== backtest: fast  config c78e62604b52  data 6408134e4b0d
OOS 2020-06-01 00:00:00+00:00 -> 2020-12-11 03:00:00+00:00  (3313 bars, 1 fold(s), train 8663 / test 3313 bars, purge 0, embargo 24, n_trials 23)

book          sharpe  ci_lower  ci_upper    psr    dsr    cagr  ann_vol  max_drawdown  n_trades  total_costs  n_halt_episodes
-----------------------------------------------------------------------------------------------------------------------------
combined       1.019    -1.380     3.641  0.776  0.114   0.059    0.055        -0.026        57      710.878                0
...
```

### `aurum walkforward`

The main research command. It refits every trainable piece per fold, stitches the
out-of-sample forecasts, runs one continuous backtest on them, and reports the statistics
with DSR and PBO. If `walkforward.holdout_start` is set, it evaluates the final holdout
once at the end.

```text
aurum walkforward [-h] --config CONFIG [--set KEY=VALUE] [--out OUT] [--jobs JOBS]
                  [--executor {auto,process,thread,serial}] [--no-write] [--no-tearsheet]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--out` | path | `output.dir/walkforward_<UTC>_<hash8>` (or `output.dir/<output.run_name>`) | Run directory. |
| `--jobs` | int | `walkforward.n_jobs` (config default 0) | Parallel workers. 0 = auto: `min(cpu_count, 8)`, capped by the number of tasks. |
| `--executor` | `auto`, `process`, `thread`, `serial` | `walkforward.executor` (`auto`) | Worker pool. `auto` runs serially with one worker, uses processes on large samples or heavy ML fits, and threads otherwise. Results do not depend on the executor. |
| `--no-write` | flag | off | Do not write the run directory. |
| `--no-tearsheet` | flag | off | Skip `tearsheet.html`. |

`--jobs` and `--executor` do not change the config hash.

**Holdout ledger.** Every holdout evaluation is appended to `holdout_ledger.jsonl`. It sits
next to the run directory, so default runs use `output.dir/holdout_ledger.jsonl`. With
`--no-write` the look is still recorded, in `output.dir/holdout_ledger.jsonl`, because a
human still saw the result. If a *different* config evaluated an overlapping window before,
the run warns and raises the DSR `n_trials`. Both shipped research configs evaluate the
2025–26 holdout, which [RESULTS.md](RESULTS.md) has already spent. To stay out of it, pass
`--set data.end=2024-12-31T23:59:59Z --set walkforward.holdout_start=null`.

A written run directory contains `summary.json`, `stats.csv`, `folds.csv`, `weights.csv`,
`fold_sharpe.csv`, `costs.csv`, `oos_forecasts.parquet`, `config.yaml`, `provenance.json`
(config hash, data hashes, git SHA and dirty flag, package versions), `books/`,
`tearsheet.html` unless `--no-tearsheet`, and `holdout/` when a holdout was evaluated.

```bash
aurum walkforward -c configs/fast.yaml \
  --set walkforward.train=4000 --set walkforward.test=2000 --set walkforward.holdout_start=null \
  --no-tearsheet
```

Output, trimmed (`...`). The WARNING lines about the missing `fedfunds` series are left
out; the synthetic store has none.

```text
loaded 12000 H1 bars 2019-01-07 00:00:00+00:00 -> 2020-12-11 03:00:00+00:00 (0.3s); 7 strategies
== walkforward: fast  config 2ebac749e954  data 6408134e4b0d
OOS 2019-08-30 01:00:00+00:00 -> 2020-12-11 03:00:00+00:00  (7976 bars, 4 fold(s), train 4000 / test 2000 bars, purge 0, embargo 24, n_trials 23)

book                  sharpe  ci_lower  ci_upper    psr    dsr    cagr  ann_vol  max_drawdown  n_trades  total_costs  n_halt_episodes
-------------------------------------------------------------------------------------------------------------------------------------
combined               0.550    -1.057     2.281  0.736  0.093   0.029    0.053        -0.041       175        2,475                0
...
benchmark             -1.477    -3.068     0.271  0.045  0.045  -0.225    0.158        -0.322         1       13.416                0

PBO (CSCV, 16 splits, 7 strategies): 0.342  prob OOS loss of IS-best 0.306

notes:
  - fold 0 combiner: inactive in train (zero weight): ['intraday_seasonality']
  ...
  - 1 fold(s) used equal combiner weights (not enough earlier OOS history)

timing: features 0.0s, strategies 0.1s, combine 0.0s, backtests 4.5s, stats 0.2s, write 1.0s, wall 6.1s

report dir: runs/walkforward_20260928T102535Z_2ebac749
```

Columns: `sharpe` is the daily Sharpe and `ci_lower`/`ci_upper` its stationary-bootstrap
95% interval. `psr`/`dsr` are the probabilistic and deflated Sharpe ratios. `n_halt_episodes`
counts risk-manager halts. Statistic definitions are in [research.md](research.md).

### `aurum report`

Prints the summary of an existing run directory, meaning any directory with a
`summary.json` written by `walkforward` or `backtest`.

```text
aurum report [-h] --run RUN [--json]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--run` | path | required | Run directory. |
| `--json` | flag | off | Print the raw `summary.json` (keys include `stats`, `settings`, `folds`, `weights`, `pbo`, `holdout`, `provenance`, `notes`, `timing`) instead of the table. |

A directory without `summary.json` exits with code 2:
`aurum: <dir>/summary.json not found (is <dir> a walkforward/backtest run directory?)`.

## `aurum train-final`

Builds the production trading artifact that [`live run`](#aurum-live-run) loads. Everything
is fitted on **all** bars closed at or before `--cutoff`. The combiner weights come from
**out-of-sample** walk-forward forecasts, never from in-sample ones. This is the recommended
way to produce a live artifact. See [live-trading.md](live-trading.md).

```text
aurum train-final [-h] --config CONFIG [--set KEY=VALUE] --out OUT [--cutoff CUTOFF]
                  [--from-run FROM_RUN] [--allow-config-mismatch] [--overwrite] [--jobs JOBS]
                  [--executor {auto,process,thread,serial}] [--no-write] [--no-tearsheet]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--out` | path | required | Artifact directory, loadable by `aurum.live.runner.load_artifact`. |
| `--cutoff` | timestamp | last bar | Use only bars closed (`available_at`) at or before this UTC time, and macro rows published by then. |
| `--from-run` | path | none | Walk-forward run directory whose OOS forecasts (research folds plus `holdout/` if present) give the combiner weights. Without it, a walk-forward is run now on the same data. |
| `--allow-config-mismatch` | flag | off | Accept a `--from-run` produced by a different config hash (prints a warning). |
| `--overwrite` | flag | off | Replace an existing `--out`. Without it an existing directory is refused with exit 2, before any fitting. |
| `--jobs` | int | `walkforward.n_jobs` | Workers for the in-process walk-forward (0 = auto). |
| `--executor` | `auto`, `process`, `thread`, `serial` | `walkforward.executor` | Executor for the in-process walk-forward. |
| `--no-write` | flag | off | Do not write the in-process walk-forward's run directory. |
| `--no-tearsheet` | flag | off | No tearsheet for the in-process walk-forward. |

Behaviour:

- **With `--from-run`**, the run's config hash must equal this config's hash (exit 2
  otherwise, unless `--allow-config-mismatch`).
- **Without `--from-run`**, the walk-forward is run and written like `aurum walkforward`,
  including a holdout evaluation and its ledger entry. If the cutoff falls before
  `walkforward.holdout_start`, that walk-forward runs without a holdout. With
  `live_paper.yaml` (14 strategies) this takes a few minutes.
- The pipeline scaler and trainable strategies are fitted on the training window that ends
  at the cutoff. Its length follows `walkforward.train`/`anchored`, with no purge and no
  holdout held back. The combiner is fitted on the stitched OOS history under the same fold
  rule, net of costs. It falls back to equal weights, with a warning, when the history is
  shorter than `walkforward.combiner_min_obs`.
  `walkforward.combiner_fit: train` is ignored here (a note says so).
- The artifact holds the fitted strategies (`strategies.pkl`, `strategies.json`), the
  combiner (`combiner.pkl`) and the OOS statistics (`backtest.json`). When strategies use the
  shared feature pipeline it also holds `pipeline.json` and `feature_reference.json`, the
  PSI reference for the live drift monitor. `manifest.json` records file hashes, package
  versions, the git SHA and a `training` block with the cutoff, the fit window, the resolved
  config and the provenance.
- Other exit-2 cases: fewer than two bars before the cutoff, OOS forecasts missing a
  configured strategy, and a sizer the live runner cannot reproduce.

```bash
aurum train-final -c configs/fast.yaml \
  --set walkforward.train=4000 --set walkforward.test=2000 --set walkforward.holdout_start=null \
  --from-run runs/walkforward_20260928T102535Z_2ebac749 --out artifacts/fast
```

```text
train-final: 12000 H1 bars 2019-01-07 00:00:00+00:00 -> 2020-12-11 03:00:00+00:00 (last close 2020-12-11 04:00:00+00:00); 7 strategies

artifact: artifacts/fast
  fit window 2020-04-21 03:00:00+00:00 -> 2020-12-11 03:00:00+00:00; combiner oos_history(4000 bars) [net of estimated costs c_t=(spread_eff/2+slippage+commission)/(close*vol)]
  weights: tsmom=0.00, ema_cross=0.00, donchian=0.00, zscore_fade=0.40, rsi2=0.00, macro_factor=0.40, intraday_seasonality=0.00  (sum 0.80, fdm 1.41)
  feature reference: 0 columns; OOS source: runs/walkforward_20260928T102535Z_2ebac749
```

No strategy in `fast.yaml` uses the shared pipeline, so this artifact has no
`pipeline.json` and no feature reference. **The artifact contains pickles: load only
artifacts you produced yourself.** Set `AURUM_ARTIFACT_KEY` to HMAC-sign them (see
[SECURITY.md](../SECURITY.md)).

## Registries

### `aurum strategies list`

```text
aurum strategies list [-h] [--json]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--json` | flag | off | Print `{name: {trainable, warmup_bars, description, params}}`, where `params` holds the default parameters. |

```text
strategy              trainable  warmup_bars                                                                                 description
----------------------------------------------------------------------------------------------------------------------------------------
bollinger_revert          False           49  Bollinger(48, 2) reversion: enter when price closes back inside a band, target the middle
donchian                  False          241  Turtle-style Donchian breakout (120-bar entry, 60-bar exit channel, 2.5 ATR stop on H1); s
...
tsmom                     False         1441  Multi-horizon time-series momentum (1w/1m/3m on H1): vol-normalised past returns, Carver-s
vol_squeeze               False           67  TTM-style squeeze: when Bollinger(20,2) exits Keltner(20,1.5 ATR) after a compression, tra
zscore_fade               False           50  Fade |z| > 2 of log price vs its 48-bar mean, only when Kaufman's efficiency ratio says th
```

Fifteen strategies are registered. Descriptions are cut at 90 characters. See
[strategies.md](strategies.md) and [ml-and-rl.md](ml-and-rl.md).

### `aurum features list`

```text
aurum features list [-h] [--json]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--json` | flag | off | Print `{group: {family, lookback, requires, doc}}`. |

```text
group                   family  lookback  requires                                                                               doc
------------------------------------------------------------------------------------------------------------------------------------
returns                returns        49         -                                              Vol-normalised trailing log returns.
...
macro                    macro      2136         -           Per-series changes & level z-scores, and rolling gold correlation/beta.
calendar              calendar         0    events        Event-proximity features for events with ``importance >= min_importance``.
regime                  regime      5820         -                                 Causal, sliding-window-stable regime descriptors.
```

Groups are listed in the pipeline's canonical order. `lookback` is the registered warm-up
in bars. `requires` shows the registry's declared inputs (`requires_macro`,
`requires_events`). The `macro` group declares none, but it reads `md.macro`, and it
returns no columns when macro data is missing. See [features.md](features.md).

## desk

The LLM trading desk is described in [llm-desk.md](llm-desk.md). `desk run` and
`desk replay` (without `--fake`) **call the paid Claude API with your own
`ANTHROPIC_API_KEY`**. The key comes only from the environment. A missing key exits with
code 2: `desk run` checks it before loading data, and `desk replay` checks it after the
confirmation banner, before anything is fitted or called.

### `aurum desk demo`

One complete desk cycle, fully offline. It uses synthetic anonymised data and a scripted
fake Claude client: the Chief consults two specialists, creates an ad-hoc agent and scales
the quant forecast by 0.5. It needs no key and no `anthropic` package.

```text
aurum desk demo [-h] [--journal-dir JOURNAL_DIR]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--journal-dir` | path | none (no journal) | Write the cycle's JSONL journal here. |

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

The `cost` line is computed from the scripted client's made-up token counts. Nothing is
billed.

### `aurum desk run`

**One paid desk cycle** on the latest bar close, or on `--at`. It fits the quant book on the
data up to that time, runs one Chief/specialist cycle on **non-anonymised** data, and prints
the decision and the size the vol-target sizer would take at `backtest.initial_equity`,
before risk checks. It is analysis only: no orders and no positions.

```text
aurum desk run [-h] --config CONFIG [--set KEY=VALUE] [--at AT] [--journal-dir JOURNAL_DIR]
               [--from-run FROM_RUN]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--at` | timestamp | latest bar close | Decision time (UTC). Only bars closed at or before it are used. Exit 2 if there are none. |
| `--journal-dir` | path | `agents.journal_dir` (`runs/desk_journal`) | Where the cycle's JSONL journal goes. |
| `--from-run` | path | none | Walk-forward run directory: combiner weights from its OOS forecasts. Without it, and with `walkforward.combiner_fit: oos`, the combiner has no OOS history and uses equal weights. |

The per-cycle budget cap is `agents.desk.max_cost_usd_per_cycle`: $2 in
`configs/desk_overlay.yaml`, and the `DeskConfig` default of $3 when unset.

```bash
export ANTHROPIC_API_KEY=...               # your own key; never in YAML
aurum desk run -c configs/desk_overlay.yaml
```

### `aurum desk replay`

Replays the desk over a historical window through **the same sizer, risk manager and
simulator** as a backtest. It also backtests the quant-only book on the same window for
comparison. The quant book is fitted on data ending before `--start`, minus purge and
embargo.

```text
aurum desk replay [-h] --config CONFIG [--set KEY=VALUE] --start START --end END
                  [--every EVERY] [--max-cost MAX_COST] [--yes] [--fake]
                  [--journal-dir JOURNAL_DIR] [--from-run FROM_RUN] [--out OUT]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--start` | timestamp | required | First decision bar. |
| `--end` | timestamp | required | Last bar. A date-only value includes the whole day. The window needs at least two bars (exit 2). |
| `--every` | int ≥ 1 | `agents.replay_every` (24) | Consult the desk every N bars. In between, the last decision is re-applied by the `DecisionPolicy` to the current quant forecast at every bar. |
| `--max-cost` | float (USD) > 0 | `agents.replay_max_cost_usd` (25.0) | Hard budget for the whole replay. |
| `--yes` | flag | off | Confirm the printed cost estimate and start. Without it the command exits with code 4. |
| `--fake` | flag | off | Use an offline scripted client that always halves the quant forecast. No API calls and no key needed; for testing the plumbing. |
| `--journal-dir` | path | `agents.journal_dir` | JSONL journal directory (one file per cycle). |
| `--from-run` | path | none | Walk-forward run directory: combiner weights from its OOS forecasts. |
| `--out` | path | `output.dir/desk_replay_<UTC>_<hash8>` | Replay report directory. |

Before anything is fitted or called, the command prints a banner with the number of cycles,
`n_cycles × agents.expected_cost_per_cycle_usd` as the estimate, the worst case, and the hard
budget. It then stops unless `--yes` is given. This is free to run:

```text
$ aurum desk replay -c configs/desk_overlay.yaml --start 2024-10-01 --end 2024-12-31 --every 48
==============================================================================
LLM DESK REPLAY - THIS CALLS THE PAID CLAUDE API
window 2024-10-01 00:00:00+00:00 -> 2024-12-31 21:00:00+00:00: 1486 bars, one desk cycle every 48 bars = 31 cycles
estimated cost ~$18.60 (at ~$0.60/cycle); worst case $25.00 (per-cycle cap $2.0, capped by the budget)
hard budget for this replay: $25.00 - every cycle may only spend what is left of it (an API call already in flight can overshoot by its own cost); once it is spent the desk is no longer called and the quant forecast is used for the remaining bars
==============================================================================
refusing to start without --yes
```

Budget and safety rules:

- Each cycle may spend at most the smaller of the per-cycle cap and what is left of
  `--max-cost`. An API call already in flight can overshoot by its own cost.
- Once the budget is spent, the desk is no longer called and the quant forecast is used for
  the remaining bars. The summary records `budget_hit_at`.
- A window that reaches into `walkforward.holdout_start` prints a warning, because
  evaluating the desk there spends the holdout.
- Replay uses `agents.anonymise` (true in the shipped configs: dates shifted, prices
  rebased). LLMs may still have memorised history. Treat replays as illustrative, not as
  evidence.

Output: a `desk` versus `quant_only` statistics table, then the report directory. The
directory holds `desk_decisions.csv`, `books/desk/`, `books/quant_only/`, `summary.json`
(cycles, cost, budget, window, stats, provenance) and `tearsheet.html` when
`output.tearsheet` is true. The replay always writes this directory. `--no-write` does not
exist here; use `--set output.tearsheet=false` to skip only the tearsheet.

```bash
# offline plumbing check on your own data: no key, no cost
aurum desk replay -c configs/fast.yaml --start 2020-10-01 --end 2020-10-31 --every 48 --fake --yes \
  --journal-dir /tmp/desk_journal --out /tmp/desk_replay_fake
```

## `aurum rl train`

Trains a PPO policy on the shared simulator using the config's `rl` section. It needs the
`rl` extra (`pip install 'aurum[rl]'`: torch, gymnasium, stable-baselines3). See
[ml-and-rl.md](ml-and-rl.md).

```text
aurum rl train [-h] --config CONFIG [--set KEY=VALUE]
```

| Config key | Default (`configs/default.yaml`) | Meaning |
|---|---|---|
| `rl.train_start` | `"2013-01-01"` | Training data starts here (null = first bar). |
| `rl.train_end` | `"2019-12-31"` | Last training day, inclusive. |
| `rl.val_end` | `"2021-12-31"` | Validation runs from the day after `train_end` through this day. |
| `rl.out_dir` | `runs/rl/ppo` | Artifact directory. |
| `rl.params` | `{total_timesteps: 200000, n_envs: 4}` | `aurum.rl.train.RLTrainConfig` fields. `seed` defaults to the config's `seed`. |

The command prints the split and the artifact path, then the numeric validation metrics as
JSON. Exit codes: 3 if the RL extras cannot be imported. 2 if there are fewer than 1,000
training or 100 validation bars (`rl: too little data (train N bars, validation N bars)`),
or if `rl.params` has an unknown or invalid key (`rl.params: unknown RLTrainConfig keys:
[...]`).

## live

The live stack is described in [live-trading.md](live-trading.md). Read the safety model
there and in [SECURITY.md](../SECURITY.md) before pointing it at a broker.

### `aurum live artifact`

Fits the enabled strategies and the combiner on the latest data (or up to `--at`) and writes
a trading artifact. It is quicker than `train-final` but weaker. **Without `--from-run`, and
with `walkforward.combiner_fit: oos` (the default), the combiner has no OOS history and falls
back to equal weights.** For a production artifact, prefer `train-final`.

```text
aurum live artifact [-h] --config CONFIG [--set KEY=VALUE] [--out OUT] [--at AT]
                    [--from-run FROM_RUN] [--overwrite]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--out` | path | `live.artifact_dir` (`artifacts/live`) | Artifact directory. |
| `--at` | timestamp | last bar | Fit on bars closed at or before this UTC time. |
| `--from-run` | path | none | Walk-forward run directory: combiner weights from its OOS forecasts, and its OOS statistics are embedded. Unlike `train-final`, the config hash is not checked. |
| `--overwrite` | flag | off | Replace an existing artifact. Without it an existing directory fails **after** fitting, with exit 1 (`FileExistsError`). |

```text
fitting 7 strategies + combiner on data up to 2020-09-29 23:00:00+00:00 ...
artifact: artifacts/la  (train 2019-01-07 00:00:00+00:00 -> 2020-09-29 23:00:00+00:00, fdm 1.49)
weights: tsmom=0.17, ema_cross=0.17, donchian=0.17, zscore_fade=0.17, rsi2=0.17, macro_factor=0.17, intraday_seasonality=0.00
```

### `aurum live run`

Runs the bar-close live loop (`aurum.live.runner`). At each bar it goes artifact →
strategies → combiner → optional desk → sizer → risk manager → OMS → broker. **It is a dry
run by default**: orders are planned and logged, never sent.

```text
aurum live run [-h] --config CONFIG [--set KEY=VALUE] [--i-understand-real-money]
               [--max-cycles MAX_CYCLES] [--until UNTIL]
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--i-understand-real-money` | flag | off | Second opt-in, required **together with** `live.allow_live_real: true`, to trade a non-demo account. On its own it has no effect (a note says so). |
| `--max-cycles` | int | unlimited | Stop after this many bar decisions. |
| `--until` | timestamp (ISO) | end of replay data (paper replay); otherwise run until stopped | Stop at this broker time (UTC). |

What happens:

1. The config is translated into the runner's settings (`AurumConfig.live_runner_mapping`),
   using the **same** instrument, costs, sizer, live risk limits and desk policy as research.
   This fails with exit 2 if the sizer cannot be reproduced live, or if `live.options`
   tries to override a typed or safety field (`dry_run`, `allow_live_real`, `broker`,
   `magic`, the `risk`/`sizer`/`costs` sections) or holds credentials:

   ```text
   aurum: configuration error: live.options.dry_run: overrides a value the typed config already sets (True); set it in its own config field instead
   ```

2. It prints `live: broker=... dry_run=... allow_live_real=... magic=... artifact=...`. Here
   `allow_live_real` is true only when both opt-ins are present.
3. The resolved settings, without secrets, are written to
   `<live.state_dir>/aurum_live_config.yaml`, and `aurum.live.runner.main` is called with
   that file plus `--i-understand-real-money`, `--max-cycles` and `--until` as given.
4. The runner checks the account. A non-demo account without both opt-ins is refused
   (`REFUSED: ...` on stderr, exit 2), including an account that turns non-demo mid-run.

`dry_run` and `allow_live_real` come only from the typed `live` section. Nothing on the
command line turns dry-run off: set `live.dry_run: false` in the config, or pass
`--set live.dry_run=false`, to fill on the paper broker.

State lives in `live.state_dir`: `decisions.jsonl`, `heartbeat.json`, `risk_state.json`,
`runner_state.json`, `runner.lock` and, for the paper broker, `paper_broker.json`. A second
run with the same `state_dir` resumes after the last processed bar. Use a fresh `state_dir`
to start over. Two runners cannot share one state directory. SIGINT/SIGTERM finish the
current cycle and stop cleanly.

A dry run on a paper replay, using the artifact from the `train-final` example above:

```bash
aurum live run -c configs/live_paper.yaml --set live.artifact_dir=artifacts/fast \
  --set live.state_dir=runs/live --set live.options.paper.warmup_bars=11000 --max-cycles 5
```

```text
live: broker=paper dry_run=True allow_live_real=False magic=20260926 artifact=artifacts/fast
resolved runner config: runs/live/aurum_live_config.yaml
2026-09-28 10:49:11,164 WARNING aurum.execution.costs: PaperBroker: rate financing without a 'fedfunds' series; using fallback_rate=0.0300 (pass rates=md.macro or call set_rates)
aurum live runner [DRY-RUN PAPER] XAUUSD H1 magic=20260926 state=runs/live
```

The warning appears because the synthetic data store has no `fedfunds` series. Add `-v` to
see one log line per decision. That artifact was fitted on all the data it replays, so the
replay is **in sample**. It tests the plumbing, not the strategy (see
[live-trading.md](live-trading.md) for out-of-sample paper replays with
`train-final --cutoff`).

## Environment variables

Secrets are read **only** from the environment. YAML files and `--set` cannot hold them.
Aurum does not read `.env` files itself; see [SECURITY.md](../SECURITY.md) and
`.env.example`.

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `desk run`, `desk replay` (not with `--fake`), `live run` with `live.use_desk: true` |
| `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH` | `live run` with `live.broker: mt5` or `paper.data: mt5` (Windows, `mt5` extra) |
| `AURUM_ALERT_WEBHOOK_URL`, `AURUM_ALERT_TELEGRAM_CHAT_ID` | optional live alert sinks |
| `AURUM_ARTIFACT_KEY` | HMAC signing and verification of pickled live artifacts |

No other config value can be overridden from the environment, so a run is fully described
by its YAML plus `--set` overrides.

## Where the CLI writes

All of these locations are git-ignored.

| Path (default) | Written by | Set with |
|---|---|---|
| `data_store/` | `data download` | `--out`, `data.dir` |
| `cache/dukascopy/`, `cache/macro/` | `data download` | `--cache`, `--macro-cache` |
| `runs/walkforward_<UTC>_<hash8>/`, `runs/backtest_<UTC>_<hash8>/` | `walkforward`, `backtest`, `train-final` (its walk-forward) | `--out`, `output.dir`, `output.run_name` |
| `runs/holdout_ledger.jsonl` | any holdout evaluation | next to the run directory |
| `runs/desk_journal/` | `desk run`, `desk replay` | `--journal-dir`, `agents.journal_dir` |
| `runs/desk_replay_<UTC>_<hash8>/` | `desk replay` | `--out`, `output.dir` |
| `runs/rl/ppo/` | `rl train` | `rl.out_dir` |
| `artifacts/live/` | `live artifact` | `--out`, `live.artifact_dir` |
| `runs/live/` (`runs/live/paper` in `live_paper.yaml`) | `live run` | `live.state_dir` |

## A typical session

```bash
aurum data download                                    # once; resumable
aurum data info
aurum config validate -c configs/trend_core.yaml
aurum walkforward -c configs/trend_core.yaml \
  --set data.end=2024-12-31T23:59:59Z --set walkforward.holdout_start=null
aurum report --run runs/walkforward_<utc>_<hash>
aurum desk demo                                        # free, offline
aurum train-final -c configs/live_paper.yaml --out artifacts/live_paper   # runs its own walk-forward
aurum live run    -c configs/live_paper.yaml --max-cycles 24   # dry run on the paper broker
```

Without `--from-run`, `train-final` runs a walk-forward that evaluates the configured
holdout and records the look in the ledger (see [`train-final`](#aurum-train-final)).

Next: [getting-started.md](getting-started.md) walks through a first install,
[configuration.md](configuration.md) lists every config key, and
[development.md](development.md) covers tests and contributing.
