# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package version in
`pyproject.toml` follows [Semantic Versioning](https://semver.org/). Results are tied to
commits and data hashes rather than to version numbers: see
[docs/RESULTS.md](docs/RESULTS.md) and each run's `provenance.json`.

## [2.1.0] - Unreleased

Research analytics (parameter sweeps, Monte Carlo and stress, explainability), two new
strategies, an interactive dashboard, stricter configuration validation, and live-runner
hardening. Nothing in this release changes the verdict of [docs/RESULTS.md](docs/RESULTS.md):
the new configurations were run as exploratory additional trials on data ending 2024-12-31,
and the used-up 2025-26 holdout was not touched.

### Added

- **Parameter sweeps** (`aurum sweep`, `aurum.research.sweep.run_sweep`). Each grid point of
  one strategy is a full walk-forward of a standalone single-strategy book (out of sample,
  net of costs). The sweep reports DSR/PSR/CI of the best point with `n_trials` = valid grid
  points + earlier overlapping sweeps in `sweep_ledger.jsonl`, PBO (CSCV) across the grid,
  and a plateau score: the mean of the best cell's Chebyshev neighbours over the best value
  (below 0.5 = knife-edge, 0.75 or more = robust plateau), the plateau centre and a grid-edge
  flag. A heatmap (`sweep.png`) and tidy tables are written. The holdout is never swept: bars
  from `walkforward.holdout_start` and every window already in a holdout ledger are cut.
- **Monte Carlo and stress** (`aurum stress`, `aurum.research.montecarlo`). A stationary
  block bootstrap of the book's daily returns (Politis-Romano, automatic block length) gives
  CAGR / max-drawdown / time-to-recover distributions, P(drawdown > x), risk of ruin and fan
  bands (10k paths x 2 years in well under a second). Deterministic stress replays the
  book's worst k-day windows, applies gap shocks to the current and peak exposure, and
  applies gold's worst historical k-day moves to the current position.
- **Explainability** (`aurum explain`, `aurum.research.attribution`). A decision waterfall
  (`c_i = w_i * f_i * FDM`, the clip, the combined forecast) with the sizing chain replayed
  point-in-time from the run's bars and checked against the saved request; PnL attribution
  by signed contribution shares that sums exactly to the book's PnL (price, costs, swap,
  net, by year); a desk-journal summary. `aurum explain --live STATE_DIR` does the same for
  a live or paper decision from the runner's `decisions.jsonl` (read-only), including the
  LLM-desk step.
- **Interactive dashboard** (`aurum dashboard`, `aurum.viz.dashboard.build_dashboard`, extra
  `viz`): KPI tiles with pass/fail badges, equity vs benchmark, drawdown, rolling Sharpe,
  per-fold Sharpe, forecast small multiples, combiner weights, cost waterfall, monthly
  heatmap, trades, holdout and an optional desk timeline, with a light/dark toggle.
- **Figure style** (`aurum.viz.style`): one palette in light and dark variants,
  `use_style`, `save_figure` (writes `-light`/`-dark` PNGs) and plotting helpers for the
  documentation figures.
- **Strategies** (now 17): `managed_gold` (long-only strategic allocation with a trend tilt
  and a latched de-risk switch: price below its EMA and fast/slow volatility above a
  threshold) and `regime_switch` (a base trend forecast scaled by a 2-state Gaussian HMM's
  causal filtered probability of the calm regime, fitted on training returns only).
- **Configs** `d1_trend.yaml` (daily NY-close bars, monthly-horizon trend rules) and
  `managed_gold.yaml`. Both are exploratory and hard-wired to end on 2024-12-31 with no
  holdout.
- **Live operations.** `aurum live status` (read-only snapshot of lock, heartbeat, kill
  switch, OMS, paper book, last decision and alerts; `--json`) and `aurum live reset-halt`
  (clears the kill switch only with `--confirm RESET`, refuses while a runner holds the lock,
  takes the runner's lock, logs who/when/why to `decisions.jsonl` and `alerts.jsonl`).
- **Decision attribution in the live log.** Every decision record carries `attribution`
  (per-strategy forecast, weight and contribution, FDM, raw and clipped combined forecast,
  unallocated risk, the desk's delta).
- **`aurum --version`**, `aurum.__version__`, and an extension hook: `aurum.cli_ext` adds
  `sweep`, `stress`, `explain` and `dashboard` to the core parser (guarded; a broken
  extension never breaks the core commands).
- **Holdout-ledger coverage.** `run_single_backtest(include_holdout=True)` and LLM-desk
  replays that reach `walkforward.holdout_start` record their look (the replay under its own
  `:desk` config hash, so it counts as a separate trial in later walk-forwards); ledger
  entries record `data_source` (real/synthetic) and `data_hash`; synthetic smoke tests no
  longer count against the real holdout.
- `aurum.rl.train.rl_config_from_aurum`: the RL environment takes costs, sizing, research
  risk, instrument and initial equity from the top-level config (used by `aurum rl train`).
- Walk-forward attribution API: `forecast_contributions`, `WalkForwardReport.contributions()`
  and friends.
- Lazy exports of the new APIs from `aurum.research` and `aurum.rl`.

### Changed

- **Configuration validation reports every problem at once** (`AurumConfig.check` ->
  `ValidationReport`): every `strategies[]` entry is built (unknown or misspelt params are
  errors with a did-you-mean hint), the feature pipeline, instrument ranges, the live
  runner's own `LiveConfig` and the RL config are checked. Every CLI command validates
  strictly before loading data. `aurum config validate|show` print warnings, skipped checks
  and what was checked.
- `live.use_desk: true` requires `agents.enabled: true`; paid `desk run` / `desk replay`
  refuse without it (`desk replay --fake` stays allowed).
- `live_runner_mapping()` passes the full `agents.desk` (nested chief/specialist/role models
  and prices) to the runner as `desk_config`.
- `rl_ppo` with `artifact_dir` loads and re-verifies the artifact in `fit` instead of
  retraining (`retrain`, `artifact_sha256` params), with a look-ahead guard: an artifact
  trained on any bar after the fit window is refused.
- `ml_gbm`, `meta_label` and `rl_ppo` reject unknown params.
- `resample_bars` drops a leading partial bucket, so higher-timeframe values never depend on
  where the history starts (the Dukascopy dataset build is byte-identical).
- GARCH(1,1) annualises with nominal bars per year (5,796 for H1), like the EWMA estimator.
- The runner rewrites `heartbeat.json` every `heartbeat_seconds` (default `live.poll_seconds`)
  while it sleeps between bars; `check_heartbeat(path)` derives its threshold from the file.
- The runner logs exactly one concise INFO line per decision, shown by `aurum live run` at
  the default verbosity, and no longer reconfigures a host's logging.
- `aurum live artifact` checks an existing `--out` before loading data; `backtest` without
  `--start` is labelled IN-SAMPLE; equal-weight combiner fallbacks print a warning.
- Tearsheets use the config's `walkforward.n_boot` and `seed` for the bootstrap.
- `risk_report` takes the stress price from `last_price`, run metadata or the bars before
  the last fill (`report["position"]["price_source"]`).
- `aurum features list` shows `macro(any)` for the macro group.

### Fixed / safety

- **Daily dating of walk-forward statistics.** `book_statistics` (Sharpe, PSR, DSR, CI in
  `stats.csv`), the DSR trial Sharpes, the per-fold Sharpes (`folds.csv`, `fold_sharpe.csv`)
  and the PBO matrix now date each equity mark by its bar's close, with the run's bar
  duration (the bars' timeframe, else `available_at - open`), like `metrics.json` already
  did. They used the bar's open: on New York close D1 bars (open 22:00 UTC) the Sunday and
  Monday bars merged, a week had four daily returns (1,862 instead of 2,322 over 2016-2024)
  and the D1 Sharpes were 10-13% too large in magnitude. H1 bars never straddle midnight:
  every published protocol number is unchanged. Both exploratory D1 runs were re-run; the
  corrected numbers are below and in [docs/RESULTS.md](docs/RESULTS.md). The tearsheet's
  daily statistics, `aurum stress`, the sweep's daily return matrix, the dashboard's daily
  equity and rolling Sharpe and the `aurum rl train` validation Sharpe date by the close
  the same way (new `aurum.backtest.metrics.resolve_bar_duration`).
- `aurum backtest --include-holdout --no-write` records its holdout look in
  `output.dir/holdout_ledger.jsonl`, as `walkforward --no-write` does (it recorded nothing).
- `aurum sweep` resolves `--strategy` and checks the `--grid` parameter names before it loads
  any data (exit 2 with a did-you-mean hint, instead of failing after the load).
- Config validation checks the keys of `live.options.mt5` against `MT5Broker`'s signature
  (without building the broker; `symbol`, `instrument`, `connect` and the code-level
  arguments are refused) and refuses unknown keys in `live.options.desk`; the runner's
  `LiveConfig` refuses desk keys outside the new `aurum.live.runner.DESK_KEYS`. Both kinds
  of typo used to pass validation (a desk typo was silently ignored, an MT5 typo failed only
  when `aurum live run` built the broker).
- `aurum data info --help` no longer claims `--config` defaults to the built-in defaults.
- The MkDocs hook also rewrites links that leave `docs/` inside figure captions (they broke
  `mkdocs build --strict`).
- CI runs on pushes to `main` (and pull requests), asserts all 17 strategies, and has a
  `docs` job: figure checks, docs asset/link tests and a strict MkDocs build.
- The test suite refuses writes into the repository's `data_store/` (the aurum.data
  writers and the generic parquet writers are guarded from collection on; a swallowed
  refusal still fails the test).
- The LLM desk's prerequisites (`anthropic` importable, `ANTHROPIC_API_KEY` set) are checked
  before the runner touches the broker, and by `aurum live run` before it writes anything.
- `desk.on_error` must be one of `follow_quant|flat|hold`, and `desk.enabled` must be a real
  boolean (a quoted `"false"` used to enable the desk).
- MT5 `server_tz: auto` without a fresh tick falls back to the zone last detected on the same
  account server (or `server_tz_fallback`) UNCONFIRMED: dry-run decisions continue, every
  order is deferred until a fresh tick confirms the offset, and a different offset is adopted
  and that order rejected.
- `aurum live run` no longer has a `run_from_config` side door that bypassed the runner's own
  config validation.
- `import aurum.live` no longer imports the runner (`python -m aurum.live.runner` printed a
  RuntimeWarning).
- Loader `attrs["source"]` is deterministic for in-memory buffers; `load_mt5_csv` warns when
  the default server time zone is assumed.
- Sweep heatmap labels pick their colour from the rendered cell (unreadable labels on
  all-negative grids).

### Research (additional trials, not pre-registered)

Exploratory runs on real data ending 2024-12-31, holdout disabled, DSR with `n_trials` = 30:
daily trend (`d1_trend`) combined Sharpe -0.26 (DSR 0.002); `managed_gold` combined 0.11
(DSR 0.040; `managed_gold` alone 0.17 vs buy-and-hold 0.47); an `ema_cross` sweep's best
point 16/64 Sharpe 0.24 with DSR 0.19 and a knife-edge plateau (ratio 0.49). No book comes
close to the protocol's bar.

### Documentation

- **Every algorithm explained with its math and a picture.** Each mechanism on the
  component pages follows one pattern: the intuition, the exact formula the code evaluates
  (display equations with every symbol defined, matching the code's windows, clipping,
  annualisation and defaults), a figure, the function with a runnable example and its real
  output, a parameter table, and the pitfalls. 205 display equations and 83 Mermaid diagrams
  (flows, sequences, state machines) across the pages.
- **New pages:** [docs/models.md](docs/models.md) (EWMA, GARCH(1,1), HAR-RV, the variance
  blend and the Gaussian HMM, moved out of the portfolio page and extended) and
  [docs/analytics.md](docs/analytics.md) (sweeps and plateau scores, Monte Carlo stress,
  decision explanations and PnL attribution, the dashboard). Both are in the MkDocs nav.
- **Figure pipeline** (`tools/figures`). 126 figures, each drawn by the real aurum code from
  a registered function in `tools/figures/fig_<page>.py`, rendered in a light and a dark
  variant (252 PNGs, about 30 MB) with `aurum.viz.style`. Data figures read the real data
  store and run outputs through loaders that stop at 2024-12-31 and refuse the holdout, a
  figure whose inputs are missing is skipped and keeps its committed PNGs, and re-renders
  are byte-identical (`python tools/figures/build.py`, `--check`). A writer guide is in
  [tools/figures/README.md](tools/figures/README.md).
- **Theme-aware embeds.** Figures are embedded as `<picture>` blocks that follow the
  reader's theme on GitHub; the MkDocs hook turns them into light/dark image pairs with a
  caption that follow the site's theme toggle, and escapes stray currency dollars so they
  are not read as math.
- **Docs asset test** (`tests/test_docs_assets.py`): every registered figure has both PNGs,
  every PNG is registered and referenced, every image reference and relative link (with
  `#anchor`) resolves, the embeds are well formed, figure modules load no data at import,
  and the figure loaders stop at the cutoff.
- The README has a visual documentation section, and the site's links to the source now
  point at the `main` branch.

## [2.0.0] - 2026-09-28

Aurum v2 is a ground-up rebuild of this repository as the `aurum` Python package, a
point-in-time research and paper-trading stack for spot gold (XAUUSD). The v1 code is kept
in `legacy/` for reference only. The full documentation starts at
[docs/index.md](docs/index.md).

Under the pre-registered protocol ([docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md))
no strategy or book showed a statistically demonstrated edge, so v2 ships as a research and
paper-trading platform ([docs/RESULTS.md](docs/RESULTS.md)).

### Added

- **Package and CLI.** The `aurum` package (Python 3.10 or newer) with optional extras
  `data`, `rl`, `agents`, `mt5`, `dev`, `docs` and `all`, and the `aurum` command (also
  `python -m aurum`) with the commands `data`, `config`, `backtest`, `walkforward`,
  `train-final`, `report`, `strategies`, `features`, `desk`, `rl` and `live`
  ([docs/cli.md](docs/cli.md)).
- **Typed configuration.** YAML configs with `extends:` inheritance, `--set` overrides,
  strict validation with "did you mean" hints, secrets read only from the environment, and
  a config hash that names runs and keys the holdout ledger. Shipped configs: `default`,
  `fast`, `trend_core`, `desk_overlay`, `live_paper`
  ([docs/configuration.md](docs/configuration.md)).
- **Point-in-time data layer.** A canonical bars schema in which every row carries
  `available_at`; `asof_join`; resampling that emits complete buckets only; a resumable
  Dukascopy bid/ask downloader with an adaptive rate limiter, content-hashed parquet store,
  `manifest.json` and data-quality report; MT5 and CSV loaders with broker time-zone specs
  (including the DST-aware `NY+7`); Yahoo and FRED macro series stamped with publication
  lags; a rule-based NFP/FOMC calendar; synthetic data generators
  ([docs/data.md](docs/data.md)).
- **Features.** Twelve causal feature groups and `FeaturePipeline`, whose scaler is fitted
  on training rows only and which persists as JSON for the live runner; warm-ups that scale
  with the bar size ([docs/features.md](docs/features.md)).
- **Strategies.** Fifteen registered strategies: `tsmom`, `ema_cross`, `donchian`,
  `kalman_trend`, `zscore_fade`, `rsi2`, `bollinger_revert`, `vol_squeeze`, `orb`,
  `macro_factor`, `risk_off`, `intraday_seasonality`, `ml_gbm`, `meta_label` and `rl_ppo`,
  all emitting forecasts in [-1, 1] with Carver-style scaling
  ([docs/strategies.md](docs/strategies.md)).
- **Machine learning and RL.** Triple-barrier labels with uniqueness weights, a calibrated
  gradient-boosting classifier behind a validation skill gate, meta-labelling with bet
  sizing, a Gymnasium environment on the shared execution stack, PPO training
  (`aurum rl train`) and the `rl_ppo` strategy adapter ([docs/ml-and-rl.md](docs/ml-and-rl.md)).
- **Portfolio and risk.** A forecast combiner (`sharpe_shrink`, `equal`, `inverse_vol`,
  `hrp`, `fixed`) that scores strategies net of estimated trading costs and can leave risk
  unallocated; volatility-targeting and fixed-fractional sizers with drawdown de-risking; a
  reduce-only risk manager with pre-trade limits, event blackouts and a kill switch;
  VaR/ES and stress tests; EWMA, GARCH(1,1), HAR-RV and Gaussian-HMM models
  ([docs/portfolio-and-risk.md](docs/portfolio-and-risk.md)).
- **Execution and backtesting.** One bar-level `ExecutionSimulator` (fill at the next open,
  mid ± half spread ± slippage, commission, intrabar stop/take-profit with the stop first)
  with rate-based overnight financing (Fed funds read point-in-time plus a markup, triple
  charge on Wednesdays); the backtest engine, `BacktestResult` with a PnL reconciliation
  identity, and daily-return metrics ([docs/execution-and-costs.md](docs/execution-and-costs.md)).
- **Research protocol tooling.** Walk-forward with per-fold refits, purge and embargo,
  combiner weights fitted on earlier folds' out-of-sample forecasts, one stitched backtest
  per book and a buy-and-hold benchmark; bootstrap confidence intervals, PSR, Deflated
  Sharpe, minimum track record and PBO; a one-time holdout with a ledger that raises
  `n_trials` when a different config looks again; HTML tearsheets; provenance records;
  purged k-fold and CPCV splitters ([docs/research.md](docs/research.md)).
- **LLM trading desk.** A Claude-powered Chief Investment Officer, four standing specialists
  and ad-hoc agents created at runtime, read-only point-in-time tools, a `DecisionPolicy`
  (`overlay`, `advisory`, `discretionary`) whose output still goes through the sizer and
  risk manager, per-cycle and per-replay cost caps, JSONL journals, an offline scripted
  client, and the `desk demo`, `desk run` and `desk replay` commands
  ([docs/llm-desk.md](docs/llm-desk.md)).
- **Live and paper trading.** Trading artifacts (`train-final`, `live artifact`) with a
  SHA-256 manifest and optional HMAC signing; a bar-close `LiveRunner`; a paper broker held
  to the simulator's equity path by tests; a MetaTrader 5 adapter with magic-number
  isolation; an idempotent order manager; PSI drift, slippage, PnL-band and heartbeat
  monitoring with log, JSONL and webhook alerts ([docs/live-trading.md](docs/live-trading.md)).
- **Research record.** The pre-registered protocol and its results
  ([docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md), [docs/RESULTS.md](docs/RESULTS.md)).
- **Documentation.** A documentation set under `docs/` (home page, getting started,
  architecture, one page per component, CLI and configuration references, a development
  guide), `mkdocs.yml` for browsing it as a site (`pip install -e ".[docs]"`, then
  `mkdocs serve`), `SECURITY.md`, `.env.example` and this changelog.
- **Tests and CI.** About 1,450 tests that run offline on synthetic data: future-perturbation
  leakage harnesses for every feature group, every strategy and the whole walk-forward,
  random-walk no-edge tests, simulator, paper-broker and RL parity tests, kill-switch
  persistence, OMS idempotency, an MT5 fake terminal and LLM-desk cycles against a scripted
  client. A GitHub Actions workflow runs lint, an import check without optional
  dependencies, a CLI smoke test and the offline suite on Python 3.10 and 3.12
  ([docs/development.md](docs/development.md)).

### Changed

- The repository is now the installable `aurum` package (`pyproject.toml`, version 2.0.0).
  The v1 Python code moved to `legacy/`, unmaintained, unpackaged and untested.
- The README was rewritten around the v2 architecture and the honest research status.

### Removed

- The v1 README's performance claims (for example "80–120% annual returns"), which had no
  evidence behind them. The v1 README is kept as `legacy/README_v1.md`.
- The v1 training, backtest and live scripts are no longer part of the supported code; they
  remain in `legacy/` for reference only.

### Security

- Secrets are read only from environment variables (`ANTHROPIC_API_KEY`, `MT5_*`,
  `AURUM_ALERT_*`, `AURUM_ARTIFACT_KEY`). YAML files and `--set` overrides that try to set
  them are rejected, and they are excluded from saved configs and the config hash.
- `aurum live run` is dry-run by default. A non-demo account needs both
  `live.allow_live_real: true` and `--i-understand-real-money`; the check is repeated at
  every decision, and an MT5 account of unknown trade mode is treated as real.
- The kill switch persists across restarts, including when its state file is deleted, and
  only `reset_halt(confirm="RESET")` clears it. A single-runner lock, idempotent order ids
  and magic-number isolation protect against double sends and against touching other
  strategies' positions.
- Live artifacts contain pickles: load only your own, and set `AURUM_ARTIFACT_KEY` to sign
  and verify them.
- LLM-desk prompts treat market text as untrusted data, and desk output is bounded by the
  policy and the risk manager.
- The test suite refuses outbound internet connections outside `@pytest.mark.network`
  tests and removes credentials from the environment.
- [SECURITY.md](SECURITY.md) discloses that early v1 commits contain a MetaAPI token and
  account id in the public history. That token must be treated as compromised and revoked.

### Fixed

Problems found by the audit of v1 that the rebuild addresses:

- Higher-timeframe and macro features leaked the future (v1 forward-filled H1 bars onto M5
  bars from their open, leaking up to 55 minutes); every join is now an as-of join on
  `available_at`.
- The feature scaler was fitted on the full sample; it is now fitted on training rows only.
- RL reward shaping was compounded into the reported equity; reward and equity are now
  separate, and equity comes from the shared simulator.
- The backtester fed random noise to a mock agent, and crisis validation was hard-coded to
  pass.
- Live trading had no stop-losses or kill switch and could close other EAs' positions.

Documentation-only fixes made while writing the v2.0.0 docs (no behaviour changes): the
`aurum data download --offline` help text now says that macro series are still fetched
unless `--no-macro` is passed; stale docstrings in `aurum/core/config.py`,
`aurum/backtest/engine.py`, `aurum/execution/simulator.py`, `aurum/strategies/base.py` and
`aurum/strategies/ml.py`, and the `rl_ppo` comment in `configs/default.yaml`, were corrected;
outdated entries in `docs/INTERFACES.md` were updated.

## v1.x (December 2025 to July 2026)

The first incarnation of this repository was a deep-reinforcement-learning gold trading bot
(Dreamer and PPO agents, MetaAPI and MetaTrader 5 live scripts, Colab training notebooks)
plus `GoldHedgerPro v4`, a standalone MQL5 grid/martingale expert advisor that received
fixes between May and July 2026. v1 was never versioned or tagged. An audit found that
nothing in it could produce trustworthy numbers, so it was replaced by v2. Its Python code
is kept in `legacy/` and the EA in `mt5_ea/`, both unmaintained. The EA has effectively
unbounded tail risk and is not recommended for real money. See the README and
[docs/development.md](docs/development.md#legacy-and-mt5_ea).

[2.0.0]: https://github.com/zero-was-here/tradingbot/tree/aurum-v2
