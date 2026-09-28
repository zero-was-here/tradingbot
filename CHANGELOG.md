# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package version in
`pyproject.toml` follows [Semantic Versioning](https://semver.org/). Results are tied to
commits and data hashes rather than to version numbers: see
[docs/RESULTS.md](docs/RESULTS.md) and each run's `provenance.json`.

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
