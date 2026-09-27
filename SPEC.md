# Aurum v2 — Engineering Specification

This is the binding contract between modules. If code and SPEC disagree, fix one of them
in the same change. Python >= 3.10, pandas >= 2.1 (must also run on pandas 3.x — never use
removed APIs like `fillna(method=...)`, `'H'`/`'T'` offset aliases; use `'h'`, `'min'`).

## 0. Principles

1. **Point-in-time or nothing.** A value may influence a decision at time `T` only if it
   was available at `T`. Every frame that is joined onto trading bars carries an
   `available_at` column and is aligned with `aurum.data.pit.asof_join`.
2. **One simulator.** Research backtests, the RL environment, paper trading and the LLM
   desk replay all use `aurum.execution.simulator.ExecutionSimulator` and the same sizer and
   risk manager objects. No duplicated PnL code.
3. **Alpha ≠ sizing ≠ risk.** Strategies emit forecasts in [-1, 1]; the combiner blends
   them; the sizer turns a forecast into lots via volatility targeting; the risk manager
   can only *reduce* risk (or halt). Nothing downstream of risk may increase exposure.
4. **Deterministic & reproducible.** Every stochastic component takes a `seed`. Every run
   writes its config, data hash and git SHA next to its results.
5. **Safe by default.** Live trading defaults to dry-run; real accounts require an explicit
   opt-in flag *and* config value. LLM agents can never bypass the risk manager.
6. **Tested.** Each module ships `tests/test_<module>.py`. Tests must not hit the network
   unless marked `@pytest.mark.network`, must run in < 60 s total per module, and use
   `aurum.data.synthetic` for data.

## 1. Conventions

- **Time**: all timestamps tz-aware UTC. Bars are indexed by OPEN time; `available_at`
  = open + timeframe (see `aurum/data/schema.py`).
- **Decision timing**: decisions are made at the close of bar `t` (`available_at[t]`), and
  orders fill at the OPEN of bar `t+1` (mid ± spread/2 ± slippage). PnL of bar `t+1` is
  `lots * contract_size * (close[t+1] - fill_price)` for the new position, etc.
- **Prices**: `open/high/low/close` are MID prices. `spread` is the full bid/ask spread in
  price units (USD/oz). A buy fills at `mid + spread/2 + slippage`, a sell at
  `mid - spread/2 - slippage`.
- **Units**: equity/PnL/costs in USD. Positions in signed lots (1 lot = 100 oz for
  XAUUSD, `aurum.core.instrument.XAUUSD`). Volatility is an annualised fraction.
- **Forecasts**: float in [-1, 1] decided at close of `t` (see `aurum/strategies/base.py`).
- **Annualisation**: `aurum.core.timeframes.infer_bars_per_year(index)` for per-bar series;
  daily series use 252. Headline Sharpe is computed on DAILY returns (UTC dates) — per-bar
  Sharpe is reported separately.
- **Registries**: features via `@register_feature`, strategies via `@register_strategy`.
- **Logging**: `logging.getLogger(__name__)`; never `print` in library code (CLI may).

## 2. Package map & ownership

```
aurum/
  core/        timeframes.py instrument.py types.py interfaces.py config.py
  data/        schema.py pit.py resample.py  (done)
               loaders.py dukascopy.py macro.py synthetic.py calendar.py store.py
  features/    base.py (done) technical.py volatility.py microstructure.py
               multi_timeframe.py macro.py calendar.py regime.py pipeline.py
  models/      volatility.py (ewma done; add garch/har) regime.py (HMM)
  labels/      triple_barrier.py
  strategies/  base.py (done) trend.py mean_reversion.py breakout.py macro.py
               seasonal.py ml.py rl.py
  portfolio/   combiner.py sizing.py
  risk/        manager.py var.py
  execution/   costs.py simulator.py
  backtest/    result.py (done) engine.py metrics.py
  research/    splits.py stats.py report.py walkforward.py
  rl/          env.py train.py
  agents/      LLM trading desk (see §10)
  live/        broker.py paper.py mt5.py oms.py runner.py monitor.py
  cli.py
```

## 3. Data layer (`aurum.data`)

### 3.1 `loaders.py`
- `load_mt5_csv(path, timeframe, *, server_tz="Etc/GMT-2", point_size=0.01) -> bars`
  Parses MT5 History Center exports (`<DATE> <TIME> <OPEN>...<TICKVOL> <VOL> <SPREAD>`,
  tab or comma). `<SPREAD>` is in points → convert with `point_size` to price units.
  `server_tz` may be an IANA name, `"Etc/GMT-2"` style, or the special value `"NY+7"`
  meaning New-York-close-aligned servers (UTC+2 winter / UTC+3 summer, following US DST).
  Convert to UTC. Validate with `make_bars`.
- `load_csv(path, timeframe, *, time_col="time", tz="UTC", default_spread=None) -> bars`.
- `bars_from_ohlc(df, timeframe, default_spread)` — thin wrapper around `make_bars`.

### 3.2 `dukascopy.py`  (primary free source of real bid/ask history)
- `download_dukascopy(symbol="XAUUSD", start, end, timeframe="M1", *, cache_dir="cache/dukascopy",
  price_scale=None, max_workers=8, session=None) -> bars`
  Fetches daily candle files `https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM-1:02d}/{DD:02d}/{BID|ASK}_candles_min_1.bi5`
  (LZMA-compressed, month is ZERO-based, 24-byte big-endian records
  `>IIIIIf` = seconds-from-midnight-UTC, open, close, low, high, volume — note O,C,L,H order).
  Price = int / price_scale (XAUUSD point scale is 1000; verify empirically against
  plausible gold prices and auto-detect if `price_scale=None`). Builds mid OHLC =
  (bid+ask)/2 per field, `spread = ask_close - bid_close` averaged sensibly (use mean of
  open & close spreads, floor at 0), volume = bid volume. Skips empty (weekend) files.
  Caches raw files on disk; resamples to `timeframe` with `resample_bars` if not M1.
  Retries with backoff; polite rate (max_workers <= 8).
- `decode_bi5_candles(raw: bytes, day: date, price_scale) -> DataFrame` (pure; unit-tested
  with a synthetic LZMA blob).

### 3.3 `macro.py`
- `DEFAULT_YAHOO = {"dxy": "DX-Y.NYB", "us10y": "^TNX", "vix": "^VIX", "spx": "^GSPC",
  "silver": "SI=F", "gold_fut": "GC=F", "oil": "CL=F"}`
- `DEFAULT_FRED = {"real10y": "DFII10", "breakeven10y": "T10YIE", "fedfunds": "DFF"}`
- `fetch_yahoo_daily(tickers=DEFAULT_YAHOO, start, end, cache_dir) -> dict[str, DataFrame]`
- `fetch_fred(series=DEFAULT_FRED, start, end, cache_dir) -> dict[str, DataFrame]`
  via `https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES`.
- Every returned frame: index = observation date (UTC midnight, tz-aware), column
  `value` (close / level) plus optional extras, and `available_at`:
  Yahoo US markets: date + 21:30 UTC (after the US close, conservative across DST);
  FRED: date + 1 day + 21:30 UTC (publication lag). Both configurable via `availability_lag`.
- `load_macro_dir(path) -> dict[str, DataFrame]` / `save_macro_dir(dict, path)` (parquet).

### 3.4 `synthetic.py`
- `make_synthetic_bars(n=5000, timeframe="H1", *, seed=0, model="gbm", start="2020-01-06",
  annual_vol=0.16, drift=0.0, spread=0.30, start_price=1800.0, weekend_gaps=True,
  regime_params=None) -> bars`. Models: `"gbm"` (random walk, no edge — used for leakage
  tests: any strategy's Sharpe on it must be ~0), `"trend"` (AR(1) drift in returns),
  `"mean_revert"` (OU on log price), `"regime"` (Markov-switching vol/drift), `"jump"`
  (GBM + Poisson jumps). Realistic OHLC (high/low from a Brownian bridge approximation),
  spread noise, volume. Skips Sat/Sun when `weekend_gaps`.
- `make_synthetic_macro(bars, *, seed=0) -> dict[str, DataFrame]` — daily frames in the
  §3.3 format (dxy, us10y, real10y, vix, spx), mildly correlated with gold returns.
- `make_synthetic_events(start, end) -> DataFrame` in the §3.5 format.

### 3.5 `calendar.py`
- Event frame columns: `time` (UTC scheduled release), `name` (e.g. "NFP", "CPI", "FOMC"),
  `currency`, `importance` (1..3), `source`, `approximate` (bool). Index = RangeIndex.
  Scheduled times are published in advance, so using *future scheduled times* is not
  leakage; using *outcomes* (actual/surprise) is only allowed from `time` onward.
- `load_calendar_csv(path)` (generic: time,name,currency,importance[,actual,forecast]).
- `generate_rule_based_calendar(start, end)`: NFP = first Friday of each month 08:30
  America/New_York (DST-aware → UTC), marked `approximate=True` (holiday shifts exist);
  FOMC statement days at 14:00 America/New_York from a hard-coded list — ONLY include dates
  you can verify (try fetching federalreserve.gov FOMC calendars; if you cannot verify a
  year, omit it and document). CPI is not rule-based; leave it to CSV import.

### 3.6 `store.py`
- `save_bars(bars, path)` / `load_bars(path) -> bars` (parquet; preserves UTC tz,
  `available_at`, `attrs["timeframe"]` via parquet metadata).
- `frame_hash(df) -> str` stable sha256 of values+index (used for run provenance).

## 4. Features (`aurum.features`)

See `aurum/features/base.py` for the causal contract. Each module registers one or more
feature groups; each group returns many columns prefixed with its family.

| module | registry names | notes |
|---|---|---|
| technical.py | `returns`, `trend`, `momentum`, `meanrev`, `range` | vol-normalised returns (1..48 bars), EMA slopes & distance in ATR, MACD/ATR, ADX, RSI(2,14), stochastic, TSMOM strength at several horizons, z-scores vs rolling mean, Bollinger %B, Donchian position, candle anatomy |
| volatility.py | `volatility` | close-close, Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang rolling vols, vol ratios, vol-of-vol, ATR%. Also export plain helpers `atr(bars, n)`, `true_range(bars)`, `yang_zhang_vol(bars, n, bars_per_year)`, `parkinson_vol(...)` for other modules |
| microstructure.py | `microstructure`, `session` | spread in bps & rolling z, range/spread, volume z, gap vs prev close; UTC hour & weekday cyclical encodings, Asia/London/NY/overlap flags (DST-aware using zoneinfo for London & New York local hours) |
| multi_timeframe.py | `mtf` | for each of `htfs=("H4","D1")`: resample with `resample_bars`, compute compact trend/momentum/vol features on the HTF bars, then `align_htf`. Also previous completed day's high/low/close & distance to them in ATR |
| macro.py | `macro` | requires `md.macro`; per series: 1/5/20-day changes (log returns for prices, bp changes for yields), 250-day rolling z of level; rolling 60-day correlation & beta of daily gold returns vs dxy/real10y (gold daily built from bars via `resample_bars(...,"D1")` so it is point-in-time); all aligned with `asof_join(bars.available_at, ...)`. Missing series → skip those columns (log at INFO); no macro at all → empty frame with the right index |
| calendar.py | `calendar` | requires `md.events`; hours until next event (importance>=3, capped 72), hours since last, in-window flags (±30 min, ±2 h), next-event-type one-hots (NFP/CPI/FOMC) |
| regime.py | `regime` | BOUNDED rolling percentile rank of realised vol over `rank_years` years of bars (default 1.0; `rank_window=<bars>` overrides; NaN until the window is full, so the group's warm-up `vol_window + W` is also where values stop depending on the history start — live/research parity), high/low-vol flags from it (`expanding=True` adds the legacy expanding rank `regime_vol_pctrank_exp`), Kaufman efficiency ratio, rolling variance-ratio / Hurst proxy, trend strength normalised by the RMS of the same window (`vol_halflife=` restores the EWMA normaliser); NO fitted models here (HMM lives in `aurum.models.regime`) |

### 4.1 `pipeline.py`
```python
class FeaturePipeline:
    def __init__(self, groups: list[str] | None = None, overrides: dict[str, dict] | None = None,
                 scaler: str = "robust", clip: float = 5.0): ...
    def compute(self, md) -> pd.DataFrame          # raw concat of all groups (causal)
    def fit(self, raw_train: pd.DataFrame) -> "FeaturePipeline"   # per-column location/scale
    def transform(self, raw: pd.DataFrame) -> pd.DataFrame        # scale, clip, NaN->0
    def fit_transform(self, raw) -> pd.DataFrame
    @property
    def max_lookback(self) -> int
    @property
    def columns(self) -> list[str]
    def save(self, path) / @classmethod load(cls, path)   # JSON: groups, overrides, columns, stats, version
    def parity_report(self, a: pd.DataFrame, b: pd.DataFrame, atol=1e-8) -> pd.DataFrame
```
Scaler stats come ONLY from `raw_train`; columns constant in train are dropped; unseen
columns at transform time raise.

## 5. Models (`aurum.models`)
- `volatility.py`: keep `ewma_volatility`; add `Garch11` (`fit(returns)`, `forecast(returns)`
  → causal conditional vol series annualised, params stored; MLE via scipy with
  stationarity constraints) and `har_rv_forecast(daily_rv)`; `blend_vol(*series, weights)`.
- `regime.py`: `GaussianHMM(n_states=2, seed=0)` with `fit(x)` (Baum–Welch on TRAIN),
  `filter(x) -> DataFrame` of forward-filtered state probabilities (causal: uses x[:t+1]
  only), `predict_next(x)`; states ordered by variance (state 0 = calm).

## 6. Strategies (`aurum.strategies`)
Base class in `strategies/base.py`. Each strategy documents its economic rationale. All are
long/short unless noted. Required families (each file may hold several classes):
- `trend.py`: `tsmom` (multi-horizon time-series momentum, vol-scaled sign/strength, à la
  Moskowitz–Ooi–Pedersen), `ema_cross` (fast/slow EMA with continuous forecast = scaled
  distance), `donchian` (turtle-style breakout with ATR exits, stateful), `kalman_trend`
  (local-linear-trend Kalman filter slope t-stat).
- `mean_reversion.py`: `zscore_fade` (fade |z|>k vs rolling mean, gated by low
  efficiency ratio / non-trending regime), `rsi2` (Connors RSI(2) with trend filter),
  `bollinger_revert`.
- `breakout.py`: `vol_squeeze` (Bollinger inside Keltner → breakout direction),
  `orb` (session opening-range breakout for London and New York opens, DST-aware).
- `macro.py`: `macro_factor` (gold vs DXY and real-yield momentum: long gold when dollar
  and real yields fall; uses `md.macro`, point-in-time), `risk_off` (VIX spike regime).
- `seasonal.py`: `intraday_seasonality` (learned hour-of-week mean returns on TRAIN with
  shrinkage; trainable). Positions are chosen NET of costs (`cost_aware=True`): a periodic
  mean-variance problem with proportional costs over the weekly cycle (DP on a position
  grid; costs from the TRAIN bars' spreads/ranges via `params.costs`, default `CostModel()`;
  `cost_multiplier` = required edge/cost margin, default 2; `dead_zone="auto"`).
  `cost_aware=False` or `cost_multiplier=0` gives the frictionless table.
- `ml.py`: `ml_gbm` (HistGradientBoosting on FeaturePipeline features predicting sign of
  forward vol-normalised return with purged training, probability → forecast via
  `2p-1` with calibration & dead-zone), `meta_label` (triple-barrier meta-labelling of a
  primary strategy: ML model predicts whether the primary signal will hit TP before SL,
  forecast = primary * size-from-probability). Both trainable, both use `aurum.labels`.
- `rl.py`: `rl_ppo` adapter that loads a trained SB3 policy from `aurum.rl` (lazy torch
  import — importing `aurum.strategies.rl` must NOT require torch). Attaching an artifact
  embeds its files' bytes + SHA-256 fingerprint in the strategy (self-contained pickles;
  `aurum.rl.train.read_artifact_bytes` / `load_artifact_bytes`); tampered bytes are refused.

## 7. Portfolio & risk
- `portfolio/combiner.py`: `ForecastCombiner(method="sharpe_shrink"|"equal"|"inverse_vol"|"hrp",
  shrinkage=0.5, max_weight=0.4, fdm_cap=2.5, ..., allow_unallocated=True, cost_multiplier=1.0)`;
  `fit(forecasts, close, *, bars=None, spread=None, costs=None, instrument=None,
  cost_per_turnover=None)` computes each strategy's unit-vol return stream
  `f_i[t] * r[t+1] / vol[t]` on TRAIN, NET of `c_t * |f_i[t] - f_i[t-1]|` with
  `c_t = (spread_eff/2 + slippage + commission/oz) / (close[t] * vol[t])` from the cost model
  when cost information is given (gross otherwise, noted), weights by method, and a forecast
  diversification multiplier (`sum(w)/sqrt(w'Hw)`); `combine(forecasts) -> Series` in [-1,1];
  `weights_` attribute; `explain()` dict for agents/reports. With `allow_unallocated=True`
  (config `combiner.allow_unallocated`, default) strategies with a non-positive NET Sharpe get
  zero weight and the cap never forces weight onto them, so weights may sum to < 1 (less
  risk; no positive net Sharpe -> flat book); `False` restores sum-to-1.
- `portfolio/sizing.py`: `VolTargetSizer(target_vol=0.10, max_leverage=2.0, max_lots=None,
  rebalance_band=0.10, kelly_cap=None, drawdown_derisk=((0.10, 0.5), (0.15, 0.25)))`
  implementing `aurum.core.interfaces.PositionSizer`:
  `notional = forecast * target_vol / vol_ann * equity`, capped at `max_leverage*equity`,
  scaled down by drawdown steps, → lots (`/ (contract_size*price)`), `round_lots`; if
  `|target-current| < rebalance_band*max(|target|,|current|)` keep `current` (turnover
  control). Also `FixedFractionalSizer(risk_per_trade=0.005, stop_atr=2.0)` for stop-based
  sizing.
- `risk/manager.py`: `RiskLimits` dataclass (`max_lots`, `max_leverage`, `max_daily_loss=0.03`,
  `max_drawdown=0.20` (hard kill), `max_spread=None` (price units; block NEW risk when
  exceeded), `event_blackout_before_min=30`, `event_blackout_after_min=30`,
  `event_min_importance=3`, `blackout_mode="no_new_risk"|"flatten"`, `max_trades_per_day=None`,
  `stale_data_seconds=None`, `max_margin_utilisation=0.5`) and `StandardRiskManager(limits,
  instrument, state_path=None)` implementing `aurum.core.interfaces.RiskManager`.
  Semantics: risk can only move `approved_lots` toward 0 relative to `target_lots` and never
  beyond `current_lots` in the risk-increasing direction when a "no new risk" rule fires.
  Daily loss uses equity at the first bar of each UTC day (or broker rollover — configurable).
  Hard kill (max drawdown from peak, daily loss) sets `halted=True` → approved 0; `halted`
  persists (JSON state file if `state_path`) until `reset_halt(confirm="RESET")`.
- `risk/var.py`: historical / Gaussian / Cornish–Fisher VaR & ES, stress scenarios
  (`gap_shock(position_lots, price, pct)`), `risk_report(result) -> dict`.

## 8. Execution & backtest
- `execution/costs.py`: `CostModel(spread_multiplier=1.0, min_spread=0.10, slippage_fixed=0.02,
  slippage_range_frac=0.02, impact_coef=0.0, commission_per_lot=None (→ instrument),
  financing=FinancingModel())` with `fill_price(side, mid, spread, bar_range, lots)` →
  (price, spread_cost, slippage_cost), `commission(lots)`, `swap(lots, nights, *, instrument,
  price=None, rate_nights=None)` and `swap_between(lots, start, end, *, instrument, price=None,
  rates=None)` (delegate to the financing model; nights are triple-weighted on
  `triple_swap_weekday`). Rollover happens when a position is held across
  `instrument.rollover_hour_utc` on a weekday. `CostModel.zero()` has no financing either.
  `FinancingModel(mode="rate"|"fixed"|"none", markup_long=0.025, markup_short=0.025,
  lease_rate=0.0, rate_series="fedfunds", rate_unit="percent", fallback_rate=0.03,
  day_count=360)`: `"rate"` (default) charges per financing night
  `-lots * contract_size * P * (r - lease ± markup) / day_count` (long `+markup_long`, short
  `-markup_short`), `P` the mid at the rollover, `r` the benchmark from `md.macro[rate_series]`
  read POINT-IN-TIME as of the rollover instant `R` (`RateCurve`: latest `available_at <= R`,
  `fallback_rate` before the first print / without a series, warned once); `"fixed"` = the
  instrument's per-lot swaps (legacy, bit-identical); `"none"`. Config: `costs.financing`.
- `execution/simulator.py`:
```python
class ExecutionSimulator:
    def __init__(self, bars, instrument=XAUUSD, costs=CostModel(), initial_equity=100_000.0,
                 *, validate=True, rates=None): ...   # rates: md.macro / frame / Series / RateCurve
    def reset(self, start: int = 0, equity: float | None = None) -> None
    index: int; equity: float; position: float; done: bool; cash etc.
    def step(self, target_lots: float, *, stop_price: float | None = None,
             take_profit: float | None = None) -> StepResult
        # Called at the close of bar `index`. Trades to target at open of index+1, applies
        # costs, checks intrabar stop/TP on bar index+1 (stop first if both touched; a gap
        # through the stop fills at the open; `intrabar_exit`, shared with the paper broker and
        # the live runner), applies financing on rollover (rate mode: notional at close[t+1],
        # close[t] for a rollover in the gap), marks to market at close of index+1, advances
        # index. StepResult has equity, pnl, costs dict, fills, position, done, exit_reason.
    def result(self) -> BacktestResult   # assemble series/trades so far
```
- `backtest/engine.py`:
```python
def run_backtest(md, forecast: pd.Series, *, sizer, risk=None, instrument=XAUUSD,
                 costs=CostModel(), initial_equity=100_000.0, vol=None,
                 stop_atr_mult: float | None = None, start=None, end=None,
                 financing=None, rates=None) -> BacktestResult
def run_target_lots(md, target_lots: pd.Series, ...) -> BacktestResult   # for pre-sized paths
```
  `financing=` overrides `costs.financing` (model, kwargs or mode name); `rates=` defaults to
  `md.macro` (read as of each rollover). Same for `buy_and_hold_benchmark` (frictionless =
  no financing). The RL env (`GoldTradingEnv(..., rates=)`) and `PaperBroker(..., rates=)`
  must get the same rate source to stay in parity with the engine.
  Loop over bars: `vol` defaults to `ewma_volatility(close)`; at each close compute
  sizer target → risk.evaluate → sim.step. Record risk events.
- `backtest/metrics.py`: `compute_metrics(result_or_returns, *, bars_per_year=None, trades=None,
  positions=None, costs=None, equity=None) -> dict` with: total_return, cagr, ann_vol, sharpe
  (daily), sharpe_bar, sortino, calmar, max_drawdown, max_dd_duration_days, skew, kurtosis,
  var_95_daily, cvar_95_daily, tail_ratio, n_trades, trades_per_year, win_rate, profit_factor,
  avg_win, avg_loss, expectancy, avg_hold_bars, exposure, turnover_lots_per_year,
  total_costs, cost_drag_ann, swap_total, best_day, worst_day. Also `daily_returns(equity)`,
  `drawdown_series(equity)`.

## 9. Research (`aurum.research`)
- `splits.py`: `walk_forward_splits(n, train, test, step=None, anchored=False, purge=0,
  embargo=0)` → list of `(train_idx, test_idx)` numpy arrays; `purged_kfold(n, k, purge,
  embargo)`; `cpcv_splits(n, n_groups=6, k_test=2, purge, embargo)` → splits + path map.
- `stats.py`: `sharpe(returns, periods)`, `probabilistic_sharpe(sr, n, skew, kurt, sr_star=0)`,
  `deflated_sharpe(sr, n, skew, kurt, trial_srs)`, `expected_max_sharpe(n_trials, var)`,
  `min_track_record_length(...)`, `pbo_cscv(perf_matrix, n_splits=16)` (Bailey et al.),
  `stationary_bootstrap(returns, stat_fn, n_boot=2000, mean_block=None, seed=0) -> CI dict`,
  `sharpe_ci(...)`. Pure numpy/scipy, documented with references.
- `report.py`: `write_tearsheet(result, path, *, benchmark=None, title="", extra=None) -> Path`
  single self-contained HTML (inline base64 PNGs via matplotlib Agg): equity vs benchmark
  (log), underwater drawdown, rolling 6-month Sharpe, monthly return table, return
  histogram, position/exposure, cost attribution, metrics table, trade stats, notes/extra.
- `walkforward.py` (wave 2): orchestrates features → strategies → combiner → backtest per
  fold with refits; stitches OOS forecasts; runs ONE continuous backtest on the stitched
  OOS; returns `WalkForwardReport` with per-strategy OOS metrics, combined metrics, DSR/PBO.
  `Strategy.fit` gets macro rows published by the last training bar's close only. The
  combiner is fitted net of costs (bars + `costs` + `instrument` of the config). Holdout
  evaluations are appended to the holdout ledger (`<output root>/holdout_ledger.jsonl`:
  timestamp, config/data hash, window, strategies, metrics); it is read BEFORE the holdout
  is evaluated, and overlapping windows evaluated by OTHER config hashes raise a WARNING +
  report note and add to the DSR `n_trials` (= `walkforward.n_trials` or #strategies +
  distinct prior other-config looks). Holdout OOS forecasts are saved
  (`holdout/oos_forecasts.parquet`).
- `aurum train-final --config C --out DIR [--cutoff TS] [--from-run WF_DIR]`: production
  artifact fitted on data closed at/before the cutoff (macro truncated point-in-time); the
  combiner weights come from the stitched OUT-OF-SAMPLE forecasts (research + holdout) of a
  walk-forward run with the same config hash (or one run in-process); loadable by
  `aurum.live.runner.load_artifact` (with the TRAIN-window `feature_reference`).

## 10. LLM trading desk (`aurum.agents`)
Multi-agent layer driven by the Claude API (`anthropic` SDK). A **Chief Investment Officer**
agent runs a decision cycle at each bar close (or on demand): it can consult predefined
specialist agents (Macro Strategist, Quant Analyst, Risk Officer, Execution Trader) and can
**create new ad-hoc specialist agents** at runtime (role + instructions + a subset of
whitelisted data tools). Specialists run as separate Claude tool-use loops (possibly in
parallel) and return memos. The Chief ends every cycle by calling `submit_decision`.
- The decision is converted by a `DecisionPolicy` (`mode="advisory"|"overlay"|"discretionary"`)
  into a forecast/scale that goes through the SAME sizer and risk manager — the LLM cannot
  bypass `RiskManager`. Default mode `overlay` (Chief may scale the quant forecast in [0, 1]
  or veto; may not flip direction or add risk). `discretionary` allows a bounded forecast in
  `[-max_abs_forecast, max_abs_forecast]`.
- All data reaches agents through a `DeskDataProvider` protocol (JSON-serialisable
  snapshots: market, quant signals, risk, macro, calendar, backtest stats, positions). A
  point-in-time `HistoricalDeskDataProvider` enables replay backtests (with the documented
  caveat that LLMs may have memorised historical prices; optional anonymisation of dates
  and price levels mitigates this).
- Every cycle is journaled (JSONL): prompts, tool calls, memos, final decision, token usage.
- Tests use a scripted fake client — no network, no API key.

## 11. Live (`aurum.live`)
- `broker.py`: `Broker` protocol (`account()`, `positions(symbol, magic)`, `place_order(order)`,
  `close_all(symbol, magic)`, `latest_bars(symbol, timeframe, n)` returning CLOSED bars only,
  `is_demo()`); `paper.py` `PaperBroker` backed by `ExecutionSimulator` semantics (same
  `intrabar_exit`, same financing valuation; `rates=` / `set_rates()` — the runner passes and
  refreshes `macro_dir` frames);
  `mt5.py` `MT5Broker` (lazy `import MetaTrader5`; magic-number isolation; broker-side SL;
  retcode handling; server-time → UTC).
- `oms.py`: idempotent client ids, reconciliation (target vs actual), retries, rejects.
- `runner.py`: bar-close scheduler; builds `MarketData` from the broker, runs the SAME
  feature pipeline (loaded from artifact), strategies, combiner, optional LLM desk, sizer,
  risk manager, OMS. `dry_run=True` default; refuses real (non-demo) accounts unless
  `allow_live_real=True` in config AND `--i-understand-real-money` CLI flag. The final
  forecast of the last decision is persisted (`runner_state.json` `prev_final_forecast`) and
  passed to the desk as `previous_forecast` (so `on_failure="hold"` survives restarts);
  `backtest.stop_cooldown_bars` is honoured live.
- `monitor.py`: feature drift (PSI) vs training stats, live vs backtest slippage, heartbeat,
  optional webhook alerts.
