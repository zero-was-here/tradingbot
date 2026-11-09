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
- **Annualisation**: `aurum.core.timeframes.infer_bars_per_year(index)` for per-bar series
  in reports; causal estimators (EWMA and GARCH volatility, sizing, features) use the
  data-independent `nominal_bars_per_year` (252·23 h / bar, 5,796 on H1); daily series
  use 252. Headline Sharpe is computed on DAILY returns (UTC dates) — per-bar
  Sharpe is reported separately.
- **Registries**: features via `@register_feature`, strategies via `@register_strategy`.
- **Logging**: `logging.getLogger(__name__)`; never `print` in library code (CLI may).

## 2. Package map & ownership

```
aurum/
  core/        timeframes.py instrument.py types.py interfaces.py config.py
  data/        schema.py pit.py resample.py loaders.py dukascopy.py macro.py synthetic.py
               calendar.py store.py
  features/    base.py technical.py volatility.py microstructure.py
               multi_timeframe.py macro.py calendar.py regime.py pipeline.py
  models/      volatility.py (EWMA / GARCH / HAR) regime.py (HMM)
  labels/      triple_barrier.py
  strategies/  base.py trend.py mean_reversion.py breakout.py macro.py
               seasonal.py ml.py rl.py allocation.py regime.py
  portfolio/   combiner.py sizing.py
  risk/        manager.py var.py
  execution/   costs.py simulator.py
  backtest/    result.py engine.py metrics.py
  research/    splits.py stats.py report.py walkforward.py
               sweep.py montecarlo.py attribution.py
  rl/          env.py train.py
  agents/      LLM trading desk (see §10)
  live/        broker.py paper.py mt5.py oms.py runner.py monitor.py state.py
  viz/         style.py (figure style) dashboard.py (HTML dashboard) cli.py
  cli.py       core commands; cli_ext.py adds sweep / stress / explain (+ dashboard) (§12)
```

## 3. Data layer (`aurum.data`)

### 3.1 `loaders.py`
- `load_mt5_csv(path, timeframe, *, server_tz=DEFAULT_MT5_SERVER_TZ, point_size=0.01,
  default_spread=None) -> bars`
  Parses MT5 History Center exports (`<DATE> <TIME> <OPEN>...<TICKVOL> <VOL> <SPREAD>`,
  tab, comma or semicolon; header-less MT4 files too). `<SPREAD>` is in points → convert
  with `point_size` to price units (without a `<SPREAD>` column `default_spread` is
  required). `server_tz` may be an IANA name, `"Etc/GMT-2"` style, or the special value
  `"NY+7"` meaning New-York-close-aligned servers (UTC+2 winter / UTC+3 summer, following
  US DST). Convert to UTC. Validate with `make_bars`.
  `DEFAULT_MT5_SERVER_TZ = "Etc/GMT-2"` (fixed UTC+2) is kept for backward compatibility;
  **relying on it logs a WARNING** (on a New-York-close server it shifts every US-summer bar
  by one hour: pass `"NY+7"` or the broker's zone).
- `load_csv(path, timeframe, *, time_col="time", tz="UTC", default_spread=None) -> bars`.
- `bars_from_ohlc(df, timeframe, default_spread)` — thin wrapper around `make_bars`.
- `attrs["source"]` is deterministic: `"mt5:<file name>"` / `"csv:<file name>"`, with
  `source_name(path_or_buffer)` returning the base name of a path or named file object and
  `BUFFER_SOURCE_NAME = "<buffer>"` for an anonymous buffer.

### 3.2 `dukascopy.py`  (primary free source of real bid/ask history)
- `download_dukascopy(symbol="XAUUSD", start, end, timeframe="M1", *, cache_dir="cache/dukascopy",
  price_scale=None, max_workers=8, session=None, source_resolution="M1", offline=False, ...) -> bars`
  Fetches daily candle files `https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM-1:02d}/{DD:02d}/{BID|ASK}_candles_min_1.bi5`
  (or monthly `{BID|ASK}_candles_hour_1.bi5` files with `source_resolution="H1"`, ~24x
  fewer requests) (LZMA-compressed, month is ZERO-based, 24-byte big-endian records
  `>IIIIIf` = seconds-from-period-start-UTC, open, close, low, high, volume — note O,C,L,H
  order). Price = int / price_scale (XAUUSD point scale is 1000; auto-detected if
  `price_scale=None`). Builds mid OHLC = (bid+ask)/2 per field,
  `spread = max(0, ((ask_o - bid_o) + (ask_c - bid_c)) / 2)`, volume = bid volume. Drops
  the flat zero-volume filler candles (weekends, daily break, holidays).
  Caches raw files on disk; resamples to `timeframe` with `resample_bars` if not M1.
  Retries with backoff (honouring `Retry-After`); at most 8 workers with an adaptive
  concurrency limit (the free feed throttles hard).
- `prefetch_cache(...)` (time-budgeted, resumable, newest-first cache fill) and
  `build_dataset(out_dir, ...)` (writes `{sym}_M15/H1/H4/D1/D1_nyclose` parquet files and a
  `manifest.json` with hashes and a quality report) back `aurum data download`.
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
  Yahoo cash indices (`^GSPC`, `^VIX`, `^TNX`): date + 21:30 UTC (after the US close,
  conservative across DST); Yahoo futures (`=F`) and ICE (`.NYB`, the dollar index):
  date + 22:30 UTC (their session ends 17:00 New York = 22:00 UTC in winter);
  FRED: the NEXT US federal business day + 21:30 UTC (publication lag; a Friday print is
  usable on Monday). Rows still provisional at fetch time are dropped. Configurable via
  `availability_lag` (for FRED the days part counts business days).
- Yahoo needs the optional `yfinance` dependency (`pip install aurum[data]`); without it the
  Yahoo series are skipped with a warning.
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
- `load_calendar_csv(path, *, tz="UTC", source=None)` (generic:
  time,name,currency,importance[,actual,forecast,previous]).
- `generate_rule_based_calendar(start, end, *, include=("NFP", "FOMC"), nfp_rule="first_friday")`:
  NFP at 08:30 America/New_York (DST-aware → UTC) on the rule's date, marked
  `approximate=True` (holiday shifts exist); FOMC statement releases from the hard-coded
  `FOMC_STATEMENTS` list (2012–2027, federalreserve.gov calendars retrieved 2026-09-26):
  times verified from the press releases for 2016-01 .. 2026-09 (`approximate=False`),
  Fed practice before 2016 and 14:00 ET for future meetings (`approximate=True`);
  unscheduled actions excluded; outside 2012–2027 no FOMC rows (warning). CPI is not
  rule-based; leave it to CSV import.

### 3.6 `store.py`
- `save_bars(bars, path)` / `load_bars(path) -> bars` (parquet; preserves UTC tz,
  `available_at`, `attrs["timeframe"]` via parquet metadata).
- `frame_hash(df) -> str` stable sha256 of values+index (used for run provenance).

### 3.7 `resample.py`
- `resample_bars(bars, to, *, daily_anchor_hour_utc=0, complete_until=None,
  complete_from="auto") -> bars`. An HTF bar's `available_at` is its bucket end.
  `daily_anchor_hour_utc` shifts D1/H4 bucket boundaries (22 = New York close).
- **Tail rule**: a bucket is emitted only once complete (`complete_until` may assert that
  historical data is final up to an instant, so a dataset ending on a Friday keeps its
  Friday bar; it never makes a bar available earlier).
- **Head rule** (`complete_from`): the LEADING bucket is emitted only if its first base bar
  opens exactly at the bucket open, or the bucket opens at or after `complete_from`;
  otherwise it holds part of its period and is dropped. `None` = strict; `"auto"`
  (`COMPLETE_FROM_AUTO`, default) = strict unless `complete_until` is given (the Dukascopy
  build, which cuts its own head; datasets are byte-identical).
- Guarantee: `resample_bars(bars[k:j])` is a subset of `resample_bars(bars)` with identical
  values for every emitted bar, so HTF features never depend on where the history starts.

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
- `volatility.py`: `ewma_volatility(close, *, halflife_bars=48, bars_per_year=None,
  min_periods=20, floor=0.03, cap=2.0)` (annualised, causal, clipped to `[floor, cap]`,
  warm-up rows 0.20); `Garch11(*, mean="zero", bars_per_year=None, ...)` (`fit(returns)`,
  `forecast(returns)` → causal conditional vol series annualised, params stored; MLE via
  scipy with stationarity constraints). **Annualisation is nominal**: without
  `bars_per_year`, `fit` uses `nominal_bars_per_year(index_bar_minutes(train index))` =
  252·23·60 / bar minutes intraday (5,796 for H1), 252 for D1 — the same convention as
  `ewma_volatility`, never the realised bar density of the sample.
  `daily_realised_variance`, `HarRV(lags=(1, 5, 22))` / `har_rv_forecast(daily_rv)`;
  `blend_vol(*series, weights)`.
- `regime.py`: `GaussianHMM(n_states=2, seed=0, *, n_iter=200, tol=1e-4, ...)` with
  `fit(x)` (Baum–Welch on TRAIN; the smoother is used only inside `fit`),
  `filter(x) -> DataFrame` of forward-filtered state probabilities (causal: uses x[:t+1]
  only), `predict_next(x)`, `filtered_state(x)`, `score(x)`, `stationary_distribution_`,
  `expected_durations_`, `summary()`; states ordered by variance (state 0 = calm). Used by
  the `regime_switch` strategy (§6).

## 6. Strategies (`aurum.strategies`)
Base class in `strategies/base.py`. Each strategy documents its economic rationale. All are
long/short unless noted. The registry holds 17 strategies (the fixed module list below).
A strategy is built only from parameters it declares in `default_params()`: an unknown
parameter is a configuration error (`AurumConfig.build_strategies` and the config validation
of every `strategies[]` entry; `ml_gbm`, `meta_label` and `rl_ppo` also reject unknown params
themselves). Required families (each file may hold several classes):
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
- `ml.py`: `ml_gbm` (HistGradientBoosting on its own per-fold FeaturePipeline features —
  the 10 price groups plus `macro`/`calendar` when available — predicting the
  triple-barrier label by default (`target="triple_barrier"`; or the sign of the forward
  vol-normalised return) with purged, uniqueness-weighted training, probability → forecast
  via calibrated `2p-1` with dead-zone and Carver scaling), `meta_label` (triple-barrier
  meta-labelling of a primary strategy, default `tsmom`: ML model predicts whether the
  primary signal will hit TP before SL, forecast = primary * size-from-probability). Both
  trainable, both use `aurum.labels`, and both have a validation **skill gate**
  (`skill_gate_z=2`): unless the validation AUC beats 0.5 by 2 standard errors the
  forecast is 0.
- `rl.py`: `rl_ppo` adapter that loads a trained SB3 policy from `aurum.rl` (lazy torch
  import — importing `aurum.strategies.rl` must NOT require torch). Attaching an artifact
  embeds its files' bytes + SHA-256 fingerprint in the strategy (self-contained pickles;
  `aurum.rl.train.read_artifact_bytes` / `load_artifact_bytes`); tampered bytes are refused.
  `retrain=None` resolves to False when `artifact_dir` is given at construction (True
  otherwise); `artifact_sha256` (>= 8 hex digits) pins the artifact. With `retrain=False`,
  `fit` verifies the artifact (embedded bytes vs SHA-256, the directory, the pin, the bar
  size) instead of training. **Look-ahead guard**: `fit` raises if the artifact's recorded
  train/validation end (`config.json` `data_ranges`, fingerprinted) is after the fit window's
  last bar, and `generate` refuses bars after `fit_cutoff_` that are not strictly after the
  artifact's last bar — a pre-trained policy can only be walk-forwarded on folds whose
  training ends at or after its validation end.
- `allocation.py`: `managed_gold` (long-only by default; `allow_short` allows −1): a
  strategic long `base_weight` tilted by Carver-scaled multi-horizon momentum,
  `raw = base_weight + tilt_weight · clip(FDM · mean_h(0.6267 · z_h), −1, 1)` with
  `z_h = ln(C_t/C_{t−h}) / (σ_t √h)`; a de-risk latch with hysteresis (enter when
  `ln C < EMA(trend_halflife)` AND `vol_fast/vol_slow >= vol_enter`, exit when
  `ln C >= EMA` OR the ratio `< vol_exit`) caps `raw` at `floor`; `f = clip(raw, 0, 1)`.
  H1 defaults: horizons 480/1440/2880/5796, `trend_halflife` 1600, vol half-lives 120/2880,
  `vol_enter`/`vol_exit` 1.25/1.0, `floor` 0. `components(md)` exposes every intermediate.
- `regime.py`: `regime_switch` (trainable): a base forecast (default `tsmom`; a trainable
  base is fitted too) scaled by a 2-state `GaussianHMM` fitted on TRAINING daily returns
  (`100·Δln close`, daily bars via `resample_bars(D1, anchor 22 UTC)` for intraday data, bar
  returns for D1): `scale = clip((p_ok − p_flat)/(p_full − p_flat), 0, 1)` with
  `p_ok = 1 − P(chaotic | r_≤t)` from the forward filter, each day's value usable from that
  day's `available_at` (`asof_join`); defaults `p_flat` 0.5, `p_full` 0.9. Abstains
  (`f = 0`) with fewer than `min_train_obs` (60) training returns.
  `regime_probabilities(md)`, `filtered_probabilities(md)`, `summary()`.
  The latch and the HMM filter are path-dependent: a short live window converges to the
  full-history value after the first latch exit / a few expected regime durations (not
  look-ahead).

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
  risk; no positive net Sharpe -> flat book); `False` restores sum-to-1. The config also
  accepts `combiner.method: fixed` (weights from `strategies[].weight`, handled by
  `aurum.core.config` / `aurum.research.walkforward`, not by `ForecastCombiner`).
- `portfolio/sizing.py`: `VolTargetSizer(target_vol=0.10, max_leverage=2.0, max_lots=None,
  rebalance_band=0.10, kelly_cap=None, drawdown_derisk=((0.10, 0.5), (0.15, 0.25)), *,
  min_vol=0.02)` implementing `aurum.core.interfaces.PositionSizer`:
  `notional = forecast * target_vol / max(vol_ann, min_vol) * equity`, capped at
  `max_leverage*equity` (or the Kelly cap), scaled down by drawdown steps, → lots
  (`/ (contract_size*price)`), `round_lots`; if
  `|target-current| < rebalance_band*max(|target|,|current|)` keep `current` (turnover
  control). Also `FixedFractionalSizer(risk_per_trade=0.005, stop_atr=2.0, *, ...)` for
  stop-based sizing. **Its stop is external**: the sizer assumes a stop `stop_atr x ATR`
  away but never places one; stops come only from `backtest.stop_atr_mult` on the engine's
  bar ATR (use `stop_atr_mult` ≈ `stop_atr · sqrt(23)` ≈ 4.8 on H1, or pass `atr=`).
- `risk/manager.py`: `RiskLimits` dataclass (`max_lots=None`, `max_leverage=3.0`,
  `max_daily_loss=0.03`, `max_drawdown=0.20` (hard kill), `max_spread=None` (price units;
  block NEW risk when exceeded), `event_blackout_before_min=30`, `event_blackout_after_min=30`,
  `event_min_importance=3`, `blackout_mode="no_new_risk"|"flatten"`, `max_trades_per_day=None`,
  `stale_data_seconds=None`, `max_margin_utilisation=0.5`, `daily_reset="utc"|"rollover"`,
  `daily_loss_persistent=True`, `event_lookahead_min=0`) and `StandardRiskManager(limits,
  instrument, state_path=None, *, events=None)` implementing
  `aurum.core.interfaces.RiskManager`.
  Semantics: risk can only move `approved_lots` toward 0 relative to `target_lots` and never
  beyond `current_lots` in the risk-increasing direction when a "no new risk" rule fires.
  Daily loss uses equity at the first bar of each UTC day (or broker rollover — configurable).
  Hard kill (max drawdown from peak, daily loss) sets `halted=True` → approved 0; `halted`
  persists (JSON state file if `state_path`) until `reset_halt(confirm="RESET")`, which
  re-bases the equity peak and day start to the last observed equity (CLI:
  `aurum live reset-halt`, §11). With
  `daily_loss_persistent=False` a daily-loss halt clears at the next day instead. Research
  runs use `RESEARCH_RISK_DEFAULTS` (`aurum.core.config`: `daily_loss_persistent=False`,
  `max_drawdown=None`; the kill switch is a live control), live runs the `risk.live` limits.
- `risk/var.py`: historical / Gaussian / Cornish–Fisher VaR & ES, stress scenarios
  (`gap_shock(position_lots, price, pct)`, `stress_test`, `DEFAULT_STRESS_SCENARIOS`),
  `risk_report(result, *, level=0.95, price=None, last_price=None, bars=None,
  instrument=XAUUSD, var_window=250, scenarios=None) -> dict`. The stress price is the first
  available of `last_price`/`price` > `result.meta["last_price"]` > the bars close at or
  before the last equity mark > the last fill, recorded in
  `report["position"]["price_source"]`.

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
  Further keyword arguments: `take_profit_atr_mult`, `atr_period=14`, `stop_cooldown_bars=0`
  (same-side re-entry block after a stop), `event_horizon_hours=24`, `bars_per_year`,
  `compute_metrics=True`, and (`run_backtest` only) `forecast_hook` / `hook_every` (a
  callback that may replace the forecast at each close; used by the LLM-desk replay).
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
- `report.py`: `write_tearsheet(result, path, *, benchmark=None, title="", extra=None,
  n_trials=None, dark_charts=True, n_boot=None, seed=None) -> Path`
  single self-contained HTML (inline base64 PNGs via matplotlib Agg): equity vs benchmark
  (log), underwater drawdown, rolling 6-month Sharpe, monthly return table, return
  histogram, position/exposure, cost attribution, metrics table, trade stats, notes/extra.
  `n_boot`/`seed` resolve explicit > `extra["config"]` (`walkforward.n_boot`, `seed`) >
  `DEFAULT_N_BOOT` 1000 / `DEFAULT_SEED` 0.
- `walkforward.py`: orchestrates features → strategies → combiner → backtest per
  fold with refits; stitches OOS forecasts; runs ONE continuous backtest on the stitched
  OOS; returns `WalkForwardReport` with per-strategy OOS metrics, combined metrics, DSR/PBO.
  `purge: auto` = the largest label horizon of the trainable strategies (24 bars for
  `ml_gbm` / `meta_label`, 0 without them). `Strategy.fit` gets macro rows published by the
  last training bar's close only. The combiner is fitted net of costs (bars + `costs` +
  `instrument` of the config); with `walkforward.combiner_fit: oos` (default) fold k's
  combiner is fitted on the stitched OOS forecasts of EARLIER folds (equal weights until
  `combiner_min_obs` OOS bars exist), with `train` on the in-sample training-window
  forecasts (biased toward trainable strategies; noted in the report). Holdout
  evaluations are appended to the holdout ledger (`<output root>/holdout_ledger.jsonl`:
  timestamp, kind, config/data hash, `data_source` (`real`/`synthetic`,
  `data_source_kind`), window, strategies, metrics with `strategy_sharpe` excluding the
  benchmark and a separate `benchmark_sharpe`); it is read BEFORE the holdout is evaluated,
  and overlapping windows evaluated by OTHER config hashes on the same kind of data raise a
  WARNING + report note and add to the DSR `n_trials` (= `walkforward.n_trials` or
  #strategies + distinct prior other-config looks). Holdout OOS forecasts are saved
  (`holdout/oos_forecasts.parquet`). Ledger writers: `run_walk_forward` (holdout),
  `run_single_backtest(include_holdout=True)` (`kind="backtest"` + `evaluated_start/end`,
  `in_sample`, `metrics_window`), the public `record_holdout_look(output_dir, entry, *,
  cfg=None, md=None)`, and LLM-desk replays that reach `walkforward.holdout_start`
  (`kind="desk_replay"`, config hash suffixed `:desk` / `:desk-fake`, so the desk is a
  separate trial). Readers: the walk-forward DSR and the sweep's holdout guard.
- Walk-forward attribution: with `k` the fold of bar `t` and `f` clipped to [−1, 1],
  `c_i[t] = FDM_k · w_{k,i} · f_i[t]`, `combined[t] = clip(Σ_i c_i[t], −1, 1)`;
  `forecast_contributions(forecasts, weights, *, fdm=None, residual=False)` (`residual` adds
  `CLIP_COLUMN = "clip_adjustment"` so rows sum to `combined`),
  `WalkForwardReport.strategy_keys / strategy_forecasts() / fold_weights() / fold_fdm() /
  bar_weights(include_fdm=True) / contributions(residual=False)`,
  `HoldoutReport.strategy_forecasts() / contributions()`.
- `sweep.py`: `run_sweep(md, cfg, strategy_name, grid, *, metric="sharpe", maximize=None,
  walkforward=True, start=None, end=None, n_jobs=0, executor="auto", out_dir=None, write=None,
  ledger=None, holdout_ledgers=None, respect_holdout_ledgers=True, prior_trials=0,
  knife_edge_ratio=0.5, n_boot=None, seed=None) -> SweepResult` (`label="exploratory"`).
  Each grid point is a standalone single-strategy book (weight 1, FDM 1) evaluated by
  `run_walk_forward` (OOS, net of costs); points the strategy rejects are `invalid`, not
  trials. `n_trials = valid points + distinct earlier configs in <out>/../sweep_ledger.jsonl
  over an overlapping window + prior_trials`; the best point gets PSR/DSR/CI (and `dsr_xs`
  with the cross-sectional variance of the grid's Sharpes); `plateau_score` (neighbours =
  valid points at Chebyshev distance 1 in grid-index space; ratio < 0.5 knife-edge, >= 0.75
  robust; plateau centre; `edge_params`); PBO = CSCV over the T x N matrix of daily OOS
  returns. **The holdout is never swept**: the window is cut at `walkforward.holdout_start`,
  at `end`, and before the earliest window in any holdout ledger in `output.dir` and its
  immediate subdirectories; every point config has `holdout_start: null`.
- `montecarlo.py`: `monte_carlo(returns, *, n_paths=10000, horizon="2Y", mean_block=None,
  seed=0, ...) -> MonteCarloResult` and `stress_run(source, *, book="combined",
  phase="research", ...) -> StressReport`. Stationary bootstrap (Politis-Romano: a block
  starts with probability `1/b` at a uniform index, then advances by one with circular wrap;
  `b` by Politis-White with the Patton-Politis-White correction; draw order identical to
  `stats.stationary_bootstrap_indices`; paths drawn in chunks of `CHUNK_PATHS = 4096`, part
  of the seeded stream). Per path: `E_t = E_0 Π(1 + r)`, CAGR `(E_H/E_0)^(252/H) − 1`,
  max drawdown from the running peak, ruin `P(min E_t <= (1 − L) E_0)` from the START,
  time to recover, longest underwater stretch; fan = per-day quantiles. Deterministic
  stress: worst rolling k-day windows (1, 5, 10, 21, 63), gap shocks on the current and peak
  exposure (`risk.var.stress_test`), gold's worst k-day moves on the current lots. Reads a
  run directory only.
- `attribution.py`: `load_run`, `contributions`, `explain_decision(source, at=None, *,
  phase, book, md=None, load_data=True) -> DecisionExplanation` (waterfall + point-in-time
  sizing replay, checked against the saved request), `pnl_attribution(source, *, phase,
  component="net") -> PnLAttribution` (`share_i(t) = c_i(t)/Σ_j c_j(t)`,
  `pnl_i(t+1) = share_i(t) · pnl(t+1)`, `unattributed` when the raw sum is 0: exact by
  construction), `desk_journal_summary(path) -> DeskSummary`, `live_decisions(...)`,
  `explain_live_decision(source, at=None)` (adds the desk step,
  `desk_delta = final − combined`). Read-only.
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
  `[-max_abs_forecast, max_abs_forecast]`. API failures, refusals and budget overruns
  resolve to `on_failure` (`follow_quant` default, or `veto` / `hold`).
- Models default to `claude-opus-5` (Chief effort `high`, specialists `medium`); caps per
  cycle: `max_specialists_per_cycle` (consulted + created), `max_cost_usd_per_cycle`
  (default 3 USD) / `max_tokens_per_cycle`, with a share reserved for the Chief. Ad-hoc agents
  get only whitelisted read-only data tools and cannot create further agents.
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
  retcode handling; server-time → UTC). With `server_tz="auto"` and no fresh tick it falls
  back to the zone persisted in `tz_state_path` (the runner's
  `<state_dir>/mt5_server_tz.json`, last detection on the SAME account server) or else to
  `server_tz_fallback` (auto only), UNCONFIRMED: bars/quotes use the provisional zone and
  every order (place, close, close_all, modify) first probes for a fresh tick, returning
  `MARKET_CLOSED` with retcode `server_tz_unconfirmed` (the OMS defers) until one confirms
  the offset; a different offset is adopted, persisted, and that order rejected
  (`server_tz_corrected`). `latest_bars` re-probes at most every `tz_reprobe_seconds` (60).
  It fails only when neither the file nor the fallback exists.
- `oms.py`: idempotent client ids, reconciliation (target vs actual), retries, rejects.
- `state.py`: atomic (fsync + `os.replace`) JSON state files and JSONL logs; an unreadable
  state file raises `StateCorruptError` instead of silently starting empty.
- Paper data: `paper.data: replay` (default) replays `paper.bars_path` on a simulated clock
  from `paper.start` or after `paper.warmup_bars`; `paper.data: mt5` paper-trades on live
  MT5 bars and quotes.
- `runner.py`: bar-close scheduler; builds `MarketData` from the broker, runs the SAME
  feature pipeline (loaded from artifact), strategies, combiner, optional LLM desk, sizer,
  risk manager, OMS. `dry_run=True` default; refuses real (non-demo) accounts unless
  `allow_live_real=True` in config AND `--i-understand-real-money` CLI flag. The final
  forecast of the last decision is persisted (`runner_state.json` `prev_final_forecast`) and
  passed to the desk as `previous_forecast` (so `on_failure="hold"` survives restarts);
  `backtest.stop_cooldown_bars` is honoured live.
  `LiveConfig` (`LiveConfig.from_mapping` rejects unknown keys; the CLI builds it only
  that way) adds `heartbeat_seconds` (None = `retry_poll_seconds`) and `desk_config` (the FULL
  `DeskConfig` mapping, merged over the legacy flat `desk.config`; a key set to different
  values in both is an error); `dry_run`, `allow_live_real`, `flatten_on_shutdown` and
  `desk.enabled` must be real booleans, and `desk.on_error` is strictly one of
  `DESK_ON_ERROR = ("follow_quant", "flat", "hold")`. With the desk enabled,
  `check_desk_prerequisites` raises `RunnerConfigError` (exit 2, `CONFIG ERROR`) when
  `anthropic` cannot be imported or `ANTHROPIC_API_KEY` is empty, before the state-dir lock
  or any broker call. `heartbeat.json` is rewritten after every loop iteration AND every
  `heartbeat_every_seconds` while sleeping (`status` running/degraded, `next_wake`,
  `heartbeat_every_s`; `broker_time` may be null). Every decision record in
  `decisions.jsonl` carries `attribution` = {method, fdm, strategies{name: forecast,
  weight, contribution}, raw_combined, combined, clipped, unallocated = max(0, 1 − Σw),
  exact, final_forecast, desk_delta = final − combined} (+ `venue_tz_confirmed` on MT5), and
  the runner logs exactly one INFO line per decision.
- `monitor.py`: feature drift (PSI) vs training stats, live vs backtest slippage, heartbeat
  (`heartbeat_max_age(heartbeat_every_s, *, periods=3, grace_seconds=120)` =
  `periods · heartbeat_every_s + grace_seconds`; `check_heartbeat(path, *,
  max_age_seconds=None, ...)` derives it from the file; the grace must cover the longest
  decision cycle), optional webhook alerts.
- Operator CLI (read and clear state; never trades): `aurum live status -c C [--json]`
  (read-only: lock probe, heartbeat, risk, OMS, paper book, last decision, last reset, last
  5 alerts) and `aurum live reset-halt -c C [--confirm RESET] [--reason TEXT]` (refuses with
  exit 2 while a runner holds `runner.lock`, exit 4 without the literal `RESET`; takes the
  runner lock, re-reads the state, `reset_halt(confirm="RESET")`, appends a `halt_reset`
  record to `decisions.jsonl` and a critical alert to `alerts.jsonl`, both fsynced).

## 12. Configuration & CLI (`aurum.core.config`, `aurum.cli`)
- One YAML (`extends` deep merge, lists replaced; `--set` dotted overrides) → typed
  `AurumConfig`. Structural problems (unknown keys with did-you-mean, types, secrets in
  YAML) raise first, all together. Semantic validation, `AurumConfig.check(*, strict=False)
  -> ValidationReport(problems, warnings, checked, skipped; .ok, .raise_if_problems())`,
  builds every downstream object and reports every problem at once: instrument ranges
  (`triple_swap_weekday` 0..4, `rollover_hour_utc` 0..23, `margin_rate` in (0, 1], ...),
  costs, the feature pipeline (groups, overrides, override params by signature), EVERY
  `strategies[]` entry (unknown param or constructor error = problem; missing file =
  warning), combiner, sizer, both risk limit sets, policy, `DeskConfig`, `live.use_desk`
  requires `agents.enabled`, the live runner's `LiveConfig` dry-built (unknown
  `live.options` keys = problem) and `rl.params` via `rl_config_from_aurum` (skipped when
  the RL stack is missing). `strict=True` turns an unknown strategy name and a value the
  runner refuses from warnings into problems. `validate(strict)`, `load_config(...,
  strict=False)`, `config_from_dict(..., strict=False)`.
- Secrets only from the environment (`SECRET_ENV`); `config_hash()` = SHA-256 of the
  canonical JSON of every result-relevant field.
- Every CLI command that takes `--config` validates with `strict=True` BEFORE loading data
  (`live status` / `live reset-halt`: keys and types only). Exit codes: 0 ok, 1 runtime
  error, 2 usage/configuration error (also a refused `live reset-halt`), 3 optional
  component missing, 4 confirmation required (`desk replay` without `--yes`,
  `live reset-halt` without `--confirm RESET`). `aurum --version`.
- Paid desk commands (`desk run`, `desk replay` without `--fake`) require
  `agents.enabled: true` and `ANTHROPIC_API_KEY`. `live artifact` / `train-final` refuse an
  existing `--out` before loading data.
- Extension hook: `aurum.cli_ext.register_all(subparsers)` is called at parser build with a
  guarded proxy (core names cannot be replaced; on any exception its parsers are rolled
  back and a WARNING logged). It adds `sweep`, `stress`, `explain [--run|--live]` and, via
  `aurum.viz.cli.register`, `dashboard`. Reference: docs/cli.md, docs/configuration.md.
