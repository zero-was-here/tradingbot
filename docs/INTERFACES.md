# Wave 1 module interfaces (as built)
Authoritative source is the code; this is a map. It was written module by module while Aurum v2 was built; entries that later changed were corrected for v2.0.0 (2026-09-28). The topic pages linked from [index.md](index.md) describe current behaviour in more depth.

## data

### Public API
```
aurum.data.loaders:
  load_mt5_csv(path, timeframe, *, server_tz="Etc/GMT-2", point_size=0.01, default_spread=None) -> bars   # tab/comma/semicolon, <SPREAD> points*point_size, UTF-16, headerless MT4, D1 w/o <TIME>
  load_csv(path, timeframe, *, time_col="time", tz="UTC", default_spread=None) -> bars
  bars_from_ohlc(df, timeframe, default_spread=None, *, tz="UTC") -> bars
  to_utc_index(times, tz="UTC") -> DatetimeIndex   # "UTC", IANA, "Etc/GMT-2", "UTC+2", "+03:00", "NY+7" (US-DST-aware NY-close server)
  parse_timestamps(values, tz="UTC") -> DatetimeIndex
  quality_report(bars, *, gap_bars=3, outlier_sigma=12.0, vol_window=500, top=10) -> dict   # gaps (weekend/daily_break/intraweek), zero/quantile/by-year spreads, causal robust return outliers, flat bars
aurum.data.dukascopy:
  download_dukascopy(symbol="XAUUSD", start, end, timeframe="M1", *, cache_dir="cache/dukascopy", price_scale=None, max_workers=8, session=None, source_resolution="M1"|"H1", skip_saturday=True, daily_anchor_hour_utc=0, retries=6, today=None, offline=False) -> bars (attrs: source, symbol, price_scale, source_resolution)
  decode_bi5_candles(raw: bytes, day: date, price_scale: float|None, *, symbol="XAUUSD", period_seconds=86400) -> DataFrame[open,high,low,close,volume]
  combine_bid_ask(bid, ask) -> DataFrame[open,high,low,close,volume,spread]   # mid OHLC, spread=max(0, mean(open,close spreads)), filler candles dropped
  detect_price_scale(raw_prices, symbol="XAUUSD") -> float
  fetch_file(symbol, day, side, *, kind="min"|"hour", cache_dir, session, retries=6, backoff=1.0, timeout=90, today=None, limiter=None, throttle_retries=40, offline=False, deadline=None) -> bytes ; fetch_day_file(...) alias
  datafeed_url(symbol, day, side, kind="min") / cache_path(cache_dir, symbol, day, side, kind="min")
  AdaptiveLimiter(max_concurrency=4, cooldown=5.0, max_cooldown=300.0).acquire(deadline=None)->bool / .release(throttled=bool)
  prefetch_cache(symbol="XAUUSD", start, end, *, cache_dir, source_resolution="M1", max_workers=8, time_budget_s=None, newest_first=True, skip_saturday=True, session=None, today=None, retries=6) -> PrefetchReport(planned, already_cached, fetched, failed, not_attempted, elapsed_s, contiguous_start, stopped_early)
  contiguous_cached_start(symbol, cache_dir, end: date, *, skip_saturday=True, earliest=date(2003,5,5)) -> date|None
  build_dataset(out_dir, *, symbol="XAUUSD", start="2012-01-01", end="2026-09-25", cache_dir, m15_start=None, max_workers=4, offline=False, ny_close_anchor_hour=22, h1_source="auto"|"M1"|"H1") -> manifest dict (+ writes {sym}_M15/H1/H4/D1/D1_nyclose.parquet + manifest.json)
aurum.data.macro:
  DEFAULT_YAHOO, DEFAULT_FRED (SPEC §3.3); YAHOO_CASH_LAG=21:30, YAHOO_FUTURES_LAG=22:30 (=F/.NYB/=X), FRED_LAG=1d21:30
  fetch_yahoo_daily(tickers=None, start="2011-01-01", end=None, cache_dir="cache/macro", *, availability_lag=None|Timedelta|{name: Timedelta}, refresh=False, skip_errors=True) -> dict[str, DataFrame]
  fetch_fred(series=None, start="2011-01-01", end=None, cache_dir="cache/macro", *, availability_lag=None, refresh=False, skip_errors=True) -> dict[str, DataFrame]
  to_macro_frame(values, *, availability_lag, value_col="close", source="") ; parse_fred_csv(text, series_id=None) ; default_yahoo_lag(ticker) ; validate_macro_frame(df, name="")
  save_macro_dir(frames, path) -> Path ; load_macro_dir(path) -> dict[str, DataFrame]   # index=date (UTC midnight), value, [open/high/low/volume], available_at
aurum.data.calendar:
  EVENT_COLUMNS, FOMC_STATEMENTS (128 scheduled statements 2012-2027), FOMC_COVERAGE=(2012, 2027)
  generate_rule_based_calendar(start, end, *, include=("NFP","FOMC"), nfp_rule="first_friday"|"bls") -> events
  nfp_release_date(year, month, rule="first_friday") -> date ; fomc_statements(start=None, end=None) -> events
  load_calendar_csv(path, *, tz="UTC", source=None) -> events ; merge_calendars(*frames) ; validate_events(ev) ; empty_events() ; ny_to_utc(day, "HH:MM")
aurum.data.store:
  save_bars(bars, path, *, metadata=None) -> Path ; load_bars(path, *, verify_hash=True) -> bars   # tz, available_at, attrs (parquet schema metadata), stored frame_hash verified
  save_frame(df, path, *, metadata=None) -> Path ; load_frame(path) -> DataFrame ; frame_hash(df) -> str (sha256, resolution/-0.0/NaN-canonical)
aurum.data.pit.asof_join (signature unchanged): now NaT decision rows -> NaN, empty right -> all-NaN, missing columns -> KeyError, empty tz-less input accepted.
aurum.data.resample.resample_bars(bars, to, *, daily_anchor_hour_utc=0, complete_until=None)  # new optional kwarg; fixed-length rule so the D1/H4 anchor is honoured on pandas 3
aurum.data.synthetic (signatures unchanged): make_synthetic_events -> FOMC-like on Wednesdays (day 15-21) 14:00 NY, NFP first Friday 08:30 NY, CPI 08:30 NY, all DST-aware; make_synthetic_macro is now point-in-time (co-moves only with gold returns known at available_at).
```

### Integration notes
The research dataset is complete from 2012 to 2026, and every file was built from Dukascopy M1 data as the SPEC asks (manifest build_path="M1"). Load bars with aurum.data.store.load_bars(path) (checks the stored hash), macro with aurum.data.macro.load_macro_dir(path), and events with aurum.data.calendar.generate_rule_based_calendar(start, end). Paths below are relative to the repo's data_store/.

| file | rows | range (UTC, bar open time) |
|---|---|---|
| xauusd_M15.parquet | 349,660 | 2012-01-01 22:45 to 2026-09-25 20:45 |
| xauusd_H1.parquet | 87,829 | 2012-01-01 22:00 to 2026-09-25 20:00 |
| xauusd_H4.parquet | 23,581 | 2012-01-01 20:00 to 2026-09-25 20:00 |
| xauusd_D1.parquet (UTC midnight days, includes short Sunday bars) | 4,585 | 2012-01-01 to 2026-09-25 |
| xauusd_D1_nyclose.parquet (days start 22:00 UTC) | 3,841 | 2012-01-01 22:00 to 2026-09-24 22:00 |

manifest.json holds row counts, sha256 hashes and a full quality report per file.

**Median M15 spread by year (USD / bps):**

| year | USD | bps |
|---|---|---|
| 2012 | 0.356 | 2.15 |
| 2013 | 0.344 | 2.38 |
| 2014 | 0.287 | 2.29 |
| 2015 | 0.303 | 2.61 |
| 2016 | 0.301 | 2.44 |
| 2017 | 0.238 | 1.89 |
| 2018 | 0.231 | 1.82 |
| 2019 | 0.290 | 2.12 |
| 2020 | 0.398 | 2.16 |
| 2021 | 0.351 | 1.96 |
| 2022 | 0.365 | 2.00 |
| 2023 | 0.328 | 1.69 |
| 2024 | 0.380 | 1.58 |
| 2025 | 0.581 | 1.73 |
| 2026 | 0.690 | 1.50 |

The dollar spread widened in 2025-26 as gold traded at $4,000-5,600, but in basis points it narrowed. M15 spread quantiles: 1% 0.20, 50% 0.33, 99% 0.99, 99.9% 2.39. The widest bar was 9.33 on 2025-11-28 08:45 (US holiday session).

**Gaps:**
- M15: 749 weekend, 2,745 daily-break (the 21-22 or 22-23 UTC maintenance hour) and 132 intraweek gaps, all holiday or early closes (Christmas, Easter and US holidays at 18:30-19:30 UTC).
- D1_nyclose: 39 missing NY-session weekdays, every one an exchange holiday (Good Friday, 25 Dec, 1 Jan and observed days).
- Zero spreads occur only in 2012.

**Outliers:** 337 M15 and 92 H1 robust-z flags. The largest match known events:
- 2013-04-15 crash (-9.6% on the day)
- 2021-08-09 flash crash
- 2016-06-24 Brexit vote
- 2020-03 COVID
- 2020-08-11 selloff
- 2025-10-21 selloff (-5.5%)
- 2026-01-30 metals crash (gold about -9.4%; silver -37% in the Yahoo data)

**Decoder check:** the real 2024-01-15 BID and ASK files decode to a mid of 2045.9-2058.7 with a median spread of $0.32 and 1,228 active minutes (MLK holiday). The price scale was detected as 1000.

**FOMC dates:** all 128 dates were re-checked on 2026-09-26 against federalreserve.gov (fomccalendars.htm for 2021-2027, fomchistorical2012-2020.htm) and all match. Unscheduled actions are excluded (2020-03-03, 2020-03-15, 2019-10-04, and the 2025-08-22 notation vote).

**Macro:** all 10 series are saved in data_store/macro/, starting 2011-01-03 (fedfunds 2011-01-01) and ending 2026-09-25 (real10y and fedfunds end 2026-09-24).

| series | rows |
|---|---|
| dxy | 3,957 |
| us10y | 3,955 |
| vix | 3,958 |
| spx | 3,956 |
| silver | 3,955 |
| gold_fut | 3,956 |
| oil | 3,956 |
| real10y | 3,934 |
| breakeven10y | 3,935 |
| fedfunds | 5,746 |

Daily spot and futures log returns correlate at about 0.89.

**Rebuild or extend:** run prefetch_cache('XAUUSD', start, end, cache_dir='cache/dukascopy', time_budget_s=...) first. It fetches newest days first, can be resumed, and never raises on individual failures. Then run build_dataset('data_store', start=..., end=..., cache_dir='cache/dukascopy', offline=True).

**PIT notes for feature and strategy authors:**
- Every bar's available_at is its open time plus the timeframe.
- H4/D1 buckets are only emitted when complete. complete_until never makes a bar available earlier than the end of its bucket.
- Macro frames must be joined with asof_join(bars['available_at'], frame).
- Event rows use scheduled times, which are published in advance. Outcome columns may only be used from `time` onward.

### Known limitations
- 2012-2013 data is lower quality. In 2012 there are 836 zero-spread M15 bars (bid equals ask, mostly 21:00-23:00 UTC and on holidays such as 25 Dec, MLK and Thanksgiving). Flat O=H=L=C bars number 347 in M15, almost all 2012-2015. Minute and hour files disagree on open/close in 529 H1 bars in 2012 and 32 in 2013, but only about 6 across 2014-2026. Starting research in 2013 or 2014, or at least applying the cost model's minimum spread, is recommended.
- Mid high/low are approximated as (bid_high + ask_high)/2 per minute. Bid and ask extremes can occur on different ticks, so the true mid range can be slightly overstated. Volume is Dukascopy's bid-side volume in its own units, so use it only for relative activity.
- Download cost: the full 2012-2026 M1 cache took about 55 minutes over three time-budgeted passes (plus partial runs earlier), with 429/503 throttling. Re-downloading from scratch is slow; the 131 MB raw cache is kept under cache/dukascopy and build_dataset(..., offline=True) rebuilds everything from it in about 11 seconds (peak memory about 1.8 GB).
- The two network tests hit live endpoints and take about 25 seconds because of throttling. Deselect them with -m "not network".
- FRED values are latest-vintage: publication lag is modelled, revisions are not (would need ALFRED). Yahoo data comes through the unofficial yfinance API.
- Macro series that break log returns: oil (CL=F) is negative on 2020-04-20 (-37.63), and real10y has 970 non-positive observations, which are genuine negative real yields. Features must use differences or basis-point changes for these, not log returns.
- The NFP calendar is rule-based (approximate=True). Shutdown delays and ad-hoc BLS moves are not captured, and there is no rule-based CPI; import CPI with load_calendar_csv.

## features

### Public API
```
Registry groups (exact SPEC names; columns prefixed '<group>_'): returns, trend, momentum, meanrev, range (technical.py); volatility (volatility.py); microstructure, session (microstructure.py); mtf (multi_timeframe.py); macro (macro.py); calendar (calendar.py, requires_events=True); regime (regime.py). 12 groups; with default params that is 174 columns on H1 bars with the five synthetic macro series and 194 with the ten series `aurum data download` fetches (see [features.md](features.md)).

aurum.features.volatility (standalone, imports no other feature module; verified in a subprocess test):
  true_range(bars) -> Series; atr(bars, n=14) -> Series (Wilder); wilder_smooth(x, n)
  close_to_close_vol(bars|close, n=20, bars_per_year=None); parkinson_vol(bars, n=20, bars_per_year=None); garman_klass_vol(...); rogers_satchell_vol(...); yang_zhang_vol(bars, n=20, bars_per_year=None)  -> annualised fraction, rolling min_periods=n
  ewma_vol(close, halflife=48, *, bars_per_year=None, min_periods=24)
  bar_minutes(bars) -> float; default_bars_per_year(bars) -> float (252*23*60/min intraday, 252 for D1); safe_div(num, den, *, fill=nan); log_pos(x)
  volatility_features(md, *, windows=(24,120), long_window=480, ewma_halflife=48, volofvol_window=120, atr_n=14, bars_per_year=None)
aurum.features.technical: ema, ewm_bar_vol, rolling_zscore, rsi(close,n)->[0,100], stochastic(bars,n,d)->(K,D), adx(bars,n)->(adx,+di,-di), rolling_linreg(y,n)->(slope,r2), tsmom_response(z), streak(logret), log_close; returns_features / trend_features / momentum_features / meanrev_features / range_features(md, **params)
aurum.features.microstructure: microstructure_features(md, *, z_window=120, atr_n=14, autocorr_window=120); session_features(md, *, asia_hours=(8,17), london_hours=(8,17), ny_hours=(8,17), rollover_hours=(16,18)); local_clock(times, tz); LONDON/NEW_YORK/TOKYO ZoneInfo
aurum.features.multi_timeframe: resample_anchored(bars, to, anchor_hour_utc=0) (pandas-3-safe anchored resample_bars); htf_compact_features(htf, ...); mtf_features(md, *, htfs=("H4","D1"), daily_anchor_hour_utc=22, prev_day=True, ...)
aurum.features.macro: macro_features(md, *, series=None, change_days=(1,5,20), z_window=250, z_min_periods=60, corr_window=60, corr_min_periods=40, beta_series=("dxy","real10y"), yield_series=DEFAULT_YIELD_SERIES, stale_days=10, daily_anchor_hour_utc=22); session_date(index, anchor_hour_utc)
aurum.features.calendar: calendar_features(md, *, min_importance=3, cap_hours=72, currencies=None, near_minutes=30, wide_hours=2, count_horizon_hours=24); classify_event(names); EVENT_PATTERNS
aurum.features.regime: regime_features(md, *, vol_window=24, rank_years=1.0, rank_window=None, expanding=False, ..., vol_halflife=None, ...); efficiency_ratio(close, n); variance_ratio(logp, q, window); rank_window_bars(rank_years, minutes); regime_lookback(params, bar_minutes); DEFAULT_RANK_YEARS
  # wave 3: default columns regime_vol_pctrank (bounded rolling rank, NaN until the rank_years window is full; 5,796 H1 / 23,184 M15 / 1,449 H4 / 252 D1 bars), regime_high_vol/low_vol from it;
  # regime_vol_pctrank_exp only with expanding=True; regime_vol_pctrank_2000 is gone (pipelines/RL policies fitted before must be retrained: FeatureSchemaError)
aurum.features.pipeline:
  class FeaturePipeline(groups=None, overrides=None, scaler="robust"|"standard"|"none", clip=5.0, *, warmup=None, bar_minutes=None)
    compute(md) -> DataFrame; fit(raw_train) -> self; transform(raw, *, strict=True) -> DataFrame; fit_transform(raw)
    max_lookback (property); columns (property); dropped_columns; stats (DataFrame loc/scale/kind); is_fitted
    save(path) -> Path; load(path) (classmethod); to_dict()/from_dict()
    parity_report(a, b, atol=1e-8) -> DataFrame[in_a,in_b,n_rows,max_abs_diff,n_mismatch,n_nan_mismatch,ok]
    snapshot(frame, at=None, columns=None) -> dict[str, float|None]  (JSON-safe, for the LLM desk)
  FeatureSchemaError(ValueError); IQR_TO_SIGMA=0.7413; FORMAT_VERSION=1
```

### Integration notes
Strategies and sizing: import atr, true_range, yang_zhang_vol, parkinson_vol, garman_klass_vol, rogers_satchell_vol from aurum.features.volatility; it does not import technical, pipeline or macro. Pass bars_per_year explicitly if it must match ewma_volatility/infer_bars_per_year. ML strategies and walk-forward: `pipe = FeaturePipeline(groups=..., overrides=...); raw = pipe.compute(md)` on FULL history (all groups are causal), then `pipe.fit(raw.iloc[train_idx])` and `X = pipe.transform(raw.iloc[test_idx])`. Warm-up rows keep NaN; call dropna() or skip pipe.max_lookback rows before training. Live runner: persist with pipe.save('artifacts/features.json') and restore with FeaturePipeline.load(...). transform raises FeatureSchemaError if live compute is missing a fitted column (e.g. macro or events not supplied), or has new columns when strict=True. Use pipe.parity_report(batch_raw, live_raw) to check batch vs incremental parity. Daily-based groups need md.macro (dict of frames with 'value' + 'available_at') and md.events (time, name, importance, currency). With no macro the macro group returns an empty frame; with events=None the pipeline skips calendar. Session features describe available_at[t] (the next bar's scheduled open). For H4/D1 bars with a broker-day anchor, resample_bars(bars, 'D1', daily_anchor_hour_utc=22) now honours the anchor on pandas 3 (it uses fixed-length Timedelta buckets); resample_anchored remains as an equivalent wrapper. Other test authors can reuse the leak checker pattern (perturbed_market / truncated_market / compare_prefix) in tests/test_features_leakage.py for strategy leakage tests.

### Known limitations
- LLM desk (SPEC §10, aurum/agents): the features module's only contribution is FeaturePipeline.snapshot(), a JSON-safe feature dict the desk's DeskDataProvider can expose to the CIO/specialist agents.
- (Updated.) max_lookback is no longer static: each built-in group attaches a lookback_fn(params, bar_minutes) evaluated on its effective parameters and the bar size (on H1: macro 2136, mtf 576, regime 5820 bars; see [features.md](features.md#warm-up)). Groups without a lookback_fn fall back to the registered lookback, raised to the largest integer override + 1 (a heuristic). Pass warmup= to set it exactly. `aurum features list` still prints the H1 registry values.
- The leakage test does not perturb the event calendar, because scheduled times are public in advance (SPEC §3.5). Unscheduled events that appear in historical calendars (e.g. emergency FOMC) cannot be told apart and would leak slightly; calendar sources should flag them.
- The perturbation test keeps the same timestamp grid; a length-dependent full-sample statistic is covered by the truncation variant. Both use one synthetic seed/model per cutoff.
- The random-walk test is a single-seed statistical test with a 1% family-wise false-positive rate by construction. It is deterministic and passes, but a future feature change could need a seed review if it trips at p just below threshold.
- Macro: a yield series whose name is not in DEFAULT_YIELD_SERIES and that has no attrs['kind'] is treated as a price (log changes), so non-positive prints give NaN changes. Staleness tolerance (10 days) turns dead feeds into NaN, which becomes 0 after warm-up.
- Wall-clock test time on this shared box was about 5 minutes (load average up to about 520) for about 16s of CPU. On an idle machine the suite should take well under 60s, but that was not observable here.

## execution_backtest

### Public API
```
aurum.execution.costs:
  FillPrice(NamedTuple): price, spread_cost, slippage_cost
  @dataclass(frozen) CostModel(spread_multiplier=1.0, min_spread=0.10, slippage_fixed=0.02, slippage_range_frac=0.02, impact_coef=0.0, commission_per_lot=None, financing=FinancingModel())   # financing also accepts kwargs / a mode name
    .zero() -> CostModel (classmethod; financing "none" too); .to_dict()
    .effective_spread(spread) -> float            # max(spread*mult, min_spread)
    .slippage(bar_range, lots) -> float           # USD/oz: fixed + frac*range + impact*sqrt(|lots|)
    .fill_price(side, mid, spread, bar_range, lots, *, instrument=XAUUSD, limit=False) -> FillPrice
    .commission(lots, *, instrument=XAUUSD) -> float   # per lot per side; None -> instrument
    .swap(lots, nights, *, instrument=XAUUSD, price=None, rate_nights=None) -> float  # signed, nights already triple-weighted; rate mode needs price
    .swap_between(lots, start, end, *, instrument=XAUUSD, price=None, rates=None) -> float
  @dataclass(frozen) FinancingModel(mode="rate"|"fixed"|"none", markup_long=0.025, markup_short=0.025, lease_rate=0.0, rate_series="fedfunds", rate_unit="percent"|"fraction"|"bps", fallback_rate=0.03, day_count=360.0)
    .fixed() / .none() / .coerce(model|mapping|mode|None); .to_dict(); .uses_rates; .curve(rates) -> RateCurve|None
    .benchmark_ns(t_ns, curve); .rate_nights_ns(t0_ns, t1_ns, instrument, curve)   # sum_R w(R)*r(R) over rollovers in (t0, t1], r as of R
    .amount(lots, nights, *, price=None, rate_nights=None, instrument) ; .nightly(lots, price, benchmark) ; .annual_rate(lots, benchmark)
    # rate: -lots*contract_size*P*(r - lease + markup_long [long] / - markup_short [short])/day_count per night
  RateCurve(available_ns, values, name).asof_ns(t_ns); RateCurve.from_source(md.macro | frame with available_at | Series indexed by availability | RateCurve, name="fedfunds", unit="percent")
  FINANCING_MODES
    .round_trip_cost(lots, spread, bar_range=0.0, *, instrument=XAUUSD) -> float
  rollover_nights(start, end, instrument=XAUUSD) -> float          # nights charged over (start, end]
  rollover_nights_ns(t0_ns, t1_ns, instrument=XAUUSD) -> np.ndarray # vectorised O(1)/interval

aurum.execution.simulator:
  ExecutionSimulator(bars, instrument=XAUUSD, costs=None, initial_equity=100_000.0, *, validate=True, rates=None)   # result().meta["financing"] = rate provenance
  intrabar_exit(pos, o, h, lo, sl, tp) -> (exit_mid, reason, is_limit) | None   # shared with PaperBroker and the live runner (ExecutionSimulator._intrabar_exit / aurum.live.paper._protective_exit are aliases)
    .reset(start=0, equity=None)
    .step(target_lots, *, stop_price=None, take_profit=None, reason="signal", stop_distance=None, take_profit_distance=None) -> StepResult   # *_distance: levels anchored at the entry fill open[t+1]
    .result(*, compute_metrics=True) -> BacktestResult
    .trades_frame(*, include_open=True) -> DataFrame; .fills_frame() -> DataFrame; .snapshot() -> dict
    attrs/properties: index, equity, position, done, peak_equity, bankrupt, n_bars, start, time, decision_time, price, spread, drawdown, margin_used, free_margin
  @dataclass StepResult(index, time, equity, pnl, ret, price_pnl, costs: dict[spread,slippage,commission,swap], fills: list[aurum.core.types.Fill], position, position_open, done, exit_reason, stop_price, take_profit)
  TRADE_COLUMNS, FILL_COLUMNS

aurum.backtest.engine:
  run_backtest(md, forecast, *, sizer, risk=None, instrument=XAUUSD, costs=None, initial_equity=100_000.0, vol=None, stop_atr_mult=None, start=None, end=None, take_profit_atr_mult=None, atr_period=14, stop_cooldown_bars=0, bars_per_year=None, event_horizon_hours=24.0, compute_metrics=True, forecast_hook=None, hook_every=1, financing=None, rates=None) -> BacktestResult
  run_target_lots(md, target_lots, *, risk=None, instrument=XAUUSD, costs=None, initial_equity=100_000.0, vol=None, stop_atr_mult=None, start=None, end=None, take_profit_atr_mult=None, atr_period=14, stop_cooldown_bars=0, bars_per_year=None, event_horizon_hours=24.0, compute_metrics=True, financing=None, rates=None) -> BacktestResult   # NaN target = hold
  buy_and_hold_benchmark(md, lots=None, notional=None, *, instrument=XAUUSD, costs=None, initial_equity=100_000.0, start=None, end=None, frictionless=False, compute_metrics=True, financing=None, rates=None) -> BacktestResult
  # financing= overrides costs.financing; rates= defaults to md.macro (rates read as of each rollover, point-in-time)
  average_true_range(bars, n=14) -> pd.Series   # Wilder, causal, never NaN
  RISK_EVENT_COLUMNS = [time, bar_time, bar, current, requested, approved, halted, reasons]; OUTCOME_COLUMNS

aurum.backtest.metrics:
  compute_metrics(result_or_returns, *, bars_per_year=None, trades=None, positions=None, costs=None, equity=None, fills=None, bar_duration=None) -> dict   # all SPEC keys + n_bars, n_days, years, bars_per_year, final_equity
  daily_returns(equity, *, initial=None, fold_weekends=True) -> Series
  daily_equity(equity, *, fold_weekends=True); trading_dates(index, *, fold_weekends=True)
  drawdown_series(equity) -> Series (<=0); max_drawdown(equity); max_drawdown_duration_days(equity)
  TRADING_DAYS_PER_YEAR = 252

aurum.backtest.result.BacktestResult (additions): pnl: DataFrame|None [price, costs, swap, net]; position_close: Series|None; reconcile() -> dict(equity_change, price_pnl, costs, swap, residual, trade_pnl)
```

### Integration notes
Checked against the real modules (not in my tests, since those are being written concurrently): VolTargetSizer() plus StandardRiskManager(RiskLimits(max_spread=0.6)) over 20k H1 bars with synthetic events ran in 0.96s. The PnL residual was 6e-10, and kill-switch halts and 'no new risk' interventions were recorded in risk_events.

How to plug in:
- **Research and walk-forward:** call run_backtest(md, combined_forecast, sizer=VolTargetSizer(...), risk=StandardRiskManager(...), stop_atr_mult=...). It uses ewma_volatility by default, computed over the full history so a start/end window begins warmed up.
- **Report and research:** result.pnl has per-bar price/costs/swap/net; result.position_close gives positions after stops; result.reconcile() checks the PnL identity; buy_and_hold_benchmark(md) gives the comparison curve (it pays CFD swap unless frictionless=True).
- **RL env:** uses ExecutionSimulator directly: reset(start) -> step(target, stop_price=, take_profit=) -> StepResult with pnl, ret, costs and fills -> result(). The constructor is O(n) and reset allocates O(n) lists, so reuse one simulator per dataset. (Updated: the live PaperBroker does not instantiate ExecutionSimulator; it keeps broker-style books with the same CostModel arithmetic, FinancingModel and intrabar_exit, and tests/test_live_paper.py holds it to the simulator's equity path.)
- **LLM desk replay (§10):** turn the DecisionPolicy output into a forecast or scale and pass it through run_backtest, so it goes through the same sizer and risk manager. Pre-sized lot paths go through run_target_lots(md, lots, risk=...). In both cases risk is applied and clamped to reduce-only, so the LLM cannot bypass it.

What risk-manager authors can rely on:
- **Call order:** at every close, on_bar(available_at[t], equity) is called first, then evaluate(ctx). on_bar is called once more at the final bar.
- **Context:** ctx.time = available_at[t]; ctx.price = close[t]; ctx.vol_ann is the same vol the sizer saw; ctx.extra = {drawdown, bar_time, forecast}.
- **Events:** upcoming_events covers [now, now+24h] with outcome columns removed (actual, surprise, outcome, result, deviation). recent_events covers [now-24h, now). Both are empty frames when nothing falls in the window, and None when md has no events.

Other conventions:
- **Sizer drawdown:** the drawdown passed to the sizer is a positive fraction measured from an equity peak the engine tracks.
- **Metrics:** headline sharpe, ann_vol and sortino use daily returns (UTC dates, weekends folded into Monday) with 252 annualisation; sharpe_bar uses bars_per_year. max_drawdown is negative.

Tests: pytest tests/test_execution_*.py tests/test_backtest_*.py

### Known limitations
- The path inside a bar is unknown. If both SL and TP are touched in one bar, the SL is assumed to fill first. In a bar where a stop or TP fired, swap is charged on the position at the bar's close, which is flat. This is exact for M1..H1 bars that end on the rollover hour, and approximate for H4/D1 bars that contain it.
- (Updated.) Engine stops are passed as stop_distance = k*ATR[t] and anchored at the entry fill (open[t+1]), so a gap at the open no longer enters and immediately stops the position. Stops are fixed for the life of the position (no trailing).
- Slippage uses the high-low range of the execution bar (t+1). This is a cost-model input only and never reaches a decision before the step. Fill prices are not rounded to the tick grid; the error is at most $0.005/oz and keeping it unrounded keeps the PnL identity exact.
- No partial fills, liquidity limits or margin stop-out model. The only guard is bankruptcy: once equity <= 0, the position is forced flat with exit_reason 'risk'. Margin utilisation limits belong to the risk manager.
- The rollover hour is a fixed UTC hour from the Instrument. It does not move with New York DST, so it can be off by an hour for about half the year.
- (Updated.) The default vol is annualised with the nominal timeframe constant (or bars_per_year= when given), so truncated runs size identically. Only the per-bar metrics (sharpe_bar, meta['bars_per_year']) use the density inferred from the whole sample's timestamps.
- The simulator works over a fixed bars frame; there is no append API. The live PaperBroker therefore keeps its own books (see the integration note above) instead of driving the simulator.
- LLM desk (SPEC §10, aurum.agents): this module gives the desk a route through the same simulator, sizer and risk manager, so it cannot bypass risk.

## research (aurum/research: splits.py, stats.py, report.py)

### Public API
```
aurum.research.splits:
  Split = tuple[np.ndarray, np.ndarray]  (sorted int64 train_idx, test_idx)
  walk_forward_splits(n, train, test, step=None, anchored=False, purge=0, embargo=0, *, min_test=1) -> list[Split]
  purged_kfold(n, k=5, purge=0, embargo=0, *, label_end=None) -> list[Split]
  cpcv_splits(n, n_groups=6, k_test=2, purge=0, embargo=0, *, label_end=None) -> CPCVSplits
  class CPCVSplits: splits, groups, test_groups, paths (n_paths x N; paths[p,g] = split id supplying group g on path p), n, meta; .n_splits, .n_paths, path_segments(p), assemble_paths(predictions) -> (n_paths, n) array; unpackable: `splits, paths = cpcv_splits(...)`
  purge_train(train_idx, test_idx, *, n, purge=0, embargo=0, label_end=None) -> np.ndarray
  contiguous_blocks(idx) -> list[(start, end)]

aurum.research.stats  (Sharpe is PER-PERIOD and kurtosis is PEARSON (normal = 3) unless a `periods` arg is taken):
  sharpe(returns, periods=252.0, *, ddof=1) -> float (annualised; periods=1 gives per-period)
  annualize_sharpe(sr, periods), deannualize_sharpe(sr_ann, periods), return_moments(returns) -> (skew, pearson_kurt)
  sharpe_std(sr, n, skew=0, kurt=3)                       # Mertens/Opdyke SE
  probabilistic_sharpe(sr, n, skew=0, kurt=3, sr_star=0) -> float
  expected_max_sharpe(n_trials, var, mean=0) -> float     # Euler-Mascheroni EV approximation
  deflated_sharpe(sr, n, skew=0, kurt=3, trial_srs=None, *, n_trials=None, trial_var=None) -> float
  min_track_record_length(sr, skew=0, kurt=3, sr_star=0, prob=0.95) -> float (observations)
  sharpe_pvalue(sr, n, skew, kurt, sr_star) ; adjust_pvalues(pvalues, method='holm'|'bonferroni'|'sidak'|'bh'|'bhy')
  haircut_sharpe(sr_ann, n_obs, n_trials, *, periods=252, method='bonferroni'|'holm'|'sidak'|'bhy') -> dict
  pbo_cscv(perf_matrix (T x N returns), n_splits=16, *, metric='sharpe'|'mean'|callable, max_combinations=None, seed=0) -> PBOResult(pbo, logits, selected, is_perf, oos_perf, oos_rank, n_combinations, n_splits, n_strategies, prob_oos_loss, degradation_slope, degradation_intercept; .to_dict())
  optimal_block_length(returns) -> {'stationary','circular','m_hat','bandwidth'}   # Politis-White 2004 + PPW 2009 correction
  stationary_bootstrap_indices(n, n_boot, mean_block, rng) -> (n_boot, n) int array
  stationary_bootstrap(returns, stat_fn, n_boot=2000, mean_block=None, seed=0, *, alpha=0.05, vectorized=False, ci_method='percentile'|'basic', return_samples=False) -> dict(estimate, lower, upper, alpha, std_error, bias, prob_le_zero, mean_block, n_boot, n, ci_method)
  sharpe_ci(returns, periods=252, *, alpha=0.05, method='bootstrap'|'analytic', n_boot=2000, mean_block=None, seed=0) -> dict (annualised)
  sharpe_summary(returns, periods=252, *, n_trials=None, trial_sharpes_ann=None, sr_star_ann=0, alpha=0.05, n_boot=1000, bootstrap=True, seed=0, prob=0.95) -> dict(sharpe, sharpe_se, psr, dsr, sr0, min_trl, min_trl_years, sharpe_haircut, haircut, ci_lower, ci_upper, bootstrap_block, skew, kurtosis, ...)
  EULER_MASCHERONI

aurum.research.report:
  write_tearsheet(result, path, *, benchmark=None, title="", extra=None, n_trials=None, dark_charts=True, n_boot=1000, seed=0) -> Path
  tearsheet_html(result, *, same kwargs) -> str
  daily_returns_from_equity(equity, initial_equity=None) -> pd.Series
  extra keys: 'folds' (DataFrame; a sharpe/oos_sharpe column gets charted), 'weights' (dict/Series/DataFrame), 'n_trials', 'trial_sharpes' (annualised), 'pbo' (float/PBOResult/dict), 'notes' (str/list), 'config' (JSON-able); any other key is rendered generically (DataFrame/Series table, dict k/v, str, matplotlib Figure).
```

### Integration notes
For walkforward.py: build folds with walk_forward_splits(n_bars, train, test, purge=label_horizon, embargo=...). Pass the stitched OOS equity's BacktestResult to write_tearsheet(result, path, extra={"folds": fold_df (with an 'oos_sharpe' column), "weights": combiner.weights_, "n_trials": number_of_configs_tried, "trial_sharpes": [annualised OOS Sharpe per config], "pbo": pbo_cscv(oos_returns_matrix_T_x_N)}). For DSR/PBO use pbo_cscv(perf_matrix) and deflated_sharpe(sr_daily, n_days, skew, pearson_kurt, trial_srs_daily), or simply sharpe_summary(daily_returns, 252, n_trials=..., trial_sharpes_ann=...), which handles units. The tearsheet computes all statistics on daily returns from aurum.backtest.metrics.daily_returns (Sat/Sun bars folded into Monday), so the headline tiles, the confidence block and the metrics table agree (verified: tile Sharpe = confidence Sharpe = 0.508 on the demo). The report never touches pyplot or the global matplotlib backend (it uses the Figure plus FigureCanvasAgg API), so it is safe to import in notebooks and servers.

### Known limitations
- report.py uses aurum.backtest.metrics.compute_metrics and daily_returns when they are importable, and falls back to minimal local versions otherwise. Values in the 'All metrics' table therefore follow that module's conventions: max_drawdown is negative, var/cvar are positive losses, kurtosis is excess (labelled as such).
- Charts are static PNGs, as the SPEC requires. There are no hover tooltips; the tables (monthly returns, costs, trades, folds, weights, metrics) act as the readable data view. Each chart is embedded twice (light and dark surface), so a typical 9,000-bar H1 report is about 0.9-1.0 MB; dark_charts=False halves that.
- On long intraday samples (>1500 bars) the position and forecast panel shows the daily average rather than bar-by-bar values, for legibility. Long series are also stride-subsampled to at most 6000 plotted points.
- Haircut 'holm' equals Bonferroni for the reported (best) strategy, and 'bhy' uses the conservative bound M*c(M)*p, because the other trials' p-values are unknown. With a full p-value vector, use adjust_pvalues.
- A single PBO estimate on pure noise has high variance across samples (0.16-0.94 seen over seeds); the test asserts the mean over 20 seeds. pbo_cscv enumerates all C(S, S/2) combinations by default; beyond S of about 20, pass max_combinations.
- Visual QA was done with headless-Chromium screenshots (desktop light and dark, 390px mobile) because the Browser pane could not open local files. Layout was checked for alignment, overflow and theme switching; charts use a palette that passed the colour-vision and contrast validator in both modes.
- walkforward.py (wave 2) was not written, as instructed.

## portfolio_risk

### Public API
```
aurum.models.volatility:
  ewma_volatility(close, *, halflife_bars=48, bars_per_year=None, min_periods=20, floor=0.03, cap=2.0) -> Series  [UNCHANGED]
  class GarchParams(mu, omega, alpha, beta)  (.persistence, .unconditional_variance, .half_life, .to_dict())
  class Garch11(*, mean="zero"|"constant", bars_per_year=None, floor=None, cap=None, max_iter=500, min_obs=250)
    .fit(returns) -> self   (SLSQP QMLE, omega>0, alpha,beta>=0, alpha+beta<=1-1e-6, 5 starts, standardised)
    .forecast(returns, *, horizon=1, annualise=True) -> Series 'garch_vol_ann'  (row t = vol for t+1..t+h given r[:t+1]; NaN-tolerant)
    .conditional_variance(returns) ; .summary() -> dict ; .aic/.bic ; attrs params_, h0_, loglik_, converged_
  simulate_garch11(n, *, omega, alpha, beta, mu=0, seed=0, burn=1000) -> (returns, h)
  daily_realised_variance(bars, *, anchor_hour_utc=0, complete_only=True) -> DataFrame[rv, n_obs, available_at]
  class HarRV(lags=(1,5,22), *, log=False): .design(rv) .fit(rv) .predict(rv) ; coef_, resid_var_, r2_
  har_rv_forecast(daily_rv, *, lags=(1,5,22), log=False, min_train=250, refit_every=21, window=None, output="vol"|"variance", periods_per_year=252) -> Series (walk-forward, causal)
  blend_vol(*series, weights=None, name="vol_blend") -> Series (variance-space, renormalises over missing)
aurum.models.regime:
  class GaussianHMM(n_states=2, seed=0, *, n_iter=200, tol=1e-4, n_init=3, var_floor=1e-3, init_stay=0.95)
    .fit(x) (Baum-Welch, Rabiner scaling, diag cov, NaN-marginalising; states sorted by variance, 0=calm)
    .filter(x) -> DataFrame p_state_k (causal forward filter) ; .predict_next(x) -> DataFrame p_next_state_k
    .filtered_state(x) ; .score(x) ; .summary() ; .stationary_distribution_ ; .expected_durations_
aurum.portfolio.sizing:
  class VolTargetSizer(target_vol=0.10, max_leverage=2.0, max_lots=None, rebalance_band=0.10, kelly_cap=None, drawdown_derisk=((0.10,0.5),(0.15,0.25)), *, min_vol=0.02)
    .target_lots(forecast, vol_ann, equity, price, instrument, *, current_lots=0.0, drawdown=0.0) -> float  [PositionSizer]
    .breakdown(...same...) -> SizingBreakdown (audit trail, .to_dict())
  class FixedFractionalSizer(risk_per_trade=0.005, stop_atr=2.0, *, max_leverage=2.0, max_lots=None, rebalance_band=0.10, drawdown_derisk=..., atr_periods_per_year=252)
    .target_lots(..., *, current_lots=0.0, drawdown=0.0, atr=None) ; .stop_distance(vol_ann, price, *, atr=None)
  drawdown_multiplier(drawdown, steps) ; DEFAULT_DRAWDOWN_DERISK
aurum.portfolio.combiner:
  class ForecastCombiner(method="sharpe_shrink"|"equal"|"inverse_vol"|"hrp", shrinkage=0.5, max_weight=0.4, fdm_cap=2.5, *, vol_halflife=48, min_periods=20, corr_floor=0.0, bars_per_year=None, allow_unallocated=True, cost_multiplier=1.0)
    .fit(forecasts, close, *, bars=None, spread=None, costs=None, instrument=None, cost_per_turnover=None) -> self   # NET of c_t*|df| when cost info is given, else gross (noted)
    .combine(forecasts) -> Series 'combined' in [-1,1] ; .fit_combine(..., **cost_kwargs) ; .unit_vol_streams(forecasts, close, *, cost_per_turnover=None) ; .turnover_cost(forecasts, close, *, bars|spread, costs, instrument) -> c_t
    .explain() -> dict(weights, weights_sum, unallocated, allow_unallocated, train_sharpe (net), train_sharpe_gross, sharpe_basis, cost_basis, avg_cost_per_turnover, train_turnover, train_cost_per_bar, train_stream_std, avg_abs_forecast, fdm, fdm_raw, avg_forecast_correlation, n_obs, train_start/end, notes)
    attrs weights_ (may sum to < 1 with allow_unallocated), fdm_, fdm_raw_, train_sharpe_, train_sharpe_gross_, corr_, stream_corr_, notes_, cost_basis_
  cap_weights(weights, max_weight, *, allow_unallocated=False) -> ndarray ; METHODS
  # config: combiner.allow_unallocated (true) / combiner.cost_multiplier (1.0)
aurum.risk.manager:
  @dataclass RiskLimits(max_lots=None, max_leverage=3.0, max_daily_loss=0.03, max_drawdown=0.20, max_spread=None, event_blackout_before_min=30, event_blackout_after_min=30, event_min_importance=3, blackout_mode="no_new_risk"|"flatten", max_trades_per_day=None, stale_data_seconds=None, max_margin_utilisation=0.5, daily_reset="utc"|"rollover", daily_loss_persistent=True, event_lookahead_min=0.0)
  class StandardRiskManager(limits=None, instrument=XAUUSD, state_path=None, *, events=None)  [RiskManager]
    .evaluate(ctx, *, commit=True) -> RiskDecision   (commit=False = side-effect-free preview)
    .on_bar(time, equity) ; .halted ; .halt_reason ; .halt(reason, *, time=None) ; .reset_halt(confirm="RESET", *, equity=None)
    .set_events(df) ; .events_frame() -> DataFrame[time, requested, current, approved, halted, reasons] ; .snapshot() -> dict
  no_new_risk(approved, current) ; toward_zero(approved, target) ; RiskState
aurum.risk.var:
  historical_var_es(r, level=0.95) ; gaussian_var_es(r=None, level, *, mu, sigma) ; cornish_fisher_var_es(r, level, *, mu, sigma, skew, excess_kurt) ; cornish_fisher_quantile(alpha, skew, exkurt)
  var_es(r, level=0.95, method="historical"|"gaussian"|"cornish_fisher", *, horizon=1) -> dict
  kupiec_pof(n_obs, n_breaches, level) ; rolling_var_backtest(r, level, *, window=250, method)
  position_var(lots, price, vol_ann, *, level=0.99, horizon_days=1, instrument, periods_per_year=252)
  gap_shock(position_lots, price, pct, *, instrument=XAUUSD, spread=0.0) -> USD pnl
  stress_test(lots, price, equity, *, scenarios=None, instrument, spread=0.0) -> DataFrame ; DEFAULT_STRESS_SCENARIOS
  daily_returns_from_equity(equity) ; risk_report(result_or_equity, *, level=0.95, price=None, instrument, var_window=250, scenarios=None) -> dict
```

### Integration notes
Engine and live runner: call risk.on_bar(bars.available_at[t], equity), then risk.evaluate(RiskContext(time=available_at[t], ...)). on_bar and evaluate are idempotent for the same (time, equity), so calling both, in either order, is safe. aurum/backtest/engine.py does this (run_backtest with VolTargetSizer() and StandardRiskManager(RiskLimits()) on 8,000 synthetic H1 bars with events takes about 1.1 s). RiskContext.upcoming_events/recent_events frames need a 'time' column (UTC) and ideally 'importance'; a missing or NaN importance counts as high. Alternatively pass the whole calendar once with StandardRiskManager(events=df) for an O(log n) lookup. For live use, set state_path so kill switches survive restarts; a corrupt file starts HALTED (fail-safe). Clearing a halt needs a human call to reset_halt(confirm='RESET'), which re-bases peak and day-start equity. The sizer expects drawdown as a positive fraction below peak (the engine passes it that way; negative values are read by magnitude) and returns lots already rounded toward zero. Combiner: fit on TRAIN forecasts and the matching close; close is reindexed to the forecasts' index, so passing a longer close series cannot leak. Then combine() is row-wise, so causal on any later window. explain() is JSON-serialisable for the LLM desk and reports. Volatility: Garch11().fit(train_returns).forecast(all_returns) gives a causal annualised vol aligned to the input index, which can be blended with ewma_volatility via blend_vol for the sizer. GaussianHMM().fit(train).filter(x) gives causal regime probabilities; state 0 is the calm regime. risk_report(BacktestResult) is JSON-friendly (json.dumps(..., default=float)) for tearsheets and agents.

### Known limitations
- GARCH(1,1): Gaussian QMLE only (no Student-t innovations, no standard errors or Hessian); the mean is estimated two-step; fit() drops non-finite returns, which joins returns across gaps; the variance recursion starts from the training-sample variance (h0_), which is also used when forecasting later data (fitted on train, so causal).
- HAR-RV: plain OLS with no HAC standard errors; forecasts are floored at a tiny positive variance in levels mode. daily_realised_variance drops a trailing day whose last bar ends before the day boundary, so a Friday 21:00 close that is the very last row is dropped too (conservative).
- GaussianHMM: diagonal covariances only. The 2-state recursion is hand-unrolled and fast (about 50 ms fit for T=3000); the general K>2 path is a numpy loop and slower (about 2.4 s for K=3, T=2000, 3 restarts). There is deliberately no public smoother or Viterbi (look-ahead). Emissions are Gaussian, so non-Gaussian features such as |returns| degrade regime recovery (seen during testing; univariate returns reach about 96% accuracy).
- ForecastCombiner FDM follows Carver in assuming forecasts share a common scale; strategies on [-1,1] with very different typical magnitudes are not rescaled (avg_abs_forecast is reported instead). (Updated.) With the default allow_unallocated=True a low max_weight with few positive-edge strategies leaves the excess risk unallocated instead of forcing weight onto losers; only allow_unallocated=False (the legacy contract) still pushes weight onto them (flagged in notes).
- Risk manager: the context event-frame cache is keyed on object identity, so frames passed in RiskContext must not be mutated in place (the engine's cached slices are fine). Blackouts use scheduled times only. The stale-data check only fires when ctx.data_age_seconds is set (live). With state_path set, the JSON state is rewritten on every bar (atomic tmp + os.replace). Measured cost is about 16 µs/bar without event frames and about 35 µs/bar under cProfile with frames; a plain timing run including first-call parsing showed about 88 µs.
- VaR: historical-method horizon scaling uses sqrt(h), an approximation. Cornish-Fisher is only reliable for moderate skew and kurtosis: on Student-t(5) it overshot the 99% quantile, as documented and tested. The named historical stress scenarios (2013-04-15 about -9%, 2011-09-23 about -6%, 2016 Brexit about +5%) are rounded, approximate magnitudes from memory, not verified data. risk_report takes the stress price from price=, then meta['last_price'], then the last fill price.
- LLM desk (SPEC §10, aurum/agents): this module offers JSON-friendly hooks: StandardRiskManager.snapshot(), evaluate(ctx, commit=False) previews, halt(reason), ForecastCombiner.explain(), VolTargetSizer.breakdown(), Garch11.summary(), GaussianHMM.summary() and risk_report(). (Updated: nothing in aurum/agents calls evaluate(commit=False) or halt(); the only halt() caller is the live runner. The desk sees risk through the runner's risk.snapshot().) The LLM path still goes through the same sizer and risk manager and cannot bypass them.

## agents

### Public API
```
aurum.agents (importing it does not import anthropic; the SDK is loaded lazily only when a TradingDesk is built without a client)
- TradingDesk(provider, *, client=None, config: DeskConfig|None=None, policy: DecisionPolicy|None=None, journal_dir="runs/desk_journal")
  .run_cycle(now: tz-aware Timestamp, quant_forecast: float, *, previous_forecast=None, context: dict|None=None) -> DeskResult
  .last_final_forecast
- DeskResult(cycle_id, now, quant_forecast, decision: Decision|None, final_forecast, memos: list[Memo], usage: {"total","by_agent","budget"}, journal_path, status "decided"|"failed", failure_reason, policy: PolicyOutcome, chief_turns); .cost_usd
- demo(client=None, *, journal_dir=None, n_bars=600, seed=7) -> DeskResult. Runs offline on synthetic anonymised data with a scripted fake client, or live if you pass anthropic.Anthropic().
- DecisionPolicy(mode="overlay"|"advisory"|"discretionary", max_abs_forecast=1.0, on_failure="follow_quant"|"veto"|"hold", min_confidence=0.0)
  .apply(quant_forecast, decision|None, *, previous_forecast=None) -> float
  .evaluate(...) -> PolicyOutcome(final_forecast, requested_forecast, quant_forecast, mode, action, used_fallback, constrained, notes)
- Decision(action, scale, forecast, confidence, horizon_bars, rationale, key_risks, dissent).from_tool_input(dict)
- Memo(agent_id, role, question, status, stance, confidence, key_points, risks, suggested_exposure, error, turns, tools, mandate).from_tool_input(...)/.failed(...)/.for_chief()
- RecordValidationError(problems)
- DeskDataProvider Protocol: as_of(now)->str; market_snapshot/quant_signals/risk_status/macro_snapshot/calendar/backtest_stats/positions(now)->dict
- StaticDeskDataProvider({"market"|"quant_signals"|"risk"|"macro"|"calendar"|"backtest_stats"|"positions": dict | callable(now)->dict}), .update(kind, snap)
- HistoricalDeskDataProvider(md, *, signals=None, combined=None, vol=None, backtest_stats=None, risk_status_fn=None, positions_fn=None, anonymise=False, date_shift_days=None, lookback_bars=250, recent_bars=12, calendar_horizon_hours=72, calendar_lookback_hours=24, yield_series=...), .bar_index_at(now)
- AgentModelConfig(model="claude-opus-5", effort="medium", max_tokens=16000, thinking={"type":"adaptive"}, fallbacks=True, max_turns=6)
- DeskConfig(chief=AgentModelConfig(effort="high", max_turns=8), specialist=AgentModelConfig(effort="medium", max_turns=5), role_models={}, max_specialists_per_cycle=6, max_parallel_agents=4, max_cost_usd_per_cycle=3.0, max_tokens_per_cycle=None, soft_budget_fraction=0.8, max_cycle_seconds=None, prompt_caching=True, cache_ttl="5m"|"1h", request_timeout_s=600, tool_result_max_chars=12000, journal_max_chars=4000, max_text_field_chars=2000, adhoc_tool_whitelist=None, prices=dict)
  .model_for_role(role)
- ModelPrice(input_per_mtok, output_per_mtok, cache_read_multiplier=0.1); DEFAULT_PRICES: dict[str, ModelPrice] (claude-opus-5 is $5/$25)
- UsageLedger(prices, max_cost_usd, max_tokens, cache_ttl): .record(agent_id, response, requested_model=) / .exceeded() / .fraction_used() / .summary()
- CycleJournal(path|None, *, cycle_id, max_chars).log(event, *, agent=None, full=False, **payload); read_journal(path)
Submodules:
- client: LLMClient(client=None, *, prompt_caching, cache_ttl, timeout).create(cfg=, system_prompt=, tools=, messages=), which calls client.beta.messages.create; build_request(...); make_default_client(); sanitize_assistant_content(content); fallback_events(response); block_get; block_to_dict
- loop: run_agent_loop(spec: AgentSpec, first_message, runtime: AgentRuntime) -> LoopResult(status completed|refusal|max_turns|budget_exceeded|deadline|error); ToolOutcome
- chief: DeskCycle (per-cycle data cache, specialist cap, consult_specialist / create_specialist / submit_decision handlers, chief_spec, run_chief)
- specialists: PREDEFINED_SPECIALISTS (macro_strategist, quant_analyst, risk_officer, execution_trader), SpecialistTask, run_specialist, build_specialist_spec, predefined_task, adhoc_task, slugify
- tools: DATA_TOOLS, DATA_TOOL_NAMES, chief_tool_definitions(adhoc_whitelist=None), specialist_tool_definitions(data_tools), validate_against_schema
- prompts: CHIEF_SYSTEM_PROMPT, SPECIALIST_BASE_PROMPT, ROLE_CHARTERS, ADHOC_CHARTER, MODE_DESCRIPTIONS, specialist_system_prompt(role), chief_brief(...), specialist_brief(...), adhoc_brief(...)
- testing: FakeAnthropicClient(scripts={agent_key: [FakeMessage | callable(kwargs)]}, *, router=default_router, default=None, validate=True) exposing .beta.messages.create, .messages.create, .calls, .calls_for(key), .errors, .remaining(key), .add(key, *responses); builders message(), tool_use(), text(), thinking(), refusal(), truncated(), decision_call(), memo_call(); FakeMessage/FakeUsage/FakeIterationUsage/FakeFallbackBlock (.to_api_dict()); validate_request(kwargs)
```

### Integration notes
An LLM Chief (CIO) agent that creates new specialist agents at runtime and makes the final decision.

Usage: desk = TradingDesk(provider, client=anthropic.Anthropic(), policy=DecisionPolicy("overlay")); res = desk.run_cycle(bars.available_at[t], quant_forecast=combined[t]). Then feed res.final_forecast into VolTargetSizer.target_lots(...) and StandardRiskManager.evaluate(...), exactly like the quant book.

Failures never raise. On refusal, max turns, budget, deadline, API or network error, the cycle returns status="failed" and final_forecast = DecisionPolicy.on_failure (default: follow the quant forecast). The only raises are programming errors, such as a naive timestamp or a missing anthropic package when no client is passed.

The default overlay mode guarantees |final| <= |q| and that the final forecast never takes the opposite sign to q.

Chief tools: 7 read-only data tools, consult_specialist(role, question), create_specialist(name, mandate, tools[], question) and submit_decision. Several consult/create calls in one Chief turn run concurrently on a per-cycle ThreadPoolExecutor (max_parallel_agents=4). Specialists run their own tools inline, which avoids nested-pool deadlock, and they have no agent-spawning tools (depth 1).

Caching: system prompts and tool lists are byte-stable (sorted tools, no timestamps). Each request sets an explicit cache breakpoint on the system prompt plus top-level automatic caching. For H1 cadence consider DeskConfig(cache_ttl="1h").

Journals are one JSONL file per cycle. Quick check: `from aurum.agents import demo; demo(journal_dir="runs/demo")`.

For tests, use aurum.agents.testing.FakeAnthropicClient with per-agent scripts keyed "chief", "<role>" or "adhoc:<slug>"; it validates each request against Messages API rules and collects violations in .errors.

Run the tests with: .venv/bin/python -m pytest tests/test_agents_desk.py tests/test_agents_policy.py tests/test_agents_providers.py tests/test_agents_sdk.py -q

### Known limitations
- Never exercised against the live Claude API: there was no network or key. Request serialisation, headers and response parsing were verified through the real anthropic 1.8 SDK over an httpx2.MockTransport, and the model IDs, beta header and parameters follow the bundled docs. Prompt quality and real cache hit rates still need a live smoke test.
- Budget enforcement is best-effort under concurrency. It is checked before every API call, so calls already in flight can overshoot the cap by their own cost. Costs are list-price estimates from the DEFAULT_PRICES dict. When usage.iterations is present it is summed per attempt, which can overcount declined-before-output fallback attempts (a conservative bias).
- Requests are non-streaming. max_tokens above about 21k needs a client with an explicit non-default timeout or streaming, which is not implemented. The default client built by the desk uses timeout=600s.
- Anonymisation (shifted dates, prices rebased to 100, macro levels withheld) reduces but cannot eliminate look-ahead through model memory: path shapes, event sequences and cross-asset co-movements can still be recognisable. This is documented in the providers.py module docstring. The risk_status_fn and positions_fn hook outputs pass through unchanged, so callers must keep them scale-free when anonymising.
- HistoricalDeskDataProvider cannot supply live risk or position state by itself; that needs the risk_status_fn / positions_fn hooks above. Backtest stats handed to it must already be point-in-time; that is the caller's responsibility.
- The Fake client routes scripts by the 'Agent: <id>' header in each agent's first user message. When the same role is consulted twice in parallel it gets id 'role#2' and falls back to the base role's queue, so the interleaving is nondeterministic in that case.
- No prompt or quality evaluation (evals) of the agents was done. That requires live model runs.

## wave 3 integration (financing, cost-aware combiner/research, stability)

### Public API additions
```
aurum.execution: FinancingModel, RateCurve (costs), intrabar_exit (simulator) added to the lazy exports
aurum.core.config: FinancingConfig (costs.financing, default mode "rate"); CombinerConfig.allow_unallocated=True, .cost_multiplier=1.0 (passed by build());
  live_runner_mapping() passes backtest.stop_cooldown_bars (live.options.stop_cooldown_bars is rejected)
aurum.rl.env.GoldTradingEnv(..., rates=None)   # aurum.rl.train passes md.macro in training, evaluation and rollout_artifact (parity with run_backtest)
aurum.rl.train: POLICY_FILES, read_artifact_bytes(path) -> dict[str, bytes], load_artifact_bytes(files, *, device="cpu", path=None) -> RLArtifact (also via aurum.rl)
aurum.strategies.rl.RLPolicyStrategy: embeds the artifact bytes + fingerprint (.has_embedded_policy); pickles are self-contained
aurum.strategies.seasonal: IntradaySeasonality params cost_aware=True, costs=None (CostModel or kwargs), cost_multiplier=2.0, dead_zone="auto"; periodic_cost_aware_positions(alpha, kappa, gamma)
aurum.live.paper.PaperBroker(..., rates=None).set_rates(rates)   # the runner passes macro_dir frames at construction and on every macro reload
aurum.live.runner: runner_state.json prev_final_forecast -> TradingDesk.run_cycle(previous_forecast=...); decisions.jsonl previous_forecast
aurum.research.walkforward: HOLDOUT_LEDGER, read_holdout_ledger(path), prior_holdout_looks(entries, *, start, end, config_hash, symbol=None), append_holdout_ledger(path, entry), train_feature_reference(train_features)
  run_walk_forward(md, config, *, strategies=None, out_dir=None, write=None, holdout_ledger=None|path|False)
  fit_quant_book(md, cfg, *, ..., oos_forecasts=None, combiner_fit=None, feature_reference=True) -> QuantBook(+combiner_basis, feature_reference)
  HoldoutReport(+forecasts, prior_looks, ledger_path); settings n_trials_base / n_trials_prior_looks
aurum.cli: aurum train-final --config C --out DIR [--cutoff TS] [--from-run WF_DIR] [--allow-config-mismatch] [--overwrite] [--jobs N] [--executor ...] [--no-write] [--no-tearsheet]
```

### Integration notes
- Config hashes changed (costs.financing and the combiner knobs are hashed): `--from-run` refuses runs made before this change unless `--allow-config-mismatch`.
- Anything built with the default `CostModel()` now pays rate financing (Fed funds from `md.macro` + 2.5%); tests that pin per-lot swap arithmetic use `FinancingModel.fixed()`.
- The regime group's default warm-up is about one year of bars (5,820 H1), which ML strategies with default feature groups lose from each fold's training slice; set the strategy param `feature_overrides: {regime: {rank_years: ...}}` (or `features.overrides` for the shared pipeline) if that matters. The live runner's history (3 x max_lookback) grows accordingly (about 17.5k H1 / 70k M15 bars).
