"""
Enhanced Macro Features Module

Integrates 24 features from 8 macro data sources:
- DXY (Dollar Index): 3 features
- SPX (S&P 500): 3 features
- US10Y (Treasury Yields): 3 features
- VIX (Fear Index): 3 features
- Oil (WTI Crude): 3 features
- Bitcoin: 3 features
- EURUSD: 3 features
- Silver + GLD: 3 features

Each source provides: returns, momentum, and correlation with gold
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_macro_data(data_dir='data'):
    """
    Load all macro data sources

    Returns:
        Dict of DataFrames: {name: df}
    """
    logger.info("📥 Loading macro data sources...")

    data_dir = Path(data_dir)

    macro_files = {
        'dxy': 'dxy_daily.csv',
        'spx': 'spx_daily.csv',
        'us10y': 'us10y_daily.csv',
        'vix': 'vix_daily.csv',
        'oil': 'oil_wti_daily.csv',
        'btc': 'bitcoin_daily.csv',
        'eur': 'eurusd_daily.csv',
        'silver': 'silver_daily.csv',
        'gld': 'gld_etf_daily.csv',
    }

    macro_dict = {}

    for name, filename in macro_files.items():
        filepath = data_dir / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            # Convert to timezone-naive datetime from the start
            df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
            df = df.set_index('time').sort_index()

            # Use 'close' or 'Close' column
            if 'close' in df.columns:
                macro_dict[name] = df['close']
            elif 'Close' in df.columns:
                macro_dict[name] = df['Close']
            else:
                logger.warning(f"⚠️  No close price found for {name}")
                continue

            logger.info(f"   ✅ {name.upper()}: {len(df):,} bars")
        else:
            logger.warning(f"⚠️  {name.upper()} file not found: {filename}")

    logger.info(f"\n✅ Loaded {len(macro_dict)} macro sources")

    return macro_dict


def normalize_timezone(series, reference_index):
    """
    Normalize timezone of a series to match reference index

    Args:
        series: Pandas Series with datetime index
        reference_index: Reference DatetimeIndex

    Returns:
        Series with timezone-naive index, aligned to reference
    """
    # Make series timezone-naive by converting to UTC then removing tz
    series_copy = series.copy()
    if hasattr(series_copy.index, 'tz') and series_copy.index.tz is not None:
        series_copy.index = series_copy.index.tz_convert('UTC').tz_localize(None)
    elif isinstance(series_copy.index, pd.DatetimeIndex):
        # Already timezone-naive
        pass
    else:
        # Not a DatetimeIndex, convert
        series_copy.index = pd.to_datetime(series_copy.index, utc=True).tz_localize(None)

    # Make reference timezone-naive
    if hasattr(reference_index, 'tz') and reference_index.tz is not None:
        ref_index_naive = reference_index.tz_convert('UTC').tz_localize(None)
    elif isinstance(reference_index, pd.DatetimeIndex):
        ref_index_naive = reference_index
    else:
        ref_index_naive = pd.to_datetime(reference_index, utc=True).tz_localize(None)

    # Align to reference
    aligned = series_copy.reindex(ref_index_naive, method='ffill')

    return aligned


def compute_rolling_correlation(series1, series2, window=120):
    """
    Compute rolling correlation between two series

    Args:
        series1, series2: Price series
        window: Rolling window (default: 120 days)

    Returns:
        Rolling correlation series
    """
    # Compute returns
    ret1 = series1.pct_change()
    ret2 = series2.pct_change()

    # Rolling correlation
    corr = ret1.rolling(window).corr(ret2)

    return corr.fillna(0.0)


def compute_dxy_features(gold_prices, dxy_prices):
    """
    Compute DXY (Dollar Index) features

    Returns 3 features:
    - dxy_return: Daily return
    - dxy_momentum: 20-day momentum
    - gold_dxy_correlation: Rolling 120-day correlation
    """
    result = pd.DataFrame(index=gold_prices.index)

    # Align DXY to gold timestamps
    dxy_aligned = normalize_timezone(dxy_prices, gold_prices.index)

    # Feature 1: DXY return
    result['dxy_return'] = dxy_aligned.pct_change()

    # Feature 2: DXY momentum (20-day)
    result['dxy_momentum'] = dxy_aligned.pct_change(20)

    # Feature 3: Gold-DXY correlation
    result['gold_dxy_correlation'] = compute_rolling_correlation(
        gold_prices, dxy_aligned, window=120
    )

    return result


def compute_spx_features(gold_prices, spx_prices):
    """
    Compute SPX (S&P 500) features

    Returns 3 features:
    - spx_return: Daily return
    - spx_momentum: 20-day momentum
    - gold_spx_correlation: Risk-on/risk-off correlation
    """
    result = pd.DataFrame(index=gold_prices.index)

    # Align SPX to gold timestamps
    spx_aligned = normalize_timezone(spx_prices, gold_prices.index)

    # Feature 1: SPX return
    result['spx_return'] = spx_aligned.pct_change()

    # Feature 2: SPX momentum (20-day)
    result['spx_momentum'] = spx_aligned.pct_change(20)

    # Feature 3: Gold-SPX correlation
    result['gold_spx_correlation'] = compute_rolling_correlation(
        gold_prices, spx_aligned, window=120
    )

    return result


def compute_us10y_features(gold_prices, us10y_prices):
    """
    Compute US10Y (Treasury Yields) features

    Returns 3 features:
    - us10y_change: Daily change
    - us10y_momentum: 20-day momentum
    - gold_yields_correlation: Inverse relationship
    """
    result = pd.DataFrame(index=gold_prices.index)

    # Align US10Y to gold timestamps
    us10y_aligned = normalize_timezone(us10y_prices, gold_prices.index)

    # Feature 1: US10Y change
    result['us10y_change'] = us10y_aligned.diff()

    # Feature 2: US10Y momentum (20-day)
    result['us10y_momentum'] = us10y_aligned.diff(20)

    # Feature 3: Gold-Yields correlation
    result['gold_yields_correlation'] = compute_rolling_correlation(
        gold_prices, us10y_aligned, window=120
    )

    return result


def compute_vix_features(gold_prices, vix_prices):
    """
    Compute VIX (Fear Index) features

    Returns 3 features:
    - vix_level: Current VIX level (normalized)
    - vix_change: Daily change
    - vix_regime: High fear (>20) or low fear
    """
    result = pd.DataFrame(index=gold_prices.index)

    # Align VIX to gold timestamps
    vix_aligned = normalize_timezone(vix_prices, gold_prices.index)

    # Feature 1: VIX level (normalized by dividing by 50)
    result['vix_level'] = vix_aligned / 50.0

    # Feature 2: VIX change
    result['vix_change'] = vix_aligned.diff()

    # Feature 3: VIX regime (high fear = 1, low fear = -1)
    result['vix_regime'] = np.where(vix_aligned > 20, 1.0, -1.0)

    return result


def compute_oil_features(gold_prices, oil_prices):
    """
    Compute Oil (WTI Crude) features

    Returns 3 features:
    - oil_return: Daily return
    - oil_momentum: 20-day momentum
    - gold_oil_correlation: Commodity correlation
    """
    result = pd.DataFrame(index=gold_prices.index)

    # Align Oil to gold timestamps
    oil_aligned = normalize_timezone(oil_prices, gold_prices.index)

    # Feature 1: Oil return
    result['oil_return'] = oil_aligned.pct_change()

    # Feature 2: Oil momentum (20-day)
    result['oil_momentum'] = oil_aligned.pct_change(20)

    # Feature 3: Gold-Oil correlation
    result['gold_oil_correlation'] = compute_rolling_correlation(
        gold_prices, oil_aligned, window=120
    )

    return result


def compute_btc_features(gold_prices, btc_prices):
    """
    Compute Bitcoin features

    Returns 3 features:
    - btc_return: Daily return
    - btc_momentum: 20-day momentum
    - gold_btc_correlation: Risk sentiment
    """
    result = pd.DataFrame(index=gold_prices.index)

    # Align Bitcoin to gold timestamps
    btc_aligned = normalize_timezone(btc_prices, gold_prices.index)

    # Feature 1: BTC return
    result['btc_return'] = btc_aligned.pct_change()

    # Feature 2: BTC momentum (20-day)
    result['btc_momentum'] = btc_aligned.pct_change(20)

    # Feature 3: Gold-BTC correlation
    result['gold_btc_correlation'] = compute_rolling_correlation(
        gold_prices, btc_aligned, window=120
    )

    return result


def compute_eur_features(gold_prices, eur_prices):
    """
    Compute EURUSD features

    Returns 3 features:
    - eur_return: Daily return
    - eur_momentum: 20-day momentum
    - gold_eur_correlation: Dollar proxy
    """
    result = pd.DataFrame(index=gold_prices.index)

    # Align EURUSD to gold timestamps
    eur_aligned = normalize_timezone(eur_prices, gold_prices.index)

    # Feature 1: EUR return
    result['eur_return'] = eur_aligned.pct_change()

    # Feature 2: EUR momentum (20-day)
    result['eur_momentum'] = eur_aligned.pct_change(20)

    # Feature 3: Gold-EUR correlation
    result['gold_eur_correlation'] = compute_rolling_correlation(
        gold_prices, eur_aligned, window=120
    )

    return result


def compute_silver_gld_features(gold_prices, silver_prices, gld_prices):
    """
    Compute Silver and GLD features

    Returns 3 features:
    - gold_silver_ratio: Gold/Silver price ratio
    - gold_silver_correlation: Precious metals correlation
    - gld_flow: Institutional flows (volume-weighted if available)
    """
    result = pd.DataFrame(index=gold_prices.index)

    # Align Silver to gold timestamps
    silver_aligned = normalize_timezone(silver_prices, gold_prices.index)

    # Feature 1: Gold/Silver ratio
    result['gold_silver_ratio'] = gold_prices / (silver_aligned + 1e-8)
    # Normalize by dividing by typical ratio (80)
    result['gold_silver_ratio'] = result['gold_silver_ratio'] / 80.0

    # Feature 2: Gold-Silver correlation
    result['gold_silver_correlation'] = compute_rolling_correlation(
        gold_prices, silver_aligned, window=120
    )

    # Feature 3: GLD flow proxy (using daily returns as proxy)
    if gld_prices is not None:
        gld_aligned = normalize_timezone(gld_prices, gold_prices.index)
        result['gld_flow'] = gld_aligned.pct_change()
    else:
        result['gld_flow'] = 0.0

    return result


def compute_macro_features(df_gold, macro_dict):
    """
    Compute all 24 macro features

    Args:
        df_gold: Gold price DataFrame (M5 or H1 data)
        macro_dict: Dict of macro price series from load_macro_data()

    Returns:
        DataFrame with 24 macro features aligned to gold timestamps
    """
    logger.info("="*70)
    logger.info("🌍 COMPUTING MACRO FEATURES")
    logger.info("="*70)

    # Get gold daily prices (resample if needed)
    if len(df_gold) > 10000:  # If intraday data
        logger.info("Resampling gold to daily for macro alignment...")
        gold_daily = df_gold['close'].resample('1D').last().dropna()
    else:
        gold_daily = df_gold['close']

    # Compute features for each macro source
    feature_dfs = []

    if 'dxy' in macro_dict:
        logger.info("Computing DXY features...")
        feature_dfs.append(compute_dxy_features(gold_daily, macro_dict['dxy']))

    if 'spx' in macro_dict:
        logger.info("Computing SPX features...")
        feature_dfs.append(compute_spx_features(gold_daily, macro_dict['spx']))

    if 'us10y' in macro_dict:
        logger.info("Computing US10Y features...")
        feature_dfs.append(compute_us10y_features(gold_daily, macro_dict['us10y']))

    if 'vix' in macro_dict:
        logger.info("Computing VIX features...")
        feature_dfs.append(compute_vix_features(gold_daily, macro_dict['vix']))

    if 'oil' in macro_dict:
        logger.info("Computing Oil features...")
        feature_dfs.append(compute_oil_features(gold_daily, macro_dict['oil']))

    if 'btc' in macro_dict:
        logger.info("Computing Bitcoin features...")
        feature_dfs.append(compute_btc_features(gold_daily, macro_dict['btc']))

    if 'eur' in macro_dict:
        logger.info("Computing EURUSD features...")
        feature_dfs.append(compute_eur_features(gold_daily, macro_dict['eur']))

    if 'silver' in macro_dict:
        logger.info("Computing Silver/GLD features...")
        gld = macro_dict.get('gld', None)
        feature_dfs.append(compute_silver_gld_features(gold_daily, macro_dict['silver'], gld))

    # Combine all features
    if feature_dfs:
        macro_features_daily = pd.concat(feature_dfs, axis=1)
        macro_features_daily = macro_features_daily.fillna(0.0)

        # Align back to original gold timeframe (forward-fill daily data)
        macro_features = macro_features_daily.reindex(df_gold.index, method='ffill')
        macro_features = macro_features.fillna(0.0)
    else:
        # No macro data available
        macro_features = pd.DataFrame(index=df_gold.index)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ MACRO FEATURES COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Generated {macro_features.shape[1]} macro features")
    logger.info(f"✅ Aligned to {len(macro_features):,} bars")

    # List features
    logger.info("\n📊 Features created:")
    for col in macro_features.columns:
        logger.info(f"   • {col}")

    return macro_features


def test_macro_features():
    """
    Test function to verify macro features work correctly
    """
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING MACRO FEATURES")
    logger.info("="*70)

    try:
        # Load timeframe features
        from features.timeframe_features import load_and_compute_all_timeframes

        logger.info("\n1️⃣ Loading timeframe features...")
        tf_features = load_and_compute_all_timeframes(base_timeframe='M5')

        logger.info("\n2️⃣ Loading macro data...")
        macro_dict = load_macro_data()

        logger.info("\n3️⃣ Computing macro features...")
        # Use M5 data with close prices
        import pandas as pd
        df_gold_m5 = pd.read_csv('data/xauusd_m5.csv')
        df_gold_m5['time'] = pd.to_datetime(df_gold_m5['time'])
        df_gold_m5 = df_gold_m5.set_index('time').sort_index()

        macro_features = compute_macro_features(df_gold_m5, macro_dict)

        logger.info("\n✅ Macro features computed successfully!")

        # Check for NaNs
        nan_count = macro_features.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️  {nan_count} NaN values found")
        else:
            logger.info("✅ No NaN values")

        # Show sample
        logger.info("\n📊 Sample data:")
        logger.info(macro_features.head())

        return macro_features

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test
    macro_feat = test_macro_features()

    logger.info("\n" + "="*70)
    logger.info("✅ MACRO FEATURES MODULE READY")
    logger.info("="*70)

    logger.info("""
📋 USAGE:
    from features.macro_features import load_macro_data, compute_macro_features

    # Load macro data
    macro_dict = load_macro_data()

    # Compute macro features
    macro_features = compute_macro_features(df_gold, macro_dict)
    """)
