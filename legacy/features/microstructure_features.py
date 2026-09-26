"""
Market Microstructure Features Module

Computes 12 features related to market microstructure:
- Session Effects (4): Asian, London, NY, overlap detection
- Time Effects (4): hour, day of week, week of month, month of year
- Volume Analysis (2): volume profile, volume imbalance
- Liquidity (2): spread proxy, liquidity regime

These features capture intraday patterns and market mechanics.
"""

import pandas as pd
import numpy as np
import logging
from datetime import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_session_features(df):
    """
    Compute trading session features

    Returns 4 features:
    - session_asian: Asian session (1/0)
    - session_london: London session (1/0)
    - session_ny: New York session (1/0)
    - session_overlap: Session overlap period (1/0)
    """
    logger.info("Computing session features...")

    result = pd.DataFrame(index=df.index)

    # Extract hour (in UTC)
    hours = df.index.hour

    # Session definitions (UTC times)
    # Asian: 00:00 - 09:00 UTC
    # London: 08:00 - 17:00 UTC
    # New York: 13:00 - 22:00 UTC

    result['session_asian'] = ((hours >= 0) & (hours < 9)).astype(float)
    result['session_london'] = ((hours >= 8) & (hours < 17)).astype(float)
    result['session_ny'] = ((hours >= 13) & (hours < 22)).astype(float)

    # Session overlaps (London + NY: 13:00 - 17:00 UTC)
    result['session_overlap'] = ((hours >= 13) & (hours < 17)).astype(float)

    return result


def compute_time_features(df):
    """
    Compute time-based features

    Returns 4 features:
    - hour_of_day: Hour (0-23, normalized)
    - day_of_week: Monday=0 to Friday=4 (normalized)
    - week_of_month: First/Last week effects
    - month_of_year: Seasonal patterns (0-11, normalized)
    """
    logger.info("Computing time features...")

    result = pd.DataFrame(index=df.index)

    # Feature 1: Hour of day (normalized to 0-1)
    result['hour_of_day'] = df.index.hour / 23.0

    # Feature 2: Day of week (normalized to 0-1)
    # Monday=0, Tuesday=1, ..., Sunday=6
    result['day_of_week'] = df.index.dayofweek / 6.0

    # Feature 3: Week of month (1-5 typically)
    # First week = 0, Last week = 1
    day_of_month = df.index.day
    result['week_of_month'] = np.where(
        day_of_month <= 7, 0.0,  # First week
        np.where(day_of_month >= 22, 1.0, 0.5)  # Last week, or middle
    )

    # Feature 4: Month of year (normalized to 0-1)
    result['month_of_year'] = (df.index.month - 1) / 11.0

    return result


def compute_volume_features(df):
    """
    Compute volume analysis features

    Returns 2 features:
    - volume_profile: Current volume percentile
    - volume_imbalance: Buy volume - Sell volume proxy
    """
    logger.info("Computing volume features...")

    result = pd.DataFrame(index=df.index)

    if 'volume' not in df.columns:
        logger.warning("⚠️  No volume column found, using defaults")
        result['volume_profile'] = 0.5
        result['volume_imbalance'] = 0.0
        return result

    volume = df['volume']

    # Feature 1: Volume profile (percentile rank)
    # Rolling 100-period percentile
    result['volume_profile'] = volume.rolling(100).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x) if len(x) > 0 else 0.5,
        raw=False
    )

    # Feature 2: Volume imbalance proxy
    # Use price change direction as proxy for buy/sell pressure
    price_change = df['close'].diff()
    volume_signed = volume * np.sign(price_change)

    # Rolling net volume
    volume_net = volume_signed.rolling(20).sum()
    volume_total = volume.rolling(20).sum()

    result['volume_imbalance'] = volume_net / (volume_total + 1e-8)

    # Fill NaNs
    result = result.fillna(0.5)

    return result


def compute_liquidity_features(df):
    """
    Compute liquidity features

    Returns 2 features:
    - spread_m5: Bid-ask spread proxy (high-low as % of close)
    - liquidity_regime: High/Low liquidity period
    """
    logger.info("Computing liquidity features...")

    result = pd.DataFrame(index=df.index)

    # Feature 1: Spread proxy (high - low) / close
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        spread = (df['high'] - df['low']) / df['close']
        result['spread_m5'] = spread
    else:
        logger.warning("⚠️  Missing OHLC columns for spread calculation")
        result['spread_m5'] = 0.001  # Default small spread

    # Feature 2: Liquidity regime
    # Low spread = high liquidity = 1
    # High spread = low liquidity = -1
    if 'spread_m5' in result.columns:
        avg_spread = result['spread_m5'].rolling(100).mean()
        result['liquidity_regime'] = np.where(
            result['spread_m5'] < avg_spread, 1.0, -1.0
        )
    else:
        result['liquidity_regime'] = 1.0

    # Fill NaNs
    result = result.fillna(0.0)

    return result


def compute_all_microstructure_features(df):
    """
    Main function: Compute all 12 microstructure features

    Args:
        df: DataFrame with OHLCV data and DatetimeIndex

    Returns:
        DataFrame with 12 microstructure features
    """
    logger.info("="*70)
    logger.info("🏛️  COMPUTING MARKET MICROSTRUCTURE FEATURES")
    logger.info("="*70)

    # Compute each feature group
    session_features = compute_session_features(df)
    time_features = compute_time_features(df)
    volume_features = compute_volume_features(df)
    liquidity_features = compute_liquidity_features(df)

    # Combine all features
    all_features = pd.concat([
        session_features,
        time_features,
        volume_features,
        liquidity_features
    ], axis=1)

    # Fill any NaNs
    all_features = all_features.fillna(0.0)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ MICROSTRUCTURE FEATURES COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Generated {all_features.shape[1]} microstructure features")
    logger.info(f"✅ Feature count: {len(all_features):,} bars")

    # Session statistics
    asian_pct = all_features['session_asian'].mean() * 100
    london_pct = all_features['session_london'].mean() * 100
    ny_pct = all_features['session_ny'].mean() * 100
    overlap_pct = all_features['session_overlap'].mean() * 100

    logger.info(f"\n📊 Session distribution:")
    logger.info(f"   • Asian session: {asian_pct:.1f}%")
    logger.info(f"   • London session: {london_pct:.1f}%")
    logger.info(f"   • New York session: {ny_pct:.1f}%")
    logger.info(f"   • Overlap period: {overlap_pct:.1f}%")

    # List features
    logger.info("\n📊 Features created:")
    for col in all_features.columns:
        logger.info(f"   • {col}")

    return all_features


def test_microstructure_features():
    """
    Test function to verify microstructure features work correctly
    """
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING MICROSTRUCTURE FEATURES")
    logger.info("="*70)

    try:
        # Load gold data
        logger.info("\n1️⃣ Loading gold data...")
        df_gold = pd.read_csv('data/xauusd_m5.csv')
        df_gold['time'] = pd.to_datetime(df_gold['time'])
        df_gold = df_gold.set_index('time').sort_index()

        logger.info("\n2️⃣ Computing microstructure features...")
        micro_features = compute_all_microstructure_features(df_gold)

        logger.info("\n✅ Microstructure features computed successfully!")

        # Check for NaNs
        nan_count = micro_features.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️  {nan_count} NaN values found")
        else:
            logger.info("✅ No NaN values")

        # Show sample
        logger.info("\n📊 Sample data:")
        logger.info(micro_features.head())

        return micro_features

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test
    micro_feat = test_microstructure_features()

    logger.info("\n" + "="*70)
    logger.info("✅ MICROSTRUCTURE FEATURES MODULE READY")
    logger.info("="*70)

    logger.info("""
📋 USAGE:
    from features.microstructure_features import compute_all_microstructure_features

    # Load data with OHLCV and DatetimeIndex
    df = pd.read_csv('data/xauusd_m5.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # Compute microstructure features
    micro_features = compute_all_microstructure_features(df)
    """)
