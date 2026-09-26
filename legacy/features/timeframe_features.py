"""
Timeframe Feature Engineering Module

Computes 16 features for each timeframe:
- Price Action (5): returns, volatility, momentum (3 periods)
- Trend Indicators (4): MA fast, MA slow, MA diff, trend direction
- Technical Indicators (4): RSI, MACD, ATR, BB position
- Volume & S/R (3): volume ratio, distance to high/low

This module is designed to work with any timeframe (M5, M15, H1, H4, D1, W1).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_timeframe_features(df, tf_name):
    """
    Compute 16 features for a single timeframe

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        tf_name: Timeframe name ('M5', 'M15', 'H1', 'H4', 'D1', 'W1')

    Returns:
        DataFrame with 16 features, columns prefixed with tf_name
    """
    logger.info(f"Computing features for {tf_name}...")

    result = pd.DataFrame(index=df.index)

    # ========== PRICE ACTION FEATURES (5) ==========

    # 1. Price return
    result[f'{tf_name}_return'] = df['close'].pct_change()

    # 2. Volatility (20-period rolling std)
    result[f'{tf_name}_volatility'] = df['close'].pct_change().rolling(20).std()

    # 3-5. Momentum (multiple periods)
    result[f'{tf_name}_momentum_5'] = df['close'].pct_change(5)
    result[f'{tf_name}_momentum_10'] = df['close'].pct_change(10)
    result[f'{tf_name}_momentum_20'] = df['close'].pct_change(20)

    # ========== TREND INDICATORS (4) ==========

    # 6. Fast moving average (10-period)
    ma_fast = df['close'].rolling(10).mean()
    result[f'{tf_name}_ma_fast'] = (df['close'] - ma_fast) / df['close']

    # 7. Slow moving average (50-period)
    ma_slow = df['close'].rolling(50).mean()
    result[f'{tf_name}_ma_slow'] = (df['close'] - ma_slow) / df['close']

    # 8. MA difference (normalized)
    result[f'{tf_name}_ma_diff'] = (ma_fast - ma_slow) / df['close']

    # 9. Trend direction (+1 or -1)
    result[f'{tf_name}_trend'] = np.where(ma_fast > ma_slow, 1.0, -1.0)

    # ========== TECHNICAL INDICATORS (4) ==========

    # 10. RSI (14-period, normalized to 0-1)
    result[f'{tf_name}_rsi'] = compute_rsi(df['close'], period=14) / 100.0

    # 11. MACD signal
    result[f'{tf_name}_macd'] = compute_macd(df['close'])

    # 12. ATR as % of price
    result[f'{tf_name}_atr_pct'] = compute_atr(df, period=14) / df['close']

    # 13. Bollinger Band position (0 = lower band, 1 = upper band)
    result[f'{tf_name}_bb_position'] = compute_bb_position(df['close'], period=20)

    # ========== VOLUME & SUPPORT/RESISTANCE (3) ==========

    # 14. Volume ratio (current / 20-period average)
    avg_volume = df['volume'].rolling(20).mean()
    result[f'{tf_name}_volume_ratio'] = df['volume'] / avg_volume

    # 15. Distance to recent high (50-period)
    recent_high = df['high'].rolling(50).max()
    result[f'{tf_name}_dist_to_high'] = (df['close'] - recent_high) / df['close']

    # 16. Distance to recent low (50-period)
    recent_low = df['low'].rolling(50).min()
    result[f'{tf_name}_dist_to_low'] = (df['close'] - recent_low) / df['close']

    # Fill initial NaNs with 0
    result = result.fillna(0.0)

    logger.info(f"   ✅ {tf_name}: {result.shape[1]} features computed")

    return result


def compute_rsi(prices, period=14):
    """
    Compute Relative Strength Index

    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50.0)


def compute_macd(prices, fast=12, slow=26, signal=9):
    """
    Compute MACD signal (normalized)

    Returns:
        MACD histogram normalized by price
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Normalize by price
    macd_normalized = macd_hist / prices

    return macd_normalized.fillna(0.0)


def compute_atr(df, period=14):
    """
    Compute Average True Range

    Returns:
        ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr.fillna(0.0)


def compute_bb_position(prices, period=20, num_std=2):
    """
    Compute position within Bollinger Bands

    Returns:
        Position (0 = lower band, 0.5 = middle, 1 = upper band)
    """
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()

    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)

    # Position within bands (0 to 1)
    bb_position = (prices - lower_band) / (upper_band - lower_band)

    # Clip to 0-1 range
    bb_position = bb_position.clip(0, 1)

    return bb_position.fillna(0.5)


def load_timeframe_data(filepath):
    """
    Load a single timeframe data file

    Args:
        filepath: Path to CSV file with columns: time, open, high, low, close, volume

    Returns:
        DataFrame with datetime index
    """
    df = pd.read_csv(filepath)

    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.sort_index()

    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def align_timeframes(tf_dict, base_timeframe='M5'):
    """
    Align all timeframes to the base timeframe's timestamps

    Args:
        tf_dict: Dict of DataFrames {tf_name: df_features}
        base_timeframe: Base timeframe to align to (default: 'M5')

    Returns:
        Dict of aligned DataFrames
    """
    logger.info(f"Aligning all timeframes to {base_timeframe}...")

    base_index = tf_dict[base_timeframe].index
    aligned = {}

    for tf_name, df in tf_dict.items():
        if tf_name == base_timeframe:
            aligned[tf_name] = df
        else:
            # Forward-fill higher timeframe data to base timeframe
            df_reindexed = df.reindex(base_index, method='ffill')
            aligned[tf_name] = df_reindexed

    logger.info(f"   ✅ All timeframes aligned to {len(base_index):,} bars")

    return aligned


def load_and_compute_all_timeframes(base_timeframe='M5', data_dir='data'):
    """
    Load all timeframe data and compute features

    Args:
        base_timeframe: Base timeframe to align everything to ('M5' recommended)
        data_dir: Directory containing data files

    Returns:
        Dict of aligned feature DataFrames: {tf_name: df_features}
    """
    logger.info("="*70)
    logger.info("📊 LOADING AND COMPUTING ALL TIMEFRAME FEATURES")
    logger.info("="*70)

    data_dir = Path(data_dir)

    # Timeframe file mappings
    timeframe_files = {
        'M5': 'xauusd_m5.csv',
        'M15': 'xauusd_m15.csv',
        'H1': 'xauusd_h1_from_m1.csv',
        'H4': 'xauusd_h4_from_m1.csv',
        'D1': 'xauusd_d1_from_m1.csv',
        'W1': 'xauusd_w1.csv',  # Optional - will skip if not found
    }

    # Load and compute features for each timeframe
    tf_features = {}

    for tf_name, filename in timeframe_files.items():
        filepath = data_dir / filename

        if not filepath.exists():
            if tf_name == 'W1':
                logger.warning(f"⚠️  {tf_name} file not found, skipping (optional)")
                continue
            else:
                raise FileNotFoundError(f"Required file not found: {filepath}")

        # Load data
        logger.info(f"\n📥 Loading {tf_name} from {filename}...")
        df = load_timeframe_data(filepath)
        logger.info(f"   ✅ Loaded {len(df):,} bars")

        # Compute features
        features = compute_timeframe_features(df, tf_name)
        tf_features[tf_name] = features

    # Align all timeframes to base
    logger.info(f"\n🔄 Aligning all timeframes to {base_timeframe}...")
    aligned_features = align_timeframes(tf_features, base_timeframe)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ TIMEFRAME FEATURES COMPLETE")
    logger.info("="*70)
    logger.info(f"\n📊 Generated timeframes:")

    total_features = 0
    for tf_name, df in aligned_features.items():
        num_features = df.shape[1]
        total_features += num_features
        logger.info(f"   • {tf_name:4} {num_features} features × {len(df):,} bars")

    logger.info(f"\n✅ Total timeframe features: {total_features}")
    logger.info(f"✅ Aligned to: {len(aligned_features[base_timeframe]):,} bars")

    return aligned_features


def test_timeframe_features():
    """
    Test function to verify timeframe features work correctly
    """
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING TIMEFRAME FEATURES")
    logger.info("="*70)

    try:
        # Load and compute all features
        tf_features = load_and_compute_all_timeframes(base_timeframe='M5')

        # Verify
        logger.info("\n✅ All timeframe features loaded successfully!")

        # Check for NaNs
        for tf_name, df in tf_features.items():
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"⚠️  {tf_name}: {nan_count} NaN values found")
            else:
                logger.info(f"✅ {tf_name}: No NaN values")

        return tf_features

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test
    tf_features = test_timeframe_features()

    logger.info("\n" + "="*70)
    logger.info("✅ TIMEFRAME FEATURES MODULE READY")
    logger.info("="*70)

    logger.info("""
📋 USAGE:
    from features.timeframe_features import load_and_compute_all_timeframes

    # Load all timeframes
    tf_features = load_and_compute_all_timeframes(base_timeframe='M5')

    # Access individual timeframes
    m5_features = tf_features['M5']
    h1_features = tf_features['H1']
    """)
