"""
ULTIMATE 150+ FEATURE SYSTEM - MAIN INTEGRATION

This is the master module that combines ALL feature sources into
a complete 150+ feature dataset ready for training.

Feature Breakdown:
- Timeframe features: 96 (16 × 6 timeframes: M5, M15, H1, H4, D1, W1)
- Cross-timeframe: 12
- Macro correlations: 24
- Economic calendar: 8
- Market microstructure: 12
Total: 152 features

This represents the maximum intelligence possible from available data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_ultimate_features(base_timeframe='M5', data_dir='data'):
    """
    Create complete 150+ feature set

    Args:
        base_timeframe: Base timeframe to use ('M5' recommended for speed)
        data_dir: Directory containing data files

    Returns:
        features (ndarray): Shape (N, 152+), dtype float32
        returns (ndarray): Shape (N,), target returns
        timestamps (DatetimeIndex): Shape (N,), timestamps for each sample
    """
    logger.info("="*70)
    logger.info("🚀 ULTIMATE 150+ FEATURE SYSTEM")
    logger.info("="*70)
    logger.info(f"Base timeframe: {base_timeframe}")
    logger.info(f"Data directory: {data_dir}")
    logger.info("")

    # ========== STEP 1: LOAD TIMEFRAME FEATURES (96) ==========
    logger.info("📊 STEP 1/5: Loading timeframe features...")
    logger.info("-" * 70)

    from features.timeframe_features import load_and_compute_all_timeframes

    tf_features = load_and_compute_all_timeframes(
        base_timeframe=base_timeframe,
        data_dir=data_dir
    )

    logger.info(f"✅ Loaded {len(tf_features)} timeframes")
    total_tf_features = sum(df.shape[1] for df in tf_features.values())
    logger.info(f"✅ Total timeframe features: {total_tf_features}")

    # ========== STEP 2: COMPUTE CROSS-TIMEFRAME FEATURES (12) ==========
    logger.info("\n🔄 STEP 2/5: Computing cross-timeframe features...")
    logger.info("-" * 70)

    from features.cross_timeframe import compute_all_cross_tf_features

    cross_tf_features = compute_all_cross_tf_features(tf_features)

    logger.info(f"✅ Cross-timeframe features: {cross_tf_features.shape[1]}")

    # ========== STEP 3: COMPUTE MACRO FEATURES (24) ==========
    logger.info("\n🌍 STEP 3/5: Computing macro features...")
    logger.info("-" * 70)

    from features.macro_features import load_macro_data, compute_macro_features

    # Load base timeframe data with close prices
    base_data_file = {
        'M5': 'xauusd_m5.csv',
        'M15': 'xauusd_m15.csv',
        'H1': 'xauusd_h1_from_m1.csv',
    }.get(base_timeframe, 'xauusd_m5.csv')

    df_gold = pd.read_csv(f"{data_dir}/{base_data_file}")
    df_gold['time'] = pd.to_datetime(df_gold['time'])
    df_gold = df_gold.set_index('time').sort_index()

    macro_data = load_macro_data(data_dir=data_dir)
    macro_features = compute_macro_features(df_gold, macro_data)

    logger.info(f"✅ Macro features: {macro_features.shape[1]}")

    # ========== STEP 4: COMPUTE CALENDAR FEATURES (8) ==========
    logger.info("\n📅 STEP 4/5: Computing economic calendar features...")
    logger.info("-" * 70)

    from features.calendar_features import load_economic_calendar, compute_calendar_features

    calendar = load_economic_calendar(filepath=f"{data_dir}/economic_events_2015_2025.json")

    # Use the base timeframe index
    base_index = tf_features[base_timeframe].index

    calendar_features = compute_calendar_features(base_index, calendar)

    logger.info(f"✅ Calendar features: {calendar_features.shape[1]}")

    # ========== STEP 5: COMPUTE MICROSTRUCTURE FEATURES (12) ==========
    logger.info("\n🏛️  STEP 5/5: Computing market microstructure features...")
    logger.info("-" * 70)

    from features.microstructure_features import compute_all_microstructure_features

    microstructure_features = compute_all_microstructure_features(df_gold)

    logger.info(f"✅ Microstructure features: {microstructure_features.shape[1]}")

    # ========== STEP 6: COMBINE ALL FEATURES ==========
    logger.info("\n🔗 COMBINING ALL FEATURES...")
    logger.info("-" * 70)

    # Align all feature DataFrames to the same index (base_index)
    all_feature_dfs = []

    # Add all timeframe features
    for tf_name in sorted(tf_features.keys()):
        df_tf = tf_features[tf_name]
        df_aligned = df_tf.reindex(base_index, method='ffill')
        all_feature_dfs.append(df_aligned)
        logger.info(f"   • {tf_name}: {df_aligned.shape[1]} features")

    # Add cross-timeframe
    cross_tf_aligned = cross_tf_features.reindex(base_index, method='ffill')
    all_feature_dfs.append(cross_tf_aligned)
    logger.info(f"   • Cross-TF: {cross_tf_aligned.shape[1]} features")

    # Add macro
    macro_aligned = macro_features.reindex(base_index, method='ffill')
    all_feature_dfs.append(macro_aligned)
    logger.info(f"   • Macro: {macro_aligned.shape[1]} features")

    # Add calendar
    calendar_aligned = calendar_features.reindex(base_index, method='ffill')
    all_feature_dfs.append(calendar_aligned)
    logger.info(f"   • Calendar: {calendar_aligned.shape[1]} features")

    # Add microstructure
    micro_aligned = microstructure_features.reindex(base_index, method='ffill')
    all_feature_dfs.append(micro_aligned)
    logger.info(f"   • Microstructure: {micro_aligned.shape[1]} features")

    # Concatenate everything
    all_features = pd.concat(all_feature_dfs, axis=1)

    # ========== STEP 7: CLEAN AND PREPARE ==========
    logger.info("\n🧹 CLEANING DATA...")
    logger.info("-" * 70)

    # Fill any remaining NaNs with 0
    nan_count_before = all_features.isna().sum().sum()
    if nan_count_before > 0:
        logger.info(f"   • Filling {nan_count_before:,} NaN values with 0")
        all_features = all_features.fillna(0.0)

    # Replace inf values
    inf_count = np.isinf(all_features.values).sum()
    if inf_count > 0:
        logger.info(f"   • Replacing {inf_count:,} inf values with 0")
        all_features = all_features.replace([np.inf, -np.inf], 0.0)

    # Convert to float32 for memory efficiency
    all_features = all_features.astype(np.float32)

    # ========== STEP 8: COMPUTE TARGET RETURNS ==========
    logger.info("\n🎯 COMPUTING TARGET RETURNS...")
    logger.info("-" * 70)

    # Use base timeframe close prices for returns
    df_gold_aligned = df_gold.reindex(base_index, method='ffill')
    returns = df_gold_aligned['close'].pct_change().fillna(0.0).values.astype(np.float32)

    logger.info(f"   • Return samples: {len(returns):,}")

    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*70)
    logger.info("✅ ULTIMATE FEATURES CREATED!")
    logger.info("="*70)

    logger.info(f"\n📊 Feature Summary:")
    logger.info(f"   • Total features: {all_features.shape[1]}")
    logger.info(f"   • Total samples: {len(all_features):,}")
    logger.info(f"   • Memory usage: {all_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    logger.info(f"   • Date range: {base_index[0]} to {base_index[-1]}")

    # Feature breakdown
    logger.info(f"\n📈 Feature Breakdown:")

    feature_counts = {
        'Timeframe (M5)': 16,
        'Timeframe (M15)': 16,
        'Timeframe (H1)': 16,
        'Timeframe (H4)': 16,
        'Timeframe (D1)': 16,
        'Timeframe (W1)': 16 if 'W1' in tf_features else 0,
        'Cross-Timeframe': cross_tf_aligned.shape[1],
        'Macro': macro_aligned.shape[1],
        'Calendar': calendar_aligned.shape[1],
        'Microstructure': micro_aligned.shape[1],
    }

    for name, count in feature_counts.items():
        if count > 0:
            logger.info(f"   • {name:20} {count:3} features")

    logger.info(f"\n🎯 Ready for training!")
    logger.info(f"   • Observation space: {all_features.shape[1]} features")
    logger.info(f"   • Action space: 3 (buy/hold/sell)")
    logger.info(f"   • Training samples: {len(all_features):,}")

    # Return as numpy arrays
    return (
        all_features.values,  # Features (N, 152+)
        returns,              # Returns (N,)
        all_features.index    # Timestamps (N,)
    )


def test_ultimate_features():
    """
    Quick test to verify the complete system works
    """
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING ULTIMATE FEATURE SYSTEM")
    logger.info("="*70)

    try:
        # Generate features
        X, r, timestamps = make_ultimate_features(base_timeframe='M5')

        logger.info("\n✅ Ultimate feature system test PASSED!")

        logger.info(f"\n📊 Output shapes:")
        logger.info(f"   • Features (X): {X.shape}")
        logger.info(f"   • Returns (r): {r.shape}")
        logger.info(f"   • Timestamps: {len(timestamps)}")

        logger.info(f"\n📈 Feature statistics:")
        logger.info(f"   • Mean: {X.mean():.6f}")
        logger.info(f"   • Std: {X.std():.6f}")
        logger.info(f"   • Min: {X.min():.6f}")
        logger.info(f"   • Max: {X.max():.6f}")
        logger.info(f"   • NaN count: {np.isnan(X).sum()}")
        logger.info(f"   • Inf count: {np.isinf(X).sum()}")

        logger.info(f"\n🎯 Return statistics:")
        logger.info(f"   • Mean return: {r.mean():.6f}")
        logger.info(f"   • Std return: {r.std():.6f}")
        logger.info(f"   • Sharpe (approx): {r.mean() / (r.std() + 1e-8):.4f}")

        return X, r, timestamps

    except Exception as e:
        logger.error(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run full system test
    X, r, timestamps = test_ultimate_features()

    logger.info("\n" + "="*70)
    logger.info("🚀 ULTIMATE 150+ FEATURE SYSTEM READY!")
    logger.info("="*70)

    logger.info("""
📋 USAGE IN TRAINING:
    from features.ultimate_150_features import make_ultimate_features

    # Generate all 150+ features
    X, returns, timestamps = make_ultimate_features(base_timeframe='M5')

    # X is now ready for training with shape (N, 152+)
    # Use with DreamerV3 or any RL algorithm

    # Example:
    # env = TradingEnvironment(X, returns)
    # agent.train(env, steps=1000000)
    """)

    logger.info("\n🎉 You're ready to train the GOD MODE AI!")
