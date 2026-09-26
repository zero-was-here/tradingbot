"""
Cross-Timeframe Intelligence Module

Computes 12 advanced features that capture relationships across timeframes:
- Trend Alignment (3): Multi-TF trend agreement
- Momentum Cascade (3): Higher → Lower TF momentum flow
- Volatility Regime (3): Vol clustering and spikes
- Pattern Confluence (3): Support/resistance alignment

These features capture the hierarchical structure of market dynamics.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_trend_alignment(tf_dict):
    """
    Compute trend alignment across all timeframes

    Returns 3 features:
    - trend_alignment_all: Overall trend agreement (-1 to +1)
    - trend_strength_cascade: Trend strength from higher to lower TF
    - trend_divergence: Conflicting trends score
    """
    logger.info("Computing trend alignment features...")

    result = pd.DataFrame(index=tf_dict['M5'].index)

    # Extract trend from each timeframe
    trends = {}
    for tf_name in ['M5', 'M15', 'H1', 'H4', 'D1']:
        if tf_name in tf_dict:
            trend_col = f'{tf_name}_trend'
            if trend_col in tf_dict[tf_name].columns:
                trends[tf_name] = tf_dict[tf_name][trend_col]

    # Feature 1: Overall trend alignment
    # Average of all trends (-1 to +1)
    if trends:
        trend_df = pd.DataFrame(trends)
        result['trend_alignment_all'] = trend_df.mean(axis=1)
    else:
        result['trend_alignment_all'] = 0.0

    # Feature 2: Trend strength cascade
    # Multiply trends from higher to lower TF (amplifies agreement)
    cascade_tfs = ['D1', 'H4', 'H1', 'M15']
    cascade = 1.0
    for tf in cascade_tfs:
        if tf in trends:
            cascade = cascade * trends[tf]
    result['trend_strength_cascade'] = cascade

    # Feature 3: Trend divergence
    # Standard deviation of trends (higher = more divergence)
    if trends:
        result['trend_divergence'] = trend_df.std(axis=1)
    else:
        result['trend_divergence'] = 0.0

    return result


def compute_momentum_cascade(tf_dict):
    """
    Compute momentum cascade across timeframes

    Returns 3 features:
    - momentum_d1_h1: Daily × Hourly momentum interaction
    - momentum_h4_h1: 4H × Hourly momentum interaction
    - momentum_h1_m15: Hourly × 15min momentum interaction
    """
    logger.info("Computing momentum cascade features...")

    result = pd.DataFrame(index=tf_dict['M5'].index)

    # Helper function to get momentum
    def get_momentum(tf_name, period='10'):
        mom_col = f'{tf_name}_momentum_{period}'
        if tf_name in tf_dict and mom_col in tf_dict[tf_name].columns:
            return tf_dict[tf_name][mom_col]
        return pd.Series(0.0, index=result.index)

    # Feature 1: D1 × H1 momentum
    d1_mom = get_momentum('D1', '10')
    h1_mom = get_momentum('H1', '10')
    result['momentum_d1_h1'] = d1_mom * h1_mom

    # Feature 2: H4 × H1 momentum
    h4_mom = get_momentum('H4', '10')
    result['momentum_h4_h1'] = h4_mom * h1_mom

    # Feature 3: H1 × M15 momentum
    m15_mom = get_momentum('M15', '10')
    result['momentum_h1_m15'] = h1_mom * m15_mom

    return result


def compute_volatility_regime(tf_dict):
    """
    Detect volatility regime changes across timeframes

    Returns 3 features:
    - volatility_regime: Current vol vs long-term average
    - volatility_spike: Sudden volatility increase detection
    - volatility_compression: Low volatility (breakout pending)
    """
    logger.info("Computing volatility regime features...")

    result = pd.DataFrame(index=tf_dict['M5'].index)

    # Use H1 volatility as primary
    if 'H1' in tf_dict and 'H1_volatility' in tf_dict['H1'].columns:
        vol = tf_dict['H1']['H1_volatility']

        # Feature 1: Current vol vs long-term (100-period average)
        vol_longterm = vol.rolling(100).mean()
        result['volatility_regime'] = vol / (vol_longterm + 1e-8)

        # Feature 2: Volatility spike (current > 2x recent average)
        vol_recent = vol.rolling(20).mean()
        result['volatility_spike'] = (vol > 2.0 * vol_recent).astype(float)

        # Feature 3: Volatility compression (current < 0.5x recent average)
        result['volatility_compression'] = (vol < 0.5 * vol_recent).astype(float)

    else:
        result['volatility_regime'] = 1.0
        result['volatility_spike'] = 0.0
        result['volatility_compression'] = 0.0

    return result


def compute_pattern_confluence(tf_dict):
    """
    Detect support/resistance confluence across timeframes

    Returns 3 features:
    - support_confluence: Multiple TFs near support
    - resistance_confluence: Multiple TFs near resistance
    - breakout_alignment: All TFs confirm breakout
    """
    logger.info("Computing pattern confluence features...")

    result = pd.DataFrame(index=tf_dict['M5'].index)

    # Collect distance to high/low from each timeframe
    dist_to_high = []
    dist_to_low = []

    for tf_name in ['M5', 'M15', 'H1', 'H4', 'D1']:
        if tf_name in tf_dict:
            high_col = f'{tf_name}_dist_to_high'
            low_col = f'{tf_name}_dist_to_low'

            if high_col in tf_dict[tf_name].columns:
                dist_to_high.append(tf_dict[tf_name][high_col])
            if low_col in tf_dict[tf_name].columns:
                dist_to_low.append(tf_dict[tf_name][low_col])

    # Feature 1: Support confluence
    # Count how many TFs are near support (within 2% of recent low)
    if dist_to_low:
        near_support = sum((d > -0.02).astype(float) for d in dist_to_low)
        result['support_confluence'] = near_support / len(dist_to_low)
    else:
        result['support_confluence'] = 0.0

    # Feature 2: Resistance confluence
    # Count how many TFs are near resistance (within 2% of recent high)
    if dist_to_high:
        near_resistance = sum((d < 0.02).astype(float) for d in dist_to_high)
        result['resistance_confluence'] = near_resistance / len(dist_to_high)
    else:
        result['resistance_confluence'] = 0.0

    # Feature 3: Breakout alignment
    # All TFs showing momentum in same direction
    momentum_signals = []
    for tf_name in ['M15', 'H1', 'H4']:
        if tf_name in tf_dict:
            mom_col = f'{tf_name}_momentum_10'
            if mom_col in tf_dict[tf_name].columns:
                momentum_signals.append(tf_dict[tf_name][mom_col])

    if momentum_signals:
        mom_df = pd.DataFrame(momentum_signals).T
        # Breakout = all positive or all negative
        all_positive = (mom_df > 0).all(axis=1).astype(float)
        all_negative = (mom_df < 0).all(axis=1).astype(float)
        result['breakout_alignment'] = all_positive - all_negative
    else:
        result['breakout_alignment'] = 0.0

    return result


def compute_all_cross_tf_features(tf_dict):
    """
    Main function: Compute all 12 cross-timeframe features

    Args:
        tf_dict: Dict of feature DataFrames from timeframe_features module
                 {tf_name: df_features}

    Returns:
        DataFrame with 12 cross-TF features
    """
    logger.info("="*70)
    logger.info("🔄 COMPUTING CROSS-TIMEFRAME FEATURES")
    logger.info("="*70)

    # Compute each feature group
    trend_features = compute_trend_alignment(tf_dict)
    momentum_features = compute_momentum_cascade(tf_dict)
    volatility_features = compute_volatility_regime(tf_dict)
    pattern_features = compute_pattern_confluence(tf_dict)

    # Combine all features
    all_features = pd.concat([
        trend_features,
        momentum_features,
        volatility_features,
        pattern_features
    ], axis=1)

    # Fill any NaNs
    all_features = all_features.fillna(0.0)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ CROSS-TIMEFRAME FEATURES COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Generated {all_features.shape[1]} cross-TF features")
    logger.info(f"✅ Feature count: {len(all_features):,} bars")

    # List features
    logger.info("\n📊 Features created:")
    for col in all_features.columns:
        logger.info(f"   • {col}")

    return all_features


def test_cross_tf_features():
    """
    Test function to verify cross-timeframe features work correctly
    """
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING CROSS-TIMEFRAME FEATURES")
    logger.info("="*70)

    try:
        # Load timeframe features first
        from features.timeframe_features import load_and_compute_all_timeframes

        logger.info("\n1️⃣ Loading timeframe features...")
        tf_features = load_and_compute_all_timeframes(base_timeframe='M5')

        logger.info("\n2️⃣ Computing cross-timeframe features...")
        cross_tf_features = compute_all_cross_tf_features(tf_features)

        logger.info("\n✅ Cross-timeframe features computed successfully!")

        # Check for NaNs
        nan_count = cross_tf_features.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️  {nan_count} NaN values found")
        else:
            logger.info("✅ No NaN values")

        # Show sample
        logger.info("\n📊 Sample data:")
        logger.info(cross_tf_features.head())

        return cross_tf_features

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test
    cross_tf = test_cross_tf_features()

    logger.info("\n" + "="*70)
    logger.info("✅ CROSS-TIMEFRAME MODULE READY")
    logger.info("="*70)

    logger.info("""
📋 USAGE:
    from features.timeframe_features import load_and_compute_all_timeframes
    from features.cross_timeframe import compute_all_cross_tf_features

    # Load timeframe features
    tf_features = load_and_compute_all_timeframes()

    # Compute cross-timeframe features
    cross_tf = compute_all_cross_tf_features(tf_features)
    """)
