"""
Multi-Timeframe Feature Engineering

Trading on a single timeframe is like looking through a straw.
You need the full picture across multiple timeframes.

Timeframes:
- M5 (5-minute): Intraday momentum, entry timing
- M15 (15-minute): Short-term trends
- H1 (1-hour): Current (existing)
- H4 (4-hour): Swing trading context
- D1 (Daily): Major trend direction
- W1 (Weekly): Structural support/resistance

This gives the AI:
- M5: "Is this a good entry right now?"
- H1: "What's the current trend?"
- D1: "What's the big picture direction?"
- All aligned → Strong signal
- Conflicting → Weak signal / don't trade
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTimeframeFeatures:
    """
    Create features across multiple timeframes

    Like a professional trader checking:
    - M5 for entry timing
    - H1 for current trend
    - H4 for swing context
    - D1 for major direction
    """

    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1']

    def __init__(self, timeframes=None):
        """
        Args:
            timeframes: List of timeframes to use (default: M5, M15, H1, H4, D1)
        """
        self.timeframes = timeframes or self.TIMEFRAMES

        logger.info(f"📊 Multi-Timeframe Features initialized")
        logger.info(f"   Timeframes: {', '.join(self.timeframes)}")

    def create_features(self, data_dict):
        """
        Create multi-timeframe features

        Args:
            data_dict: {'M5': df_m5, 'H1': df_h1, 'H4': df_h4, 'D1': df_d1}

        Returns:
            DataFrame with multi-timeframe features aligned to fastest timeframe
        """

        logger.info("📊 Creating multi-timeframe features...")

        # Process each timeframe
        tf_features = {}

        for tf in self.timeframes:
            if tf not in data_dict:
                logger.warning(f"⚠️ {tf} data not found, skipping")
                continue

            df = data_dict[tf]

            # Compute features for this timeframe
            feats = self._compute_tf_features(df, tf)

            # Add timeframe prefix to all columns
            for col in feats.columns:
                tf_features[f'{tf}_{col}'] = feats[col]

        # Combine all features
        all_features = pd.DataFrame(tf_features)

        # Add cross-timeframe features
        cross_features = self._compute_cross_tf_features(data_dict)

        for col, values in cross_features.items():
            all_features[col] = values

        logger.info(f"✅ Created {len(all_features.columns)} multi-timeframe features")

        return all_features

    def _compute_tf_features(self, df, timeframe):
        """
        Compute standard features for one timeframe

        Args:
            df: DataFrame for this timeframe
            timeframe: 'M5', 'H1', etc.

        Returns:
            DataFrame with features
        """

        feats = pd.DataFrame(index=df.index)

        # Price action
        feats['close'] = df['close']
        feats['ret'] = df['close'].pct_change()
        feats['vol'] = feats['ret'].rolling(20).std()
        feats['mom_5'] = df['close'].pct_change(5)
        feats['mom_10'] = df['close'].pct_change(10)
        feats['mom_20'] = df['close'].pct_change(20)

        # Moving averages (scaled to timeframe)
        window_fast = self._get_ma_window(timeframe, 'fast')
        window_slow = self._get_ma_window(timeframe, 'slow')

        feats['ma_fast'] = df['close'].rolling(window_fast).mean()
        feats['ma_slow'] = df['close'].rolling(window_slow).mean()
        feats['ma_diff'] = (feats['ma_fast'] - feats['ma_slow']) / df['close']

        # Trend strength
        feats['trend'] = np.where(feats['ma_fast'] > feats['ma_slow'], 1, -1)
        feats['trend_strength'] = abs(feats['ma_diff'])

        # RSI
        feats['rsi'] = self._compute_rsi(df['close'], period=14)

        # ATR (volatility)
        feats['atr'] = self._compute_atr(df, period=14)
        feats['atr_pct'] = feats['atr'] / df['close']

        # Bollinger Bands
        bb_period = 20
        bb_std = df['close'].rolling(bb_period).std()
        bb_mid = df['close'].rolling(bb_period).mean()
        feats['bb_upper'] = bb_mid + 2 * bb_std
        feats['bb_lower'] = bb_mid - 2 * bb_std
        feats['bb_position'] = (df['close'] - feats['bb_lower']) / (feats['bb_upper'] - feats['bb_lower'])

        # Volume (if available)
        if 'volume' in df.columns:
            feats['volume'] = df['volume']
            feats['volume_ma'] = df['volume'].rolling(20).mean()
            feats['volume_ratio'] = df['volume'] / feats['volume_ma']
        else:
            feats['volume'] = 0
            feats['volume_ratio'] = 1.0

        return feats

    def _compute_cross_tf_features(self, data_dict):
        """
        Cross-timeframe features

        Examples:
        - Trend alignment (all TFs pointing same direction)
        - Momentum cascade (higher TF momentum filtering to lower)
        - Volatility regime comparison
        """

        cross_features = {}

        # 1. Trend Alignment
        trend_alignment = self._trend_alignment(data_dict)
        cross_features['trend_alignment'] = trend_alignment

        # 2. Momentum Cascade
        momentum_cascade = self._momentum_cascade(data_dict)
        cross_features['momentum_cascade'] = momentum_cascade

        # 3. Volatility Regime
        vol_regime = self._volatility_regime(data_dict)
        cross_features['volatility_regime'] = vol_regime

        # 4. Multi-TF Support/Resistance
        sr_confluence = self._support_resistance_confluence(data_dict)
        cross_features['sr_confluence'] = sr_confluence

        return cross_features

    def _trend_alignment(self, data_dict):
        """
        Check if trends align across timeframes

        Strong signal: All TFs bullish or all bearish
        Weak signal: Conflicting trends
        """

        trends = {}

        for tf, df in data_dict.items():
            if len(df) < 50:
                continue

            # Fast vs slow MA
            ma_fast = df['close'].rolling(20).mean()
            ma_slow = df['close'].rolling(50).mean()

            # Current trend
            if len(ma_fast) > 0 and len(ma_slow) > 0:
                trends[tf] = 1.0 if ma_fast.iloc[-1] > ma_slow.iloc[-1] else -1.0
            else:
                trends[tf] = 0.0

        if not trends:
            return pd.Series([0.0] * len(list(data_dict.values())[0]))

        # Alignment score: -1 (all bearish) to +1 (all bullish)
        alignment = sum(trends.values()) / len(trends)

        # Broadcast to series
        first_df = list(data_dict.values())[0]
        return pd.Series([alignment] * len(first_df), index=first_df.index)

    def _momentum_cascade(self, data_dict):
        """
        Higher timeframe momentum filtering to lower timeframe

        If D1 is bullish, H1 long signals are stronger
        If D1 is bearish, H1 long signals are weaker
        """

        if 'D1' not in data_dict or 'H1' not in data_dict:
            first_df = list(data_dict.values())[0]
            return pd.Series([0.0] * len(first_df), index=first_df.index)

        # D1 momentum
        d1_mom = data_dict['D1']['close'].pct_change(5).fillna(0)

        # H1 momentum
        h1_mom = data_dict['H1']['close'].pct_change(5).fillna(0)

        # Cascade: D1 momentum * H1 momentum
        # Both positive → strong positive
        # Opposite signs → weak
        cascade = d1_mom * h1_mom

        return cascade

    def _volatility_regime(self, data_dict):
        """
        Compare current volatility to higher timeframe

        If H1 vol > D1 vol → High volatility regime
        """

        if 'H1' not in data_dict:
            first_df = list(data_dict.values())[0]
            return pd.Series([1.0] * len(first_df), index=first_df.index)

        h1_df = data_dict['H1']

        # Current volatility (20-period)
        h1_vol = h1_df['close'].pct_change().rolling(20).std()

        # Long-term volatility (100-period)
        h1_vol_long = h1_df['close'].pct_change().rolling(100).std()

        # Regime: current / long-term
        vol_regime = h1_vol / h1_vol_long

        return vol_regime.fillna(1.0)

    def _support_resistance_confluence(self, data_dict):
        """
        Multi-timeframe support/resistance confluence

        When price is at support on multiple timeframes → Strong support
        """

        # Simplified version - check if price is near recent high/low across TFs
        confluence = pd.Series([0.0] * len(list(data_dict.values())[0]),
                              index=list(data_dict.values())[0].index)

        for tf, df in data_dict.items():
            if len(df) < 50:
                continue

            # Recent high/low
            high_50 = df['high'].rolling(50).max()
            low_50 = df['low'].rolling(50).min()

            current_price = df['close']

            # Distance to high/low as percentage
            dist_to_high = (high_50 - current_price) / current_price
            dist_to_low = (current_price - low_50) / current_price

            # Near support/resistance (within 1%)
            near_resistance = (dist_to_high < 0.01).astype(float)
            near_support = (dist_to_low < 0.01).astype(float)

            # Add to confluence
            confluence += near_resistance - near_support

        return confluence

    def _get_ma_window(self, timeframe, speed):
        """Get appropriate MA window for timeframe"""

        windows = {
            'M5': {'fast': 20, 'slow': 50},
            'M15': {'fast': 20, 'slow': 50},
            'H1': {'fast': 24, 'slow': 120},
            'H4': {'fast': 24, 'slow': 100},
            'D1': {'fast': 20, 'slow': 50},
            'W1': {'fast': 10, 'slow': 26},
        }

        return windows.get(timeframe, {'fast': 20, 'slow': 50})[speed]

    def _compute_rsi(self, prices, period=14):
        """Compute RSI indicator"""

        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def _compute_atr(self, df, period=14):
        """Compute Average True Range"""

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()

        return atr.fillna(method='bfill')


# Helper function to resample data to different timeframes
def create_multi_timeframe_data(df_base, base_tf='H1'):
    """
    Create multiple timeframe datasets from base data

    Args:
        df_base: Base DataFrame (e.g., H1 data)
        base_tf: Base timeframe ('H1')

    Returns:
        dict with {'M5': df_m5, 'H1': df_h1, 'H4': df_h4, 'D1': df_d1}
    """

    logger.info(f"📊 Creating multi-timeframe datasets from {base_tf}...")

    # Ensure datetime index
    if 'time' in df_base.columns:
        df_base = df_base.set_index('time')

    df_base.index = pd.to_datetime(df_base.index)

    # Resample rules
    resample_rules = {
        'M5': '5T',   # 5 minutes
        'M15': '15T', # 15 minutes
        'H1': '1H',   # 1 hour
        'H4': '4H',   # 4 hours
        'D1': '1D',   # 1 day
    }

    data_dict = {}

    for tf, rule in resample_rules.items():
        # Resample OHLCV
        resampled = df_base.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df_base.columns else lambda x: 0,
        }).dropna()

        data_dict[tf] = resampled

        logger.info(f"   {tf}: {len(resampled)} bars")

    return data_dict


# Example usage
if __name__ == "__main__":
    print("📊 Multi-Timeframe Features Demo\n")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')

    df_h1 = pd.DataFrame({
        'time': dates,
        'open': 2000 + np.random.randn(1000).cumsum() * 10,
        'high': 2000 + np.random.randn(1000).cumsum() * 10 + 5,
        'low': 2000 + np.random.randn(1000).cumsum() * 10 - 5,
        'close': 2000 + np.random.randn(1000).cumsum() * 10,
        'volume': np.random.randint(1000, 10000, 1000),
    })

    # Create multi-timeframe data
    data_dict = create_multi_timeframe_data(df_h1, base_tf='H1')

    # Create features
    mtf = MultiTimeframeFeatures()
    features = mtf.create_features(data_dict)

    print(f"✅ Created {len(features.columns)} features across {len(data_dict)} timeframes")
    print(f"\nSample features:")
    print(features.columns.tolist()[:20])

    print(f"\nCross-timeframe features:")
    cross_features = [c for c in features.columns if not c.startswith(('M5_', 'H1_', 'H4_', 'D1_'))]
    print(cross_features)

    print("\n✅ Multi-timeframe feature engineering working!")
