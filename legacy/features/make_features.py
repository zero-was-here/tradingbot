# features/make_features.py
import numpy as np
import pandas as pd
from data.load_data import load_ohlc_csv

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_features(df):
    df = df.copy()
    
    # 1. Gold Features
    df["ret"] = np.log(df["close"]).diff().fillna(0.0)
    df["vol"] = df["ret"].rolling(24).std().fillna(0.0)         
    df["mom"] = df["close"].pct_change(24).fillna(0.0)          
    
    # Moving Averages
    df["ma_fast"] = df["close"].rolling(24).mean()
    df["ma_slow"] = df["close"].rolling(120).mean()
    df["ma_diff"] = ((df["ma_fast"] - df["ma_slow"]) / df["close"]).fillna(0.0)
    
    df["rsi"] = compute_rsi(df["close"], period=14) / 100.0

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_diff"] = (macd - signal) / df["close"]
    df["macd_diff"] = df["macd_diff"].fillna(0.0)

    # 2. MACRO FEATURES (The "God Mode" Inputs)
    if "dxy_close" in df.columns:
        # DXY Returns
        df["dxy_ret"] = np.log(df["dxy_close"]).diff().fillna(0.0)
        # SPX Returns
        df["spx_ret"] = np.log(df["spx_close"]).diff().fillna(0.0)
        # US10Y Change
        df["us10y_chg"] = df["us10y_close"].diff().fillna(0.0)
        
        # Correlations (Is Gold moving with or against the Dollar?)
        # 24h rolling correlation
        df["corr_dxy"] = df["ret"].rolling(24).corr(df["dxy_ret"]).fillna(0.0)
        df["corr_spx"] = df["ret"].rolling(24).corr(df["spx_ret"]).fillna(0.0)
        
        feature_cols = [
            "ret", "vol", "mom", "ma_diff", "rsi", "macd_diff",
            "dxy_ret", "spx_ret", "us10y_chg", "corr_dxy", "corr_spx"
        ]
    else:
        # Fallback if macro data missing
        feature_cols = ["ret", "vol", "mom", "ma_diff", "rsi", "macd_diff"]

    # Clean data
    df = df.iloc[120:].reset_index(drop=True)

    feats = df[feature_cols].to_numpy(dtype=np.float32)
    rets = df["ret"].to_numpy(dtype=np.float32)

    # Force cleanup
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    mu = feats.mean(axis=0, keepdims=True)
    sig = feats.std(axis=0, keepdims=True) + 1e-8
    feats = (feats - mu) / sig
    
    return df, feats, rets

def make_features(csv_path: str, window: int = 64):
    df = load_ohlc_csv(csv_path)
    return compute_features(df)
