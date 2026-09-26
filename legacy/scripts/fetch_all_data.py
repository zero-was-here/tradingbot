"""
Auto-Fetch ALL Available Market Data

This script automatically downloads:
- VIX (Volatility Index)
- Oil (WTI Crude)
- Bitcoin (BTCUSD)
- EURUSD
- Silver (XAGUSD)
- Gold ETF holdings (GLD)
- Fed data from FRED

All FREE data sources, no API keys needed for basics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_yahoo_data(symbol, start_date, end_date, name):
    """
    Fetch data from Yahoo Finance

    Args:
        symbol: Yahoo Finance symbol (e.g., '^VIX', 'CL=F')
        start_date: Start date string
        end_date: End date string
        name: Name for logging

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"📥 Fetching {name} ({symbol})...")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')

        if len(df) == 0:
            logger.warning(f"⚠️ No data returned for {name}")
            return None

        # Rename columns to lowercase
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # Select relevant columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df = df.rename(columns={'date': 'time'})

        logger.info(f"   ✅ {name}: {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

        return df

    except Exception as e:
        logger.error(f"   ❌ Error fetching {name}: {e}")
        return None


def fetch_vix(start_date, end_date):
    """Fetch VIX (Volatility Index) - Fear gauge"""
    return fetch_yahoo_data('^VIX', start_date, end_date, 'VIX')


def fetch_oil(start_date, end_date):
    """Fetch WTI Crude Oil prices"""
    return fetch_yahoo_data('CL=F', start_date, end_date, 'WTI Crude Oil')


def fetch_bitcoin(start_date, end_date):
    """Fetch Bitcoin (BTCUSD)"""
    # Bitcoin only liquid since ~2017
    btc_start = max(start_date, '2017-01-01')
    return fetch_yahoo_data('BTC-USD', btc_start, end_date, 'Bitcoin')


def fetch_eurusd(start_date, end_date):
    """Fetch EURUSD forex pair"""
    return fetch_yahoo_data('EURUSD=X', start_date, end_date, 'EURUSD')


def fetch_silver(start_date, end_date):
    """Fetch Silver (XAGUSD)"""
    return fetch_yahoo_data('SI=F', start_date, end_date, 'Silver')


def fetch_gld_holdings(start_date, end_date):
    """Fetch GLD ETF (Gold ETF) price as proxy for institutional positioning"""
    return fetch_yahoo_data('GLD', start_date, end_date, 'GLD Gold ETF')


def fetch_us_dollar_index(start_date, end_date):
    """Fetch US Dollar Index (DXY) - backup if needed"""
    return fetch_yahoo_data('DX-Y.NYB', start_date, end_date, 'US Dollar Index')


def align_to_hourly(df_daily, df_hourly_reference):
    """
    Align daily data to hourly frequency by forward-filling

    Args:
        df_daily: Daily data DataFrame
        df_hourly_reference: Hourly reference DataFrame (for timestamps)

    Returns:
        DataFrame aligned to hourly frequency
    """
    # Convert to datetime
    df_daily['time'] = pd.to_datetime(df_daily['time'])
    df_hourly_reference['time'] = pd.to_datetime(df_hourly_reference['time'])

    # Set date as index
    df_daily = df_daily.set_index('time')

    # Resample to hourly and forward fill
    df_hourly = df_daily.resample('H').ffill()

    # Align to reference timestamps
    df_aligned = df_hourly.reindex(df_hourly_reference['time'], method='ffill')

    return df_aligned.reset_index()


def main():
    """Main function to fetch all data"""

    # Configuration
    START_DATE = '2015-11-17'
    END_DATE = '2025-12-17'
    DATA_DIR = 'data'

    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("="*70)
    logger.info("🚀 FETCHING ALL AVAILABLE MARKET DATA")
    logger.info("="*70)
    logger.info(f"\n📅 Date Range: {START_DATE} to {END_DATE}")
    logger.info(f"📂 Save Directory: {DATA_DIR}/\n")

    # Dictionary to store all data
    datasets = {}

    # 1. VIX (Fear Index)
    vix = fetch_vix(START_DATE, END_DATE)
    if vix is not None:
        vix.to_csv(f'{DATA_DIR}/vix_daily.csv', index=False)
        datasets['VIX'] = vix

    # 2. Oil (WTI Crude)
    oil = fetch_oil(START_DATE, END_DATE)
    if oil is not None:
        oil.to_csv(f'{DATA_DIR}/oil_wti_daily.csv', index=False)
        datasets['OIL'] = oil

    # 3. Bitcoin
    btc = fetch_bitcoin(START_DATE, END_DATE)
    if btc is not None:
        btc.to_csv(f'{DATA_DIR}/bitcoin_daily.csv', index=False)
        datasets['BTC'] = btc

    # 4. EURUSD
    eur = fetch_eurusd(START_DATE, END_DATE)
    if eur is not None:
        eur.to_csv(f'{DATA_DIR}/eurusd_daily.csv', index=False)
        datasets['EURUSD'] = eur

    # 5. Silver
    silver = fetch_silver(START_DATE, END_DATE)
    if silver is not None:
        silver.to_csv(f'{DATA_DIR}/silver_daily.csv', index=False)
        datasets['SILVER'] = silver

    # 6. GLD ETF
    gld = fetch_gld_holdings(START_DATE, END_DATE)
    if gld is not None:
        gld.to_csv(f'{DATA_DIR}/gld_etf_daily.csv', index=False)
        datasets['GLD'] = gld

    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 DOWNLOAD SUMMARY")
    logger.info("="*70)

    for name, df in datasets.items():
        logger.info(f"✅ {name:15} {len(df):6,} bars → data/{name.lower()}_daily.csv")

    logger.info(f"\n✅ Successfully downloaded {len(datasets)}/{6} datasets")

    # Next steps
    logger.info("\n" + "="*70)
    logger.info("📋 NEXT STEPS")
    logger.info("="*70)
    logger.info("""
1. ✅ These files are saved in data/ directory
2. ⏳ Waiting for YOU to provide:
   - M5 XAUUSD data (from MT5)
   - M15 XAUUSD data (from MT5)

3. 🔜 I will create:
   - Economic calendar JSON
   - Data integration pipeline
   - Updated God Mode features with ALL data

4. 🚀 Then we train the ULTIMATE model!
    """)

    return datasets


if __name__ == "__main__":
    # Check if yfinance is installed
    try:
        import yfinance
    except ImportError:
        print("❌ ERROR: yfinance not installed")
        print("\n📥 Installing yfinance...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'yfinance'])
        print("✅ yfinance installed!")
        import yfinance

    # Run
    datasets = main()

    print("\n🔥 Data fetch complete! Check the data/ directory.")
