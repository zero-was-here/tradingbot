import yfinance as yf
import pandas as pd
import os

def fetch_data():
    print("🚀 Fetching Macro Data...")
    
    # 1. Gold Futures (Reference)
    # 2. DXY (US Dollar Index) - Critical for Gold
    # 3. SPX (S&P 500) - Risk Sentiment
    # 4. US10Y (Treasury Yields) - Opportunity Cost of Gold
    
    tickers = {
        "GC=F": "gold_futures", 
        "DX-Y.NYB": "dxy", 
        "^GSPC": "spx", 
        "^TNX": "us10y"
    }
    
    for symbol, name in tickers.items():
        print(f"📥 Downloading {name} ({symbol})...")
        try:
            # FORCE DAILY DATA to get long history (15 years)
            # We will forward-fill this to hourly later
            df = yf.download(symbol, period="15y", interval="1d", progress=False)
            
            if not df.empty:
                # Flat columns if MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Standardize
                df = df.reset_index()
                # Rename columns to lowercase
                df.columns = [c.lower() for c in df.columns]
                
                # Rename 'date' or 'datetime' to 'time'
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'time'})
                elif 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'time'})
                
                # Save
                path = f"data/{name}.csv"
                df.to_csv(path, index=False)
                print(f"✅ Saved {path} ({len(df)} rows)")
            else:
                print(f"❌ Failed to download {name}")
                
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")

if __name__ == "__main__":
    fetch_data()
