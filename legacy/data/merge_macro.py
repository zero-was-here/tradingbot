import pandas as pd
from data.load_data import load_ohlc_csv

def merge_data():
    print("🔄 Merging Macro Data (Daily -> Hourly)...")
    
    # 1. Load Master (Gold from Broker)
    master = load_ohlc_csv("data/xauusd_1h.csv")
    
    # Ensure master time is naive for easy merging
    master["time"] = pd.to_datetime(master["time"]).dt.tz_localize(None)
    master = master.sort_values("time").set_index("time")
    
    print(f"Master (XAUUSD): {len(master)} rows")

    # 2. Load Aux Data
    aux_files = {
        "dxy": "data/dxy.csv",
        "spx": "data/spx.csv",
        "us10y": "data/us10y.csv"
    }
    
    for name, path in aux_files.items():
        try:
            # Load Daily Data
            df = pd.read_csv(path)
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df = df.set_index("time").sort_index()
            
            # Keep Close only
            df = df[["close"]].rename(columns={"close": f"{name}_close"})
            
            # Upsample to Hourly to match Master
            # Reindex to the Master's index (timestamps)
            # method='ffill' propagates the last known daily price to the current hour
            aligned = df.reindex(master.index, method='ffill')
            
            # Merge
            master = master.merge(aligned, left_index=True, right_index=True, how="left")
            
            # Fill any remaining gaps (e.g. holidays)
            master[f"{name}_close"] = master[f"{name}_close"].ffill().bfill()
            
            print(f"✅ Merged {name}")
            
        except Exception as e:
            print(f"❌ Failed to merge {name}: {e}")

    # Drop any rows that are still NaN (should be none due to bfill)
    master = master.dropna()
    
    # Save
    master = master.reset_index()
    output_path = "data/xauusd_1h_macro.csv"
    master.to_csv(output_path, index=False)
    print(f"🎉 Saved Macro Dataset to {output_path} ({len(master)} rows)")
    print(master.head())

if __name__ == "__main__":
    merge_data()
