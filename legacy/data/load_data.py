from pathlib import Path
import pandas as pd


def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python")  # auto-detect tab/comma

    # ---- Rename MT5 angle-bracket columns ----
    rename_map = {
        "<DATE>": "date",
        "<TIME>": "clock",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "tick_volume",  # Standardized name
        "<VOL>": "volume",
        "<SPREAD>": "spread",
    }
    df = df.rename(columns=rename_map)

    # Check if we already have "time" column (macro data)
    if "time" not in df.columns:
        required = {"date", "clock", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

        # ---- Combine date + time ----
        df["time"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["clock"].astype(str),
            errors="coerce"
        )
    else:
        # Already has time column (e.g., macro data)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # ---- Ensure numeric OHLC ----
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Ensure tick_volume is numeric if present
    if "tick_volume" in df.columns:
        df["tick_volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0)

    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()

    # ---- Sort & dedupe ----
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    # ---- Sanity checks ----
    if not (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all():
        raise ValueError("Invalid OHLC: high < max(open, close, low)")
    if not (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all():
        raise ValueError("Invalid OHLC: low > min(open, close, high)")

    # Optional: keep only columns we need right now
    cols_to_keep = ["time", "open", "high", "low", "close"]
    if "tick_volume" in df.columns:
        cols_to_keep.append("tick_volume")

    # Keep macro columns if they exist
    for col in df.columns:
        if col.endswith("_close") or col.endswith("_ret") or col.endswith("_chg"):
            if col not in cols_to_keep:
                cols_to_keep.append(col)

    out = df[[c for c in cols_to_keep if c in df.columns]].copy()
    return out


if __name__ == "__main__":
    df = load_ohlc_csv("data/xauusd_1h.csv")
    print(df.head())
    print(df.tail())
    print("-" * 40)
    print("Rows:", len(df))
    print("From:", df["time"].iloc[0])
    print("To:", df["time"].iloc[-1])
