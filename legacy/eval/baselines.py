import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.load_data import load_ohlc_csv

# Load
df = load_ohlc_csv("data/xauusd_1h.csv").copy()

# Returns
df["ret"] = df["close"].pct_change().fillna(0.0)

# -----------------------
# Baseline 1: Buy & Hold
# -----------------------
df["bh_equity"] = (1.0 + df["ret"]).cumprod()

# -----------------------
# Baseline 2: Random policy (long/flat/short)
# -----------------------
np.random.seed(42)
df["rand_pos"] = np.random.choice([-1, 0, 1], size=len(df))
df["rand_equity"] = (1.0 + df["rand_pos"] * df["ret"]).cumprod()

# -----------------------
# Baseline 3: MA crossover (20/50)
# -----------------------
df["ma_fast"] = df["close"].rolling(20).mean()
df["ma_slow"] = df["close"].rolling(50).mean()

# Position: +1 if fast>slow else -1, shift by 1 to avoid look-ahead
df["ma_pos"] = np.where(df["ma_fast"] > df["ma_slow"], 1, -1)
df["ma_pos"] = pd.Series(df["ma_pos"]).shift(1).fillna(0)

df["ma_equity"] = (1.0 + df["ma_pos"] * df["ret"]).cumprod()

# Print final multiples
print("Rows:", len(df))
print("From:", df["time"].iloc[0], "To:", df["time"].iloc[-1])
print("Buy & Hold final multiple:", df["bh_equity"].iloc[-1])
print("Random final multiple:", df["rand_equity"].iloc[-1])
print("MA(20/50) final multiple:", df["ma_equity"].iloc[-1])

# Plot equity curves
plt.figure()
plt.plot(df["time"], df["bh_equity"], label="Buy & Hold")
plt.plot(df["time"], df["rand_equity"], label="Random")
plt.plot(df["time"], df["ma_equity"], label="MA 20/50")
plt.legend()
plt.title("Baseline Equity Curves (No Costs Yet)")
plt.xlabel("Time")
plt.ylabel("Equity (start=1.0)")
plt.show()
