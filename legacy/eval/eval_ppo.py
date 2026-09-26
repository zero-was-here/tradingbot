import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from features.make_features import make_features
from env.xauusd_env import XAUUSDTradingEnv

WINDOW = 64
COST = 0.0001
MODEL_PATH = "train/ppo_xauusd_latest.zip"
SPLIT_DATE = "2022-01-01"


def main():
    df, X, r = make_features("data/xauusd_1h.csv", window=WINDOW)

    split_t = np.searchsorted(df["time"].to_numpy(), np.datetime64(SPLIT_DATE))
    df_test = df.iloc[split_t:].reset_index(drop=True)
    X_test = X[split_t:]
    r_test = r[split_t:]

    # --- PPO roll-out ---
    env = XAUUSDTradingEnv(X_test, r_test, window=WINDOW, cost_per_trade=COST)
    model = PPO.load(MODEL_PATH)

    obs, _ = env.reset()
    equity = [1.0]
    pos = [0]
    rew = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        rew.append(reward)
        equity.append(info["equity"])
        pos.append(info["pos"])
        if term or trunc:
            break

    equity = np.array(equity[1:])
    pos = np.array(pos[1:])

    # --- Baselines on SAME test period (no RL) ---
    df_test = df_test.copy()
    df_test["ret_simple"] = df_test["close"].pct_change().fillna(0.0)

    # Buy & Hold
    df_test["bh_equity"] = (1.0 + df_test["ret_simple"]).cumprod()

    # MA 20/50
    df_test["ma_fast"] = df_test["close"].rolling(20).mean()
    df_test["ma_slow"] = df_test["close"].rolling(50).mean()
    df_test["ma_pos"] = np.where(df_test["ma_fast"] > df_test["ma_slow"], 1, -1)
    df_test["ma_pos"] = pd.Series(df_test["ma_pos"]).shift(1).fillna(0)
    df_test["ma_equity"] = (1.0 + df_test["ma_pos"] * df_test["ret_simple"]).cumprod()

    # --- Stats ---
    trades = int(np.sum(np.abs(np.diff(pos)) > 0))
    pct_time_long = float(np.mean(pos == 1))
    pct_time_short = float(np.mean(pos == -1))

    print("MODEL:", MODEL_PATH)
    print("PPO TEST final equity:", float(equity[-1]))
    print("PPO TEST trades:", trades)
    print("PPO % time long:", pct_time_long)
    print("PPO % time short:", pct_time_short)
    print("Buy&Hold TEST final equity:", float(df_test["bh_equity"].iloc[-1]))
    print("MA 20/50 TEST final equity:", float(df_test["ma_equity"].iloc[-1]))

    # --- Plot equity curves ---
    plt.figure()
    plt.plot(df_test["time"].iloc[:len(equity)], equity, label="PPO (with costs)")
    plt.plot(df_test["time"], df_test["bh_equity"], label="Buy & Hold (no costs)")
    plt.plot(df_test["time"], df_test["ma_equity"], label="MA 20/50 (no costs)")
    plt.legend()
    plt.title("Test Period Equity Curves")
    plt.xlabel("Time")
    plt.ylabel("Equity (start=1.0)")
    plt.show()

    # --- Plot positions ---
    plt.figure()
    plt.plot(df_test["time"].iloc[:len(pos)], pos)
    plt.title("PPO Positions Over Time (Test)")
    plt.xlabel("Time")
    plt.ylabel("Position (0 flat, 1 long)")
    plt.show()


if __name__ == "__main__":
    main()
