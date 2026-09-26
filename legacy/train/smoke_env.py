import numpy as np
from features.make_features import make_features
from env.xauusd_env import XAUUSDTradingEnv

df, X, r = make_features("data/xauusd_1h.csv", window=64)
env = XAUUSDTradingEnv(X, r, window=64, cost_per_trade=0.0002)

obs, _ = env.reset()
total = 0.0
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total += reward
    if term or trunc:
        break

print("Smoke test total reward:", total)
print("Equity:", info["equity"], "Last pos:", info["pos"])
