import os
import numpy as np
import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from features.make_features import make_features
from env.xauusd_env import XAUUSDTradingEnv

WINDOW = 64
COST = 0.0001          # KEEP CONSISTENT everywhere
N_ENVS = 8             # try 4/8 depending on your Mac

TRAIN_END_DATE = "2022-01-01"

# Training schedule
CHUNK_STEPS = 50_000  # timesteps per chunk
N_CHUNKS = 10          # 10 chunks => 1,000,000 total

SAVE_DIR = "train"
SAVE_PREFIX = "ppo_xauusd"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    df, X, r = make_features("data/xauusd_1h.csv", window=WINDOW)

    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64(TRAIN_END_DATE))
    X_train, r_train = X[:train_end], r[:train_end]
    X_test, r_test = X[train_end:], r[train_end:]

    def make_train_env():
        return XAUUSDTradingEnv(
            X_train,
            r_train,
            window=WINDOW,
            cost_per_trade=COST,
            max_episode_steps=20_000,
        )

    # parallel envs
    train_env = SubprocVecEnv([make_train_env for _ in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=1024,     # per env (total per update ~ n_steps * N_ENVS)
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )

    total = 0
    for i in range(N_CHUNKS):
        model.learn(total_timesteps=CHUNK_STEPS, reset_num_timesteps=False)
        total += CHUNK_STEPS

        ckpt_path = f"{SAVE_DIR}/{SAVE_PREFIX}_{total//1000}k"
        model.save(ckpt_path)
        print(f"\n✅ Saved checkpoint: {ckpt_path}.zip\n")

    # also save "latest" for convenience
    model.save(f"{SAVE_DIR}/{SAVE_PREFIX}_latest")
    print(f"\n✅ Saved latest: {SAVE_DIR}/{SAVE_PREFIX}_latest.zip\n")

    # quick small evaluation at the end (optional)
    test_env = XAUUSDTradingEnv(X_test, r_test, window=WINDOW, cost_per_trade=COST, max_episode_steps=None)
    obs, _ = test_env.reset()
    equities, positions = [], []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = test_env.step(action)
        equities.append(info["equity"])
        positions.append(info["pos"])
        if term or trunc:
            break

    trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    pct_time_long = float(np.mean(np.array(positions) == 1))

    print("FINAL QUICK TEST equity:", float(equities[-1]))
    print("FINAL QUICK TEST trades:", trades)
    print("FINAL QUICK TEST % time long:", pct_time_long)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # macOS safe
    main()
