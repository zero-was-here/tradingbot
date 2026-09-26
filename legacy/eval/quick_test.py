import numpy as np
from stable_baselines3 import PPO
from env.xauusd_env_aggressive import XAUUSDTradingEnvAggressive
from features.make_features import make_features

WINDOW = 120
COST = 0.0001
TEST_START_DATE = "2024-01-01"

def run_test():
    # Load Data (H1)
    print("Loading data...")
    df, X, r = make_features("data/xauusd_1h.csv", window=WINDOW)
    
    # Split
    test_start = np.searchsorted(df["time"].to_numpy(), np.datetime64(TEST_START_DATE))
    X_test, r_test = X[test_start:], r[test_start:]
    
    # Setup Env
    test_env = XAUUSDTradingEnvAggressive(
        X_test, r_test,
        window=WINDOW,
        cost_per_trade=COST,
        turnover_coef=0.0,
        leverage=1.0,
        max_episode_steps=None
    )
    
    # Load Model
    model_path = "train/ppo_xauusd_swing_latest"
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Run
    print("Running backtest on 2024-2025 data...")
    obs, _ = test_env.reset()
    equities = []
    positions = []
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        equities.append(info["equity"])
        positions.append(info["pos"])
        
        if terminated or truncated:
            break
            
    # Stats
    trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    pct_long = np.mean(np.array(positions) == 1)
    pct_short = np.mean(np.array(positions) == -1)
    pct_flat = np.mean(np.array(positions) == 0)
    final_eq = equities[-1] if equities else 1.0
    
    print("\n" + "="*60)
    print("📊 FINAL SWING TRADER TEST RESULTS (2024-2025)")
    print("="*60)
    print(f"Final Equity:     {final_eq:.4f}x")
    print(f"Total Return:     {(final_eq-1)*100:.2f}%")
    print(f"Total Trades:     {trades}")
    print(f"% Time Long:      {pct_long*100:.1f}%")
    print(f"% Time Short:     {pct_short*100:.1f}%")
    print(f"% Time Flat:      {pct_flat*100:.1f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_test()
