"""
Train DreamerV3 Agent on XAUUSD Trading

This is the "Baby Stockfish" - Phase 1 of PROJECT GOD MODE
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.make_features import make_features
from models.dreamer_agent import DreamerV3Agent


WINDOW = 64
COST = 0.0001
TRAIN_END_DATE = "2022-01-01"

# DreamerV3 hyperparameters
BATCH_SIZE = 16
PREFILL_STEPS = 5_000  # Random exploration to fill buffer
TRAIN_STEPS = 100_000  # Training steps
TRAIN_EVERY = 4  # Train every N environment steps
SAVE_EVERY = 10_000

SAVE_DIR = "train/dreamer"
SAVE_PREFIX = "dreamer_xauusd"


class TradingEnvironment:
    """
    Simple trading environment for DreamerV3
    Similar to XAUUSDTradingEnv but returns flat numpy arrays
    """
    def __init__(self, features, returns, window=64, cost_per_trade=0.0001):
        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.T = len(self.r)

        self.reset()

    def reset(self):
        """Reset environment"""
        self.t = self.window
        self.pos = 0  # 0 = flat, 1 = long
        self.equity = 1.0

        return self._get_obs()

    def _get_obs(self):
        """Get current observation"""
        w = self.X[self.t - self.window : self.t]  # (window, F)
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def step(self, action_onehot):
        """
        Execute action

        Args:
            action_onehot: one-hot encoded action [flat, long, short] or [flat, long]

        Returns:
            obs, reward, done, info
        """
        # Decode action (for long-only: 0=flat, 1=long)
        # For long/short: 0=flat, 1=long, 2=short
        if len(action_onehot) == 2:
            new_pos = int(np.argmax(action_onehot))  # 0 or 1
        else:
            action_idx = int(np.argmax(action_onehot))
            if action_idx == 0:
                new_pos = 0  # flat
            elif action_idx == 1:
                new_pos = 1  # long
            else:
                new_pos = -1  # short (not used for long-only)

        # Ensure long-only
        new_pos = max(0, min(1, new_pos))

        # Position change
        delta = abs(new_pos - self.pos)

        # Costs
        trade_cost = self.cost * delta

        # PnL from holding previous position
        pnl = self.pos * self.r[self.t]

        # Reward
        reward = pnl - trade_cost

        # Update equity
        self.equity *= (1.0 + reward)

        # Update position
        self.pos = new_pos

        # Advance time
        self.t += 1

        # Check if done
        done = self.t >= self.T

        # Get next observation
        if not done:
            obs = self._get_obs()
        else:
            obs = np.zeros_like(self._get_obs())

        info = {
            "equity": float(self.equity),
            "pos": int(self.pos),
        }

        return obs, float(reward), done, info


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DreamerV3 on XAUUSD')
    parser.add_argument('--steps', type=int, default=TRAIN_STEPS,
                        help='Number of training steps (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size (default: 16, recommended: 64 for GPU)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto-detect)')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Set device
    if args.device == 'auto':
        # Auto-detect best available (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"🚀 Using device: {device}")
    if device == 'mps':
        print("   ⚡ Apple Metal GPU acceleration enabled!")
    elif device == 'cuda':
        print("   ⚡ NVIDIA CUDA GPU acceleration enabled!")

    # Use local variables instead of modifying globals
    batch_size = args.batch_size
    train_steps = args.steps

    print(f"📊 Training config:")
    print(f"   Steps: {train_steps:,}")
    print(f"   Batch size: {batch_size}")

    # Load data
    print("Loading data...")
    # Try to use macro data first
    if os.path.exists("data/xauusd_1h_macro.csv"):
        print("Using MACRO data (DXY, SPX, US10Y) 🚀")
        df, X, r = make_features("data/xauusd_1h_macro.csv", window=WINDOW)
    else:
        print("Using basic XAUUSD data (no macro)")
        df, X, r = make_features("data/xauusd_1h.csv", window=WINDOW)

    # Split train/test
    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64(TRAIN_END_DATE))
    X_train, r_train = X[:train_end], r[:train_end]
    X_test, r_test = X[train_end:], r[train_end:]

    print(f"Train: {len(X_train)} bars | Test: {len(X_test)} bars")

    # Create environment
    env = TradingEnvironment(X_train, r_train, window=WINDOW, cost_per_trade=COST)

    # Observation dimension
    obs_dim = env._get_obs().shape[0]
    print(f"Observation dimension: {obs_dim}")

    # Create agent
    print("\nInitializing DreamerV3 Agent...")
    agent = DreamerV3Agent(
        obs_dim=obs_dim,
        action_dim=2,  # flat, long (long-only for now)
        device=device,
        embed_dim=256,
        hidden_dim=512,
        stoch_dim=32,
        num_categories=32,
        lr_world_model=3e-4,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.99,
        lambda_=0.95,
        horizon=15,
    )

    print("\n" + "="*60)
    print("PHASE 1: Prefill Replay Buffer (Random Exploration)")
    print("="*60)

    obs = env.reset()
    h, z = None, None

    for step in tqdm(range(PREFILL_STEPS), desc="Prefilling"):
        # Random action
        action_onehot = np.zeros(2, dtype=np.float32)
        action_onehot[np.random.randint(0, 2)] = 1.0

        # Step
        next_obs, reward, done, info = env.step(action_onehot)

        # Store in replay buffer
        agent.replay_buffer.add(obs, action_onehot, reward, done)

        # Next
        obs = next_obs
        if done:
            obs = env.reset()
            h, z = None, None

    print(f"✅ Replay buffer filled with {len(agent.replay_buffer)} transitions")

    print("\n" + "="*60)
    print("PHASE 2: Train DreamerV3 World Model + Policy")
    print("="*60)

    obs = env.reset()
    h, z = None, None
    episode_reward = 0
    episode_count = 0
    step_count = 0

    for train_step in tqdm(range(train_steps), desc="Training"):
        # Act in environment
        action_onehot, (h, z) = agent.act(obs, h, z, deterministic=False)

        # Step
        next_obs, reward, done, info = env.step(action_onehot)

        # Store in replay buffer
        agent.replay_buffer.add(obs, action_onehot, reward, done)

        episode_reward += reward
        step_count += 1

        # Next
        obs = next_obs

        if done:
            print(f"\n  Episode {episode_count}: Reward={episode_reward:.4f}, Equity={info['equity']:.4f}, Steps={step_count}")
            episode_count += 1
            episode_reward = 0
            step_count = 0
            obs = env.reset()
            h, z = None, None

        # Train
        if train_step % TRAIN_EVERY == 0:
            losses = agent.train_step(batch_size=batch_size)

            if losses and train_step % 1000 == 0:
                print(f"\n  Step {train_step}:")
                print(f"    World Model Loss: {losses['world_model_loss']:.4f}")
                print(f"    - Recon: {losses['recon_loss']:.4f}")
                print(f"    - Reward: {losses['reward_loss']:.4f}")
                print(f"    - KL: {losses['kl_loss']:.4f}")
                print(f"    Value Loss: {losses['value_loss']:.4f}")
                print(f"    Policy Loss: {losses['policy_loss']:.4f}")

        # Save checkpoint
        if (train_step + 1) % SAVE_EVERY == 0:
            ckpt_path = f"{SAVE_DIR}/{SAVE_PREFIX}_{(train_step+1)//1000}k.pt"
            agent.save(ckpt_path)
            print(f"\n✅ Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = f"{SAVE_DIR}/{SAVE_PREFIX}_final.pt"
    agent.save(final_path)
    print(f"\n✅ Saved final model: {final_path}")

    print("\n" + "="*60)
    print("PHASE 3: Evaluation on Test Set")
    print("="*60)

    test_env = TradingEnvironment(X_test, r_test, window=WINDOW, cost_per_trade=COST)
    obs = test_env.reset()
    h, z = None, None

    equities = []
    positions = []

    while True:
        action_onehot, (h, z) = agent.act(obs, h, z, deterministic=True)
        obs, reward, done, info = test_env.step(action_onehot)

        equities.append(info["equity"])
        positions.append(info["pos"])

        if done:
            break

    trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    pct_time_long = float(np.mean(np.array(positions) == 1))
    final_equity = float(equities[-1])

    print(f"\n📊 Test Results:")
    print(f"   Final Equity: {final_equity:.4f}")
    print(f"   Return: {(final_equity - 1.0) * 100:.2f}%")
    print(f"   Trades: {trades}")
    print(f"   % Time Long: {pct_time_long * 100:.1f}%")

    print("\n🎉 Training Complete! The World Model has learned the Physics of the Market.")
    print("   Next: Implement MCTS to achieve true 'Stockfish' lookahead capability.")


if __name__ == "__main__":
    main()
