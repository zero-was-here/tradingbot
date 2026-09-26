"""
Train DreamerV3 Agent with FULL GOD MODE Features

This integrates ALL Phase 1-6 components:
- Multi-timeframe features (H1, H4, D1)
- Macro correlations (DXY, SPX, US10Y)
- Economic calendar awareness
- 100+ features total

This is the TRUE God Mode training script.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.god_mode_features import make_features
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
SAVE_PREFIX = "god_mode_xauusd"


class TradingEnvironment:
    """
    Trading environment for DreamerV3 with God Mode features
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
        """
        Get current observation

        Returns:
            Flattened observation with:
            - Last WINDOW timesteps of features
            - Current position
        """
        # Get window of features
        w = self.X[self.t - self.window : self.t]  # (window, num_features)

        # Flatten
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])

        return obs.astype(np.float32)

    def step(self, action_onehot):
        """
        Execute action

        Args:
            action_onehot: one-hot encoded action [flat, long]

        Returns:
            obs, reward, done, info
        """
        # Decode action (for long-only: 0=flat, 1=long)
        new_pos = int(np.argmax(action_onehot))  # 0 or 1

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
    parser = argparse.ArgumentParser(description='Train DreamerV3 with God Mode features')
    parser.add_argument('--steps', type=int, default=TRAIN_STEPS,
                        help='Number of training steps (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size (default: 16, recommended: 64-128 for GPU)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--multi-timeframe', action='store_true', default=True,
                        help='Use multi-timeframe features (default: True)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
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

    print("\n" + "="*70)
    print("🔥 GOD MODE TRAINING - The Stockfish of Trading")
    print("="*70)
    print(f"\n🚀 Device: {device}")
    if device == 'mps':
        print("   ⚡ Apple Metal GPU acceleration enabled!")
    elif device == 'cuda':
        print("   ⚡ NVIDIA CUDA GPU acceleration enabled!")

    batch_size = args.batch_size
    train_steps = args.steps

    print(f"\n📊 Training Configuration:")
    print(f"   Steps: {train_steps:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Multi-timeframe: {args.multi_timeframe}")
    print(f"   Cost per trade: {COST*100:.2f}%")

    # Load data with GOD MODE features
    print(f"\n{'='*70}")
    print("PHASE 0: Loading Data & Creating God Mode Features")
    print("="*70)

    csv_path = "data/xauusd_1h_macro.csv"
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: Data file not found: {csv_path}")
        print("   Please ensure your data file exists with columns:")
        print("   time, open, high, low, close, volume, dxy_close, spx_close, us10y_close")
        return

    # Create features using God Mode feature engineering
    X, r = make_features(csv_path, use_multi_timeframe=args.multi_timeframe)

    # Load time column for train/test split
    import pandas as pd
    df_time = pd.read_csv(csv_path, usecols=['time'])

    # Split train/test
    train_end = np.searchsorted(df_time["time"].to_numpy(), TRAIN_END_DATE)
    X_train, r_train = X[:train_end], r[:train_end]
    X_test, r_test = X[train_end:], r[train_end:]

    print(f"\n✅ Data loaded successfully!")
    print(f"   Train: {len(X_train)} bars ({df_time['time'].iloc[0]} to {TRAIN_END_DATE})")
    print(f"   Test: {len(X_test)} bars ({TRAIN_END_DATE} to {df_time['time'].iloc[-1]})")
    print(f"   Features: {X.shape[1]} (God Mode enabled)")

    # Create environment
    env = TradingEnvironment(X_train, r_train, window=WINDOW, cost_per_trade=COST)

    # Observation dimension
    obs_dim = env._get_obs().shape[0]
    print(f"\n🧠 Model Configuration:")
    print(f"   Observation dim: {obs_dim:,}")
    print(f"   Action space: 2 (flat, long)")
    print(f"   Lookback window: {WINDOW} timesteps")

    # Create agent
    print(f"\n{'='*70}")
    print("Initializing DreamerV3 Agent...")
    print("="*70)

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

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n📥 Loading checkpoint: {args.resume}")
            agent.load(args.resume)
            print(f"✅ Resumed from checkpoint")
        else:
            print(f"\n⚠️ WARNING: Checkpoint not found: {args.resume}")
            print("   Starting from scratch...")

    print("\n" + "="*70)
    print("PHASE 1: Prefill Replay Buffer (Random Exploration)")
    print("="*70)

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

    print(f"✅ Replay buffer filled with {len(agent.replay_buffer.buffer)} transitions")

    print("\n" + "="*70)
    print("PHASE 2: Train DreamerV3 World Model + Policy")
    print("="*70)

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
            episode_count += 1
            episode_reward = 0
            step_count = 0
            obs = env.reset()
            h, z = None, None

        # Train
        if train_step % TRAIN_EVERY == 0:
            losses = agent.train_step(batch_size=batch_size)

            if losses and train_step % 1000 == 0:
                tqdm.write(f"\n  Step {train_step}:")
                tqdm.write(f"    World Model Loss: {losses['world_model_loss']:.4f}")
                tqdm.write(f"    - Recon: {losses['recon_loss']:.4f}")
                tqdm.write(f"    - Reward: {losses['reward_loss']:.4f}")
                tqdm.write(f"    - KL: {losses['kl_loss']:.4f}")
                tqdm.write(f"    Value Loss: {losses['value_loss']:.4f}")
                tqdm.write(f"    Policy Loss: {losses['policy_loss']:.4f}")

        # Save checkpoint
        if (train_step + 1) % SAVE_EVERY == 0:
            ckpt_path = f"{SAVE_DIR}/{SAVE_PREFIX}_step_{train_step+1}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"\n💾 Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = f"{SAVE_DIR}/{SAVE_PREFIX}_final.pt"
    agent.save(final_path)
    print(f"\n✅ Final model saved: {final_path}")

    print("\n" + "="*70)
    print("PHASE 3: Evaluation on Test Set")
    print("="*70)

    test_env = TradingEnvironment(X_test, r_test, window=WINDOW, cost_per_trade=COST)
    obs = test_env.reset()
    h, z = None, None

    test_rewards = []
    test_positions = []

    while True:
        action_onehot, (h, z) = agent.act(obs, h, z, deterministic=True)
        obs, reward, done, info = test_env.step(action_onehot)

        test_rewards.append(reward)
        test_positions.append(info['pos'])

        if done:
            break

    # Calculate metrics
    final_equity = test_env.equity
    total_return = (final_equity - 1.0) * 100
    num_trades = sum(np.diff([0] + test_positions) != 0)
    pct_time_long = np.mean(test_positions) * 100

    print(f"\n📊 Test Results:")
    print(f"   Final Equity: {final_equity:.4f}")
    print(f"   Return: {total_return:+.2f}%")
    print(f"   Trades: {num_trades}")
    print(f"   % Time Long: {pct_time_long:.1f}%")

    print("\n" + "="*70)
    print("🎉 GOD MODE TRAINING COMPLETE!")
    print("="*70)
    print(f"\n🔥 The World Model has learned the Physics of the Market.")
    print(f"   Features used: {X.shape[1]} (multi-timeframe + macro + calendar)")
    print(f"   Training steps: {train_steps:,}")
    print(f"   Final model: {final_path}")

    if total_return > 0:
        print(f"\n✅ Model achieved {total_return:+.2f}% on test set!")
    else:
        print(f"\n⚠️ Model needs more training (test return: {total_return:+.2f}%)")
        print(f"   Consider training for more steps (e.g., 500k-1M)")

    print(f"\n📈 Next Steps:")
    print(f"   1. Validate on crises: python eval/crisis_validation.py")
    print(f"   2. Test with MCTS: Use DreamerMCTSAgent wrapper")
    print(f"   3. Deploy to demo: Update live trading script")
    print(f"   4. Monitor 24/7: python monitoring/production_monitor.py")

    print(f"\n🚀 You're ready for the next phase!")


if __name__ == "__main__":
    main()
