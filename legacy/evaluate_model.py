"""
Evaluate Trained DreamerV3 Model

This script evaluates a trained model on validation/test data and generates
comprehensive performance metrics and visualizations.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.ultimate_150_features import make_ultimate_features
from models.dreamer_agent import DreamerV3Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnvironment:
    """Simple trading environment for evaluation"""
    def __init__(self, features, returns, window=64, cost_per_trade=0.0001):
        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.T = len(self.r)
        self.reset()

    def reset(self):
        self.t = self.window
        self.pos = 0
        self.equity = 1.0
        return self._get_obs()

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def step(self, action_onehot):
        # action_onehot: [flat, long] probabilities
        action = np.argmax(action_onehot)  # 0 = flat, 1 = long

        # Get return
        ret = self.r[self.t]

        # Calculate reward
        if action == 1:  # Long position
            reward = ret - self.cost  # Profit/loss minus transaction cost
            self.pos = 1
        else:  # Flat position
            reward = -self.cost if self.pos == 1 else 0  # Only cost if exiting position
            self.pos = 0

        # Update equity
        if action == 1:
            self.equity *= (1 + ret - self.cost)
        elif self.pos == 1:  # Closing position
            self.equity *= (1 - self.cost)

        # Move forward
        self.t += 1
        done = (self.t >= self.T)

        next_obs = self._get_obs() if not done else self._get_obs()

        return next_obs, reward, done, {'equity': self.equity, 'position': self.pos}

    @property
    def observation_space(self):
        return self.window * self.X.shape[1] + 1

    @property
    def action_space(self):
        return 2  # flat or long


def evaluate_model(agent, env, timestamps):
    """
    Evaluate model on environment

    Returns:
        metrics: Dict of performance metrics
        equity_curve: Array of equity over time
        positions: Array of positions over time
        dates: Corresponding timestamps
    """
    logger.info("🎯 Running evaluation...")

    obs = env.reset()
    h, z = None, None

    equity_curve = [1.0]
    positions = []
    rewards = []

    for step in tqdm(range(env.T - env.window), desc="Evaluating"):
        # Get action from agent
        action, (h, z) = agent.act(obs, h, z, deterministic=True)
        action_idx = np.argmax(action)
        action_onehot = np.eye(env.action_space)[action_idx]

        # Step environment
        obs, reward, done, info = env.step(action_onehot)

        equity_curve.append(info['equity'])
        positions.append(info['position'])
        rewards.append(reward)

        if done:
            break

    # Convert to arrays
    equity_curve = np.array(equity_curve)
    positions = np.array(positions)
    rewards = np.array(rewards)
    dates = timestamps[env.window:env.window + len(positions)]

    # Calculate metrics
    returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return = (equity_curve[-1] - 1) * 100

    # Annualized metrics (assuming 252 trading days)
    days = len(equity_curve) / (252 * 24 * 12)  # Convert 5-min bars to years
    annual_return = ((equity_curve[-1] ** (1 / days)) - 1) * 100 if days > 0 else 0

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 12)

    # Max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = np.min(drawdown) * 100

    # Win rate
    win_rate = np.mean(rewards > 0) * 100 if len(rewards) > 0 else 0

    # Position statistics
    long_pct = np.mean(positions) * 100 if len(positions) > 0 else 0

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_equity': equity_curve[-1],
        'num_trades': len(positions),
        'long_percentage': long_pct,
    }

    return metrics, equity_curve, positions, dates


def plot_results(equity_curve, positions, dates, metrics, save_path='results.png'):
    """Plot evaluation results"""
    logger.info("📊 Creating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Equity curve
    axes[0].plot(dates, equity_curve, linewidth=2, color='green')
    axes[0].set_title(f'Equity Curve - Final: ${equity_curve[-1]:.2f}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Equity ($)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    axes[0].legend()

    # Drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax * 100
    axes[1].fill_between(dates, drawdown, 0, color='red', alpha=0.3)
    axes[1].plot(dates, drawdown, color='darkred', linewidth=1)
    axes[1].set_title(f'Drawdown - Max: {metrics["max_drawdown"]:.2f}%', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Positions
    axes[2].fill_between(dates, positions, 0, alpha=0.3, color='blue')
    axes[2].set_title(f'Positions - Long: {metrics["long_percentage"]:.1f}%', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Position', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Plot saved to: {save_path}")

    return fig


def print_metrics(metrics, title="EVALUATION RESULTS"):
    """Pretty print metrics"""
    logger.info("\n" + "="*70)
    logger.info(f"📊 {title}")
    logger.info("="*70)
    logger.info(f"💰 Total Return:      {metrics['total_return']:>10.2f}%")
    logger.info(f"📈 Annual Return:     {metrics['annual_return']:>10.2f}%")
    logger.info(f"📉 Max Drawdown:      {metrics['max_drawdown']:>10.2f}%")
    logger.info(f"⚡ Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
    logger.info(f"🎯 Win Rate:          {metrics['win_rate']:>10.2f}%")
    logger.info(f"💵 Final Equity:      {metrics['final_equity']:>10.2f}x")
    logger.info(f"📊 Long %:            {metrics['long_percentage']:>10.2f}%")
    logger.info(f"🔄 Num Trades:        {metrics['num_trades']:>10,}")
    logger.info("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trained DreamerV3 Model')
    parser.add_argument('--checkpoint', type=str, default='train/dreamer_ultimate/ultimate_150_xauusd_final.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--period', type=str, default='validation', choices=['validation', 'test', 'all'],
                       help='Evaluation period (validation=2022-2023, test=2024-2025, all=everything)')
    parser.add_argument('--save-plot', type=str, default='evaluation_results.png',
                       help='Path to save results plot')

    args = parser.parse_args()

    # ========== LOAD FEATURES ==========
    logger.info("📊 Loading Ultimate 150+ features...")
    X, returns, timestamps = make_ultimate_features(base_timeframe='M5')

    logger.info(f"✅ Loaded {X.shape[1]} features, {len(X):,} samples")
    logger.info(f"📅 Date range: {timestamps[0]} to {timestamps[-1]}")

    # ========== SELECT PERIOD ==========
    if args.period == 'validation':
        # 2022-2023
        mask = (timestamps >= '2022-01-01') & (timestamps < '2024-01-01')
        period_name = "VALIDATION (2022-2023)"
    elif args.period == 'test':
        # 2024-2025
        mask = (timestamps >= '2024-01-01')
        period_name = "TEST (2024-2025)"
    else:
        # All data
        mask = np.ones(len(timestamps), dtype=bool)
        period_name = "ALL DATA"

    X_eval = X[mask]
    returns_eval = returns[mask]
    timestamps_eval = timestamps[mask]

    logger.info(f"\n📅 Evaluating on {period_name}")
    logger.info(f"   • Samples: {len(X_eval):,}")
    logger.info(f"   • Date range: {timestamps_eval[0]} to {timestamps_eval[-1]}")

    # ========== CREATE ENVIRONMENT ==========
    env = TradingEnvironment(X_eval, returns_eval, window=64, cost_per_trade=0.0001)

    # ========== LOAD AGENT ==========
    logger.info(f"\n🤖 Loading model from: {args.checkpoint}")

    agent = DreamerV3Agent(
        obs_dim=env.observation_space,
        action_dim=env.action_space,
        embed_dim=256,
        hidden_dim=512,
        stoch_dim=32,
        num_categories=32,
        device='cpu'  # Use CPU for evaluation
    )

    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        logger.info("✅ Model loaded successfully")
    else:
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    # ========== EVALUATE ==========
    metrics, equity_curve, positions, dates = evaluate_model(agent, env, timestamps_eval)

    # ========== PRINT RESULTS ==========
    print_metrics(metrics, title=f"EVALUATION RESULTS - {period_name}")

    # ========== PLOT RESULTS ==========
    plot_results(equity_curve, positions, dates, metrics, save_path=args.save_plot)

    # ========== SAVE DETAILED RESULTS ==========
    results_df = pd.DataFrame({
        'timestamp': dates,
        'equity': equity_curve[1:],  # Skip initial 1.0
        'position': positions,
    })

    csv_path = args.save_plot.replace('.png', '.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Detailed results saved to: {csv_path}")

    logger.info("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
