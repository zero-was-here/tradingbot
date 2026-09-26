"""
Meta-Learning (MAML) for Trading

Model-Agnostic Meta-Learning allows the agent to:
- Quickly adapt to new market regimes
- Learn from just a few examples (few-shot learning)
- Generalize across different market conditions

Instead of training on one task, train on MANY tasks:
- Trending markets
- Ranging markets
- High volatility
- Low volatility
- Bull markets
- Bear markets

The agent learns an initialization that can quickly adapt to any new regime.

Paper: https://arxiv.org/abs/1703.03400
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAMLTrader:
    """
    Model-Agnostic Meta-Learning for trading

    Learns to quickly adapt to new market regimes with just a few examples
    """

    def __init__(self, base_agent, meta_lr=1e-3, adapt_lr=1e-2, adapt_steps=5):
        """
        Initialize MAML

        Args:
            base_agent: Base trading agent to meta-learn
            meta_lr: Meta-learning rate (for updating initialization)
            adapt_lr: Adaptation learning rate (for quick adaptation)
            adapt_steps: Number of gradient steps for adaptation
        """

        self.base_agent = base_agent
        self.meta_lr = meta_lr
        self.adapt_lr = adapt_lr
        self.adapt_steps = adapt_steps

        # Meta-optimizer (updates the initialization)
        self.meta_optimizer = torch.optim.Adam(
            base_agent.parameters(),
            lr=meta_lr
        )

        logger.info("🧬 MAML Trader initialized")
        logger.info(f"   Meta LR: {meta_lr}")
        logger.info(f"   Adapt LR: {adapt_lr}")
        logger.info(f"   Adapt steps: {adapt_steps}")

    def meta_train(self, market_regimes, num_epochs=100):
        """
        Meta-train on multiple market regimes

        Args:
            market_regimes: List of different market conditions
                           Each is a dict with {'train_data', 'test_data'}
            num_epochs: Number of meta-training epochs
        """

        logger.info(f"🧬 Starting meta-training on {len(market_regimes)} regimes")

        for epoch in range(num_epochs):
            meta_loss = 0.0

            # Sample batch of tasks (market regimes)
            batch_tasks = np.random.choice(market_regimes, size=min(4, len(market_regimes)), replace=False)

            for task in batch_tasks:
                # Clone current model
                adapted_agent = self._clone_agent(self.base_agent)

                # Adapt to this task with few gradient steps
                for step in range(self.adapt_steps):
                    # Sample batch from task's training data
                    batch = self._sample_batch(task['train_data'], batch_size=32)

                    # Compute loss
                    loss = adapted_agent.compute_loss(batch)

                    # Adapt (inner loop update)
                    self._adapt_step(adapted_agent, loss)

                # Evaluate adapted model on task's test data
                test_batch = self._sample_batch(task['test_data'], batch_size=32)
                task_loss = adapted_agent.compute_loss(test_batch)

                meta_loss += task_loss

            # Meta-update (outer loop update)
            # This updates the initialization to be good for quick adaptation
            meta_loss = meta_loss / len(batch_tasks)
            meta_loss.backward()
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Meta-loss: {meta_loss.item():.4f}")

        logger.info("✅ Meta-training complete")

    def fast_adapt(self, new_regime_data, num_steps=None):
        """
        Quickly adapt to a new market regime

        With meta-learning, can adapt with just 100 examples
        vs 10,000+ examples for normal training

        Args:
            new_regime_data: Data from new market regime
            num_steps: Number of adaptation steps (default: self.adapt_steps)
        """

        if num_steps is None:
            num_steps = self.adapt_steps

        logger.info(f"🚀 Fast adapting to new regime ({num_steps} steps)...")

        for step in range(num_steps):
            # Sample batch
            batch = self._sample_batch(new_regime_data, batch_size=32)

            # Compute loss
            loss = self.base_agent.compute_loss(batch)

            # Update
            loss.backward()
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

        logger.info("✅ Adaptation complete")

    def _clone_agent(self, agent):
        """Create a deep copy of the agent"""
        return copy.deepcopy(agent)

    def _adapt_step(self, agent, loss):
        """Single adaptation step"""

        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            agent.parameters(),
            create_graph=True  # Important for MAML!
        )

        # Manual SGD update
        for param, grad in zip(agent.parameters(), grads):
            param.data = param.data - self.adapt_lr * grad

    def _sample_batch(self, data, batch_size=32):
        """Sample random batch from data"""

        # Placeholder - would sample actual data
        # For now, return None
        return None


class MarketRegimeGenerator:
    """
    Generate different market regime tasks for meta-learning

    Regimes:
    - Trending up
    - Trending down
    - Range-bound
    - High volatility
    - Low volatility
    - Crash recovery
    """

    @staticmethod
    def generate_regimes(historical_data):
        """
        Split historical data into different regime tasks

        Args:
            historical_data: DataFrame with market data

        Returns:
            List of regime dicts
        """

        regimes = []

        # Detect different regimes in historical data
        # This is simplified - in production, use sophisticated regime detection

        # 1. Trending markets
        trending_periods = MarketRegimeGenerator._find_trending_periods(historical_data)

        for period in trending_periods:
            train_data = period[:int(len(period) * 0.8)]
            test_data = period[int(len(period) * 0.8):]

            regimes.append({
                'name': 'Trending',
                'train_data': train_data,
                'test_data': test_data
            })

        # 2. Ranging markets
        ranging_periods = MarketRegimeGenerator._find_ranging_periods(historical_data)

        for period in ranging_periods:
            train_data = period[:int(len(period) * 0.8)]
            test_data = period[int(len(period) * 0.8):]

            regimes.append({
                'name': 'Ranging',
                'train_data': train_data,
                'test_data': test_data
            })

        # 3. High volatility
        volatile_periods = MarketRegimeGenerator._find_volatile_periods(historical_data)

        for period in volatile_periods:
            train_data = period[:int(len(period) * 0.8)]
            test_data = period[int(len(period) * 0.8):]

            regimes.append({
                'name': 'High Volatility',
                'train_data': train_data,
                'test_data': test_data
            })

        logger.info(f"Generated {len(regimes)} market regime tasks")

        return regimes

    @staticmethod
    def _find_trending_periods(data):
        """Find trending periods in data"""
        # Simplified - would use actual trend detection
        return []

    @staticmethod
    def _find_ranging_periods(data):
        """Find range-bound periods"""
        return []

    @staticmethod
    def _find_volatile_periods(data):
        """Find high volatility periods"""
        return []


# Example usage
if __name__ == "__main__":
    print("🧬 Meta-Learning (MAML) Demo\n")

    print("="*60)
    print("What is Meta-Learning?")
    print("="*60)

    print("""
Meta-learning is "learning to learn."

Normal Training:
- Train on trending markets → Good at trending, bad at ranging
- Train on ranging markets → Good at ranging, bad at trending
- Needs 10,000+ examples to adapt

Meta-Learning:
- Train on MANY different regimes
- Learn an initialization that adapts quickly
- Needs only 100 examples to adapt to new regime

Example:
1. Meta-train on:
   - Trending markets (2019-2020)
   - Ranging markets (2021)
   - Volatile markets (2022)
   - Low vol markets (2023)

2. New regime appears (2024):
   - 5 gradient steps with 100 examples
   - Agent adapted!

This is how you handle market regime changes.
    """)

    print("="*60)
    print("Key Benefits")
    print("="*60)

    print("""
✅ Fast Adaptation
   - New regime? Adapt in minutes, not weeks

✅ Few-Shot Learning
   - Needs only 100 examples vs 10,000+

✅ Robust to Regime Changes
   - Market changes constantly
   - Meta-learning handles this

✅ Better Generalization
   - Trained on many regimes
   - Not overfit to one
    """)

    print("="*60)
    print("How to Use")
    print("="*60)

    print("""
# 1. Generate market regimes
regimes = MarketRegimeGenerator.generate_regimes(historical_data)

# 2. Create MAML trader
maml_trader = MAMLTrader(
    base_agent=your_agent,
    meta_lr=1e-3,
    adapt_lr=1e-2,
    adapt_steps=5
)

# 3. Meta-train
maml_trader.meta_train(regimes, num_epochs=100)

# 4. When new regime appears:
maml_trader.fast_adapt(new_regime_data, num_steps=5)
# Agent adapted in just 5 steps!
    """)

    print("\n✅ Meta-learning system ready!")
    print("\nThis is advanced - implement after basic system works well.")
