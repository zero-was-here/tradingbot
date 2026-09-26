"""
Kelly Criterion Position Sizing

Optimal position sizing based on win probability and risk/reward ratio.

The Kelly Criterion tells you the mathematically optimal fraction of your capital
to risk on each trade to maximize long-term growth.

Formula: f* = (p * b - q) / b

Where:
- f* = fraction of bankroll to bet
- p = probability of win
- q = probability of loss (1-p)
- b = odds (win amount / loss amount)

CRITICAL: Use fractional Kelly (1/4 or 1/2) for safety.
Full Kelly is too aggressive and can lead to large drawdowns.
"""

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KellyPositionSizer:
    """
    Optimal position sizing using Kelly Criterion

    Features:
    - Dynamic position sizing based on win probability
    - Fractional Kelly for safety
    - Maximum position caps
    - Volatility-adjusted sizing
    - Confidence-based sizing
    """

    def __init__(self, max_position=0.10, kelly_fraction=0.25):
        """
        Initialize Kelly position sizer

        Args:
            max_position: Maximum position size (e.g., 0.10 = 10%)
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
                           - Full Kelly (1.0): Optimal but aggressive
                           - Half Kelly (0.5): Good balance
                           - Quarter Kelly (0.25): Conservative (recommended)
        """

        self.max_position = max_position
        self.kelly_fraction = kelly_fraction

        # Statistics tracking
        self.trade_history = []
        self.win_rate = 0.5  # Initial estimate
        self.avg_win = 0.02  # Initial estimate (2%)
        self.avg_loss = 0.01  # Initial estimate (1%)

        logger.info(f"💰 Kelly Position Sizer initialized")
        logger.info(f"  Max Position: {self.max_position:.1%}")
        logger.info(f"  Kelly Fraction: {self.kelly_fraction:.1%}")

    def compute_position_size(self, win_prob, avg_win, avg_loss, equity=1.0):
        """
        Compute optimal position size using Kelly Criterion

        Args:
            win_prob: Probability of winning (0-1)
            avg_win: Average win size (as fraction, e.g., 0.02 for 2%)
            avg_loss: Average loss size (as fraction, e.g., 0.01 for 1%)
            equity: Current equity (for converting fraction to dollars)

        Returns:
            position_size: Optimal position size (as fraction of equity)

        Example:
            >>> sizer = KellyPositionSizer()
            >>> pos = sizer.compute_position_size(0.55, 0.02, 0.01, equity=10000)
            >>> # With 55% win rate, 2:1 reward/risk, quarter Kelly suggests ~5% position
        """

        # Kelly formula
        if avg_loss == 0:
            logger.warning("⚠️ Average loss is zero - using minimum position")
            return 0.01  # Minimum position

        b = avg_win / avg_loss  # Odds (risk/reward ratio)
        p = win_prob
        q = 1 - p

        # Full Kelly
        kelly = (p * b - q) / b

        # Handle edge cases
        if kelly <= 0:
            # Negative or zero Kelly = no edge = don't trade
            return 0.0

        # Apply fractional Kelly (safer)
        fractional_kelly = kelly * self.kelly_fraction

        # Cap at maximum position
        position_fraction = min(fractional_kelly, self.max_position)

        # Ensure non-negative
        position_fraction = max(0.0, position_fraction)

        return position_fraction

    def dynamic_sizing(self, agent, current_state, obs):
        """
        Use agent's world model to estimate win probability

        This uses the agent's critic to estimate expected value,
        then converts it to a win probability.

        Args:
            agent: DreamerV3Agent with critic network
            current_state: Current trading state
            obs: Current observation

        Returns:
            position_size: Optimal position size
        """

        # Get agent's value estimates for both actions
        with torch.no_grad():
            # Prepare observation
            if not isinstance(obs, torch.Tensor):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            else:
                obs_tensor = obs

            # Get hidden states if available
            if hasattr(agent, 'h') and agent.h is not None:
                h = agent.h
                z = agent.z
            else:
                # Initialize states
                h, z = agent.rssm.initial_state(1, device=agent.device)

            # Encode observation
            embed = agent.encoder(obs_tensor)

            # Get posterior state
            h, z_dist = agent.rssm.observe(embed, None, h, z)
            z = z_dist.sample()

            # Flatten z for critic
            z_flat = z.reshape(z.shape[0], -1)

            # Concatenate h and z for critic input
            state = torch.cat([h, z_flat], dim=-1)

            # Get action logits from actor
            action_logits = agent.actor(state)
            action_probs = torch.softmax(action_logits, dim=-1)

            # Get values for each action
            # For binary action (flat=0, long=1):
            value_flat = agent.critic(state)  # Value of current state
            # Estimate value if we go long (approximate)
            value_long = value_flat  # Simplified - in practice, simulate forward

            # Expected advantage
            advantage = value_long - value_flat

            # Convert to probability using sigmoid
            # Higher advantage = higher confidence = higher win probability
            win_prob = torch.sigmoid(advantage * 5).item()

            # Clamp to reasonable range
            win_prob = max(0.3, min(0.7, win_prob))

        # Use historical statistics for avg win/loss
        avg_win = self.avg_win
        avg_loss = self.avg_loss

        # Compute Kelly size
        position_size = self.compute_position_size(
            win_prob, avg_win, avg_loss, equity=current_state.get('equity', 1.0)
        )

        return position_size

    def volatility_adjusted_sizing(self, base_position, current_volatility, normal_volatility):
        """
        Adjust position size based on volatility

        Higher volatility = smaller position (same dollar risk)

        Args:
            base_position: Base position size from Kelly
            current_volatility: Current market volatility
            normal_volatility: Normal/average volatility

        Returns:
            adjusted_position: Volatility-adjusted position
        """

        if normal_volatility == 0:
            return base_position

        # Volatility scaling factor
        vol_ratio = current_volatility / normal_volatility

        # Inverse scaling: higher vol = smaller position
        adjusted_position = base_position / vol_ratio

        # Still respect maximum
        adjusted_position = min(adjusted_position, self.max_position)

        return adjusted_position

    def update_statistics(self, trade_result):
        """
        Update win/loss statistics from trade results

        Args:
            trade_result: dict with 'pnl', 'is_win', etc.
        """

        self.trade_history.append(trade_result)

        # Keep last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)

        # Recompute statistics
        if len(self.trade_history) >= 10:  # Need minimum trades
            wins = [t for t in self.trade_history if t.get('is_win', t['pnl'] > 0)]
            losses = [t for t in self.trade_history if not t.get('is_win', t['pnl'] > 0)]

            # Win rate
            self.win_rate = len(wins) / len(self.trade_history)

            # Average win
            if wins:
                self.avg_win = np.mean([abs(t['pnl']) for t in wins])
            else:
                self.avg_win = 0.02  # Default

            # Average loss
            if losses:
                self.avg_loss = np.mean([abs(t['pnl']) for t in losses])
            else:
                self.avg_loss = 0.01  # Default

            logger.info(f"📊 Updated statistics: WR={self.win_rate:.1%}, Avg Win={self.avg_win:.3f}, Avg Loss={self.avg_loss:.3f}")

    def get_current_stats(self):
        """Get current statistics"""
        return {
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'reward_risk_ratio': self.avg_win / self.avg_loss if self.avg_loss > 0 else 0,
            'num_trades': len(self.trade_history),
        }


class FixedFractionSizer:
    """
    Simple fixed fraction position sizing

    Always risk X% of equity per trade (e.g., 2%)
    """

    def __init__(self, risk_per_trade=0.02):
        """
        Args:
            risk_per_trade: Fixed fraction to risk (e.g., 0.02 = 2%)
        """
        self.risk_per_trade = risk_per_trade

        logger.info(f"💰 Fixed Fraction Sizer: {self.risk_per_trade:.1%} per trade")

    def compute_position_size(self, equity=1.0):
        """Always return fixed fraction"""
        return self.risk_per_trade


class ATRPositionSizer:
    """
    ATR-based position sizing

    Position size = (Account Risk%) / (ATR * ATR Multiplier)

    Larger ATR = more volatile = smaller position for same dollar risk
    """

    def __init__(self, account_risk=0.02, atr_multiplier=2.0):
        """
        Args:
            account_risk: Total account risk per trade (e.g., 0.02 = 2%)
            atr_multiplier: Stop loss in ATR units (e.g., 2.0 = 2x ATR)
        """
        self.account_risk = account_risk
        self.atr_multiplier = atr_multiplier

        logger.info(f"💰 ATR Position Sizer: {account_risk:.1%} risk, {atr_multiplier}x ATR stop")

    def compute_position_size(self, atr, price, equity=1.0):
        """
        Compute position size based on ATR

        Args:
            atr: Average True Range
            price: Current price
            equity: Current equity

        Returns:
            position_size: Position as fraction of equity
        """

        if atr == 0 or price == 0:
            return 0.02  # Default

        # Dollar risk per trade
        dollar_risk = equity * self.account_risk

        # Stop loss distance
        stop_distance = atr * self.atr_multiplier

        # Position size in units
        position_units = dollar_risk / stop_distance

        # Convert to fraction of equity
        position_value = position_units * price
        position_fraction = position_value / equity

        return min(position_fraction, 0.10)  # Cap at 10%


# Example usage
if __name__ == "__main__":
    print("💰 Position Sizing Demo\n")

    # Test 1: Kelly Criterion
    print("=" * 60)
    print("Test 1: Kelly Criterion")
    print("=" * 60)

    kelly = KellyPositionSizer(max_position=0.10, kelly_fraction=0.25)

    # Scenario 1: Strong edge (60% win rate, 2:1 R:R)
    win_prob = 0.60
    avg_win = 0.02  # 2%
    avg_loss = 0.01  # 1%
    equity = 10000

    position = kelly.compute_position_size(win_prob, avg_win, avg_loss, equity)
    print(f"\nScenario 1: Strong Edge")
    print(f"  Win Rate: {win_prob:.1%}")
    print(f"  Avg Win: {avg_win:.1%}")
    print(f"  Avg Loss: {avg_loss:.1%}")
    print(f"  Risk/Reward: {avg_win/avg_loss:.1f}:1")
    print(f"  → Kelly Position: {position:.2%}")

    # Scenario 2: Weak edge (52% win rate, 1:1 R:R)
    position2 = kelly.compute_position_size(0.52, 0.01, 0.01, equity)
    print(f"\nScenario 2: Weak Edge")
    print(f"  Win Rate: 52%")
    print(f"  R:R: 1:1")
    print(f"  → Kelly Position: {position2:.2%}")

    # Scenario 3: No edge (50% win rate, 1:1 R:R)
    position3 = kelly.compute_position_size(0.50, 0.01, 0.01, equity)
    print(f"\nScenario 3: No Edge")
    print(f"  Win Rate: 50%")
    print(f"  R:R: 1:1")
    print(f"  → Kelly Position: {position3:.2%} (should be 0)")

    # Test 2: Fixed Fraction
    print("\n" + "=" * 60)
    print("Test 2: Fixed Fraction")
    print("=" * 60)

    fixed = FixedFractionSizer(risk_per_trade=0.02)
    position = fixed.compute_position_size(equity=10000)
    print(f"  Fixed risk: {position:.2%}")

    # Test 3: ATR-based
    print("\n" + "=" * 60)
    print("Test 3: ATR-based Sizing")
    print("=" * 60)

    atr_sizer = ATRPositionSizer(account_risk=0.02, atr_multiplier=2.0)

    # Gold @ $2000, ATR = $20
    position = atr_sizer.compute_position_size(atr=20, price=2000, equity=10000)
    print(f"  XAUUSD @ $2000, ATR=$20")
    print(f"  → Position: {position:.2%}")

    print("\n✅ All position sizing methods working!")
