"""
Risk Supervisor - Hard-coded Safety Layer

This is NOT a neural network - it's deterministic rules that override AI decisions.
Think of it as "circuit breakers" that prevent catastrophic losses.

CRITICAL: This must ALWAYS be active during live trading.
"""

from datetime import datetime, timedelta
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskSupervisor:
    """
    Deterministic safety layer that overrides AI decisions

    The AI can hallucinate or make mistakes. This prevents disasters.

    Features:
    - Daily loss limits (circuit breaker)
    - Maximum drawdown protection
    - Position size limits
    - Volatility filters
    - Correlation guards
    - Event risk filters
    - Overtrading prevention
    - Spread filters
    """

    def __init__(self, config=None):
        """
        Initialize risk supervisor with safety limits

        Args:
            config: dict with safety parameters
        """

        # Default conservative config
        if config is None:
            config = self.get_default_config()

        # Core limits
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% max daily loss
        self.max_position_size = config.get('max_position', 0.10)  # 10% of equity
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15% max drawdown
        self.volatility_threshold = config.get('vol_threshold', 3.0)  # High volatility
        self.max_spread = config.get('max_spread', 0.0005)  # 5 pips for XAUUSD
        self.max_trades_per_day = config.get('max_trades_per_day', 20)
        self.min_time_between_trades = config.get('min_trade_interval', 300)  # 5 minutes
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)

        # State tracking
        self.daily_pnl = 0.0
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.halt_until = None
        self.last_trade_time = None
        self.trade_history = []

        # Statistics
        self.total_trades_approved = 0
        self.total_trades_rejected = 0
        self.rejection_reasons = {}

        logger.info("🛡️ Risk Supervisor initialized")
        logger.info(f"  Max Daily Loss: {self.max_daily_loss:.1%}")
        logger.info(f"  Max Position: {self.max_position_size:.1%}")
        logger.info(f"  Max Drawdown: {self.max_drawdown:.1%}")

    @staticmethod
    def get_default_config():
        """Default conservative configuration"""
        return {
            'max_daily_loss': 0.02,  # 2%
            'max_position': 0.10,  # 10%
            'max_drawdown': 0.15,  # 15%
            'vol_threshold': 3.0,
            'max_spread': 0.0005,
            'max_trades_per_day': 20,
            'min_trade_interval': 300,  # seconds
            'max_consecutive_losses': 5,
        }

    def check_trade(self, action, state, market_data):
        """
        Approve or reject AI's proposed trade

        Args:
            action: AI's proposed action (0=flat, 1=long)
            state: Current trading state (position, equity, etc.)
            market_data: Current market conditions

        Returns:
            (approved: bool, reason: str)
        """

        # 1. CIRCUIT BREAKER: Daily Loss Limit
        if self.daily_pnl < -self.max_daily_loss:
            self.halt_until = datetime.now() + timedelta(hours=24)
            return self._reject("CIRCUIT_BREAKER: Daily loss limit exceeded")

        # 2. Trading Halt Check
        if self.halt_until and datetime.now() < self.halt_until:
            time_left = (self.halt_until - datetime.now()).seconds / 3600
            return self._reject(f"HALTED until {self.halt_until.strftime('%H:%M')} ({time_left:.1f}h remaining)")

        # 3. Maximum Drawdown Protection
        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            return self._reject(f"MAX_DRAWDOWN: {current_drawdown:.2%} > {self.max_drawdown:.2%}")

        # 4. Position Size Limit
        position_size = abs(action) if isinstance(action, (int, float)) else 1.0
        if position_size > self.max_position_size:
            return self._reject(f"POSITION_TOO_LARGE: {position_size:.2%} > {self.max_position_size:.2%}")

        # 5. Consecutive Losses Protection
        if self.consecutive_losses >= self.max_consecutive_losses:
            return self._reject(f"TOO_MANY_LOSSES: {self.consecutive_losses} consecutive losses")

        # 6. Volatility Filter
        current_volatility = market_data.get('volatility', 0.0)
        if current_volatility > self.volatility_threshold:
            # Only allow closing positions, no new entries
            current_position = state.get('position', 0)
            if action != 0 and current_position == 0:
                return self._reject(f"HIGH_VOLATILITY: {current_volatility:.2f} > {self.volatility_threshold}")

        # 7. Correlation Guard (Gold vs USD)
        if action == 1:  # Going long Gold
            dxy_momentum = market_data.get('dxy_momentum', 0.0)
            # Gold and USD are inversely correlated
            if dxy_momentum > 0.01:  # USD rallying strongly
                return self._reject("CORRELATION_GUARD: USD rallying (bearish for Gold)")

        # 8. Event Risk Filter (during high-impact news)
        is_high_impact_event = market_data.get('is_high_impact_event', False)
        is_event_window = market_data.get('is_event_window', False)

        if is_high_impact_event or is_event_window:
            # Reduce maximum position size during events
            max_event_position = 0.5 * self.max_position_size
            if position_size > max_event_position:
                return self._reject(f"EVENT_RISK: Max position {max_event_position:.2%} during news")

        # 9. Maximum Trades Per Day (prevent overtrading)
        if self.trades_today >= self.max_trades_per_day:
            return self._reject(f"MAX_TRADES: Daily limit reached ({self.max_trades_per_day})")

        # 10. Minimum Time Between Trades (prevent churning)
        if self.last_trade_time is not None:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.min_time_between_trades:
                remaining = self.min_time_between_trades - time_since_last
                return self._reject(f"COOLDOWN: Wait {remaining:.0f}s before next trade")

        # 11. Spread Filter (don't trade when spread is too wide)
        current_spread = market_data.get('spread', 0.0)
        if current_spread > self.max_spread:
            return self._reject(f"SPREAD_TOO_WIDE: {current_spread:.5f} > {self.max_spread:.5f}")

        # 12. Market Hours Check (optional - avoid trading at market open/close)
        if 'is_market_open' in market_data and not market_data['is_market_open']:
            return self._reject("MARKET_CLOSED: Trading outside market hours")

        # All checks passed!
        return self._approve()

    def update_state(self, pnl, equity, is_win=None):
        """
        Update supervisor state after each trade

        Args:
            pnl: Profit/loss from this trade
            equity: Current equity
            is_win: Whether trade was profitable (optional)
        """

        # Update equity tracking
        self.daily_pnl += pnl
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        # Update trade counters
        self.trades_today += 1
        self.last_trade_time = datetime.now()

        # Track consecutive losses
        if is_win is not None:
            if is_win:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

        # Log to history
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'equity': equity,
            'daily_pnl': self.daily_pnl,
            'is_win': is_win,
        })

        # Log warnings
        if self.consecutive_losses >= 3:
            logger.warning(f"⚠️ {self.consecutive_losses} consecutive losses")

        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if current_drawdown > 0.10:
            logger.warning(f"⚠️ Drawdown: {current_drawdown:.2%}")

    def reset_daily(self):
        """Reset daily counters (call at midnight UTC)"""
        logger.info(f"📊 Daily reset - PnL: {self.daily_pnl:.4f}, Trades: {self.trades_today}")

        self.daily_pnl = 0.0
        self.trades_today = 0

        # Clear halt if it was just for the day
        if self.halt_until and datetime.now() >= self.halt_until:
            self.halt_until = None
            logger.info("✅ Trading halt cleared")

    def emergency_shutdown(self):
        """
        Emergency shutdown - halt all trading

        Call this if something goes very wrong
        """
        self.halt_until = datetime.now() + timedelta(days=365)  # Halt for 1 year (manual restart needed)
        logger.critical("🚨 EMERGENCY SHUTDOWN ACTIVATED")
        logger.critical("🚨 All trading halted - manual restart required")

        return "EMERGENCY_SHUTDOWN"

    def get_statistics(self):
        """Get approval/rejection statistics"""
        total = self.total_trades_approved + self.total_trades_rejected

        if total == 0:
            return {
                'total_checks': 0,
                'approval_rate': 0.0,
                'rejection_rate': 0.0,
                'rejection_reasons': {},
            }

        return {
            'total_checks': total,
            'approved': self.total_trades_approved,
            'rejected': self.total_trades_rejected,
            'approval_rate': self.total_trades_approved / total,
            'rejection_rate': self.total_trades_rejected / total,
            'rejection_reasons': self.rejection_reasons,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'consecutive_losses': self.consecutive_losses,
        }

    def _approve(self):
        """Internal: approve trade"""
        self.total_trades_approved += 1
        return True, "APPROVED"

    def _reject(self, reason):
        """Internal: reject trade"""
        self.total_trades_rejected += 1

        # Track rejection reasons
        if reason not in self.rejection_reasons:
            self.rejection_reasons[reason] = 0
        self.rejection_reasons[reason] += 1

        logger.warning(f"🚫 Trade REJECTED: {reason}")

        return False, reason


# Example integration wrapper
class SafeTradingAgent:
    """
    Wrapper that combines AI agent with Risk Supervisor

    The AI proposes trades, Risk Supervisor approves/rejects them.
    """

    def __init__(self, ai_agent, risk_supervisor=None):
        """
        Args:
            ai_agent: Your DreamerV3 agent
            risk_supervisor: RiskSupervisor instance
        """
        self.ai_agent = ai_agent
        self.risk_supervisor = risk_supervisor or RiskSupervisor()

        logger.info("✅ Safe Trading Agent initialized")

    def act(self, obs, state, market_data):
        """
        Get action with safety checks

        Args:
            obs: Observation for AI
            state: Current trading state
            market_data: Market conditions

        Returns:
            final_action: Approved action (may be overridden to 0)
            info: Dict with approval status and reason
        """

        # Get AI's decision
        ai_action = self.ai_agent.act(obs)

        # Risk supervisor checks it
        approved, reason = self.risk_supervisor.check_trade(ai_action, state, market_data)

        if approved:
            final_action = ai_action
            logger.info(f"✅ Trade approved: {ai_action}")
        else:
            final_action = 0  # Override to flat position
            logger.warning(f"🚫 Trade rejected: {reason}")

        return final_action, {
            'approved': approved,
            'reason': reason,
            'ai_action': ai_action,
            'final_action': final_action,
        }


# Example usage
if __name__ == "__main__":
    print("🛡️ Risk Supervisor Demo\n")

    # Create supervisor
    supervisor = RiskSupervisor()

    # Simulate some checks
    state = {'position': 0, 'equity': 1.0}
    market_data = {
        'volatility': 1.5,
        'spread': 0.0003,
        'dxy_momentum': -0.005,
        'is_high_impact_event': False,
        'is_event_window': False,
    }

    # Check 1: Normal trade (should approve)
    approved, reason = supervisor.check_trade(1, state, market_data)
    print(f"Check 1 (normal): {approved} - {reason}")

    # Check 2: High volatility (should reject new entries)
    market_data['volatility'] = 4.0
    approved, reason = supervisor.check_trade(1, state, market_data)
    print(f"Check 2 (high vol): {approved} - {reason}")
    market_data['volatility'] = 1.5  # Reset

    # Check 3: Wide spread (should reject)
    market_data['spread'] = 0.001  # 10 pips
    approved, reason = supervisor.check_trade(1, state, market_data)
    print(f"Check 3 (wide spread): {approved} - {reason}")
    market_data['spread'] = 0.0003  # Reset

    # Check 4: Daily loss limit
    supervisor.daily_pnl = -0.025  # Lost 2.5%
    approved, reason = supervisor.check_trade(1, state, market_data)
    print(f"Check 4 (daily loss): {approved} - {reason}")

    # Statistics
    print("\n📊 Statistics:")
    stats = supervisor.get_statistics()
    print(f"Total checks: {stats['total_checks']}")
    print(f"Approved: {stats['approved']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Approval rate: {stats['approval_rate']:.1%}")
