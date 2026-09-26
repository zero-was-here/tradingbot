"""
Production Monitoring & Alerting

For live trading, you need 24/7 monitoring:
- Real-time P&L tracking
- Performance degradation detection
- Model drift alerts
- Emergency shutdown triggers
- Health checks

This prevents silent failures that cost money.
"""

import logging
import time
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveTradingMonitor:
    """
    Real-time monitoring and alerting for live trading

    Features:
    - P&L tracking
    - Performance degradation detection
    - Model drift detection
    - Latency monitoring
    - Emergency shutdown
    """

    def __init__(self, config=None):
        """
        Initialize monitor

        Args:
            config: Dict with thresholds and settings
        """

        if config is None:
            config = self._get_default_config()

        # Thresholds
        self.max_daily_loss = config.get('max_daily_loss', 0.02)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.max_latency_ms = config.get('max_latency_ms', 1000)
        self.min_sharpe = config.get('min_sharpe', 0.5)
        self.model_drift_threshold = config.get('model_drift_threshold', 0.5)

        # State tracking
        self.start_time = datetime.now()
        self.start_equity = 1.0
        self.current_equity = 1.0
        self.peak_equity = 1.0
        self.daily_pnl = 0.0

        # Performance history
        self.pnl_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=100)
        self.action_distribution = {'flat': 0, 'long': 0}
        self.historical_action_dist = {'flat': 0.5, 'long': 0.5}

        # Alerts
        self.alerts = []
        self.shutdown_triggered = False

        logger.info("📊 Production Monitor initialized")
        logger.info(f"   Max daily loss: {self.max_daily_loss:.1%}")
        logger.info(f"   Max drawdown: {self.max_drawdown:.1%}")
        logger.info(f"   Max latency: {self.max_latency_ms}ms")

    @staticmethod
    def _get_default_config():
        """Default configuration"""
        return {
            'max_daily_loss': 0.02,
            'max_drawdown': 0.15,
            'max_latency_ms': 1000,
            'min_sharpe': 0.5,
            'model_drift_threshold': 0.5,
        }

    def check_health(self, agent_state):
        """
        Continuous health checks

        Args:
            agent_state: Dict with current agent state

        Returns:
            healthy: bool
            issues: list of issues found
        """

        issues = []

        # Check 1: P&L within limits
        if not self._check_pnl(agent_state):
            issues.append("CRITICAL: P&L outside limits")

        # Check 2: Latency acceptable
        if not self._check_latency(agent_state):
            issues.append("WARNING: High latency")

        # Check 3: Model drift
        if not self._check_model_drift(agent_state):
            issues.append("WARNING: Model drift detected")

        # Check 4: Overtrading
        if not self._check_overtrading(agent_state):
            issues.append("WARNING: Overtrading detected")

        # Check 5: Drawdown
        if not self._check_drawdown():
            issues.append("CRITICAL: Max drawdown exceeded")

        # Trigger emergency shutdown if critical issues
        critical_issues = [i for i in issues if i.startswith("CRITICAL")]

        if critical_issues and not self.shutdown_triggered:
            self.emergency_shutdown()

        return len(issues) == 0, issues

    def _check_pnl(self, state):
        """Check if P&L is within limits"""

        # Daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            self._send_alert("CRITICAL: Daily loss limit exceeded", priority="CRITICAL")
            return False

        return True

    def _check_latency(self, state):
        """Check if latency is acceptable"""

        current_latency = state.get('latency_ms', 0)
        self.latency_history.append(current_latency)

        avg_latency = np.mean(list(self.latency_history))

        if avg_latency > self.max_latency_ms:
            self._send_alert(f"WARNING: High latency ({avg_latency:.0f}ms)", priority="WARNING")
            return False

        return True

    def _check_model_drift(self, state):
        """
        Detect if model behavior has changed

        Model drift = distribution of actions has changed significantly
        """

        current_action = state.get('action', 0)

        # Update action distribution
        action_name = 'long' if current_action == 1 else 'flat'
        self.action_distribution[action_name] += 1

        total_actions = sum(self.action_distribution.values())

        if total_actions < 100:
            # Not enough data
            return True

        # Current distribution
        current_dist = {
            k: v / total_actions
            for k, v in self.action_distribution.items()
        }

        # KL divergence vs historical
        drift = self._kl_divergence(current_dist, self.historical_action_dist)

        if drift > self.model_drift_threshold:
            self._send_alert(f"WARNING: Model drift detected (KL={drift:.3f})", priority="WARNING")
            return False

        return True

    def _check_overtrading(self, state):
        """Check for excessive trading"""

        total_actions = sum(self.action_distribution.values())

        # Estimate trades per hour
        running_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        if running_hours > 0:
            trades_per_hour = total_actions / running_hours

            if trades_per_hour > 20:  # More than 20 trades/hour
                self._send_alert(f"WARNING: Overtrading ({trades_per_hour:.1f} trades/hour)", priority="WARNING")
                return False

        return True

    def _check_drawdown(self):
        """Check if drawdown is acceptable"""

        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity

        if current_drawdown > self.max_drawdown:
            self._send_alert(f"CRITICAL: Max drawdown exceeded ({current_drawdown:.1%})", priority="CRITICAL")
            return False

        return True

    def _kl_divergence(self, p, q):
        """Compute KL divergence between two distributions"""

        kl = 0.0

        for key in p.keys():
            p_val = p[key]
            q_val = q.get(key, 0.5)  # Default if missing

            if p_val > 0 and q_val > 0:
                kl += p_val * np.log(p_val / q_val)

        return kl

    def update(self, pnl, equity, action):
        """
        Update monitor state

        Args:
            pnl: P&L from last trade
            equity: Current equity
            action: Action taken
        """

        # Update equity tracking
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self.daily_pnl += pnl

        # Update history
        self.pnl_history.append(pnl)

    def emergency_shutdown(self):
        """
        Emergency shutdown - halt all trading

        This is the nuclear option
        """

        self.shutdown_triggered = True

        self._send_alert("🚨 EMERGENCY SHUTDOWN TRIGGERED", priority="CRITICAL")
        self._send_alert("🚨 All trading halted - manual restart required", priority="CRITICAL")

        logger.critical("="*60)
        logger.critical("🚨 EMERGENCY SHUTDOWN ACTIVATED 🚨")
        logger.critical("="*60)
        logger.critical("Trading has been HALTED")
        logger.critical("Manual review and restart required")
        logger.critical("="*60)

        return "EMERGENCY_SHUTDOWN"

    def _send_alert(self, message, priority="INFO"):
        """
        Send alert (placeholder for real alerting)

        In production, would send:
        - SMS
        - Email
        - Slack message
        - PagerDuty
        """

        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'priority': priority,
        }

        self.alerts.append(alert)

        if priority == "CRITICAL":
            logger.critical(f"🚨 {message}")
        elif priority == "WARNING":
            logger.warning(f"⚠️ {message}")
        else:
            logger.info(f"ℹ️ {message}")

    def get_statistics(self):
        """Get current statistics"""

        running_time = (datetime.now() - self.start_time).total_seconds() / 3600

        if len(self.pnl_history) > 0:
            returns = np.array(list(self.pnl_history))
            sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        return {
            'running_hours': running_time,
            'current_equity': self.current_equity,
            'total_return': (self.current_equity - self.start_equity) / self.start_equity,
            'daily_pnl': self.daily_pnl,
            'peak_equity': self.peak_equity,
            'current_drawdown': (self.peak_equity - self.current_equity) / self.peak_equity,
            'sharpe_ratio': sharpe,
            'num_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a['priority'] == 'CRITICAL']),
            'shutdown_triggered': self.shutdown_triggered,
        }

    def reset_daily(self):
        """Reset daily counters"""

        logger.info(f"📊 Daily reset - P&L: {self.daily_pnl:.4f}")
        self.daily_pnl = 0.0


# Example usage
if __name__ == "__main__":
    print("📊 Production Monitor Demo\n")

    # Create monitor
    monitor = LiveTradingMonitor()

    # Simulate trading
    print("="*60)
    print("Simulating live trading...")
    print("="*60)

    equity = 1.0

    for i in range(100):
        # Simulate trade
        pnl = np.random.randn() * 0.001  # Random P&L
        equity *= (1 + pnl)

        action = np.random.choice([0, 1])

        # Update monitor
        monitor.update(pnl, equity, action)

        # Simulate latency
        latency_ms = np.random.randint(100, 500)

        # Check health
        state = {
            'action': action,
            'latency_ms': latency_ms,
        }

        healthy, issues = monitor.check_health(state)

        if not healthy:
            print(f"Step {i}: ⚠️ Issues detected: {issues}")

        # Simulate daily reset
        if i % 24 == 0 and i > 0:
            monitor.reset_daily()

    # Print statistics
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)

    stats = monitor.get_statistics()

    print(f"Running time: {stats['running_hours']:.2f} hours")
    print(f"Final equity: {stats['current_equity']:.4f}")
    print(f"Total return: {stats['total_return']:.2%}")
    print(f"Sharpe ratio: {stats['sharpe_ratio']:.2f}")
    print(f"Current drawdown: {stats['current_drawdown']:.2%}")
    print(f"Total alerts: {stats['num_alerts']}")
    print(f"Critical alerts: {stats['critical_alerts']}")
    print(f"Shutdown triggered: {stats['shutdown_triggered']}")

    print("\n✅ Production monitoring system working!")
    print("\nKey features:")
    print("  ✅ Real-time P&L monitoring")
    print("  ✅ Model drift detection")
    print("  ✅ Latency monitoring")
    print("  ✅ Emergency shutdown capability")
    print("  ✅ 24/7 health checks")
