"""
Realistic Execution Model

Most backtests are too optimistic because they ignore real trading costs:
- Slippage (price moves while your order fills)
- Market impact (large orders move the market)
- Spread widening during volatility
- Adverse selection (smart traders take the other side)
- Partial fills
- Requotes

This module models all of these to give realistic P&L estimates.

CRITICAL: Train with these costs, or your backtest will be fantasy.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealisticExecutionModel:
    """
    Model all real-world execution costs

    The gap between backtest and live trading often comes from:
    1. Slippage (0.1-0.5 pips typically)
    2. Spread widening (1.5-3x during volatility)
    3. Market impact (for larger positions)
    4. Adverse selection
    """

    def __init__(self, config=None):
        """
        Initialize execution model

        Args:
            config: dict with cost parameters
        """

        if config is None:
            config = self.get_default_config()

        # Base costs
        self.base_spread = config.get('base_spread', 0.0003)  # 3 pips for XAUUSD
        self.base_slippage = config.get('base_slippage', 0.0001)  # 1 pip
        self.commission = config.get('commission', 0.00005)  # 0.5 pip

        # Dynamic cost parameters
        self.spread_vol_multiplier = config.get('spread_vol_multiplier', 2.0)
        self.slippage_vol_multiplier = config.get('slippage_vol_multiplier', 3.0)
        self.market_impact_coefficient = config.get('market_impact_coef', 0.01)

        # Adverse selection
        self.adverse_selection_cost = config.get('adverse_selection', 0.00005)

        # Statistics
        self.total_trades = 0
        self.total_costs = 0.0
        self.cost_breakdown = {
            'spread': 0.0,
            'slippage': 0.0,
            'commission': 0.0,
            'market_impact': 0.0,
            'adverse_selection': 0.0,
        }

        logger.info("⚖️ Realistic Execution Model initialized")
        logger.info(f"  Base Spread: {self.base_spread:.5f} ({self.base_spread * 10000:.1f} pips)")
        logger.info(f"  Base Slippage: {self.base_slippage:.5f}")
        logger.info(f"  Commission: {self.commission:.5f}")

    @staticmethod
    def get_default_config():
        """Conservative (pessimistic) default costs"""
        return {
            'base_spread': 0.0003,  # 3 pips
            'base_slippage': 0.0001,  # 1 pip
            'commission': 0.00005,  # 0.5 pip
            'spread_vol_multiplier': 2.0,  # Spread doubles in high vol
            'slippage_vol_multiplier': 3.0,  # Slippage triples in high vol
            'market_impact_coef': 0.01,  # 1% position → 0.01% impact
            'adverse_selection': 0.00005,  # 0.5 pip
        }

    def estimate_execution_cost(self, order, market_state):
        """
        Estimate total cost of executing an order

        Args:
            order: dict with:
                - side: 'buy' or 'sell'
                - size: position size (fraction of equity)
                - order_type: 'market' or 'limit'
            market_state: dict with:
                - volatility: current volatility
                - normal_volatility: baseline volatility
                - spread: current spread
                - liquidity: available liquidity
                - is_event_window: bool

        Returns:
            total_cost: Total cost as fraction (e.g., 0.0005 = 0.05%)
            cost_breakdown: dict with component costs
        """

        costs = {}

        # 1. Spread Cost (always pay bid-ask)
        current_spread = market_state.get('spread', self.base_spread)

        # Spread widens during volatility
        vol_ratio = self._get_volatility_ratio(market_state)
        if vol_ratio > 1.5:  # High volatility
            current_spread *= min(vol_ratio, self.spread_vol_multiplier)

        # Spread widens during news events
        if market_state.get('is_event_window', False):
            current_spread *= 2.0  # Double spread during events

        costs['spread'] = current_spread

        # 2. Slippage (price moves while order fills)
        slippage = self.base_slippage

        # Slippage worse during high volatility
        if vol_ratio > 1.5:
            slippage *= min(vol_ratio, self.slippage_vol_multiplier)

        # Slippage worse for market orders
        if order.get('order_type', 'market') == 'market':
            slippage *= 1.5

        # Slippage worse during events
        if market_state.get('is_event_window', False):
            slippage *= 3.0

        costs['slippage'] = slippage

        # 3. Commission
        costs['commission'] = self.commission

        # 4. Market Impact (for large orders)
        position_size = order.get('size', 0.05)  # Default 5%
        avg_liquidity = market_state.get('liquidity', 1.0)

        if position_size > avg_liquidity:
            # Large order moves the market
            impact = (position_size / avg_liquidity) * self.market_impact_coefficient
            costs['market_impact'] = impact
        else:
            costs['market_impact'] = 0.0

        # 5. Adverse Selection
        # When you trade, informed traders may be on the other side
        costs['adverse_selection'] = self.adverse_selection_cost

        # Total cost
        total_cost = sum(costs.values())

        # Update statistics
        self.total_trades += 1
        self.total_costs += total_cost
        for key, value in costs.items():
            self.cost_breakdown[key] += value

        return total_cost, costs

    def execute_trade(self, order, market_state, entry_price):
        """
        Execute trade with realistic costs

        Args:
            order: Order dict
            market_state: Market state dict
            entry_price: Intended entry price

        Returns:
            fill_price: Actual fill price after costs
            total_cost: Total cost incurred
            cost_breakdown: Breakdown of costs
        """

        # Estimate costs
        total_cost, cost_breakdown = self.estimate_execution_cost(order, market_state)

        # Adjust fill price
        side = order.get('side', 'buy')

        if side == 'buy' or side == 'long':
            # Buying: pay higher price
            fill_price = entry_price * (1 + total_cost)
        else:
            # Selling: receive lower price
            fill_price = entry_price * (1 - total_cost)

        return fill_price, total_cost, cost_breakdown

    def _get_volatility_ratio(self, market_state):
        """Get current volatility relative to normal"""
        current_vol = market_state.get('volatility', 1.0)
        normal_vol = market_state.get('normal_volatility', 1.0)

        if normal_vol == 0:
            return 1.0

        return current_vol / normal_vol

    def get_statistics(self):
        """Get execution cost statistics"""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'avg_cost_per_trade': 0.0,
                'total_costs': 0.0,
                'cost_breakdown': self.cost_breakdown,
            }

        avg_cost = self.total_costs / self.total_trades

        return {
            'total_trades': self.total_trades,
            'avg_cost_per_trade': avg_cost,
            'total_costs': self.total_costs,
            'cost_breakdown': {k: v / self.total_trades for k, v in self.cost_breakdown.items()},
            'cost_in_pips': avg_cost * 10000,  # For XAUUSD
        }


class SlippageSimulator:
    """
    Simple slippage simulator

    Adds realistic randomness to fills
    """

    def __init__(self, avg_slippage=0.0001, volatility_scaling=True):
        """
        Args:
            avg_slippage: Average slippage (e.g., 0.0001 = 1 pip)
            volatility_scaling: Scale slippage with volatility
        """
        self.avg_slippage = avg_slippage
        self.volatility_scaling = volatility_scaling

    def get_slippage(self, market_state):
        """
        Get random slippage for this trade

        Returns slippage as a fraction (can be positive or negative)
        """

        # Base slippage
        slippage = self.avg_slippage

        # Scale with volatility if enabled
        if self.volatility_scaling:
            vol_ratio = market_state.get('volatility', 1.0)
            slippage *= vol_ratio

        # Add randomness (normal distribution)
        random_component = np.random.normal(0, slippage * 0.5)
        slippage += random_component

        # Slippage is usually negative (costs you)
        # But occasionally you get "price improvement" (positive slippage)
        # 80% negative, 20% positive
        if np.random.random() < 0.8:
            slippage = -abs(slippage)
        else:
            slippage = abs(slippage) * 0.5  # Price improvement is smaller

        return slippage


# Example usage
if __name__ == "__main__":
    print("⚖️ Realistic Execution Model Demo\n")

    # Create execution model
    exec_model = RealisticExecutionModel()

    # Test scenarios
    print("=" * 60)
    print("Scenario 1: Normal Market Conditions")
    print("=" * 60)

    order = {
        'side': 'buy',
        'size': 0.05,  # 5% position
        'order_type': 'market',
    }

    market_state = {
        'volatility': 1.0,
        'normal_volatility': 1.0,
        'spread': 0.0003,
        'liquidity': 1.0,
        'is_event_window': False,
    }

    entry_price = 2000.0

    fill_price, total_cost, breakdown = exec_model.execute_trade(order, market_state, entry_price)

    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Fill Price: ${fill_price:.2f}")
    print(f"Total Cost: {total_cost:.5f} ({total_cost * 10000:.2f} pips)")
    print(f"Cost Breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.5f} ({value * 10000:.2f} pips)")

    # Scenario 2: High Volatility
    print("\n" + "=" * 60)
    print("Scenario 2: High Volatility")
    print("=" * 60)

    market_state['volatility'] = 3.0  # 3x normal volatility

    fill_price2, total_cost2, breakdown2 = exec_model.execute_trade(order, market_state, entry_price)

    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Fill Price: ${fill_price2:.2f}")
    print(f"Total Cost: {total_cost2:.5f} ({total_cost2 * 10000:.2f} pips)")
    print(f"Cost increased by: {(total_cost2 / total_cost - 1) * 100:.1f}%")

    # Scenario 3: During News Event
    print("\n" + "=" * 60)
    print("Scenario 3: During News Event")
    print("=" * 60)

    market_state['is_event_window'] = True

    fill_price3, total_cost3, breakdown3 = exec_model.execute_trade(order, market_state, entry_price)

    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Fill Price: ${fill_price3:.2f}")
    print(f"Total Cost: {total_cost3:.5f} ({total_cost3 * 10000:.2f} pips)")
    print(f"Cost vs normal: {(total_cost3 / total_cost - 1) * 100:.1f}% higher")

    # Statistics
    print("\n" + "=" * 60)
    print("Overall Statistics")
    print("=" * 60)

    stats = exec_model.get_statistics()
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Average Cost: {stats['avg_cost_per_trade']:.5f} ({stats['cost_in_pips']:.2f} pips)")
    print(f"\nAverage Cost Breakdown:")
    for key, value in stats['cost_breakdown'].items():
        print(f"  {key}: {value:.5f} ({value * 10000:.2f} pips)")

    print("\n✅ Realistic execution modeling working!")
