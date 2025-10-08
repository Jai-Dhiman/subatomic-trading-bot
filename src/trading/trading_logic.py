"""
Trading logic and decision-making for energy nodes.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from src.models.battery_manager import BatteryState


@dataclass
class TradingDecision:
    """Represents a trading decision made by a node."""
    action: str
    quantity: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0


class TradingStrategy:
    """Trading strategy for a household node."""
    
    def __init__(self, config: dict):
        """
        Initialize trading strategy.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.max_single_trade = config.get('max_single_trade_kwh', 2.0)
        self.min_profit_margin = config.get('min_profit_margin', 0.05)
        
    def make_decision(
        self,
        predicted_consumption: np.ndarray,
        battery_state: BatteryState,
        market_price: float,
        pge_price: float
    ) -> TradingDecision:
        """
        Make autonomous trading decision.
        
        Args:
            predicted_consumption: Predicted consumption for next intervals (kWh)
            battery_state: Current battery state
            market_price: Current market price ($/kWh)
            pge_price: PG&E baseline price ($/kWh)
            
        Returns:
            TradingDecision with action and parameters
        """
        total_predicted = np.sum(predicted_consumption)
        battery_available = battery_state.available_energy()
        battery_capacity = battery_state.available_capacity()
        
        net_position = battery_available - total_predicted
        
        if net_position < -1.0:
            return self._decide_buy(
                deficit=abs(net_position),
                battery_capacity=battery_capacity,
                market_price=market_price,
                pge_price=pge_price
            )
        
        elif net_position > 1.5:
            return self._decide_sell(
                excess=net_position,
                market_price=market_price,
                pge_price=pge_price
            )
        
        else:
            return TradingDecision(action='hold')
    
    def _decide_buy(
        self,
        deficit: float,
        battery_capacity: float,
        market_price: float,
        pge_price: float
    ) -> TradingDecision:
        """Decide on buying energy."""
        quantity = min(
            deficit * 0.5,
            battery_capacity,
            self.max_single_trade
        )
        
        if market_price < pge_price * 0.95:
            return TradingDecision(
                action='buy',
                quantity=quantity,
                max_price=pge_price * 0.95
            )
        else:
            return TradingDecision(action='hold')
    
    def _decide_sell(
        self,
        excess: float,
        market_price: float,
        pge_price: float
    ) -> TradingDecision:
        """Decide on selling energy."""
        quantity = min(
            excess * 0.3,
            self.max_single_trade
        )
        
        cost_basis = pge_price * 0.7
        
        if market_price > cost_basis * (1 + self.min_profit_margin):
            return TradingDecision(
                action='sell',
                quantity=quantity,
                min_price=cost_basis * (1 + self.min_profit_margin)
            )
        else:
            return TradingDecision(action='hold')


class TradingConstraints:
    """Manages trading constraints and daily limits."""
    
    def __init__(self, config: dict):
        """
        Initialize trading constraints.
        
        Args:
            config: Trading configuration
        """
        self.max_sell_per_day = config.get('max_sell_per_day_kwh', 10.0)
        self.max_single_trade = config.get('max_single_trade_kwh', 2.0)
        self.sold_today = 0.0
        
    def can_sell(self, quantity: float) -> bool:
        """Check if node can sell given quantity."""
        return (
            self.sold_today + quantity <= self.max_sell_per_day
            and quantity <= self.max_single_trade
            and quantity > 0
        )
    
    def can_buy(self, quantity: float) -> bool:
        """Check if node can buy given quantity."""
        return quantity <= self.max_single_trade and quantity > 0
    
    def record_sale(self, quantity: float):
        """Record a sale transaction."""
        self.sold_today += quantity
    
    def reset_daily(self):
        """Reset daily counters."""
        self.sold_today = 0.0
    
    def get_remaining_sell_capacity(self) -> float:
        """Get remaining sell capacity for today."""
        return max(0, self.max_sell_per_day - self.sold_today)


if __name__ == "__main__":
    from src.models.battery_manager import BatteryState
    
    config = {
        'max_single_trade_kwh': 2.0,
        'min_profit_margin': 0.05,
        'max_sell_per_day_kwh': 10.0
    }
    
    strategy = TradingStrategy(config)
    constraints = TradingConstraints(config)
    
    battery = BatteryState(
        capacity_kwh=13.5,
        current_charge_kwh=6.0,
        efficiency=0.90
    )
    
    predicted = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.0])
    
    decision = strategy.make_decision(
        predicted_consumption=predicted,
        battery_state=battery,
        market_price=0.40,
        pge_price=0.51
    )
    
    print(f"Decision: {decision.action}")
    print(f"Quantity: {decision.quantity:.2f} kWh")
    print(f"Max price: ${decision.max_price:.3f}/kWh")
    print(f"Min price: ${decision.min_price:.3f}/kWh")
    
    print(f"\nCan sell 1.5 kWh: {constraints.can_sell(1.5)}")
    print(f"Remaining sell capacity: {constraints.get_remaining_sell_capacity():.2f} kWh")
