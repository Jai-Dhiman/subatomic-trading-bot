"""
Simplified household agent for demo simulation.

No database dependencies - all state managed in memory.
Uses trained Transformer model for predictions and autonomous trading decisions.
"""

import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BatteryState:
    """Battery state information."""
    capacity_kwh: float = 13.5
    current_charge_kwh: float = 6.75  # Start at 50%
    soc_percent: float = 50.0
    soh_percent: float = 100.0
    cycle_count: float = 0.0
    efficiency: float = 0.90
    max_charge_rate_kw: float = 5.0
    max_discharge_rate_kw: float = 5.0
    min_reserve_percent: float = 10.0
    max_capacity_percent: float = 80.0


@dataclass
class TradeDecision:
    """Trading decision from household."""
    action: str  # 'buy', 'sell', or 'hold'
    quantity: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0


class DemoHouseholdAgent:
    """
    Simplified household agent for demo.
    
    Manages predictions, battery state, trading decisions, and cost tracking.
    """
    
    def __init__(
        self,
        household_id: int,
        model: torch.nn.Module,
        feature_engineer,
        battery_config: dict = None
    ):
        """
        Initialize household agent.
        
        Args:
            household_id: Unique household identifier
            model: Trained Transformer model
            feature_engineer: FeatureEngineer instance for preprocessing
            battery_config: Battery configuration dict
        """
        self.household_id = household_id
        self.model = model
        self.feature_engineer = feature_engineer
        
        # Initialize battery
        if battery_config:
            self.battery = BatteryState(**battery_config)
        else:
            self.battery = BatteryState()
        
        # Tracking
        self.timeline = []
        self.trades = []
        self.predictions_cache = {}
        
        # Costs
        self.baseline_cost = 0.0
        self.optimized_cost = 0.0
        
        # Statistics
        self.total_charge_kwh = 0.0
        self.total_discharge_kwh = 0.0
    
    def predict_consumption(
        self,
        recent_features: np.ndarray,
        interval: int
    ) -> Dict[str, np.ndarray]:
        """
        Predict future consumption and prices.
        
        Args:
            recent_features: Recent feature history (48 timesteps × 24 features)
            interval: Current interval number
            
        Returns:
            Dict with predictions for day and week horizons
        """
        # Cache predictions to avoid redundant computation
        if interval in self.predictions_cache:
            return self.predictions_cache[interval]
        
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension
            X = torch.FloatTensor(recent_features).unsqueeze(0)
            predictions = self.model(X)
            
            # Convert to numpy
            result = {
                key: val.cpu().numpy().squeeze()
                for key, val in predictions.items()
            }
        
        self.predictions_cache[interval] = result
        return result
    
    def make_trading_decision(
        self,
        predicted_consumption_next: float,
        market_price: float,
        pge_price: float,
        battery_soc: float
    ) -> TradeDecision:
        """
        Make autonomous trading decision.
        
        Strategy:
        - Buy if: market price < PGE price AND battery not full AND predicted high consumption
        - Sell if: market price > cost AND battery has excess AND predicted low consumption
        - Hold otherwise
        
        Args:
            predicted_consumption_next: Predicted consumption for next interval
            market_price: Current P2P market price
            pge_price: PG&E baseline price
            battery_soc: Current battery state of charge (%)
            
        Returns:
            TradeDecision with action and parameters
        """
        # Calculate battery status
        battery_available = self.battery.current_charge_kwh
        battery_space = (self.battery.max_capacity_percent / 100 * self.battery.capacity_kwh) - self.battery.current_charge_kwh
        
        # Normalize consumption prediction (typical range 0.5-8 kW)
        high_consumption = predicted_consumption_next > 3.0
        low_consumption = predicted_consumption_next < 2.0
        
        # BUY Decision: market is cheap AND (battery has space OR high consumption coming)
        if market_price < pge_price * 0.9 and (battery_space > 1.0 or high_consumption):
            # Buy amount based on prediction and battery space
            if high_consumption:
                quantity = min(2.0, predicted_consumption_next * 0.5, battery_space)
            else:
                quantity = min(1.5, battery_space * 0.5)
            
            if quantity > 0.1:  # Min trade size
                return TradeDecision(
                    action='buy',
                    quantity=quantity,
                    max_price=pge_price * 0.95  # Willing to pay up to 95% of PGE price
                )
        
        # SELL Decision: market is profitable AND battery has excess AND low consumption
        elif market_price > pge_price * 0.8 and battery_available > 3.0 and low_consumption:
            # Sell excess while maintaining reserve
            available_to_sell = battery_available - (self.battery.min_reserve_percent / 100 * self.battery.capacity_kwh)
            quantity = min(2.0, available_to_sell * 0.3)  # Sell conservatively
            
            if quantity > 0.1:
                return TradeDecision(
                    action='sell',
                    quantity=quantity,
                    min_price=pge_price * 0.75  # Accept 75% of PGE price minimum
                )
        
        # HOLD by default
        return TradeDecision(action='hold')
    
    def execute_trade(
        self,
        action: str,
        quantity: float,
        price: float,
        counterparty_id: int,
        interval: int
    ):
        """
        Execute a trade and update battery/costs.
        
        Args:
            action: 'buy' or 'sell'
            quantity: Energy quantity (kWh)
            price: Price per kWh
            counterparty_id: Other household ID
            interval: Current interval
        """
        if action == 'buy':
            # Charge battery
            actual_charge = min(quantity, (self.battery.max_capacity_percent / 100 * self.battery.capacity_kwh) - self.battery.current_charge_kwh)
            self.battery.current_charge_kwh += actual_charge * self.battery.efficiency
            self.total_charge_kwh += actual_charge
            
            # Pay for energy
            cost = quantity * price
            self.optimized_cost += cost
            
        elif action == 'sell':
            # Discharge battery
            actual_discharge = min(
                quantity,
                self.battery.current_charge_kwh - (self.battery.min_reserve_percent / 100 * self.battery.capacity_kwh)
            )
            self.battery.current_charge_kwh -= actual_discharge
            self.total_discharge_kwh += actual_discharge / self.battery.efficiency
            
            # Receive payment
            revenue = quantity * price
            self.optimized_cost -= revenue
        
        # Update SoC
        self.battery.soc_percent = (self.battery.current_charge_kwh / self.battery.capacity_kwh) * 100
        
        # Record trade
        self.trades.append({
            'interval': interval,
            'action': action,
            'counterparty': counterparty_id,
            'quantity_kwh': quantity,
            'price_per_kwh': price,
            'total_cost': quantity * price
        })
    
    def consume_energy(
        self,
        actual_consumption: float,
        pge_price: float,
        interval: int
    ):
        """
        Handle energy consumption for the interval.
        
        Args:
            actual_consumption: Actual consumption (kWh)
            pge_price: PG&E baseline price
            interval: Current interval
        """
        # Baseline cost (what we would pay without optimization)
        self.baseline_cost += actual_consumption * pge_price
        
        # Try to use battery first
        battery_available = max(0, self.battery.current_charge_kwh - (self.battery.min_reserve_percent / 100 * self.battery.capacity_kwh))
        energy_from_battery = min(battery_available, actual_consumption)
        
        if energy_from_battery > 0:
            self.battery.current_charge_kwh -= energy_from_battery
            self.total_discharge_kwh += energy_from_battery / self.battery.efficiency
            remaining = actual_consumption - energy_from_battery
        else:
            remaining = actual_consumption
        
        # Buy remaining from grid at PGE price
        if remaining > 0:
            self.optimized_cost += remaining * pge_price
        
        # Update SoC
        self.battery.soc_percent = (self.battery.current_charge_kwh / self.battery.capacity_kwh) * 100
    
    def record_interval(
        self,
        interval: int,
        timestamp: datetime,
        predicted: float,
        actual: float,
        market_price: float,
        pge_price: float,
        action: str
    ):
        """Record metrics for this interval."""
        self.timeline.append({
            'interval': interval,
            'timestamp': timestamp.isoformat(),
            'predicted_consumption_kwh': round(predicted, 3),
            'actual_consumption_kwh': round(actual, 3),
            'battery_soc_percent': round(self.battery.soc_percent, 2),
            'battery_charge_kwh': round(self.battery.current_charge_kwh, 3),
            'market_price': round(market_price, 4),
            'pge_baseline_price': round(pge_price, 4),
            'action': action
        })
    
    def get_results(self) -> Dict:
        """Get complete results for this household."""
        savings = self.baseline_cost - self.optimized_cost
        savings_percent = (savings / self.baseline_cost * 100) if self.baseline_cost > 0 else 0
        
        # Update battery cycle count (approximate)
        self.battery.cycle_count = self.total_discharge_kwh / self.battery.capacity_kwh
        
        return {
            'id': self.household_id,
            'timeline': self.timeline,
            'trades': self.trades,
            'costs': {
                'baseline_pge_total': round(self.baseline_cost, 2),
                'optimized_total': round(self.optimized_cost, 2),
                'savings': round(savings, 2),
                'savings_percent': round(savings_percent, 1)
            },
            'battery_stats': {
                'cycles_completed': round(self.battery.cycle_count, 3),
                'total_charge_kwh': round(self.total_charge_kwh, 2),
                'total_discharge_kwh': round(self.total_discharge_kwh, 2),
                'final_soc_percent': round(self.battery.soc_percent, 2),
                'efficiency_percent': round(self.battery.efficiency * 100, 1)
            }
        }
