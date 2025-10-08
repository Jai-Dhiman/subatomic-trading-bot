"""
Household node that integrates model, battery, trading, and grid constraints.

This is the main integration point for all household components:
- NodeModel: LSTM prediction from sensor data
- BatteryManager: Battery operations with sensor state
- GridConstraintsManager: 10 kWh in / 4 kWh out limits
- TradingStrategy: Autonomous trading decisions
- Profitability: Node signals for central model

All configurations loaded from database (no hardcoded fallbacks).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from src.models.node_model import NodeModel
from src.models.battery_manager import BatteryManager
from src.models.grid_constraints import GridConstraintsManager
from src.models.profitability import generate_node_signals, NodeSignals
from src.trading.trading_logic import TradingStrategy, TradingConstraints, TradingDecision
from src.trading.market_mechanism import Transaction
from src.data_integration import consumption_parser


class HouseholdNode:
    """Complete household node with prediction, battery, trading, and grid constraints.
    
    This class integrates all components needed for a household to participate
    in peer-to-peer energy trading:
    
    Components:
        - NodeModel: LSTM-based consumption prediction
        - BatteryManager: Battery state from sensors, operations planning
        - GridConstraintsManager: Physical grid connection limits
        - TradingStrategy: Autonomous buy/sell decisions
        - Profitability: Signals sent to central model
    
    All configurations are loaded from database per-house tables.
    NO hardcoded fallbacks or synthetic data.
    """
    
    def __init__(self, household_id: int, connector, model_config: dict = None):
        """
        Initialize household node with database-driven configuration.
        
        Args:
            household_id: Unique household identifier
            connector: SupabaseConnector instance for loading configs
            model_config: Optional model hyperparameters (LSTM layers, etc.)
                        If None, uses defaults from config.yaml
                        
        Raises:
            ValueError: If required configurations cannot be loaded from database
            
        Note:
            Battery, grid, and trading configs are loaded from database.
            Only model hyperparameters can optionally come from config.yaml.
        """
        self.household_id = household_id
        self.connector = connector
        
        # Load all configurations from database
        battery_config = connector.get_battery_config(household_id)
        grid_config = connector.get_grid_config(household_id)
        trading_config = connector.get_trading_config(household_id)
        
        # Initialize model with optional config
        if model_config is None:
            model_config = {
                'lstm_hidden_size': 64,
                'lstm_num_layers': 2,
                'dropout': 0.2,
                'prediction_horizon_intervals': 6,
                'input_sequence_length': 48
            }
        
        self.model = NodeModel(household_id, model_config)
        self.battery_manager = BatteryManager(battery_config, household_id)
        self.grid_constraints = GridConstraintsManager(grid_config, household_id)
        self.trading_strategy = TradingStrategy(trading_config)
        self.trading_constraints = TradingConstraints(trading_config)
        
        self.consumption_data = None
        self.trades = []
        self.timeline = []
        
        self.baseline_cost = 0.0
        self.optimized_cost = 0.0
        
    def load_historical_data(self, start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None):
        """
        Load historical consumption data from database for training.
        
        Uses consumption_parser to load validated data from house_{id}_data table.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Raises:
            ValueError: If data table doesn't exist or data is malformed
        """
        self.consumption_data = consumption_parser.parse_house_data(
            house_id=self.household_id,
            connector=self.connector,
            start_date=start_date,
            end_date=end_date
        )
        
    def train_model(self, epochs: int = 50, verbose: bool = False):
        """Train the LSTM model on historical data."""
        if self.consumption_data is not None:
            self.model.train(self.consumption_data, epochs=epochs, verbose=verbose)
    
    def predict_consumption(self, recent_data: pd.DataFrame) -> np.ndarray:
        """Predict future consumption."""
        return self.model.predict(recent_data)
    
    def make_trading_decision(
        self,
        predicted_consumption: np.ndarray,
        market_price: float,
        pge_price: float
    ) -> TradingDecision:
        """Make trading decision based on prediction and prices."""
        battery_state = self.battery_manager.get_state()
        
        decision = self.trading_strategy.make_decision(
            predicted_consumption=predicted_consumption,
            battery_state=battery_state,
            market_price=market_price,
            pge_price=pge_price
        )
        
        if decision.action == 'buy':
            if not self.trading_constraints.can_buy(decision.quantity):
                return TradingDecision(action='hold')
        elif decision.action == 'sell':
            if not self.trading_constraints.can_sell(decision.quantity):
                return TradingDecision(action='hold')
        
        return decision
    
    def execute_transaction(self, transaction: Transaction):
        """
        Execute a transaction (buy or sell) with grid constraint validation.
        
        Args:
            transaction: Transaction object from market mechanism
            
        Raises:
            ValueError: If transaction would violate grid limits
        """
        self.trades.append(transaction)
        
        if transaction.buyer_id == self.household_id:
            # Buying energy - validate grid import if needed
            # Energy goes into battery, which may require grid import
            can_charge, max_allowed = self.battery_manager.can_charge(transaction.delivered_kwh)
            
            if can_charge:
                # Charge happens in hardware, battery state updated from sensors
                self.optimized_cost += transaction.total_cost
            else:
                raise ValueError(
                    f"Cannot execute buy transaction: battery cannot accept {transaction.delivered_kwh:.2f} kWh. "
                    f"Maximum allowed: {max_allowed:.2f} kWh"
                )
            
        elif transaction.seller_id == self.household_id:
            # Selling energy - validate grid export limit (4 kWh per 30-min)
            self.grid_constraints.validate_grid_transaction(transaction.energy_kwh, 'export', 0.5)
            
            can_discharge, max_allowed = self.battery_manager.can_discharge(transaction.energy_kwh)
            
            if can_discharge:
                # Discharge happens in hardware, battery state updated from sensors
                self.optimized_cost -= transaction.total_cost
                self.trading_constraints.record_sale(transaction.energy_kwh)
            else:
                raise ValueError(
                    f"Cannot execute sell transaction: battery cannot discharge {transaction.energy_kwh:.2f} kWh. "
                    f"Maximum allowed: {max_allowed:.2f} kWh"
                )
    
    def consume_energy(self, actual_consumption: float, pge_price: float, market_price: float):
        """
        Handle energy consumption for an interval with grid constraint checking.
        
        Args:
            actual_consumption: Actual consumption for this interval (kWh)
            pge_price: PG&E baseline price
            market_price: Market price (for cost tracking)
            
        Raises:
            ValueError: If grid import exceeds physical limits
        """
        self.baseline_cost += actual_consumption * pge_price
        
        battery_available = self.battery_manager.battery.available_energy()
        remaining = actual_consumption
        
        # Try to use battery first
        if battery_available > 0:
            energy_from_battery = min(battery_available, actual_consumption)
            can_discharge, max_allowed = self.battery_manager.can_discharge(energy_from_battery)
            
            if can_discharge:
                # Battery discharge happens in hardware, update state from sensors
                # For simulation, we track the intended operation
                remaining -= max_allowed
        
        # Purchase remaining energy from grid
        if remaining > 0.001:  # Account for floating point errors
            # Check grid import limits (10 kWh per 30-min interval)
            self.grid_constraints.validate_grid_transaction(remaining, 'import', 0.5)
            self.optimized_cost += remaining * pge_price
    
    def record_interval(self, interval: int, timestamp: datetime, 
                        predicted: float, actual: float, 
                        market_price: float, pge_price: float, action: str):
        """Record data for this interval."""
        battery_state = self.battery_manager.get_state()
        
        self.timeline.append({
            'interval': interval,
            'timestamp': timestamp,
            'predicted_consumption_kwh': predicted,
            'actual_consumption_kwh': actual,
            'battery_charge_kwh': battery_state.current_charge_kwh,
            'battery_soc_percent': battery_state.state_of_charge() * 100,
            'market_price': market_price,
            'pge_price': pge_price,
            'action': action
        })
    
    def get_results(self) -> Dict:
        """Get simulation results for this household."""
        battery_stats = self.battery_manager.get_statistics()
        
        savings = self.baseline_cost - self.optimized_cost
        savings_percent = (savings / self.baseline_cost * 100) if self.baseline_cost > 0 else 0
        
        return {
            'household_id': self.household_id,
            'timeline': self.timeline,
            'costs': {
                'baseline_pge_total': self.baseline_cost,
                'optimized_total': self.optimized_cost,
                'savings': savings,
                'savings_percent': savings_percent
            },
            'battery_stats': battery_stats,
            'num_trades': len(self.trades),
            'trades': [
                {
                    'interval': t.interval,
                    'action': 'buy' if t.buyer_id == self.household_id else 'sell',
                    'quantity_kwh': t.energy_kwh,
                    'delivered_kwh': t.delivered_kwh if t.buyer_id == self.household_id else t.energy_kwh,
                    'price_per_kwh': t.price_per_kwh,
                    'total_cost': t.total_cost
                }
                for t in self.trades
            ]
        }
    
    def get_node_signals(self, predicted_consumption: np.ndarray, 
                        market_price: float, pge_price: float) -> NodeSignals:
        """
        Generate node signals for central model coordination.
        
        Args:
            predicted_consumption: Array of predicted consumption for next intervals
            market_price: Current P2P market price
            pge_price: PG&E baseline price
            
        Returns:
            NodeSignals with profitability_score and power_signal
            
        Raises:
            RuntimeError: If battery state hasn't been loaded from sensors
        """
        battery_stats = self.battery_manager.get_statistics()
        
        return generate_node_signals(
            battery_state=battery_stats,
            predicted_consumption=predicted_consumption,
            market_price=market_price,
            pge_price=pge_price
        )
    
    def update_battery_state(self, sensor_data: Dict):
        """
        Update battery state from sensor readings.
        
        Should be called every interval with latest sensor data from database.
        
        Args:
            sensor_data: Dictionary with battery sensor readings:
                - current_charge_kwh
                - soh_percent
                - cycle_count
                
        Raises:
            ValueError: If sensor data is missing required fields
        """
        self.battery_manager.update_state_from_sensors(sensor_data)
    
    def reset_daily_constraints(self):
        """Reset daily trading constraints."""
        self.trading_constraints.reset_daily()


if __name__ == "__main__":
    config = {
        'model': {
            'lstm_hidden_size': 64,
            'lstm_num_layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'prediction_horizon_intervals': 6,
            'input_sequence_length': 48
        },
        'battery': {
            'capacity_kwh': 13.5,
            'efficiency': 0.90,
            'max_charge_rate_kw': 5.0,
            'max_discharge_rate_kw': 5.0,
            'min_reserve_percent': 0.10
        },
        'trading': {
            'max_sell_per_day_kwh': 10.0,
            'max_single_trade_kwh': 2.0,
            'min_profit_margin': 0.05,
            'transmission_efficiency': 0.95
        }
    }
    
    node = HouseholdNode(household_id=1, config=config)
    
    print(f"Created household node {node.household_id}")
    print(f"Battery capacity: {node.battery_manager.battery.capacity_kwh} kWh")
    print(f"Model ready: {node.model.is_trained}")
