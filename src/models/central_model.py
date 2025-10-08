"""
Central model for price signals, market coordination, and federated learning.
"""

import numpy as np
from typing import List, Dict
from datetime import datetime


class CentralModel:
    """Central model that coordinates market and federated learning."""

    def __init__(self, config: dict):
        """
        Initialize central model.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pricing_config = config.get("pricing", {})
        self.base_price = self.pricing_config.get("base_price", 0.35)
        self.price_bounds = config.get("trading", {}).get(
            "price_bounds", {"min": 0.10, "max": 1.00}
        )

        self.global_model_weights = None

    def generate_price_signal(
        self, aggregate_demand: float, aggregate_supply: float, timestamp: datetime
    ) -> float:
        """
        Generate dynamic market price signal based on supply/demand.

        Args:
            aggregate_demand: Total energy demand from all nodes (kWh)
            aggregate_supply: Total energy available for sale (kWh)
            timestamp: Current timestamp

        Returns:
            Market price in $/kWh
        """
        hour = timestamp.hour
        month = timestamp.month

        if 16 <= hour < 20:
            tod_multiplier = 1.4
        elif 0 <= hour < 6:
            tod_multiplier = 0.8
        else:
            tod_multiplier = 1.0

        if aggregate_demand > 0:
            imbalance = (aggregate_demand - aggregate_supply) / aggregate_demand
        else:
            imbalance = 0

        if imbalance > 0.2:
            imbalance_multiplier = 1.0 + (imbalance * 0.5)
        elif imbalance < -0.2:
            imbalance_multiplier = 1.0 + (imbalance * 0.3)
        else:
            imbalance_multiplier = 1.0

        price = self.base_price * tod_multiplier * imbalance_multiplier

        price = max(self.price_bounds["min"], min(self.price_bounds["max"], price))

        return price

    def get_pge_rate(self, timestamp: datetime) -> float:
        """
        Get PG&E baseline rate for comparison.

        Args:
            timestamp: Current timestamp

        Returns:
            PG&E rate in $/kWh
        """
        hour = timestamp.hour
        month = timestamp.month

        is_summer = 6 <= month <= 9

        if 0 <= hour < 6:
            return 0.28
        elif 16 <= hour < 20:
            return 0.51 if is_summer else 0.40
        else:
            return 0.35 if is_summer else 0.33

    def enforce_market_rules(
        self, node_id: int, action: str, quantity: float, daily_sold: Dict[int, float]
    ) -> bool:
        """
        Enforce market rules and regulations.

        Args:
            node_id: Node identifier
            action: 'buy' or 'sell'
            quantity: Energy quantity (kWh)
            daily_sold: Dictionary tracking daily sales per node

        Returns:
            True if trade is allowed, False otherwise
        """
        max_sell_per_day = self.config.get("trading", {}).get("max_sell_per_day_kwh", 10.0)
        max_single_trade = self.config.get("trading", {}).get("max_single_trade_kwh", 2.0)

        if quantity > max_single_trade:
            return False

        if action == "sell":
            current_sold = daily_sold.get(node_id, 0.0)
            if current_sold + quantity > max_sell_per_day:
                return False

        return True

    def update_global_weights(self, weights: dict):
        """Update global model weights after federated aggregation."""
        self.global_model_weights = weights

    def get_global_weights(self) -> dict:
        """Get current global model weights."""
        return self.global_model_weights


if __name__ == "__main__":
    from datetime import datetime

    config = {
        "pricing": {"base_price": 0.35},
        "trading": {
            "max_sell_per_day_kwh": 10.0,
            "max_single_trade_kwh": 2.0,
            "price_bounds": {"min": 0.10, "max": 1.00},
        },
    }

    central = CentralModel(config)

    ts = datetime(2024, 7, 15, 16, 30)

    price = central.generate_price_signal(
        aggregate_demand=15.0, aggregate_supply=10.0, timestamp=ts
    )

    print(f"Market price at {ts}: ${price:.3f}/kWh")
    print(f"PG&E rate at {ts}: ${central.get_pge_rate(ts):.3f}/kWh")

    is_allowed = central.enforce_market_rules(
        node_id=1, action="sell", quantity=1.5, daily_sold={1: 8.0}
    )

    print(f"Trade allowed: {is_allowed}")
