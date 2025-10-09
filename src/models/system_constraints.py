"""
Physical System Constraints for Battery Trading.

Defines realistic bounds based on actual hardware specifications:
- Battery capacity, charge/discharge rates
- Time interval durations
- State of charge operating ranges
- Energy flow limits
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BatteryConstraints:
    """
    Physical constraints for Subatomic Battery system.
    
    Based on hardware specifications:
    - Houses 1-9: 1 battery (40 kWh capacity)
    - Houses 10-11: 2 batteries (80 kWh capacity)
    - Charge rate: 10 kW
    - Discharge rate: 8 kW
    - Operating range: 20-90% SoC
    """
    
    # Battery capacity
    capacity_kwh: float = 40.0  # kWh (40 for single, 80 for double)
    
    # Power limits (rates that allow specified energy per 30-min interval)
    max_charge_rate_kw: float = 20.0  # kW (20 kW × 0.5h = 10 kWh per 30min)
    max_discharge_rate_kw: float = 16.0  # kW (16 kW × 0.5h = 8 kWh per 30min)
    
    # State of charge limits (business rules for longevity)
    min_soc: float = 0.20  # 20% minimum
    max_soc: float = 0.90  # 90% maximum
    
    # Efficiency
    charge_efficiency: float = 0.95  # 95% round-trip efficiency
    discharge_efficiency: float = 0.95
    
    # Time interval
    interval_hours: float = 0.5  # 30 minutes
    
    def __post_init__(self):
        """Calculate derived constraints."""
        # Maximum energy per interval
        self.max_charge_per_interval_kwh = (
            self.max_charge_rate_kw * self.interval_hours
        )  # 10 kW × 0.5h = 5 kWh
        
        self.max_discharge_per_interval_kwh = (
            self.max_discharge_rate_kw * self.interval_hours
        )  # 8 kW × 0.5h = 4 kWh
        
        # Usable capacity
        self.usable_capacity_kwh = self.capacity_kwh * (self.max_soc - self.min_soc)
        # 40 kWh × (0.90 - 0.20) = 28 kWh usable
        
        # Absolute bounds
        self.min_charge_kwh = self.capacity_kwh * self.min_soc  # 8 kWh
        self.max_charge_kwh = self.capacity_kwh * self.max_soc  # 36 kWh
    
    def get_max_buy_quantity(
        self, 
        current_charge_kwh: float,
        consumption_kwh: float = 0.0
    ) -> float:
        """
        Calculate maximum kWh that can be purchased given current battery state.
        
        Accounts for:
        - Available battery capacity (max_charge - current_charge)
        - Charge rate limit (max 5 kWh per interval)
        - Consumption that must be covered
        - Charge efficiency losses
        
        Args:
            current_charge_kwh: Current battery charge level
            consumption_kwh: Expected consumption this interval
            
        Returns:
            Maximum kWh that can be bought from grid
        """
        # Room in battery after accounting for consumption
        available_space = self.max_charge_kwh - current_charge_kwh + consumption_kwh
        
        # Limited by charge rate
        rate_limited = self.max_charge_per_interval_kwh
        
        # Take minimum (most restrictive)
        max_buy = min(available_space / self.charge_efficiency, rate_limited)
        
        return max(0.0, max_buy)
    
    def get_max_sell_quantity(
        self, 
        current_charge_kwh: float,
        consumption_kwh: float = 0.0
    ) -> float:
        """
        Calculate maximum kWh that can be sold given current battery state.
        
        Accounts for:
        - Available battery charge (current_charge - min_charge)
        - Discharge rate limit (max 4 kWh per interval)
        - Consumption that must be covered first
        - Discharge efficiency losses
        
        Args:
            current_charge_kwh: Current battery charge level
            consumption_kwh: Expected consumption this interval (must cover this first)
            
        Returns:
            Maximum kWh that can be sold to grid
        """
        # Available charge minus what we need for consumption
        available_charge = current_charge_kwh - self.min_charge_kwh - consumption_kwh
        
        # Limited by discharge rate
        rate_limited = self.max_discharge_per_interval_kwh
        
        # Take minimum (most restrictive)
        max_sell = min(available_charge * self.discharge_efficiency, rate_limited)
        
        return max(0.0, max_sell)
    
    def validate_quantity(
        self,
        quantity_kwh: float,
        action: str,  # 'buy', 'sell', or 'hold'
        current_charge_kwh: float,
        consumption_kwh: float = 0.0
    ) -> tuple[bool, float, str]:
        """
        Validate and clip a proposed trade quantity.
        
        Args:
            quantity_kwh: Proposed trade quantity
            action: 'buy', 'sell', or 'hold'
            current_charge_kwh: Current battery charge
            consumption_kwh: Expected consumption
            
        Returns:
            (is_valid, clipped_quantity, reason)
        """
        if action == 'hold':
            return True, 0.0, "Hold action"
        
        if action == 'buy':
            max_qty = self.get_max_buy_quantity(current_charge_kwh, consumption_kwh)
            if quantity_kwh > max_qty:
                return False, max_qty, f"Exceeds max buy {max_qty:.2f} kWh"
            return True, quantity_kwh, "Valid"
        
        if action == 'sell':
            max_qty = self.get_max_sell_quantity(current_charge_kwh, consumption_kwh)
            if quantity_kwh > max_qty:
                return False, max_qty, f"Exceeds max sell {max_qty:.2f} kWh"
            return True, quantity_kwh, "Valid"
        
        return False, 0.0, f"Unknown action: {action}"
    
    def simulate_state_change(
        self,
        current_charge_kwh: float,
        action: str,
        quantity_kwh: float,
        consumption_kwh: float
    ) -> tuple[float, dict]:
        """
        Simulate battery state change for an action.
        
        Args:
            current_charge_kwh: Starting battery charge
            action: 'buy', 'sell', or 'hold'
            quantity_kwh: Trade quantity
            consumption_kwh: Household consumption
            
        Returns:
            (new_charge_kwh, info_dict)
        """
        info = {
            'initial_charge': current_charge_kwh,
            'initial_soc': current_charge_kwh / self.capacity_kwh,
            'action': action,
            'quantity': quantity_kwh,
            'consumption': consumption_kwh
        }
        
        # Start with current charge
        new_charge = current_charge_kwh
        
        # Apply action
        if action == 'buy':
            # Add purchased energy (with efficiency loss)
            new_charge += quantity_kwh * self.charge_efficiency
            info['energy_added'] = quantity_kwh * self.charge_efficiency
        elif action == 'sell':
            # Remove sold energy (with efficiency loss)
            new_charge -= quantity_kwh / self.discharge_efficiency
            info['energy_removed'] = quantity_kwh / self.discharge_efficiency
        
        # Subtract consumption
        new_charge -= consumption_kwh
        info['after_consumption'] = new_charge
        
        # Clip to valid range
        new_charge = np.clip(new_charge, self.min_charge_kwh, self.max_charge_kwh)
        
        info['final_charge'] = new_charge
        info['final_soc'] = new_charge / self.capacity_kwh
        info['violated_bounds'] = (
            new_charge == self.min_charge_kwh or 
            new_charge == self.max_charge_kwh
        )
        
        return new_charge, info


# Pre-defined constraint sets
SINGLE_BATTERY_CONSTRAINTS = BatteryConstraints(
    capacity_kwh=40.0,
    max_charge_rate_kw=20.0,  # 20 kW allows 10 kWh per 30min
    max_discharge_rate_kw=16.0,  # 16 kW allows 8 kWh per 30min
    min_soc=0.20,
    max_soc=0.90,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    interval_hours=0.5
)

DOUBLE_BATTERY_CONSTRAINTS = BatteryConstraints(
    capacity_kwh=80.0,
    max_charge_rate_kw=40.0,  # 2x batteries: 40 kW allows 20 kWh per 30min
    max_discharge_rate_kw=32.0,  # 2x batteries: 32 kW allows 16 kWh per 30min
    min_soc=0.20,
    max_soc=0.90,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    interval_hours=0.5
)


def get_constraints_for_house(house_id: int) -> BatteryConstraints:
    """Get battery constraints for a specific house."""
    if house_id in [10, 11]:
        return DOUBLE_BATTERY_CONSTRAINTS
    else:
        return SINGLE_BATTERY_CONSTRAINTS


# Model training constraints
class ModelTrainingConstraints:
    """
    Constraints for neural network model outputs.
    
    These should match the physical system but may be slightly
    conservative for safety margin during inference.
    """
    
    # Quantity bounds for model output
    # Using single battery as baseline (most restrictive)
    MAX_TRADE_QUANTITY_KWH = 10.0  # Max charge per 30-min interval
    
    # Price bounds (from market data)
    MIN_PRICE_KWH = -0.05  # Occasionally negative (surplus)
    MAX_PRICE_KWH = 0.30   # Typical max ~$0.18, allow headroom
    
    # Decision classes
    DECISIONS = {
        'buy': 0,
        'hold': 1,
        'sell': 2
    }
    
    @classmethod
    def clip_model_outputs(
        cls,
        predicted_prices: np.ndarray,
        predicted_quantities: np.ndarray,
        predicted_decisions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Clip model outputs to physically valid ranges.
        
        Args:
            predicted_prices: Price predictions ($/kWh)
            predicted_quantities: Quantity predictions (kWh)
            predicted_decisions: Decision class predictions (0/1/2)
            
        Returns:
            (clipped_prices, clipped_quantities, clipped_decisions)
        """
        prices = np.clip(predicted_prices, cls.MIN_PRICE_KWH, cls.MAX_PRICE_KWH)
        quantities = np.clip(predicted_quantities, 0.0, cls.MAX_TRADE_QUANTITY_KWH)
        decisions = np.clip(predicted_decisions, 0, 2)
        
        return prices, quantities, decisions


if __name__ == "__main__":
    print("="*70)
    print("BATTERY SYSTEM CONSTRAINTS TEST")
    print("="*70)
    
    # Test single battery
    print("\n1. Single Battery (40 kWh):")
    single = SINGLE_BATTERY_CONSTRAINTS
    print(f"   Capacity: {single.capacity_kwh} kWh")
    print(f"   Usable capacity: {single.usable_capacity_kwh} kWh")
    print(f"   Max charge per interval: {single.max_charge_per_interval_kwh} kWh")
    print(f"   Max discharge per interval: {single.max_discharge_per_interval_kwh} kWh")
    print(f"   Operating range: {single.min_charge_kwh:.1f} - {single.max_charge_kwh:.1f} kWh")
    
    # Test buy/sell limits
    print("\n2. Testing buy/sell limits at 50% SoC:")
    current_charge = 20.0  # 50% of 40 kWh
    consumption = 1.0  # 1 kWh consumption
    
    max_buy = single.get_max_buy_quantity(current_charge, consumption)
    max_sell = single.get_max_sell_quantity(current_charge, consumption)
    
    print(f"   Current charge: {current_charge} kWh (50% SoC)")
    print(f"   Consumption: {consumption} kWh")
    print(f"   Max buy: {max_buy:.2f} kWh")
    print(f"   Max sell: {max_sell:.2f} kWh")
    
    # Test state simulation
    print("\n3. Simulating buy action:")
    new_charge, info = single.simulate_state_change(
        current_charge_kwh=20.0,
        action='buy',
        quantity_kwh=5.0,
        consumption_kwh=1.0
    )
    print(f"   Initial: {info['initial_charge']:.2f} kWh ({info['initial_soc']*100:.1f}% SoC)")
    print(f"   Buy: {info['quantity']:.2f} kWh")
    print(f"   Energy added: {info['energy_added']:.2f} kWh (after efficiency)")
    print(f"   After consumption: {info['after_consumption']:.2f} kWh")
    print(f"   Final: {info['final_charge']:.2f} kWh ({info['final_soc']*100:.1f}% SoC)")
    print(f"   Bounds violated: {info['violated_bounds']}")
    
    # Test validation
    print("\n4. Testing quantity validation:")
    
    # Valid buy
    valid, qty, reason = single.validate_quantity(3.0, 'buy', 20.0, 1.0)
    print(f"   Buy 3 kWh: valid={valid}, qty={qty:.2f}, reason='{reason}'")
    
    # Invalid buy (too much)
    valid, qty, reason = single.validate_quantity(10.0, 'buy', 20.0, 1.0)
    print(f"   Buy 10 kWh: valid={valid}, qty={qty:.2f}, reason='{reason}'")
    
    # Valid sell
    valid, qty, reason = single.validate_quantity(2.0, 'sell', 20.0, 1.0)
    print(f"   Sell 2 kWh: valid={valid}, qty={qty:.2f}, reason='{reason}'")
    
    # Test model constraints
    print("\n5. Model Training Constraints:")
    print(f"   Max quantity for model: {ModelTrainingConstraints.MAX_TRADE_QUANTITY_KWH} kWh")
    print(f"   Price range: ${ModelTrainingConstraints.MIN_PRICE_KWH:.2f} to ${ModelTrainingConstraints.MAX_PRICE_KWH:.2f}")
    
    # Test edge cases
    print("\n6. Edge Case: Near max SoC:")
    near_max = single.max_charge_kwh - 1.0  # 35 kWh (88% SoC)
    max_buy = single.get_max_buy_quantity(near_max, 1.0)
    max_sell = single.get_max_sell_quantity(near_max, 1.0)
    print(f"   At {near_max} kWh ({near_max/single.capacity_kwh*100:.1f}% SoC):")
    print(f"   Max buy: {max_buy:.2f} kWh (limited by capacity)")
    print(f"   Max sell: {max_sell:.2f} kWh (plenty of room)")
    
    print("\n7. Edge Case: Near min SoC:")
    near_min = single.min_charge_kwh + 1.0  # 9 kWh (22% SoC)
    max_buy = single.get_max_buy_quantity(near_min, 1.0)
    max_sell = single.get_max_sell_quantity(near_min, 1.0)
    print(f"   At {near_min} kWh ({near_min/single.capacity_kwh*100:.1f}% SoC):")
    print(f"   Max buy: {max_buy:.2f} kWh (plenty of room)")
    print(f"   Max sell: {max_sell:.2f} kWh (limited by min SoC)")
    
    print("\n" + "="*70)
    print("✅ CONSTRAINTS TEST COMPLETE")
    print("="*70)
