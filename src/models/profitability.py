"""
Profitability calculator and node signaling for central model.

Calculates:
- Profitability score: How profitable this node is for trading
- Power signal: GREEN (sufficient power) or RED (needs power)

These signals are sent to the central model to coordinate trading decisions.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class NodeSignals:
    """Signals that a node sends to the central model.
    
    These signals help the central model coordinate trading and
    prioritize which nodes should trade with each other.
    """
    
    profitability_score: float  # 0-100, higher = more profitable to trade
    power_signal: str           # "GREEN" or "RED"
    battery_soc_percent: float  # Current state of charge
    predicted_deficit_kwh: float  # Predicted energy deficit (negative = surplus)
    
    def to_dict(self) -> Dict:
        """Convert signals to dictionary for transmission."""
        return {
            'profitability_score': self.profitability_score,
            'power_signal': self.power_signal,
            'battery_soc_percent': self.battery_soc_percent,
            'predicted_deficit_kwh': self.predicted_deficit_kwh,
        }


def calculate_power_need_signal(
    battery_soc_percent: float,
    predicted_consumption_kwh: float,
    battery_available_energy_kwh: float,
    soc_red_threshold: float = 30.0,
    soc_green_threshold: float = 60.0
) -> str:
    """
    Calculate power need signal (GREEN or RED) based on battery and predictions.
    
    Logic:
    - RED: Battery SoC < 30% OR predicted consumption > available energy
    - GREEN: Battery SoC >= 60% AND sufficient energy for consumption
    - 30-60%: Based on predicted deficit
    
    Args:
        battery_soc_percent: Current battery state of charge (0-100)
        predicted_consumption_kwh: Total predicted consumption for next intervals
        battery_available_energy_kwh: Energy available in battery (above reserve)
        soc_red_threshold: SoC threshold for RED signal (default 30%)
        soc_green_threshold: SoC threshold for GREEN signal (default 60%)
        
    Returns:
        "RED" if needs power, "GREEN" if sufficient
        
    Example:
        >>> calculate_power_need_signal(25.0, 5.0, 3.0)
        "RED"  # Low SoC
        
        >>> calculate_power_need_signal(75.0, 5.0, 8.0)
        "GREEN"  # High SoC, sufficient energy
    """
    # Critical: Low battery
    if battery_soc_percent < soc_red_threshold:
        return "RED"
    
    # Comfortable: High battery
    if battery_soc_percent >= soc_green_threshold:
        if battery_available_energy_kwh >= predicted_consumption_kwh:
            return "GREEN"
    
    # Middle range: Check predicted deficit
    predicted_deficit = predicted_consumption_kwh - battery_available_energy_kwh
    
    if predicted_deficit > 1.0:  # Will need >1 kWh from grid
        return "RED"
    else:
        return "GREEN"


def calculate_profitability_metric(
    battery_soc_percent: float,
    battery_capacity_kwh: float,
    predicted_consumption_kwh: float,
    predicted_surplus_kwh: float,
    market_price: float,
    pge_price: float,
    battery_soh_percent: float = 100.0
) -> float:
    """
    Calculate profitability score for this node (0-100).
    
    Higher scores indicate better trading opportunities:
    - Nodes with large surplus = high profitability (good sellers)
    - Nodes with large deficit + capacity = high profitability (good buyers)
    - Nodes with healthy batteries = higher profitability
    - Favorable market conditions = higher profitability
    
    Args:
        battery_soc_percent: Current battery SoC (0-100)
        battery_capacity_kwh: Total battery capacity
        predicted_consumption_kwh: Predicted consumption for next intervals
        predicted_surplus_kwh: Predicted surplus (positive) or deficit (negative)
        market_price: Current market price ($/kWh)
        pge_price: PG&E baseline price ($/kWh)
        battery_soh_percent: Battery health (0-100, default 100)
        
    Returns:
        Profitability score (0-100), higher = more profitable
        
    Components:
    - Trading potential (40%): Based on surplus/deficit magnitude
    - Battery health (20%): Healthier batteries = more reliable trading
    - Market conditions (20%): Price differential vs PG&E
    - Capacity utilization (20%): Using battery effectively
    """
    # Component 1: Trading Potential (40 points)
    # Large surplus or deficit = high trading potential
    surplus_magnitude = abs(predicted_surplus_kwh)
    max_tradeable = min(surplus_magnitude, 10.0)  # Cap at 10 kWh max
    trading_potential = (max_tradeable / 10.0) * 40.0
    
    # Component 2: Battery Health (20 points)
    battery_health_score = (battery_soh_percent / 100.0) * 20.0
    
    # Component 3: Market Conditions (20 points)
    # Favorable if market price significantly different from PG&E
    if predicted_surplus_kwh > 0:
        # Seller: wants high market price
        price_ratio = market_price / pge_price if pge_price > 0 else 1.0
        market_score = min(price_ratio * 10.0, 20.0)
    else:
        # Buyer: wants low market price
        price_ratio = pge_price / market_price if market_price > 0 else 1.0
        market_score = min(price_ratio * 10.0, 20.0)
    
    # Component 4: Capacity Utilization (20 points)
    # Using battery capacity effectively
    if battery_capacity_kwh > 0:
        # Optimal SoC range: 40-70%
        if 40 <= battery_soc_percent <= 70:
            capacity_score = 20.0
        elif 30 <= battery_soc_percent <= 80:
            capacity_score = 15.0
        elif 20 <= battery_soc_percent <= 90:
            capacity_score = 10.0
        else:
            capacity_score = 5.0
    else:
        capacity_score = 0.0
    
    # Total profitability score
    profitability = (
        trading_potential +
        battery_health_score +
        market_score +
        capacity_score
    )
    
    # Ensure within 0-100 range
    profitability = max(0.0, min(100.0, profitability))
    
    return profitability


def generate_node_signals(
    battery_state: Dict,
    predicted_consumption: np.ndarray,
    market_price: float,
    pge_price: float
) -> NodeSignals:
    """
    Generate complete node signals for central model.
    
    This is the main function that combines all signal calculations
    and creates the NodeSignals object to send to central model.
    
    Args:
        battery_state: Dictionary from BatteryManager.get_statistics()
            Must include: state_of_charge_percent, available_energy_kwh,
                         capacity_kwh, state_of_health_percent
        predicted_consumption: Array of predicted consumption (kWh per interval)
        market_price: Current P2P market price ($/kWh)
        pge_price: PG&E baseline price ($/kWh)
        
    Returns:
        NodeSignals object with all calculated signals
        
    Raises:
        ValueError: If required battery state fields are missing
        
    Example:
        >>> battery_state = {
        ...     'state_of_charge_percent': 55.0,
        ...     'available_energy_kwh': 5.0,
        ...     'capacity_kwh': 13.5,
        ...     'state_of_health_percent': 98.5
        ... }
        >>> predicted = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.0])
        >>> signals = generate_node_signals(battery_state, predicted, 0.40, 0.51)
        >>> signals.power_signal
        'GREEN'
    """
    # Validate required fields
    required_fields = [
        'state_of_charge_percent',
        'available_energy_kwh',
        'capacity_kwh',
        'state_of_health_percent'
    ]
    missing = [f for f in required_fields if f not in battery_state]
    if missing:
        raise ValueError(
            f"Missing required battery state fields: {missing}. "
            f"Required: {required_fields}. "
            f"Get from BatteryManager.get_statistics()"
        )
    
    # Extract battery state
    battery_soc = battery_state['state_of_charge_percent']
    battery_available = battery_state['available_energy_kwh']
    battery_capacity = battery_state['capacity_kwh']
    battery_soh = battery_state['state_of_health_percent']
    
    # Calculate predicted totals
    total_predicted_consumption = float(np.sum(predicted_consumption))
    predicted_deficit = total_predicted_consumption - battery_available
    predicted_surplus = -predicted_deficit  # Positive if surplus
    
    # Calculate power signal
    power_signal = calculate_power_need_signal(
        battery_soc_percent=battery_soc,
        predicted_consumption_kwh=total_predicted_consumption,
        battery_available_energy_kwh=battery_available
    )
    
    # Calculate profitability score
    profitability_score = calculate_profitability_metric(
        battery_soc_percent=battery_soc,
        battery_capacity_kwh=battery_capacity,
        predicted_consumption_kwh=total_predicted_consumption,
        predicted_surplus_kwh=predicted_surplus,
        market_price=market_price,
        pge_price=pge_price,
        battery_soh_percent=battery_soh
    )
    
    # Create signals object
    signals = NodeSignals(
        profitability_score=profitability_score,
        power_signal=power_signal,
        battery_soc_percent=battery_soc,
        predicted_deficit_kwh=predicted_deficit
    )
    
    return signals


if __name__ == "__main__":
    print("Profitability Calculator & Node Signals")
    print("=" * 60)
    
    # Simulate battery state from BatteryManager
    battery_state_high = {
        'state_of_charge_percent': 75.0,
        'available_energy_kwh': 7.0,
        'capacity_kwh': 13.5,
        'state_of_health_percent': 98.5,
    }
    
    battery_state_low = {
        'state_of_charge_percent': 25.0,
        'available_energy_kwh': 2.0,
        'capacity_kwh': 13.5,
        'state_of_health_percent': 98.5,
    }
    
    battery_state_medium = {
        'state_of_charge_percent': 55.0,
        'available_energy_kwh': 5.0,
        'capacity_kwh': 13.5,
        'state_of_health_percent': 98.5,
    }
    
    # Predicted consumption (6 intervals = 3 hours)
    predicted_low = np.array([0.5, 0.6, 0.7, 0.6, 0.5, 0.6])  # 3.5 kWh total
    predicted_high = np.array([1.2, 1.3, 1.5, 1.4, 1.3, 1.2])  # 7.9 kWh total
    
    # Market conditions
    market_price_low = 0.32  # Low market price (good for buyers)
    market_price_high = 0.48  # High market price (good for sellers)
    pge_price = 0.51
    
    print("\n" + "=" * 60)
    print("Scenario 1: High Battery, Low Consumption (Seller)")
    print("=" * 60)
    signals = generate_node_signals(
        battery_state_high, predicted_low, market_price_high, pge_price
    )
    print(f"  Battery SoC: {signals.battery_soc_percent:.1f}%")
    print(f"  Predicted consumption: {np.sum(predicted_low):.2f} kWh")
    print(f"  Available energy: {battery_state_high['available_energy_kwh']:.2f} kWh")
    print(f"  Predicted surplus: {-signals.predicted_deficit_kwh:.2f} kWh")
    print(f"\n  Power Signal: {signals.power_signal}")
    print(f"  Profitability Score: {signals.profitability_score:.1f}/100")
    print(f"  → Good SELLER opportunity (high SoC, surplus energy)")
    
    print("\n" + "=" * 60)
    print("Scenario 2: Low Battery, High Consumption (Buyer)")
    print("=" * 60)
    signals = generate_node_signals(
        battery_state_low, predicted_high, market_price_low, pge_price
    )
    print(f"  Battery SoC: {signals.battery_soc_percent:.1f}%")
    print(f"  Predicted consumption: {np.sum(predicted_high):.2f} kWh")
    print(f"  Available energy: {battery_state_low['available_energy_kwh']:.2f} kWh")
    print(f"  Predicted deficit: {signals.predicted_deficit_kwh:.2f} kWh")
    print(f"\n  Power Signal: {signals.power_signal}")
    print(f"  Profitability Score: {signals.profitability_score:.1f}/100")
    print(f"  → Good BUYER opportunity (low SoC, needs power)")
    
    print("\n" + "=" * 60)
    print("Scenario 3: Medium Battery, Medium Consumption (Balanced)")
    print("=" * 60)
    predicted_medium = np.array([0.8, 0.9, 1.0, 1.0, 0.9, 0.8])  # 5.4 kWh total
    signals = generate_node_signals(
        battery_state_medium, predicted_medium, market_price_low, pge_price
    )
    print(f"  Battery SoC: {signals.battery_soc_percent:.1f}%")
    print(f"  Predicted consumption: {np.sum(predicted_medium):.2f} kWh")
    print(f"  Available energy: {battery_state_medium['available_energy_kwh']:.2f} kWh")
    print(f"  Predicted deficit: {signals.predicted_deficit_kwh:.2f} kWh")
    print(f"\n  Power Signal: {signals.power_signal}")
    print(f"  Profitability Score: {signals.profitability_score:.1f}/100")
    print(f"  → Moderate trading opportunity")
    
    print("\n" + "=" * 60)
    print("Signal Component Breakdown")
    print("=" * 60)
    print("\nProfitability Score Components (0-100):")
    print("  • Trading Potential: 0-40 points")
    print("    - Based on magnitude of surplus/deficit")
    print("  • Battery Health: 0-20 points")
    print("    - Higher SoH = more reliable trading")
    print("  • Market Conditions: 0-20 points")
    print("    - Price differential vs PG&E baseline")
    print("  • Capacity Utilization: 0-20 points")
    print("    - Optimal SoC range = higher score")
    
    print("\nPower Signal Logic:")
    print("  • RED: SoC < 30% OR deficit > 1 kWh")
    print("  • GREEN: SoC >= 60% AND sufficient energy")
    
    print("\n" + "=" * 60)
    print("Signals sent to Central Model for coordination")
    print("=" * 60)
