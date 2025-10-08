"""
Stage 2: Energy Requirement Signal Generation

This module analyzes consumption predictions from the transformer (Stage 1)
and generates energy requirement signals for the central node.

Signals indicate whether each household needs more energy, less energy,
or is okay for each 30-min interval across day/week/month horizons.

These signals replace the direct trading decisions and allow the central
node to coordinate energy distribution across all households.
"""

import numpy as np
from typing import Dict, Tuple
from enum import Enum
from dataclasses import dataclass


class EnergySignal(Enum):
    """Energy requirement signal types."""
    NEED_MORE = "NEED_MORE"    # Household needs additional energy
    NEED_LESS = "NEED_LESS"    # Household has excess energy to provide
    OK = "OK"                  # Household is balanced


@dataclass
class BatteryState:
    """Simple battery state for Stage 2 calculations."""
    current_charge_kwh: float
    capacity_kwh: float
    min_reserve_percent: float = 0.10
    max_charge_percent: float = 0.80
    
    def available_energy(self) -> float:
        """Energy available for discharge (above reserve)."""
        reserve = self.capacity_kwh * self.min_reserve_percent
        return max(0, self.current_charge_kwh - reserve)
    
    def available_capacity(self) -> float:
        """Capacity available for charging (below max)."""
        max_charge = self.capacity_kwh * self.max_charge_percent
        return max(0, max_charge - self.current_charge_kwh)
    
    def state_of_charge(self) -> float:
        """Current state of charge as percentage."""
        return self.current_charge_kwh / self.capacity_kwh


def generate_energy_signals(
    consumption_predictions: np.ndarray,
    battery_state: BatteryState,
    threshold_need_more: float = 1.0,
    threshold_need_less: float = 1.5
) -> np.ndarray:
    """
    Generate energy requirement signals for predicted consumption.
    
    Args:
        consumption_predictions: Array of predicted consumption (kWh) for future intervals
                                Shape: (n_intervals,) where n_intervals is 48, 336, or 1440
        battery_state: Current battery state
        threshold_need_more: Deficit threshold (kWh) to trigger NEED_MORE signal
        threshold_need_less: Excess threshold (kWh) to trigger NEED_LESS signal
        
    Returns:
        Array of signals for each interval (0=NEED_MORE, 1=OK, 2=NEED_LESS)
        Shape: (n_intervals,)
    """
    n_intervals = len(consumption_predictions)
    signals = np.zeros(n_intervals, dtype=int)
    
    # Start with current battery state
    current_battery = battery_state.current_charge_kwh
    
    for i in range(n_intervals):
        predicted_consumption = consumption_predictions[i]
        
        # Calculate net position for this interval
        # Positive = have excess, Negative = need energy
        net_position = current_battery - predicted_consumption
        
        # Determine signal based on net position
        if net_position < -threshold_need_more:
            # Need more energy - battery would go below reserve
            signals[i] = 0  # NEED_MORE
        elif net_position > threshold_need_less:
            # Have excess energy - can provide to others
            signals[i] = 2  # NEED_LESS
        else:
            # Balanced - no action needed
            signals[i] = 1  # OK
        
        # Update battery state for next interval simulation
        # (This is a simple simulation, actual state comes from hardware)
        current_battery = max(0, min(
            battery_state.capacity_kwh * battery_state.max_charge_percent,
            current_battery - predicted_consumption
        ))
    
    return signals


def generate_multi_horizon_signals(
    predictions_dict: Dict[str, np.ndarray],
    battery_state: BatteryState
) -> Dict[str, np.ndarray]:
    """
    Generate energy requirement signals for all prediction horizons.
    
    Args:
        predictions_dict: Dict with keys 'consumption_day', 'consumption_week', 'consumption_month'
                         Each value is np.ndarray of shape (n_intervals,)
        battery_state: Current battery state
        
    Returns:
        Dict with same keys, values are signal arrays
    """
    signals_dict = {}
    
    for horizon_key, predictions in predictions_dict.items():
        signals = generate_energy_signals(
            consumption_predictions=predictions,
            battery_state=battery_state
        )
        
        # Replace 'consumption_' with 'signal_' in key name
        signal_key = horizon_key.replace('consumption_', 'signal_')
        signals_dict[signal_key] = signals
    
    return signals_dict


def signals_to_string(signals: np.ndarray) -> list:
    """
    Convert numerical signals to human-readable strings.
    
    Args:
        signals: Array of signal values (0, 1, or 2)
        
    Returns:
        List of signal strings
    """
    signal_map = {
        0: EnergySignal.NEED_MORE.value,
        1: EnergySignal.OK.value,
        2: EnergySignal.NEED_LESS.value
    }
    
    return [signal_map[int(s)] for s in signals]


def analyze_signal_distribution(signals: np.ndarray) -> Dict[str, float]:
    """
    Analyze the distribution of signals.
    
    Args:
        signals: Array of signal values
        
    Returns:
        Dict with percentages of each signal type
    """
    total = len(signals)
    need_more_count = np.sum(signals == 0)
    ok_count = np.sum(signals == 1)
    need_less_count = np.sum(signals == 2)
    
    return {
        'need_more_percent': (need_more_count / total) * 100,
        'ok_percent': (ok_count / total) * 100,
        'need_less_percent': (need_less_count / total) * 100,
        'need_more_intervals': int(need_more_count),
        'ok_intervals': int(ok_count),
        'need_less_intervals': int(need_less_count)
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Energy Requirement Signal Generation (Stage 2)")
    print("=" * 60)
    
    # Test with example battery state
    battery = BatteryState(
        current_charge_kwh=6.0,
        capacity_kwh=13.5,
        min_reserve_percent=0.10,
        max_charge_percent=0.80
    )
    
    print(f"\n1. Battery State:")
    print(f"   Current charge: {battery.current_charge_kwh} kWh")
    print(f"   Available energy: {battery.available_energy():.2f} kWh")
    print(f"   Available capacity: {battery.available_capacity():.2f} kWh")
    print(f"   SoC: {battery.state_of_charge() * 100:.1f}%")
    
    # Generate example consumption predictions (day horizon - 48 intervals)
    # Simulate varying consumption: high during day, low at night
    print(f"\n2. Generating example consumption predictions...")
    hours = np.arange(48)
    base_consumption = 0.5
    peak_consumption = 2.0
    
    # Sinusoidal pattern with peaks during the day
    consumption_pattern = base_consumption + (peak_consumption - base_consumption) * (
        0.5 + 0.5 * np.sin(2 * np.pi * hours / 48 - np.pi / 2)
    )
    
    print(f"   Consumption range: {consumption_pattern.min():.2f} - {consumption_pattern.max():.2f} kWh")
    print(f"   Mean consumption: {consumption_pattern.mean():.2f} kWh")
    
    # Generate signals
    print(f"\n3. Generating energy requirement signals...")
    signals = generate_energy_signals(
        consumption_predictions=consumption_pattern,
        battery_state=battery
    )
    
    # Analyze distribution
    distribution = analyze_signal_distribution(signals)
    
    print(f"\n4. Signal Distribution:")
    print(f"   NEED_MORE: {distribution['need_more_percent']:.1f}% ({distribution['need_more_intervals']} intervals)")
    print(f"   OK:        {distribution['ok_percent']:.1f}% ({distribution['ok_intervals']} intervals)")
    print(f"   NEED_LESS: {distribution['need_less_percent']:.1f}% ({distribution['need_less_intervals']} intervals)")
    
    # Show first 12 intervals (6 hours)
    print(f"\n5. First 12 intervals (6 hours):")
    signal_strings = signals_to_string(signals[:12])
    for i in range(12):
        hour = i * 0.5
        print(f"   Interval {i:2d} (t+{hour:4.1f}h): Consumption={consumption_pattern[i]:4.2f} kWh -> {signal_strings[i]}")
    
    # Test multi-horizon signals
    print(f"\n6. Testing multi-horizon signals...")
    predictions_dict = {
        'consumption_day': consumption_pattern,
        'consumption_week': np.tile(consumption_pattern, 7)[:336],  # Repeat for week
        'consumption_month': np.tile(consumption_pattern, 30)[:1440]  # Repeat for month
    }
    
    signals_dict = generate_multi_horizon_signals(predictions_dict, battery)
    
    print(f"   Generated signals for {len(signals_dict)} horizons:")
    for key, sigs in signals_dict.items():
        dist = analyze_signal_distribution(sigs)
        print(f"     {key}: {len(sigs)} intervals, {dist['need_more_percent']:.1f}% NEED_MORE, {dist['ok_percent']:.1f}% OK, {dist['need_less_percent']:.1f}% NEED_LESS")
    
    print("\n" + "=" * 60)
    print("✅ Energy requirement signal generation test complete!")
    print("✅ Stage 2 ready for integration with transformer predictions")
    print("=" * 60)
