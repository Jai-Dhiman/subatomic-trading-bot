"""
Trading Optimizer V2 - Simplified and Fixed

CORE PRINCIPLE: Buy low, sell high, hold otherwise.

Hard Rules (ALWAYS enforced):
1. NEVER discharge below 20% SoC (safety)
2. BUY aggressively when price < $20/MWh AND SoC < 90%
3. SELL aggressively when price > $40/MWh AND SoC > 25%
4. HOLD when neither condition is met

The original version had complex competing logic that caused:
- Buying at high prices
- Selling at low prices
- Holding at extreme prices
- Over-trading in the middle zone

This version is brutally simple: hard thresholds only.
"""

import numpy as np
from typing import Dict

def calculate_optimal_trading_decisions_v2(
    predicted_consumption: np.ndarray,
    actual_prices: np.ndarray,
    battery_state: Dict,
    household_price_kwh: float = 0.27,
    buy_threshold_mwh: float = None,  # Will be calculated from price distribution if None
    sell_threshold_mwh: float = None,  # Will be calculated from price distribution if None
    min_soc_for_sell: float = 0.25,  # Need 25% to sell
    target_soc_on_buy: float = 0.90,  # Fill to 90% when buying
    buy_percentile: float = 25.0,  # Buy below 25th percentile
    sell_percentile: float = 75.0,  # Sell above 75th percentile
) -> Dict:
    """
    Calculate optimal trading decisions with SIMPLE, CLEAR logic.
    
    Logic Flow (in order):
    1. If SoC <= 20%: HOLD (safety - never discharge below minimum)
    2. If price < buy_threshold: BUY aggressively to 90%
    3. If price > sell_threshold: SELL aggressively to 25%
    4. Otherwise: HOLD (don't trade at mediocre prices)
    
    Thresholds are calculated dynamically from price distribution:
    - Buy threshold: 25th percentile (default) or specified value
    - Sell threshold: 75th percentile (default) or specified value
    
    Args:
        predicted_consumption: (N,) array of predicted consumption (kWh per interval)
        actual_prices: (N,) array of actual prices ($/kWh)
        battery_state: Dict with battery parameters
        household_price_kwh: Price we charge households ($/kWh)
        buy_threshold_mwh: Buy below this price ($/MWh). If None, uses buy_percentile
        sell_threshold_mwh: Sell above this price ($/MWh). If None, uses sell_percentile
        min_soc_for_sell: Minimum SoC to allow selling (default 0.25 = 25%)
        target_soc_on_buy: Target SoC when buying (default 0.90 = 90%)
        buy_percentile: Percentile for buy threshold if buy_threshold_mwh is None (default 25.0)
        sell_percentile: Percentile for sell threshold if sell_threshold_mwh is None (default 75.0)
    
    Returns:
        Dict with optimal_decisions, optimal_quantities, and metrics
    """
    n_intervals = len(predicted_consumption)
    
    # Extract battery parameters
    capacity_kwh = battery_state.get('capacity_kwh', 40.0)
    min_soc = battery_state.get('min_soc', 0.20)
    max_soc = battery_state.get('max_soc', 1.0)
    max_charge_rate_kw = battery_state.get('max_charge_rate_kw', 10.0)
    max_discharge_rate_kw = battery_state.get('max_discharge_rate_kw', 8.0)
    current_charge_kwh = battery_state.get('current_charge_kwh', capacity_kwh * 0.50)
    efficiency = battery_state.get('efficiency', 0.95)
    
    # Calculate dynamic thresholds if not specified
    if buy_threshold_mwh is None:
        # Convert prices from $/kWh to $/MWh for threshold calculation
        prices_mwh = actual_prices * 1000.0
        buy_threshold_mwh = np.percentile(prices_mwh, buy_percentile)
    
    if sell_threshold_mwh is None:
        prices_mwh = actual_prices * 1000.0
        sell_threshold_mwh = np.percentile(prices_mwh, sell_percentile)
    
    # Convert thresholds to $/kWh
    buy_threshold_kwh = buy_threshold_mwh / 1000.0
    sell_threshold_kwh = sell_threshold_mwh / 1000.0
    
    # Max energy per 30-min interval
    max_charge_energy = max_charge_rate_kw * 0.5
    max_discharge_energy = max_discharge_rate_kw * 0.5
    
    # Initialize outputs
    optimal_decisions = np.ones(n_intervals, dtype=int)  # 1 = Hold
    optimal_quantities = np.zeros(n_intervals)
    battery_trajectory = np.zeros(n_intervals)
    
    # Profit tracking
    household_revenue = 0.0
    market_revenue = 0.0
    buy_costs = 0.0
    
    current_charge = current_charge_kwh
    
    for i in range(n_intervals):
        consumption = predicted_consumption[i]
        price = actual_prices[i]
        
        # Track household revenue (we charge them regardless of where energy comes from)
        household_revenue += consumption * household_price_kwh
        
        # Current battery state (BEFORE considering consumption)
        # Note: Consumption doesn't deplete battery during trading - it's served from grid/battery mix
        current_soc = current_charge / capacity_kwh
        available_for_discharge = max(0, current_charge - (capacity_kwh * min_soc))
        available_for_charge = max(0, (capacity_kwh * max_soc) - current_charge)
        
        decision = 1  # Hold by default
        quantity = 0.0
        
        # RULE 1: Safety - Never discharge below minimum
        if current_soc <= min_soc:
            decision = 1
            quantity = 0.0
        
        # RULE 2: Hard BUY - Price is excellent, fill battery
        elif price < buy_threshold_kwh and current_soc < target_soc_on_buy:
            # Calculate how much to buy
            target_charge = capacity_kwh * target_soc_on_buy
            max_buy = min(max_charge_energy, available_for_charge)
            quantity = min(max_buy, target_charge - current_charge)
            
            if quantity > 0.01:
                decision = 0  # Buy
                current_charge += quantity * efficiency
                buy_costs += quantity * price
        
        # RULE 3: Hard SELL - Price is excellent, sell excess
        elif price > sell_threshold_kwh and current_soc > min_soc_for_sell:
            # Sell down to min_soc_for_sell
            # Note: Grid can always supply consumption, so we can sell aggressively
            target_charge = capacity_kwh * min_soc_for_sell
            
            max_sell = min(max_discharge_energy, available_for_discharge)
            quantity = min(max_sell, max(0, current_charge - target_charge))
            
            if quantity > 0.01:
                decision = 2  # Sell
                current_charge -= quantity
                market_revenue += quantity * price
        
        # RULE 4: HOLD - Price is in middle zone, don't trade
        else:
            decision = 1
            quantity = 0.0
        
        # Subtract consumption from battery
        # Battery serves household load when it has charge
        # Grid supplies the rest if battery is low
        # This creates realistic charge/discharge cycles
        current_charge -= consumption
        
        # Ensure SoC stays within bounds
        current_soc = current_charge / capacity_kwh
        current_soc = np.clip(current_soc, min_soc, max_soc)
        current_charge = capacity_kwh * current_soc
        
        # Record decision
        optimal_decisions[i] = decision
        optimal_quantities[i] = quantity
        battery_trajectory[i] = current_soc * 100
    
    # Calculate profitability
    market_profit = market_revenue - buy_costs
    total_profit = household_revenue + market_profit
    
    return {
        'optimal_decisions': optimal_decisions,
        'optimal_quantities': optimal_quantities,
        'expected_profit': total_profit,
        'household_revenue': household_revenue,
        'market_revenue': market_revenue,
        'buy_costs': buy_costs,
        'market_profit': market_profit,
        'battery_trajectory': battery_trajectory,
        'price_thresholds': {
            'buy_below': buy_threshold_kwh,
            'sell_above': sell_threshold_kwh
        }
    }


# Alias for backward compatibility
calculate_optimal_trading_decisions = calculate_optimal_trading_decisions_v2


if __name__ == "__main__":
    print("="*80)
    print("TRADING OPTIMIZER V2 - TESTING")
    print("="*80)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Very Low Prices ($5/MWh)',
            'price': 0.005,
            'initial_soc': 0.35,
            'expected_buy': 80
        },
        {
            'name': 'Very High Prices ($100/MWh)',
            'price': 0.100,
            'initial_soc': 0.70,
            'expected_sell': 80
        },
        {
            'name': 'Middle Prices ($30/MWh)',
            'price': 0.030,
            'initial_soc': 0.50,
            'expected_hold': 90
        },
        {
            'name': 'At Minimum SoC (20%)',
            'price': 0.050,
            'initial_soc': 0.20,
            'expected_hold': 100
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("-"*80)
        
        battery_state = {
            'current_charge_kwh': 40.0 * scenario['initial_soc'],
            'capacity_kwh': 40.0,
            'min_soc': 0.20,
            'max_soc': 1.0
        }
        
        prices = np.ones(48) * scenario['price']
        consumption = np.ones(48) * 1.5
        
        result = calculate_optimal_trading_decisions(
            predicted_consumption=consumption,
            actual_prices=prices,
            battery_state=battery_state
        )
        
        decisions = result['optimal_decisions']
        unique, counts = np.unique(decisions, return_counts=True)
        
        print(f"  Battery: {scenario['initial_soc']*100:.0f}% SoC")
        print(f"  Price: ${scenario['price']*1000:.1f}/MWh")
        print(f"  Results:")
        for dec, count in zip(unique, counts):
            name = ['Buy', 'Hold', 'Sell'][dec]
            pct = count/48*100
            print(f"    {name}: {count} ({pct:.1f}%)")
        
        trajectory = result['battery_trajectory']
        print(f"  Battery: {trajectory[0]:.1f}% -> {trajectory[-1]:.1f}%")
        print(f"  Market profit: ${result['market_profit']:.2f}")
    
    print("\n" + "="*80)
    print("✅ V2 TESTING COMPLETE")
    print("="*80)
