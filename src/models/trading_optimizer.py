"""
Trading Optimizer - Clean Business Logic

Buy Rules:
- Buy when price < $0.02/kWh ($20/MWh) OR
- Buy when price < $0.27/kWh AND SoC < 33%

Sell Rules:
- Sell when price > $0.40/kWh ($400/MWh) AND SoC > 20% OR
- Sell when price > $0.27/kWh AND SoC > 66%

Otherwise: Hold

No complex reserve logic - just simple price/SoC conditions.
"""

import numpy as np
from typing import Dict

def calculate_optimal_trading_decisions(
    predicted_consumption: np.ndarray,
    actual_prices: np.ndarray,
    battery_state: Dict,
    household_price_kwh: float = 0.027,
    **kwargs  # Accept any other params for compatibility
) -> Dict:
    """
    Calculate optimal trading decisions with clean business logic.
    
    Buy Conditions:
    1. Price < $0.02/kWh (= $20/MWh, very cheap) OR
    2. Price < $0.027/kWh (= $27/MWh) AND SoC < 33% (low charge, need energy)
    
    Sell Conditions:
    1. Price > $0.40/kWh (= $40/MWh) AND SoC > 20% (premium pricing) OR  
    2. Price > $0.027/kWh (= $27/MWh) AND SoC > 66% (high charge, can sell)
    
    Otherwise: Hold (don't trade)
    
    Args:
        predicted_consumption: (N,) array of predicted consumption (kWh per interval)
        actual_prices: (N,) array of actual prices ($/kWh)
        battery_state: Dict with battery parameters
        household_price_kwh: Price threshold for conditional trades ($/kWh, default 0.027 = $27/MWh)
        **kwargs: Ignored for compatibility
    
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
    
    # Trading thresholds (your business logic)
    cheap_price = 0.02  # Always buy below $0.02/kWh (= $20/MWh)
    expensive_price = 0.04  # Always sell above $0.04/kWh (= $40/MWh)
    low_soc_threshold = 0.33  # Below this, buy if price < household rate
    high_soc_threshold = 0.66  # Above this, sell if price > household rate
    
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
        
        # BUY CONDITIONS
        should_buy = (price < cheap_price) or (price < household_price_kwh and current_soc < low_soc_threshold)
        
        if should_buy and current_soc < max_soc:
            # Buy up to 90% SoC
            target_charge = capacity_kwh * 0.90
            max_buy = min(max_charge_energy, available_for_charge)
            quantity = min(max_buy, target_charge - current_charge)
            
            if quantity > 0.01:
                decision = 0  # Buy
                current_charge += quantity * efficiency
                buy_costs += quantity * price
        
        # SELL CONDITIONS (only if not buying)
        elif (price > expensive_price and current_soc > min_soc) or \
             (price > household_price_kwh and current_soc > high_soc_threshold):
            # Sell down to 30% SoC
            target_charge = capacity_kwh * 0.30
            max_sell = min(max_discharge_energy, available_for_discharge)
            quantity = min(max_sell, max(0, current_charge - target_charge))
            
            if quantity > 0.01:
                decision = 2  # Sell
                current_charge -= quantity
                market_revenue += quantity * price
        
        # Subtract consumption from battery (serves household load)
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
            'cheap_price': cheap_price,
            'expensive_price': expensive_price,
            'household_price': household_price_kwh,
            'low_soc_threshold': low_soc_threshold,
            'high_soc_threshold': high_soc_threshold
        }
    }


# No alias needed - this IS the main function


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
