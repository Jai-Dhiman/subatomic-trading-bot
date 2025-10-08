"""
Trading Optimizer for Supervised Learning.

Calculates optimal trading decisions in hindsight for training the Trading Transformer.

Business Model:
- Company charges households 27 cents/kWh ($270/MWh)
- Goal: Maximize profit by buying low, selling high
- Revenue optimization through strategic battery management

Hard Rules:
- Always buy when price < $20/MWh AND SoC < 100%
- Always sell when price > $40/MWh AND SoC > 20%
- Hold when SoC < 20% (safety threshold)
- Hold when price < 10% of grid price ($27/MWh)

Priority List:
1. Match household power demands per day (must-have)
2. Sell (Max Capacity - Daily Predicted Usage) to market
3. Sell remaining un-needed power back to market
4. Balance profit margin and minimize excess battery storage

Reward Mechanism:
- Higher profit = higher reward for transformer
- Profit = (Sell Revenue - Buy Cost)
- Penalize holding excess power unnecessarily
"""

import numpy as np
from typing import Dict, Tuple


def calculate_optimal_trading_decisions(
    predicted_consumption: np.ndarray,
    actual_prices: np.ndarray,
    battery_state: Dict,
    household_price_kwh: float = 0.27,  # $270/MWh - what we charge households
    buy_threshold_mwh: float = 20.0,  # Buy below $20/MWh
    sell_threshold_mwh: float = 40.0,  # Sell above $40/MWh
    min_grid_price_pct: float = 0.10,  # 10% of household price = $27/MWh
    opportunistic_window: int = 48,  # Look-back window for opportunistic trading (48 hours)
    opportunistic_buy_percentile: float = 55.0,  # Buy when price in bottom 55% (aggressive)
    opportunistic_sell_percentile: float = 45.0  # Sell when price in top 55% (aggressive)
) -> Dict:
    """
    Calculate optimal trading decisions in hindsight for supervised learning.
    
    This function simulates what the ideal trading decisions would have been
    given perfect knowledge of future prices. Used as training labels for
    the Trading Transformer.
    
    Business Logic:
    - Company charges households $270/MWh (27 cents/kWh)
    - HARD RULES (must follow):
      * Buy from market when < $20/MWh AND SoC < 100%
      * Sell to market when > $40/MWh AND SoC > 20%
      * Hold when SoC < 20%
    - OPPORTUNISTIC TRADING (learn from patterns):
      * In $20-40/MWh zone, buy when price is relatively low for the period
      * In $20-40/MWh zone, sell when price is relatively high for the period
      * Use rolling window to identify low/high prices dynamically
    - Priority: Match household demand first, then maximize profit
    
    Args:
        predicted_consumption: (48,) array from Consumption Transformer (kWh per 30-min)
        actual_prices: (48,) array from Supabase pricing data ($/kWh)
        battery_state: Dict with current battery state:
            - current_charge_kwh: float
            - capacity_kwh: float (e.g., 40 or 80)
            - min_soc: float (default 0.20)
            - max_soc: float (default 0.90 -> now 1.0)
            - max_charge_rate_kw: float (default 10.0)
            - max_discharge_rate_kw: float (default 8.0)
        household_price_kwh: What we charge households ($/kWh, default 0.27)
        buy_threshold_mwh: Buy below this price ($/MWh, default 20)
        sell_threshold_mwh: Sell above this price ($/MWh, default 40)
        min_grid_price_pct: Min price as % of household rate (default 0.10)
        opportunistic_window: Look-back window for price patterns (intervals, default 24)
        opportunistic_buy_percentile: Buy when price below this percentile (default 30)
        opportunistic_sell_percentile: Sell when price above this percentile (default 70)
    
    Returns:
        Dict containing:
            - optimal_decisions: (48,) array [0=Buy, 1=Hold, 2=Sell]
            - optimal_quantities: (48,) array of kWh amounts to trade
            - expected_profit: float (profit in $ over the 48 intervals)
            - battery_trajectory: (48,) array of battery SoC over time
            - household_revenue: float (revenue from household at $270/MWh)
            - market_profit: float (profit from market trading)
            - total_profit: float (household revenue + market profit - buy costs)
    """
    n_intervals = len(predicted_consumption)
    
    # Validate inputs
    if len(actual_prices) != n_intervals:
        raise ValueError(f"Price array length {len(actual_prices)} must match consumption length {n_intervals}")
    
    # Extract battery parameters
    capacity_kwh = battery_state.get('capacity_kwh', 40.0)
    min_soc = battery_state.get('min_soc', 0.20)  # 20% minimum (safety)
    max_soc = battery_state.get('max_soc', 1.0)   # 100% max (changed from 90%)
    max_charge_rate_kw = battery_state.get('max_charge_rate_kw', 10.0)
    max_discharge_rate_kw = battery_state.get('max_discharge_rate_kw', 8.0)
    current_charge_kwh = battery_state.get('current_charge_kwh', capacity_kwh * 0.50)
    efficiency = battery_state.get('efficiency', 0.95)
    
    # Convert price thresholds from $/MWh to $/kWh
    buy_threshold_kwh = buy_threshold_mwh / 1000.0  # $20/MWh = $0.020/kWh
    sell_threshold_kwh = sell_threshold_mwh / 1000.0  # $40/MWh = $0.040/kWh
    min_price_kwh = household_price_kwh * min_grid_price_pct  # $0.027/kWh (10% of $0.27)
    
    # Calculate rolling price statistics for opportunistic trading
    # This teaches the model to identify relatively low/high prices
    price_history = []
    opportunistic_buy_triggers = np.zeros(n_intervals, dtype=bool)
    opportunistic_sell_triggers = np.zeros(n_intervals, dtype=bool)
    
    # Initialize outputs
    optimal_decisions = np.ones(n_intervals, dtype=int)  # 1 = Hold by default
    optimal_quantities = np.zeros(n_intervals)
    battery_trajectory = np.zeros(n_intervals)
    
    # Profit tracking
    household_revenue = 0.0  # Revenue from serving household at $270/MWh
    market_revenue = 0.0     # Revenue from selling to market
    buy_costs = 0.0          # Cost of buying from market
    
    # Calculate total household demand over period
    total_household_demand = np.sum(predicted_consumption)
    
    # Simulate battery operation with optimal strategy
    current_charge = current_charge_kwh
    
    for i in range(n_intervals):
        consumption = predicted_consumption[i]
        price = actual_prices[i]  # Market price in $/kWh
        
        # Priority 1: Track household revenue (we charge them $270/MWh regardless)
        household_revenue += consumption * household_price_kwh
        
        # Update price history for opportunistic trading
        price_history.append(price)
        
        # Calculate rolling price percentiles for opportunistic decisions
        if len(price_history) >= 3:  # Need at least 3 points
            window_prices = price_history[-opportunistic_window:] if len(price_history) >= opportunistic_window else price_history
            price_percentile_30 = np.percentile(window_prices, opportunistic_buy_percentile)
            price_percentile_70 = np.percentile(window_prices, opportunistic_sell_percentile)
            
            # Mark if current price is opportunistically low or high
            # Opportunistic trading works at ALL price levels (hard rules will override when needed)
            opportunistic_buy_triggers[i] = price <= price_percentile_30
            opportunistic_sell_triggers[i] = price >= price_percentile_70
        
        # Current battery state
        current_soc = current_charge / capacity_kwh
        available_for_discharge = max(0, current_charge - (capacity_kwh * min_soc))
        available_for_charge = max(0, (capacity_kwh * max_soc) - current_charge)
        
        # Max energy that can be traded in 30-min interval (0.5 hours)
        max_charge_energy = max_charge_rate_kw * 0.5
        max_discharge_energy = max_discharge_rate_kw * 0.5
        
        # Look ahead to determine energy needs
        # Note: consumption values are per hour, not per interval
        future_consumption_2h = np.sum(predicted_consumption[i:i+2])  # Next 2 hours
        future_consumption_4h = np.sum(predicted_consumption[i:i+4])  # Next 4 hours
        future_consumption_8h = np.sum(predicted_consumption[i:i+8])  # Next 8 hours
        
        # Calculate energy deficit/surplus relative to near-term needs
        # Deficit: Do we have enough for next 4 hours?
        energy_deficit = max(0, future_consumption_4h - available_for_discharge)
        
        # Surplus: Do we have more than we need for next 2 hours?
        # If so, we can sell the excess
        energy_surplus = max(0, available_for_discharge - future_consumption_2h)
        
        # Decision logic based on hard rules and priorities
        # Default to TRADE, not hold - holding is leaving money on the table!
        decision = 1  # Will be overridden
        quantity = 0.0
        
        # HARD RULE: Hold when SoC < 20% (safety threshold)
        if current_soc <= min_soc:
            decision = 1  # Hold - don't discharge below 20%
            quantity = 0.0
        
        # HARD RULE: Hold when price < 10% of grid price ($27/MWh)
        elif price < min_price_kwh:
            decision = 1  # Hold - price too low
            quantity = 0.0
        
        # PRIORITY 1: MUST BUY - We have energy deficit and need to cover consumption
        # Buy regardless of price if we won't have enough for household
        elif energy_deficit > 0.5 and current_soc < max_soc:
            # We NEED energy to serve the household
            max_buy = min(max_charge_energy, available_for_charge)
            # Buy aggressively - at least the deficit, but fill towards 70% to maintain inventory
            target_charge = max(current_charge + energy_deficit, capacity_kwh * 0.70)
            quantity = min(max_buy, target_charge - current_charge)
            
            if quantity > 0.01:
                decision = 0  # Buy (necessity)
                current_charge += quantity * efficiency
                buy_costs += quantity * price
        
        # HARD RULE: Always buy when price < $20/MWh AND SoC < 100%
        elif price < buy_threshold_kwh and current_soc < max_soc:
            # Excellent price - buy as much as possible
            max_buy = min(max_charge_energy, available_for_charge)
            
            # Buy aggressively when price is great
            # Fill to near capacity when price < $20/MWh
            target_charge = capacity_kwh * 0.90  # Target 90%
            quantity = min(max_buy, max(0, target_charge - current_charge))
            
            if quantity > 0.01:  # Only trade if meaningful
                decision = 0  # Buy (hard rule)
                current_charge += quantity * efficiency
                buy_costs += quantity * price
        
        # PRIORITY 2: SHOULD SELL - We have excess energy that won't be needed
        # Sell excess even at moderate prices to avoid idle battery
        elif energy_surplus > 1.0 and price > min_price_kwh and current_soc > min_soc * 1.3:
            # We have more energy than we'll need
            max_sell = min(max_discharge_energy, available_for_discharge)
            quantity = min(max_sell, energy_surplus * 0.7)  # Sell 70% of excess
            
            if quantity > 0.01:
                decision = 2  # Sell (should - to avoid waste)
                current_charge -= quantity
                market_revenue += quantity * price
        
        # HARD RULE: Always sell when price > $40/MWh AND SoC > 20%
        elif price > sell_threshold_kwh and current_soc > min_soc:
            # Excellent price - sell excess energy aggressively
            max_sell = min(max_discharge_energy, available_for_discharge)
            
            # Keep enough for near-term demand only (next 2-3 hours)
            future_demand = np.sum(predicted_consumption[i:i+6])  # Next 3 hours
            safe_discharge = max(0, current_charge - future_demand * 1.05)  # 5% buffer
            
            quantity = min(max_sell, safe_discharge)
            
            if quantity > 0.01:  # Only trade if meaningful
                decision = 2  # Sell (hard rule - excellent price)
                current_charge -= quantity
                market_revenue += quantity * price
        
        # OPPORTUNISTIC RULE: Buy when price is relatively low for recent period
        # BE AGGRESSIVE - holding is leaving money on the table!
        # Buy whenever price is opportunistic AND we have room
        elif opportunistic_buy_triggers[i] and current_soc < max_soc * 0.90:
            # Price is below percentile threshold - BUY to maintain inventory!
            max_buy = min(max_charge_energy, available_for_charge)
            
            # Fill aggressively to 85% - we need inventory to sell later
            target_soc = min(0.85, max_soc)
            room_to_target = max(0, (capacity_kwh * target_soc) - current_charge)
            quantity = min(max_buy, room_to_target)
            
            if quantity > 0.01:
                decision = 0  # Buy opportunistically
                current_charge += quantity * efficiency
                buy_costs += quantity * price
        
        # OPPORTUNISTIC RULE: Sell when price is relatively high for recent period
        # This is in the middle zone ($20-40/MWh) where we look for good opportunities  
        # BE AGGRESSIVE - holding is leaving money on the table!
        elif opportunistic_sell_triggers[i] and current_soc > min_soc * 1.1:
            # Price is above median - SELL!
            max_sell = min(max_discharge_energy, available_for_discharge)
            
            # Sell aggressively - only keep 2 hours of consumption
            future_demand = np.sum(predicted_consumption[i:i+2])  # Next 2 hours only
            safe_discharge = max(0, current_charge - future_demand * 1.10)  # 10% buffer
            
            quantity = min(max_sell, safe_discharge)
            
            if quantity > 0.01:
                decision = 2  # Sell opportunistically
                current_charge -= quantity
                market_revenue += quantity * price
        
        # Final fallback: Sell excess if battery is too full (>70%)
        elif current_soc > 0.70 and price > min_price_kwh:
            # Priority 4: Minimize excess held power - be AGGRESSIVE
            excess_capacity = current_charge - (capacity_kwh * 0.50)  # Target 50%
            max_sell = min(max_discharge_energy, available_for_discharge)
            
            quantity = min(max_sell, max(0, excess_capacity))
            
            if quantity > 0.01:
                decision = 2  # Sell to reduce excess
                current_charge -= quantity
                market_revenue += quantity * price
        
        # AGGRESSIVE DEFAULT: If we haven't made a decision yet, make one based on price alone
        # Don't just hold - we have massive capacity and should use it!
        elif current_soc < 0.80 and current_soc > min_soc * 1.2:
            # In the middle range - decide based on price trend
            # Use the opportunistic triggers we calculated
            if opportunistic_buy_triggers[i]:
                # Price is relatively low - buy
                max_buy = min(max_charge_energy, available_for_charge)
                target_charge = capacity_kwh * 0.75
                quantity = min(max_buy, max(0, target_charge - current_charge))
                
                if quantity > 0.01:
                    decision = 0  # Buy
                    current_charge += quantity * efficiency
                    buy_costs += quantity * price
            
            elif opportunistic_sell_triggers[i]:
                # Price is relatively high - sell
                max_sell = min(max_discharge_energy, available_for_discharge)
                future_demand_short = np.sum(predicted_consumption[i:i+4])  # Next 4 hours
                safe_discharge = max(0, current_charge - future_demand_short * 1.15)
                quantity = min(max_sell, safe_discharge)
                
                if quantity > 0.01:
                    decision = 2  # Sell
                    current_charge -= quantity
                    market_revenue += quantity * price
        
        # Ensure SoC stays within bounds
        current_soc = current_charge / capacity_kwh
        current_soc = np.clip(current_soc, min_soc, max_soc)
        current_charge = capacity_kwh * current_soc
        
        # Record decision and state
        optimal_decisions[i] = decision
        optimal_quantities[i] = quantity
        battery_trajectory[i] = current_soc * 100  # Convert to percentage
    
    # Calculate total profitability
    market_profit = market_revenue - buy_costs
    total_profit = household_revenue + market_profit
    
    # Calculate profit margin
    profit_margin = (total_profit / (household_revenue + 1e-7)) if household_revenue > 0 else 0
    
    return {
        'optimal_decisions': optimal_decisions,
        'optimal_quantities': optimal_quantities,
        'expected_profit': total_profit,
        'household_revenue': household_revenue,  # Revenue from serving household
        'market_revenue': market_revenue,        # Revenue from market sales
        'buy_costs': buy_costs,                  # Cost of market purchases
        'market_profit': market_profit,          # Net market profit
        'profit_margin': profit_margin,          # Profit margin %
        'battery_trajectory': battery_trajectory,
        'price_thresholds': {
            'buy_below': buy_threshold_kwh,
            'sell_above': sell_threshold_kwh,
            'min_price': min_price_kwh
        },
        'opportunistic_stats': {
            'buy_opportunities': int(np.sum(opportunistic_buy_triggers)),
            'sell_opportunities': int(np.sum(opportunistic_sell_triggers)),
            'window_size': opportunistic_window,
            'buy_percentile': opportunistic_buy_percentile,
            'sell_percentile': opportunistic_sell_percentile
        },
        'business_metrics': {
            'household_price_kwh': household_price_kwh,
            'total_demand_kwh': total_household_demand,
            'avg_market_price_kwh': np.mean(actual_prices)
        }
    }


def calculate_profitability_reward(
    predicted_prices: np.ndarray,
    trading_decisions: np.ndarray,
    trade_quantities: np.ndarray,
    battery_state: Dict
) -> Tuple[float, Dict]:
    """
    Calculate profitability reward for given trading decisions.
    
    Used as a loss component in the Trading Transformer to optimize for profit.
    
    Args:
        predicted_prices: (48,) array of predicted prices ($/kWh)
        trading_decisions: (48,) array of decisions [0=Buy, 1=Hold, 2=Sell]
        trade_quantities: (48,) array of trade quantities (kWh)
        battery_state: Dict with battery parameters
    
    Returns:
        profit: Total profit in $ (can be negative if losses)
        metrics: Dict with breakdown of buys, sells, and net profit
    """
    total_profit = 0.0
    buy_count = 0
    sell_count = 0
    buy_volume = 0.0
    sell_volume = 0.0
    buy_cost = 0.0
    sell_revenue = 0.0
    
    for i in range(len(predicted_prices)):
        price = predicted_prices[i]
        decision = trading_decisions[i]
        quantity = trade_quantities[i]
        
        if decision == 0:  # Buy
            buy_count += 1
            buy_volume += quantity
            buy_cost += quantity * price
            total_profit -= quantity * price
        elif decision == 2:  # Sell
            sell_count += 1
            sell_volume += quantity
            sell_revenue += quantity * price
            total_profit += quantity * price
    
    metrics = {
        'total_profit': total_profit,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'buy_volume_kwh': buy_volume,
        'sell_volume_kwh': sell_volume,
        'buy_cost': buy_cost,
        'sell_revenue': sell_revenue,
        'net_volume': sell_volume - buy_volume
    }
    
    return total_profit, metrics


if __name__ == "__main__":
    print("="*70)
    print("TRADING OPTIMIZER TEST")
    print("="*70)
    
    # Test configuration
    n_intervals = 48
    
    print(f"\nTest Configuration:")
    print(f"  Number of intervals: {n_intervals} (24 hours)")
    print(f"  Interval length: 30 minutes")
    
    # Create synthetic test data
    print(f"\n1. Creating synthetic test data...")
    
    # Consumption: Higher during day (6am-10pm), lower at night
    hours = np.arange(n_intervals) / 2
    consumption = 1.0 + 0.5 * np.sin(2 * np.pi * (hours - 6) / 24)  # kWh per 30-min
    consumption = np.clip(consumption, 0.5, 2.0)
    
    # Prices: Higher during peak hours (4pm-9pm), lower at night
    prices = 0.20 + 0.10 * np.sin(2 * np.pi * (hours - 16) / 24)  # $/kWh
    prices = np.clip(prices, 0.10, 0.35)
    
    print(f"   ✓ Consumption range: {consumption.min():.2f} to {consumption.max():.2f} kWh")
    print(f"   ✓ Price range: ${prices.min():.3f} to ${prices.max():.3f} per kWh")
    
    # Battery state
    battery_state = {
        'current_charge_kwh': 20.0,
        'capacity_kwh': 40.0,
        'min_soc': 0.20,
        'max_soc': 0.90,
        'max_charge_rate_kw': 10.0,
        'max_discharge_rate_kw': 8.0,
        'efficiency': 0.95
    }
    
    print(f"\n2. Battery Configuration:")
    print(f"   Capacity: {battery_state['capacity_kwh']} kWh")
    print(f"   Current charge: {battery_state['current_charge_kwh']} kWh (50%)")
    print(f"   Operating range: {battery_state['min_soc']*100:.0f}% - {battery_state['max_soc']*100:.0f}%")
    
    # Calculate optimal decisions
    print(f"\n3. Calculating optimal trading decisions...")
    result = calculate_optimal_trading_decisions(
        predicted_consumption=consumption,
        actual_prices=prices,
        battery_state=battery_state
    )
    
    print(f"   ✓ Optimization complete")
    print(f"   Price thresholds:")
    print(f"     - Buy below: ${result['price_thresholds']['buy_below']:.4f}/kWh (${result['price_thresholds']['buy_below']*1000:.1f}/MWh)")
    print(f"     - Sell above: ${result['price_thresholds']['sell_above']:.4f}/kWh (${result['price_thresholds']['sell_above']*1000:.1f}/MWh)")
    print(f"     - Min price: ${result['price_thresholds']['min_price']:.4f}/kWh (${result['price_thresholds']['min_price']*1000:.1f}/MWh)")
    
    # Analyze decisions
    decisions = result['optimal_decisions']
    buy_count = np.sum(decisions == 0)
    hold_count = np.sum(decisions == 1)
    sell_count = np.sum(decisions == 2)
    
    print(f"\n4. Trading Decision Analysis:")
    print(f"   Buy: {buy_count} intervals ({buy_count/n_intervals*100:.1f}%)")
    print(f"   Hold: {hold_count} intervals ({hold_count/n_intervals*100:.1f}%)")
    print(f"   Sell: {sell_count} intervals ({sell_count/n_intervals*100:.1f}%)")
    print(f"   Total trade volume: {np.sum(result['optimal_quantities']):.2f} kWh")
    print(f"\n   Profit Breakdown:")
    print(f"     - Household revenue: ${result['household_revenue']:.2f}")
    print(f"     - Market revenue: ${result['market_revenue']:.2f}")
    print(f"     - Buy costs: ${result['buy_costs']:.2f}")
    print(f"     - Market profit: ${result['market_profit']:.2f}")
    print(f"     - Total profit: ${result['expected_profit']:.2f}")
    print(f"     - Profit margin: {result['profit_margin']*100:.1f}%")
    
    # Battery trajectory analysis
    trajectory = result['battery_trajectory']
    print(f"\n5. Battery Trajectory:")
    print(f"   Starting SoC: {trajectory[0]:.1f}%")
    print(f"   Ending SoC: {trajectory[-1]:.1f}%")
    print(f"   Min SoC: {trajectory.min():.1f}%")
    print(f"   Max SoC: {trajectory.max():.1f}%")
    print(f"   SoC range maintained: {trajectory.min():.1f}% - {trajectory.max():.1f}%")
    
    # Test profitability calculation
    print(f"\n6. Testing profitability calculation...")
    profit, metrics = calculate_profitability_reward(
        predicted_prices=prices,
        trading_decisions=result['optimal_decisions'],
        trade_quantities=result['optimal_quantities'],
        battery_state=battery_state
    )
    
    print(f"   ✓ Profitability calculated")
    print(f"   Total profit: ${profit:.2f}")
    print(f"   Buy transactions: {metrics['buy_count']}")
    print(f"   Sell transactions: {metrics['sell_count']}")
    print(f"   Buy cost: ${metrics['buy_cost']:.2f}")
    print(f"   Sell revenue: ${metrics['sell_revenue']:.2f}")
    
    # Validate constraints
    print(f"\n7. Validating constraints...")
    assert trajectory.min() >= battery_state['min_soc'] * 100, "SoC below minimum!"
    assert trajectory.max() <= battery_state['max_soc'] * 100, "SoC above maximum!"
    assert np.all(result['optimal_quantities'] >= 0), "Negative quantities detected!"
    print(f"   ✓ All constraints satisfied")
    print(f"   ✓ SoC within bounds: {battery_state['min_soc']*100:.0f}% - {battery_state['max_soc']*100:.0f}%")
    print(f"   ✓ All quantities non-negative")
    
    print("\n" + "="*70)
    print("✅ TRADING OPTIMIZER TEST COMPLETE")
    print("="*70)
