"""
Debug script to understand why no sell decisions are generated.

Run this in a Colab cell to diagnose the trading optimizer issue.
"""

import numpy as np
import pandas as pd

print("="*70)
print("TRADING LABEL GENERATION DIAGNOSTICS")
print("="*70)

# This should be run after Section 5 Step 3 in the notebook
# It analyzes why no sell decisions are being generated

# Check 1: Price distribution
print("\n1. PRICE ANALYSIS")
print("-"*50)
prices_kwh = optimal_prices
prices_mwh = prices_kwh * 1000

print(f"Price statistics ($/kWh):")
print(f"  Min:  ${prices_kwh.min():.4f}")
print(f"  25%:  ${np.percentile(prices_kwh, 25):.4f}")
print(f"  50%:  ${np.percentile(prices_kwh, 50):.4f}")
print(f"  75%:  ${np.percentile(prices_kwh, 75):.4f}")
print(f"  Max:  ${prices_kwh.max():.4f}")
print(f"  Mean: ${prices_kwh.mean():.4f}")

print(f"\nPrice statistics ($/MWh):")
print(f"  Min:  ${prices_mwh.min():.2f}")
print(f"  Median: ${np.percentile(prices_mwh, 50):.2f}")
print(f"  Max:  ${prices_mwh.max():.2f}")

# Check sell threshold
sell_threshold_kwh = 0.040  # $40/MWh
prices_above_sell = (prices_kwh > sell_threshold_kwh).sum()
print(f"\nSell threshold: ${sell_threshold_kwh:.4f}/kWh (${sell_threshold_kwh*1000:.0f}/MWh)")
print(f"Prices above sell threshold: {prices_above_sell}/{len(prices_kwh)} ({prices_above_sell/len(prices_kwh)*100:.1f}%)")

if prices_above_sell == 0:
    print("  ❌ NO PRICES ABOVE SELL THRESHOLD - This is why no sells!")
    print(f"  📊 Max price is ${prices_mwh.max():.2f}/MWh, need >${sell_threshold_kwh*1000:.0f}/MWh")
    print("\n  SOLUTION: Lower sell_threshold_mwh in trading_optimizer")

# Check 2: Battery state
print("\n2. BATTERY STATE ANALYSIS")
print("-"*50)

# Get battery data from the notebook
batt_soc = battery_df['battery_soc_percent'].values[:len(optimal_decisions)]
batt_avail = battery_df['battery_available_kwh'].values[:len(optimal_decisions)]

print(f"Battery SoC statistics:")
print(f"  Min:  {batt_soc.min():.1f}%")
print(f"  25%:  {np.percentile(batt_soc, 25):.1f}%")
print(f"  50%:  {np.percentile(batt_soc, 50):.1f}%")
print(f"  75%:  {np.percentile(batt_soc, 75):.1f}%")
print(f"  Max:  {batt_soc.max():.1f}%")

print(f"\nBattery available energy:")
print(f"  Min:  {batt_avail.min():.2f} kWh")
print(f"  Mean: {batt_avail.mean():.2f} kWh")
print(f"  Max:  {batt_avail.max():.2f} kWh")

min_soc_threshold = 20.0
soc_above_min = (batt_soc > min_soc_threshold * 1.3).sum()  # Need >26% for sell
print(f"\nSoC above sell threshold ({min_soc_threshold*1.3:.0f}%): {soc_above_min}/{len(batt_soc)} ({soc_above_min/len(batt_soc)*100:.1f}%)")

if soc_above_min == 0:
    print("  ❌ BATTERY ALWAYS TOO LOW - This is why no sells!")
    print("\n  SOLUTION: Check generate_battery_data() function")

# Check 3: Consumption patterns
print("\n3. CONSUMPTION ANALYSIS")
print("-"*50)

cons_pred = consumption_predictions[:len(optimal_decisions), :]
hourly_cons = cons_pred.sum(axis=1) / 2  # Convert from 30-min to hourly

print(f"Predicted consumption (hourly avg):")
print(f"  Min:  {hourly_cons.min():.2f} kWh/h")
print(f"  Mean: {hourly_cons.mean():.2f} kWh/h")
print(f"  Max:  {hourly_cons.max():.2f} kWh/h")

# Check 4: Decision breakdown by condition
print("\n4. DECISION CONDITION ANALYSIS")
print("-"*50)

print("Let's simulate one interval to see what's happening...")
test_idx = len(optimal_decisions) // 2  # Middle of dataset

test_consumption = consumption_predictions[test_idx]
test_price_kwh = optimal_prices[test_idx]
test_battery_state = {
    'current_charge_kwh': battery_df['battery_charge_kwh'].iloc[test_idx],
    'capacity_kwh': 40.0,
    'min_soc': 0.20,
    'max_soc': 1.0
}

print(f"\nTest interval {test_idx}:")
print(f"  Price: ${test_price_kwh:.4f}/kWh (${test_price_kwh*1000:.2f}/MWh)")
print(f"  Battery charge: {test_battery_state['current_charge_kwh']:.2f} kWh")
print(f"  Battery SoC: {(test_battery_state['current_charge_kwh']/test_battery_state['capacity_kwh'])*100:.1f}%")
print(f"  Avg consumption: {test_consumption.mean():.2f} kWh")

# Check sell conditions
sell_threshold_kwh = 0.040
min_soc = 0.20
current_soc = test_battery_state['current_charge_kwh'] / test_battery_state['capacity_kwh']

print(f"\nSell condition checks:")
print(f"  Price > ${sell_threshold_kwh:.4f}? {test_price_kwh > sell_threshold_kwh} (actual: ${test_price_kwh:.4f})")
print(f"  SoC > {min_soc*100:.0f}%? {current_soc > min_soc} (actual: {current_soc*100:.1f}%)")

if test_price_kwh <= sell_threshold_kwh:
    print(f"  ❌ Price too low for sell (need >${sell_threshold_kwh*1000:.0f}/MWh, have {test_price_kwh*1000:.2f}/MWh)")
if current_soc <= min_soc:
    print(f"  ❌ Battery too low for sell (need >{min_soc*100:.0f}%, have {current_soc*100:.1f}%)")

# Check 5: Recommendations
print("\n5. RECOMMENDATIONS")
print("-"*50)

if prices_above_sell < len(prices_kwh) * 0.10:
    print("✅ RECOMMENDATION 1: Lower sell_threshold_mwh")
    print(f"   Current: $40/MWh")
    print(f"   Suggested: ${np.percentile(prices_mwh, 60):.0f}/MWh (60th percentile)")
    print(f"   This would trigger sells in {((prices_kwh > np.percentile(prices_kwh, 60)).sum())/len(prices_kwh)*100:.0f}% of cases")

if soc_above_min < len(batt_soc) * 0.50:
    print("\n✅ RECOMMENDATION 2: Increase battery SoC in generated data")
    print(f"   Currently, only {soc_above_min/len(batt_soc)*100:.0f}% of time SoC > 26%")
    print(f"   Suggestion: Initialize battery at 50-70% SoC range")

print("\n✅ RECOMMENDATION 3: Enable opportunistic selling")
print("   The opportunistic_sell_percentile is set to 45%")
print(f"   This should trigger sells when price > {np.percentile(prices_mwh, 55):.0f}/MWh")
print(f"   Verify this is working in the trading_optimizer")

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE")
print("="*70)
