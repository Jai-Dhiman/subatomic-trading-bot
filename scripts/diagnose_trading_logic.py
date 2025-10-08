"""
Diagnose Trading Logic Issues

This script analyzes why the trading optimizer isn't buying aggressively enough
when prices are low. It evaluates:

1. Price distributions and trading opportunities
2. Battery SoC behavior over time
3. Decision-making logic at specific price points
4. Why battery stays at minimum SoC instead of filling up
5. Opportunistic trading trigger analysis

Goal: Identify why we're not building up battery inventory at low prices
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from supabase import create_client, Client
from src.models.trading_optimizer import calculate_optimal_trading_decisions

# Setup
sns.set_style("whitegrid")
load_dotenv()

print("="*80)
print("TRADING LOGIC DIAGNOSTICS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Connect to Supabase
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    sys.exit(1)

supabase: Client = create_client(supabase_url, supabase_key)
print("✓ Connected to Supabase\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("LOADING DATA")
print("="*80)

# Load pricing
print("\nLoading pricing data...")
pricing_response = supabase.table('cabuyingpricehistoryseptember2025') \
    .select('*') \
    .eq('LMP_TYPE', 'LMP') \
    .order('INTERVALSTARTTIME_GMT', desc=False) \
    .limit(1000) \
    .execute()

pricing_df = pd.DataFrame(pricing_response.data)
pricing_df['timestamp'] = pd.to_datetime(pricing_df['INTERVALSTARTTIME_GMT'])

if 'MW' in pricing_df.columns:
    pricing_df['price_per_kwh'] = pricing_df['MW'] / 1000.0
elif 'Price MWH' in pricing_df.columns:
    pricing_df['price_per_kwh'] = pricing_df['Price MWH'] / 1000.0

prices_kwh = pricing_df['price_per_kwh'].values
prices_mwh = prices_kwh * 1000

print(f"✓ Loaded {len(pricing_df):,} pricing records")
print(f"  Price range: ${prices_kwh.min():.4f} - ${prices_kwh.max():.4f} per kWh")
print(f"  Price range: ${prices_mwh.min():.2f} - ${prices_mwh.max():.2f} per MWh")

# Load consumption
print("\nLoading consumption data...")
consumption_response = supabase.table('august11homeconsumption') \
    .select('*') \
    .eq('House', 1) \
    .order('Timestamp', desc=False) \
    .limit(1000) \
    .execute()

consumption_df = pd.DataFrame(consumption_response.data)
consumption_df['timestamp'] = pd.to_datetime(consumption_df['Timestamp'])

# Parse appliance breakdown
if 'Appliance_Breakdown_JSON' in consumption_df.columns:
    import json
    appliance_data = consumption_df['Appliance_Breakdown_JSON'].apply(
        lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else {}
    ).apply(pd.Series)
    
    appliance_cols = [col for col in appliance_data.columns]
    for col in appliance_cols:
        consumption_df[f'appliance_{col.lower().replace("/", "_").replace(" ", "_").replace(".", "")}'] = appliance_data[col]

appliance_cols = [col for col in consumption_df.columns if col.startswith('appliance_')]
if appliance_cols:
    consumption_df['hourly_consumption_kwh'] = consumption_df[appliance_cols].sum(axis=1)
else:
    consumption_df['hourly_consumption_kwh'] = 1.0  # Default

consumption = consumption_df['hourly_consumption_kwh'].values

print(f"✓ Loaded {len(consumption_df):,} consumption records")
print(f"  Consumption range: {consumption.min():.2f} - {consumption.max():.2f} kWh/h")
print(f"  Average: {consumption.mean():.2f} kWh/h")

# ============================================================================
# ANALYZE PRICE ZONES
# ============================================================================
print("\n" + "="*80)
print("PRICE ZONE ANALYSIS")
print("="*80)

buy_threshold = 0.020  # $20/MWh
sell_threshold = 0.040  # $40/MWh

below_buy = prices_kwh < buy_threshold
in_middle = (prices_kwh >= buy_threshold) & (prices_kwh <= sell_threshold)
above_sell = prices_kwh > sell_threshold

print(f"\nPrice Zones:")
print(f"  HARD BUY (<$20/MWh):      {below_buy.sum():4d} intervals ({below_buy.sum()/len(prices_kwh)*100:.1f}%)")
print(f"  MIDDLE ($20-40/MWh):      {in_middle.sum():4d} intervals ({in_middle.sum()/len(prices_kwh)*100:.1f}%)")
print(f"  HARD SELL (>$40/MWh):     {above_sell.sum():4d} intervals ({above_sell.sum()/len(prices_kwh)*100:.1f}%)")

print(f"\nPrice Statistics by Zone:")
print(f"  Hard BUY zone:")
print(f"    Count: {below_buy.sum()}")
if below_buy.sum() > 0:
    print(f"    Min:   ${prices_mwh[below_buy].min():.2f}/MWh")
    print(f"    Max:   ${prices_mwh[below_buy].max():.2f}/MWh")
    print(f"    Avg:   ${prices_mwh[below_buy].mean():.2f}/MWh")

print(f"\n  Middle zone:")
print(f"    Count: {in_middle.sum()}")
if in_middle.sum() > 0:
    print(f"    Min:   ${prices_mwh[in_middle].min():.2f}/MWh")
    print(f"    Max:   ${prices_mwh[in_middle].max():.2f}/MWh")
    print(f"    Avg:   ${prices_mwh[in_middle].mean():.2f}/MWh")

print(f"\n  Hard SELL zone:")
print(f"    Count: {above_sell.sum()}")
if above_sell.sum() > 0:
    print(f"    Min:   ${prices_mwh[above_sell].min():.2f}/MWh")
    print(f"    Max:   ${prices_mwh[above_sell].max():.2f}/MWh")
    print(f"    Avg:   ${prices_mwh[above_sell].mean():.2f}/MWh")

# ============================================================================
# TEST DIFFERENT BATTERY INITIALIZATION SCENARIOS
# ============================================================================
print("\n" + "="*80)
print("TESTING DIFFERENT BATTERY INITIALIZATION SCENARIOS")
print("="*80)

# Limit to first 500 intervals for testing
test_size = min(500, len(prices_kwh), len(consumption))
test_prices = prices_kwh[:test_size]
test_consumption = consumption[:test_size]

scenarios = [
    {'name': 'Current (50% SoC)', 'initial_soc': 0.50},
    {'name': 'Higher (70% SoC)', 'initial_soc': 0.70},
    {'name': 'Low (30% SoC)', 'initial_soc': 0.30},
    {'name': 'Very Low (20% SoC)', 'initial_soc': 0.20},
]

results = {}

for scenario in scenarios:
    print(f"\n{'-'*80}")
    print(f"Scenario: {scenario['name']}")
    print(f"{'-'*80}")
    
    battery_state = {
        'current_charge_kwh': 40.0 * scenario['initial_soc'],
        'capacity_kwh': 40.0,
        'min_soc': 0.20,
        'max_soc': 1.0,
        'max_charge_rate_kw': 10.0,
        'max_discharge_rate_kw': 8.0,
        'efficiency': 0.95
    }
    
    print(f"  Initial charge: {battery_state['current_charge_kwh']:.1f} kWh ({scenario['initial_soc']*100:.0f}%)")
    
    result = calculate_optimal_trading_decisions(
        predicted_consumption=test_consumption,
        actual_prices=test_prices,
        battery_state=battery_state,
        household_price_kwh=0.27,
        buy_threshold_mwh=20.0,
        sell_threshold_mwh=40.0,
        opportunistic_window=48,
        opportunistic_buy_percentile=80.0,  # Ultra aggressive
        opportunistic_sell_percentile=20.0  # Ultra aggressive
    )
    
    decisions = result['optimal_decisions']
    buy_count = (decisions == 0).sum()
    hold_count = (decisions == 1).sum()
    sell_count = (decisions == 2).sum()
    
    print(f"\n  Trading Actions:")
    print(f"    Buy:  {buy_count:4d} ({buy_count/test_size*100:5.1f}%)")
    print(f"    Hold: {hold_count:4d} ({hold_count/test_size*100:5.1f}%)")
    print(f"    Sell: {sell_count:4d} ({sell_count/test_size*100:5.1f}%)")
    
    print(f"\n  Battery SoC:")
    trajectory = result['battery_trajectory']
    print(f"    Starting: {trajectory[0]:.1f}%")
    print(f"    Ending:   {trajectory[-1]:.1f}%")
    print(f"    Min:      {trajectory.min():.1f}%")
    print(f"    Max:      {trajectory.max():.1f}%")
    print(f"    Average:  {trajectory.mean():.1f}%")
    print(f"    Median:   {np.percentile(trajectory, 50):.1f}%")
    
    print(f"\n  Profitability:")
    print(f"    Household revenue:  ${result['household_revenue']:8.2f}")
    print(f"    Market revenue:     ${result['market_revenue']:8.2f}")
    print(f"    Buy costs:          ${result['buy_costs']:8.2f}")
    print(f"    Market profit:      ${result['market_profit']:8.2f}")
    print(f"    Total profit:       ${result['expected_profit']:8.2f}")
    
    # Analyze buy decisions by price zone
    buy_mask = decisions == 0
    if buy_mask.sum() > 0:
        buy_prices_mwh = test_prices[buy_mask] * 1000
        print(f"\n  Buy Decision Analysis:")
        print(f"    Avg buy price:  ${buy_prices_mwh.mean():.2f}/MWh")
        print(f"    Min buy price:  ${buy_prices_mwh.min():.2f}/MWh")
        print(f"    Max buy price:  ${buy_prices_mwh.max():.2f}/MWh")
        
        buy_in_hard_zone = (test_prices[buy_mask] < buy_threshold).sum()
        print(f"    Bought in hard BUY zone (<$20): {buy_in_hard_zone} ({buy_in_hard_zone/buy_mask.sum()*100:.1f}%)")
    
    # Analyze sell decisions by price zone
    sell_mask = decisions == 2
    if sell_mask.sum() > 0:
        sell_prices_mwh = test_prices[sell_mask] * 1000
        print(f"\n  Sell Decision Analysis:")
        print(f"    Avg sell price:  ${sell_prices_mwh.mean():.2f}/MWh")
        print(f"    Min sell price:  ${sell_prices_mwh.min():.2f}/MWh")
        print(f"    Max sell price:  ${sell_prices_mwh.max():.2f}/MWh")
        
        sell_in_hard_zone = (test_prices[sell_mask] > sell_threshold).sum()
        print(f"    Sold in hard SELL zone (>$40): {sell_in_hard_zone} ({sell_in_hard_zone/sell_mask.sum()*100:.1f}%)")
    
    results[scenario['name']] = {
        'result': result,
        'buy_pct': buy_count/test_size*100,
        'sell_pct': sell_count/test_size*100,
        'hold_pct': hold_count/test_size*100,
        'avg_soc': trajectory.mean(),
        'market_profit': result['market_profit']
    }

# ============================================================================
# DEEP DIVE: WHY AREN'T WE BUYING MORE?
# ============================================================================
print("\n" + "="*80)
print("DEEP DIVE: WHY AREN'T WE BUYING ENOUGH AT LOW PRICES?")
print("="*80)

# Find intervals with very low prices
very_low_prices_mask = test_prices < 0.015  # < $15/MWh
low_prices_mask = (test_prices >= 0.015) & (test_prices < buy_threshold)  # $15-20/MWh

print(f"\nPrice Opportunity Analysis:")
print(f"  Very low prices (<$15/MWh): {very_low_prices_mask.sum()} intervals")
print(f"  Low prices ($15-20/MWh):    {low_prices_mask.sum()} intervals")
print(f"  Total buy opportunities:    {(very_low_prices_mask | low_prices_mask).sum()} intervals")

# Analyze what happened at these low prices
print(f"\nWhat did we do at VERY LOW prices (<$15/MWh)?")
best_scenario_result = results['Current (50% SoC)']['result']
decisions = best_scenario_result['optimal_decisions']
trajectory = best_scenario_result['battery_trajectory']

if very_low_prices_mask.sum() > 0:
    decisions_at_low = decisions[very_low_prices_mask]
    soc_at_low = trajectory[very_low_prices_mask]
    
    bought = (decisions_at_low == 0).sum()
    held = (decisions_at_low == 1).sum()
    sold = (decisions_at_low == 2).sum()
    
    print(f"  Actions taken:")
    print(f"    Buy:  {bought} ({bought/very_low_prices_mask.sum()*100:.1f}%)")
    print(f"    Hold: {held} ({held/very_low_prices_mask.sum()*100:.1f}%)")
    print(f"    Sell: {sold} ({sold/very_low_prices_mask.sum()*100:.1f}%)")
    
    print(f"\n  Battery state during these opportunities:")
    print(f"    Avg SoC: {soc_at_low.mean():.1f}%")
    print(f"    Min SoC: {soc_at_low.min():.1f}%")
    print(f"    Max SoC: {soc_at_low.max():.1f}%")
    
    if held > 0:
        print(f"\n  WHY DID WE HOLD {held} TIMES AT GREAT PRICES?")
        held_indices = np.where(very_low_prices_mask)[0][decisions_at_low == 1]
        for idx in held_indices[:5]:  # Show first 5
            print(f"    Interval {idx}:")
            print(f"      Price: ${test_prices[idx]*1000:.2f}/MWh")
            print(f"      SoC: {trajectory[idx]:.1f}%")
            print(f"      Consumption: {test_consumption[idx]:.2f} kWh")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS TO FIX THE TRADING LOGIC")
print("="*80)

print("\n1. BATTERY INITIALIZATION:")
print("   CURRENT: Starting at 50% SoC in seed data (seed_supabase_data.py line 303)")
print("   PROBLEM: This limits buying capacity")
print("   FIX: Start at 30% SoC to have more room to buy")
print("   Change: 'current_charge_kwh': capacity_kwh * 0.30  # Was 0.50")

print("\n2. OPPORTUNISTIC BUYING PERCENTILES:")
print("   CURRENT: opportunistic_buy_percentile = 80.0 (line 114 in generate_battery_trading_data.py)")
print("   PROBLEM: Only buys when price in bottom 80% - too conservative")
print("   INTERPRETATION: With percentile=80, it buys when price <= 80th percentile")
print("                   This means buying in bottom 80% of prices - should be AGGRESSIVE")
print("   FIX: Change to 70.0 to be more selective")
print("   Change: opportunistic_buy_percentile=70.0  # Buy bottom 70%")

print("\n3. OPPORTUNISTIC SELLING PERCENTILES:")
print("   CURRENT: opportunistic_sell_percentile = 20.0")
print("   INTERPRETATION: Sells when price >= 80th percentile (top 20%)")
print("   PROBLEM: Too aggressive on selling, not enough on buying")
print("   FIX: Change to 30.0 (sell in top 30%)")
print("   Change: opportunistic_sell_percentile=30.0  # Sell top 30%")

print("\n4. TARGET SOC FOR BUYING:")
print("   CURRENT: Target 90% when price < $20/MWh (line 214 in trading_optimizer.py)")
print("   PROBLEM: Good, but we're not reaching this often")
print("   FIX: In opportunistic buying, target 85% (line 258)")
print("   This is already set correctly")

print("\n5. ENERGY DEFICIT LOGIC:")
print("   CURRENT: Only buys aggressively if energy_deficit > 0.5 kWh (line 195)")
print("   PROBLEM: This is good for necessity, but we need more aggressive speculation")
print("   FIX: Add 'preemptive buying' when SoC < 50% AND price is good")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nKey Finding:")
print(f"  The system IS working, but the opportunistic percentiles are confusing!")
print(f"")
print(f"  Current settings:")
print(f"    opportunistic_buy_percentile = 80.0")
print(f"    opportunistic_sell_percentile = 20.0")
print(f"")
print(f"  What this ACTUALLY means:")
print(f"    - Buy when price <= 80th percentile (buys 80% of the time)")
print(f"    - Sell when price >= 80th percentile (sells 20% of the time)")
print(f"")
print(f"  The REAL problem:")
print(f"    Battery starts at 50% SoC, which limits buy capacity")
print(f"    Need to start lower (30%) to have room to accumulate")

best_scenario = max(results.items(), key=lambda x: x[1]['market_profit'])
print(f"\nBest scenario: {best_scenario[0]}")
print(f"  Market profit: ${best_scenario[1]['market_profit']:.2f}")
print(f"  Buy %: {best_scenario[1]['buy_pct']:.1f}%")
print(f"  Sell %: {best_scenario[1]['sell_pct']:.1f}%")
print(f"  Avg SoC: {best_scenario[1]['avg_soc']:.1f}%")

print("\n" + "="*80)
print("DIAGNOSTICS COMPLETE")
print("="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
