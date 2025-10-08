"""
Comprehensive Supabase Data Visualization and Diagnostics

This script:
1. Loads all data from Supabase (pricing, consumption, battery)
2. Generates comprehensive visualizations
3. Diagnoses why no sell decisions are being generated
4. Provides specific recommendations for fixing the trading optimizer

Run this locally with:
    python scripts/visualize_supabase_data.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from supabase import create_client, Client

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*70)
print("COMPREHENSIVE SUPABASE DATA DIAGNOSTICS")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load environment
load_dotenv()
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    sys.exit(1)

print("✓ Credentials loaded")

# Initialize Supabase client
supabase: Client = create_client(supabase_url, supabase_key)
print("✓ Supabase client initialized\n")

# ============================================================================
# SECTION 1: LOAD ALL DATA
# ============================================================================
print("="*70)
print("SECTION 1: LOADING DATA FROM SUPABASE")
print("="*70)

# 1.1 Pricing Data
print("\n1.1 Loading pricing data (cabuyingpricehistoryseptember2025)...")
try:
    pricing_response = supabase.table('cabuyingpricehistoryseptember2025') \
        .select('*') \
        .eq('LMP_TYPE', 'LMP') \
        .order('INTERVALSTARTTIME_GMT', desc=False) \
        .execute()
    
    pricing_df = pd.DataFrame(pricing_response.data)
    
    if 'INTERVALSTARTTIME_GMT' in pricing_df.columns:
        pricing_df['timestamp'] = pd.to_datetime(pricing_df['INTERVALSTARTTIME_GMT'])
    
    # Convert price to $/kWh
    if 'MW' in pricing_df.columns:
        pricing_df['price_per_kwh'] = pricing_df['MW'] / 1000.0
    elif 'Price MWH' in pricing_df.columns:
        pricing_df['price_per_kwh'] = pricing_df['Price MWH'] / 1000.0
    
    print(f"   ✓ Loaded {len(pricing_df):,} pricing records")
    print(f"   ✓ Date range: {pricing_df['timestamp'].min()} to {pricing_df['timestamp'].max()}")
    print(f"   ✓ Columns: {pricing_df.columns.tolist()}")
    
except Exception as e:
    print(f"   ❌ Error loading pricing data: {e}")
    pricing_df = None

# 1.2 Consumption Data
print("\n1.2 Loading consumption data (august11homeconsumption)...")
try:
    consumption_response = supabase.table('august11homeconsumption') \
        .select('*') \
        .order('Timestamp', desc=False) \
        .execute()
    
    consumption_df = pd.DataFrame(consumption_response.data)
    
    if 'Timestamp' in consumption_df.columns:
        consumption_df['timestamp'] = pd.to_datetime(consumption_df['Timestamp'])
    
    # Parse appliance breakdown from JSON
    if 'Appliance_Breakdown_JSON' in consumption_df.columns:
        appliance_data = consumption_df['Appliance_Breakdown_JSON'].apply(pd.Series)
        
        # Rename columns
        appliance_mapping = {
            'A/C': 'appliance_ac',
            'Washing/Drying': 'appliance_washing_drying',
            'Refrig.': 'appliance_fridge',
            'EV Charging': 'appliance_ev_charging',
            'DishWasher': 'appliance_dishwasher',
            'Computers': 'appliance_computers',
            'Stovetop': 'appliance_stove',
            'Water Heater': 'appliance_water_heater',
            'Standby/ Misc.': 'appliance_misc'
        }
        appliance_data = appliance_data.rename(columns=appliance_mapping)
        
        # Add to dataframe
        for col in appliance_mapping.values():
            if col in appliance_data.columns:
                consumption_df[col] = appliance_data[col]
    
    # Calculate actual hourly consumption from appliances
    appliance_cols = [col for col in consumption_df.columns if col.startswith('appliance_')]
    if appliance_cols:
        consumption_df['hourly_consumption_kwh'] = consumption_df[appliance_cols].sum(axis=1)
    
    print(f"   ✓ Loaded {len(consumption_df):,} consumption records")
    print(f"   ✓ Houses: {sorted(consumption_df['House'].unique())}")
    print(f"   ✓ Appliance columns: {len(appliance_cols)}")
    
    if 'hourly_consumption_kwh' in consumption_df.columns:
        print(f"   ✓ Hourly consumption range: {consumption_df['hourly_consumption_kwh'].min():.2f} - {consumption_df['hourly_consumption_kwh'].max():.2f} kWh")
    
except Exception as e:
    print(f"   ❌ Error loading consumption data: {e}")
    consumption_df = None

# 1.3 Battery Trading Data
print("\n1.3 Loading battery trading data (house1_battery, house2_battery)...")
try:
    # Load both houses
    battery_dfs = []
    
    for house_id in [1, 2]:
        try:
            battery_response = supabase.table(f'house{house_id}_battery') \
                .select('*') \
                .order('timestamp', desc=False) \
                .execute()
            
            house_df = pd.DataFrame(battery_response.data)
            if len(house_df) > 0:
                house_df['timestamp'] = pd.to_datetime(house_df['timestamp'])
                battery_dfs.append(house_df)
                print(f"   ✓ Loaded {len(house_df):,} records from house{house_id}_battery")
        except Exception as e:
            print(f"   ⚠️  Could not load house{house_id}_battery: {e}")
    
    if battery_dfs:
        battery_df = pd.concat(battery_dfs, ignore_index=True).sort_values('timestamp')
        print(f"   ✓ Total battery records: {len(battery_df):,}")
        
        # Show action distribution
        if 'action' in battery_df.columns:
            action_counts = battery_df['action'].value_counts()
            print(f"   ✓ Action distribution:")
            for action, count in action_counts.items():
                print(f"      - {action}: {count} ({count/len(battery_df)*100:.1f}%)")
    else:
        battery_df = None
        print("   ⚠️  No battery data loaded")
    
except Exception as e:
    print(f"   ❌ Error loading battery data: {e}")
    battery_df = None

print("\n✓ Data loading complete\n")

# ============================================================================
# SECTION 2: DATA ANALYSIS
# ============================================================================
print("="*70)
print("SECTION 2: DATA ANALYSIS")
print("="*70)

# 2.1 Pricing Analysis
if pricing_df is not None:
    print("\n2.1 PRICING ANALYSIS")
    print("-"*50)
    
    prices_kwh = pricing_df['price_per_kwh'].values
    prices_mwh = prices_kwh * 1000
    
    print(f"Statistics ($/kWh):")
    print(f"  Min:    ${prices_kwh.min():.4f}")
    print(f"  25%:    ${np.percentile(prices_kwh, 25):.4f}")
    print(f"  Median: ${np.percentile(prices_kwh, 50):.4f}")
    print(f"  75%:    ${np.percentile(prices_kwh, 75):.4f}")
    print(f"  Max:    ${prices_kwh.max():.4f}")
    print(f"  Mean:   ${prices_kwh.mean():.4f}")
    print(f"  Std:    ${prices_kwh.std():.4f}")
    
    print(f"\nStatistics ($/MWh):")
    print(f"  Min:    ${prices_mwh.min():.2f}")
    print(f"  Median: ${np.percentile(prices_mwh, 50):.2f}")
    print(f"  Max:    ${prices_mwh.max():.2f}")
    
    # Check against trading thresholds
    buy_threshold = 0.020  # $20/MWh
    sell_threshold = 0.040  # $40/MWh
    
    below_buy = (prices_kwh < buy_threshold).sum()
    above_sell = (prices_kwh > sell_threshold).sum()
    in_middle = len(prices_kwh) - below_buy - above_sell
    
    print(f"\nTrading Zones:")
    print(f"  Hard BUY zone (<$20/MWh):   {below_buy:,} ({below_buy/len(prices_kwh)*100:.1f}%)")
    print(f"  Middle zone ($20-40/MWh):   {in_middle:,} ({in_middle/len(prices_kwh)*100:.1f}%)")
    print(f"  Hard SELL zone (>$40/MWh):  {above_sell:,} ({above_sell/len(prices_kwh)*100:.1f}%)")
    
    if above_sell == 0:
        print(f"\n  ❌ CRITICAL: NO PRICES ABOVE SELL THRESHOLD!")
        print(f"     Max price: ${prices_mwh.max():.2f}/MWh")
        print(f"     Sell threshold: $40.00/MWh")
        print(f"     Gap: ${40 - prices_mwh.max():.2f}/MWh")
        print(f"\n  💡 SOLUTION: Lower sell_threshold_mwh to ${np.percentile(prices_mwh, 75):.0f}/MWh")
    
    # Negative prices check
    negative_prices = (prices_kwh < 0).sum()
    if negative_prices > 0:
        print(f"\n  ⚠️  Negative prices detected: {negative_prices} occurrences")
        print(f"     Min negative: ${prices_mwh.min():.2f}/MWh")

# 2.2 Consumption Analysis
if consumption_df is not None:
    print("\n2.2 CONSUMPTION ANALYSIS")
    print("-"*50)
    
    if 'hourly_consumption_kwh' in consumption_df.columns:
        consumption = consumption_df['hourly_consumption_kwh'].values
        
        print(f"Hourly consumption statistics:")
        print(f"  Min:    {consumption.min():.2f} kWh/h")
        print(f"  25%:    {np.percentile(consumption, 25):.2f} kWh/h")
        print(f"  Median: {np.percentile(consumption, 50):.2f} kWh/h")
        print(f"  75%:    {np.percentile(consumption, 75):.2f} kWh/h")
        print(f"  Max:    {consumption.max():.2f} kWh/h")
        print(f"  Mean:   {consumption.mean():.2f} kWh/h")
        
        # Daily total estimate
        daily_total = consumption.mean() * 24
        print(f"\n  Estimated daily consumption: {daily_total:.1f} kWh/day")
        
        # Battery coverage
        battery_capacity = 40.0  # kWh
        hours_coverage = battery_capacity / consumption.mean()
        print(f"  Battery coverage (40 kWh): {hours_coverage:.1f} hours at average consumption")

# 2.3 Battery Analysis
if battery_df is not None:
    print("\n2.3 BATTERY ANALYSIS")
    print("-"*50)
    
    if 'battery_soc_percent' in battery_df.columns:
        soc = battery_df['battery_soc_percent'].values
        
        print(f"Battery SoC statistics:")
        print(f"  Min:    {soc.min():.1f}%")
        print(f"  25%:    {np.percentile(soc, 25):.1f}%")
        print(f"  Median: {np.percentile(soc, 50):.1f}%")
        print(f"  75%:    {np.percentile(soc, 75):.1f}%")
        print(f"  Max:    {soc.max():.1f}%")
        
        # Check sell viability
        min_soc_for_sell = 26.0  # 20% * 1.3
        soc_above_threshold = (soc > min_soc_for_sell).sum()
        print(f"\n  SoC above sell threshold ({min_soc_for_sell:.0f}%): {soc_above_threshold:,} ({soc_above_threshold/len(soc)*100:.1f}%)")
        
        if soc_above_threshold < len(soc) * 0.5:
            print(f"  ⚠️  Battery often too low for selling")
            print(f"     Recommendation: Initialize battery at 50-70% SoC")

# ============================================================================
# SECTION 3: VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("SECTION 3: GENERATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(20, 12))

# 3.1 Pricing Distribution
if pricing_df is not None:
    print("\n3.1 Creating pricing visualizations...")
    
    # Subplot 1: Price over time
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(pricing_df['timestamp'], pricing_df['price_per_kwh'] * 1000, linewidth=0.5, alpha=0.7)
    plt.axhline(y=20, color='g', linestyle='--', label='Buy threshold ($20/MWh)', alpha=0.7)
    plt.axhline(y=40, color='r', linestyle='--', label='Sell threshold ($40/MWh)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Price ($/MWh)')
    plt.title('Electricity Prices Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Price distribution histogram
    ax2 = plt.subplot(3, 3, 2)
    plt.hist(pricing_df['price_per_kwh'] * 1000, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=20, color='g', linestyle='--', label='Buy threshold', alpha=0.7)
    plt.axvline(x=40, color='r', linestyle='--', label='Sell threshold', alpha=0.7)
    plt.xlabel('Price ($/MWh)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Price boxplot
    ax3 = plt.subplot(3, 3, 3)
    plt.boxplot(pricing_df['price_per_kwh'] * 1000, vert=True)
    plt.axhline(y=20, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=40, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('Price ($/MWh)')
    plt.title('Price Box Plot')
    plt.grid(True, alpha=0.3)

# 3.2 Consumption Patterns
if consumption_df is not None and 'hourly_consumption_kwh' in consumption_df.columns:
    print("3.2 Creating consumption visualizations...")
    
    # Subplot 4: Consumption over time
    ax4 = plt.subplot(3, 3, 4)
    for house in consumption_df['House'].unique()[:3]:  # First 3 houses
        house_data = consumption_df[consumption_df['House'] == house]
        plt.plot(house_data['timestamp'], house_data['hourly_consumption_kwh'], 
                label=f'House {house}', alpha=0.7, linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('Consumption (kWh/h)')
    plt.title('Hourly Consumption by House')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Consumption distribution
    ax5 = plt.subplot(3, 3, 5)
    plt.hist(consumption_df['hourly_consumption_kwh'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Consumption (kWh/h)')
    plt.ylabel('Frequency')
    plt.title('Consumption Distribution')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Appliance breakdown (average)
    ax6 = plt.subplot(3, 3, 6)
    appliance_cols = [col for col in consumption_df.columns if col.startswith('appliance_')]
    if appliance_cols:
        appliance_means = consumption_df[appliance_cols].mean()
        appliance_means = appliance_means.sort_values(ascending=False)
        
        plt.barh(range(len(appliance_means)), appliance_means.values)
        plt.yticks(range(len(appliance_means)), 
                  [col.replace('appliance_', '').replace('_', ' ').title() for col in appliance_means.index])
        plt.xlabel('Average Consumption (kWh/h)')
        plt.title('Average Consumption by Appliance')
        plt.grid(True, alpha=0.3, axis='x')

# 3.3 Battery State
if battery_df is not None:
    print("3.3 Creating battery visualizations...")
    
    # Subplot 7: Battery SoC over time
    ax7 = plt.subplot(3, 3, 7)
    plt.plot(battery_df['timestamp'], battery_df['battery_soc_percent'], linewidth=0.8, alpha=0.7)
    plt.axhline(y=20, color='r', linestyle='--', label='Min SoC (20%)', alpha=0.5)
    plt.axhline(y=26, color='orange', linestyle='--', label='Sell threshold (26%)', alpha=0.5)
    plt.axhline(y=100, color='g', linestyle='--', label='Max SoC (100%)', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('State of Charge (%)')
    plt.title('Battery SoC Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 8: Trading actions
    if 'action' in battery_df.columns:
        ax8 = plt.subplot(3, 3, 8)
        action_counts = battery_df['action'].value_counts()
        colors = {'buy': 'green', 'hold': 'gray', 'sell': 'red'}
        plt.bar(range(len(action_counts)), action_counts.values, 
               color=[colors.get(a.lower(), 'blue') for a in action_counts.index])
        plt.xticks(range(len(action_counts)), action_counts.index)
        plt.ylabel('Count')
        plt.title('Trading Action Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add percentages
        for i, (action, count) in enumerate(action_counts.items()):
            plt.text(i, count, f'{count/len(battery_df)*100:.1f}%', 
                    ha='center', va='bottom')
    
    # Subplot 9: Price vs SoC (trading opportunities)
    if pricing_df is not None and len(battery_df) > 0:
        ax9 = plt.subplot(3, 3, 9)
        
        # Merge data on timestamp
        merged = pd.merge_asof(
            battery_df.sort_values('timestamp'),
            pricing_df[['timestamp', 'price_per_kwh']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        if len(merged) > 0 and 'price_per_kwh' in merged.columns:
            scatter = plt.scatter(merged['battery_soc_percent'], 
                                 merged['price_per_kwh'] * 1000,
                                 c=merged['timestamp'].astype(int) / 10**9,
                                 cmap='viridis', alpha=0.5, s=10)
            
            # Add trading zones
            plt.axhline(y=20, color='g', linestyle='--', label='Buy threshold', alpha=0.5)
            plt.axhline(y=40, color='r', linestyle='--', label='Sell threshold', alpha=0.5)
            plt.axvline(x=26, color='orange', linestyle='--', label='Min SoC for sell', alpha=0.5)
            
            plt.xlabel('Battery SoC (%)')
            plt.ylabel('Price ($/MWh)')
            plt.title('Trading Opportunities: Price vs SoC')
            plt.legend()
            plt.colorbar(scatter, label='Time (Unix timestamp)')
            plt.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'data_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# ============================================================================
# SECTION 4: RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("SECTION 4: RECOMMENDATIONS FOR FIXING TRADING")
print("="*70)

if pricing_df is not None:
    prices_mwh = pricing_df['price_per_kwh'].values * 1000
    above_sell = (prices_mwh > 40).sum()
    
    print("\n📊 DIAGNOSIS:")
    print("-"*50)
    
    if above_sell == 0:
        print("❌ ROOT CAUSE: NO PRICES ABOVE SELL THRESHOLD")
        print(f"   - Current sell threshold: $40/MWh")
        print(f"   - Maximum price in data: ${prices_mwh.max():.2f}/MWh")
        print(f"   - Gap: ${40 - prices_mwh.max():.2f}/MWh")
        
        suggested_threshold = np.percentile(prices_mwh, 70)
        print(f"\n✅ RECOMMENDATION 1: Lower sell threshold")
        print(f"   Change: sell_threshold_mwh = {suggested_threshold:.0f}  # Was 40.0")
        print(f"   This will trigger sells in ~30% of cases")
        
        print(f"\n✅ RECOMMENDATION 2: Use opportunistic selling")
        print(f"   The opportunistic_sell_percentile should trigger at 55th percentile")
        print(f"   Target price: ${np.percentile(prices_mwh, 55):.2f}/MWh")
    
    elif above_sell < len(prices_mwh) * 0.1:
        print("⚠️  ISSUE: VERY FEW PRICES ABOVE SELL THRESHOLD")
        print(f"   - Only {above_sell/len(prices_mwh)*100:.1f}% of prices trigger hard sell")
        print(f"\n✅ RECOMMENDATION: Rely on opportunistic selling")

if battery_df is not None and 'battery_soc_percent' in battery_df.columns:
    soc = battery_df['battery_soc_percent'].values
    soc_above_26 = (soc > 26).sum()
    
    if soc_above_26 < len(soc) * 0.5:
        print(f"\n⚠️  ISSUE: BATTERY SOC TOO LOW")
        print(f"   - Only {soc_above_26/len(soc)*100:.0f}% of time SoC > 26%")
        print(f"   - Median SoC: {np.percentile(soc, 50):.1f}%")
        print(f"\n✅ RECOMMENDATION 3: Initialize battery at higher SoC")
        print(f"   Change battery_state in training:")
        print(f"   'current_charge_kwh': 25.0,  # 62.5% of 40 kWh (was lower)")

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE")
print("="*70)
print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Visualization saved to: {os.path.abspath(output_path)}")
print(f"\nOpen the image to see all data visualizations!")
print("="*70)

plt.show()
