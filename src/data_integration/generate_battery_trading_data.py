"""
Generate Battery Trading Data for Trading Transformer Training.

Uses existing consumption and pricing data from Supabase to generate
intelligent battery trading decisions that follow business rules:
- Buy when price < $20/MWh
- Sell when price > $40/MWh  
- Hold when SoC < 20% or price < $27/MWh
- Maximize profit while meeting household demand

This creates training labels for the Trading Transformer.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import load_consumption_data, load_pricing_data
from src.models.trading_optimizer import calculate_optimal_trading_decisions


def generate_battery_trading_for_house(
    consumption_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    house_id: int,
    battery_capacity_kwh: float = 40.0
) -> pd.DataFrame:
    """
    Generate intelligent battery trading data for one house.
    
    Args:
        consumption_df: Consumption data for this house
        pricing_df: Market pricing data
        house_id: House ID
        battery_capacity_kwh: Battery capacity (40 or 80 kWh)
    
    Returns:
        DataFrame with battery states and trading decisions
    """
    print(f"\n  Generating battery trading data for House {house_id} ({battery_capacity_kwh} kWh)...")
    
    # Filter to this house
    house_data = consumption_df[consumption_df['house_id'] == house_id].copy()
    house_data = house_data.sort_values('timestamp').reset_index(drop=True)
    
    if len(house_data) == 0:
        print(f"    ⚠ No consumption data for house {house_id}")
        return pd.DataFrame()
    
    # Calculate ACTUAL hourly consumption from appliance columns
    # Note: total_consumption_kwh shows daily total, not hourly!
    appliance_cols = [col for col in house_data.columns if col.startswith('appliance_')]
    house_data['hourly_consumption_kwh'] = house_data[appliance_cols].sum(axis=1)
    
    # Merge with pricing - use hour/dayofweek pattern matching since dates don't align
    house_data['hour'] = house_data['timestamp'].dt.hour
    house_data['dayofweek'] = house_data['timestamp'].dt.dayofweek
    
    pricing_df['hour'] = pricing_df['timestamp'].dt.hour
    pricing_df['dayofweek'] = pricing_df['timestamp'].dt.dayofweek
    
    # Get average price per hour/day pattern
    price_patterns = pricing_df.groupby(['hour', 'dayofweek'])['price_per_kwh'].mean().reset_index()
    
    house_data = house_data.merge(
        price_patterns,
        on=['hour', 'dayofweek'],
        how='left'
    )
    
    # Fill any missing prices with median
    house_data['price_per_kwh'] = house_data['price_per_kwh'].fillna(
        pricing_df['price_per_kwh'].median()
    )
    
    # Extract arrays for optimizer
    consumption = house_data['hourly_consumption_kwh'].values
    prices = house_data['price_per_kwh'].values
    
    print(f"    Processing {len(consumption):,} hourly intervals...")
    print(f"    Hourly consumption range: {consumption.min():.2f} to {consumption.max():.2f} kWh")
    print(f"    Average consumption: {consumption.mean():.2f} kWh/hour")
    print(f"    Price range: ${prices.min():.4f} to ${prices.max():.4f} per kWh")
    print(f"    Price range: ${prices.min()*1000:.1f} to ${prices.max()*1000:.1f} per MWh")
    
    # Battery configuration
    battery_state = {
        'current_charge_kwh': battery_capacity_kwh * 0.70,  # Start at 70% for active trading
        'capacity_kwh': battery_capacity_kwh,
        'min_soc': 0.20,   # 20% minimum
        'max_soc': 1.0,    # 100% max per business rules
        'max_charge_rate_kw': 10.0,
        'max_discharge_rate_kw': 8.0,
        'efficiency': 0.95
    }
    
    # Calculate optimal trading decisions using business rules
    result = calculate_optimal_trading_decisions(
        predicted_consumption=consumption,
        actual_prices=prices,
        battery_state=battery_state,
        household_price_kwh=0.27,
        buy_threshold_mwh=20.0,
        sell_threshold_mwh=40.0,
        min_grid_price_pct=0.10,
        opportunistic_window=48,  # Look back 48 hours (2 days) for price patterns  
        opportunistic_buy_percentile=80.0,  # ULTRA AGGRESSIVE: Buy in bottom 80% of prices
        opportunistic_sell_percentile=20.0  # ULTRA AGGRESSIVE: Sell in top 80% of prices
    )
    
    # Create battery records
    battery_records = []
    current_charge = battery_state['current_charge_kwh']
    soh = 98.0  # State of health
    
    for i in range(len(house_data)):
        decision = result['optimal_decisions'][i]
        quantity = result['optimal_quantities'][i]
        
        # Map decision to action
        action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        action = action_map[decision]
        
        # Update charge based on decision
        if decision == 0:  # Buy
            current_charge += quantity * battery_state['efficiency']
        elif decision == 2:  # Sell
            current_charge -= quantity
        
        # Ensure within bounds
        current_soc = current_charge / battery_capacity_kwh
        current_soc = np.clip(current_soc, battery_state['min_soc'], battery_state['max_soc'])
        current_charge = battery_capacity_kwh * current_soc
        
        available_for_discharge = max(0, current_charge - (battery_capacity_kwh * battery_state['min_soc']))
        available_for_charge = max(0, (battery_capacity_kwh * battery_state['max_soc']) - current_charge)
        
        battery_records.append({
            'timestamp': house_data.iloc[i]['timestamp'],
            'house_id': house_id,
            'battery_soc_percent': float(current_soc * 100),
            'battery_charge_kwh': float(current_charge),
            'battery_available_kwh': float(available_for_discharge),
            'battery_capacity_remaining_kwh': float(available_for_charge),
            'battery_soh_percent': float(soh),
            'battery_count': 2 if battery_capacity_kwh >= 80 else 1,
            'total_capacity_kwh': float(battery_capacity_kwh),
            'max_charge_rate_kw': 10.0,
            'max_discharge_rate_kw': 8.0,
            'action': action,
            'trade_amount_kwh': float(quantity),
            'price_per_kwh': float(house_data.iloc[i]['price_per_kwh']),
            'consumption_kwh': float(consumption[i])
        })
    
    df = pd.DataFrame(battery_records)
    
    # Print statistics
    buy_count = len(df[df['action'] == 'buy'])
    sell_count = len(df[df['action'] == 'sell'])
    hold_count = len(df[df['action'] == 'hold'])
    
    print(f"    ✓ Generated {len(df):,} battery states")
    print(f"    ✓ Trading actions:")
    print(f"      - Buy: {buy_count} ({buy_count/len(df)*100:.1f}%)")
    print(f"      - Hold: {hold_count} ({hold_count/len(df)*100:.1f}%)")
    print(f"      - Sell: {sell_count} ({sell_count/len(df)*100:.1f}%)")
    print(f"    ✓ SoC range: {df['battery_soc_percent'].min():.1f}% to {df['battery_soc_percent'].max():.1f}%")
    print(f"    ✓ Expected profit: ${result['expected_profit']:.2f}")
    print(f"    ✓ Household revenue: ${result['household_revenue']:.2f}")
    print(f"    ✓ Market profit: ${result['market_profit']:.2f}")
    
    # Verify business rules
    if buy_count > 0:
        buy_prices = df[df['action'] == 'buy']['price_per_kwh'] * 1000
        print(f"    ✓ Avg buy price: ${buy_prices.mean():.1f}/MWh (should be < $20)")
    
    if sell_count > 0:
        sell_prices = df[df['action'] == 'sell']['price_per_kwh'] * 1000
        print(f"    ✓ Avg sell price: ${sell_prices.mean():.1f}/MWh (should be > $40)")
    
    return df


def upload_battery_data(
    battery_df: pd.DataFrame,
    house_id: int,
    connector: SupabaseConnector,
    batch_size: int = 500
) -> bool:
    """
    Upload battery data to Supabase.
    
    Args:
        battery_df: Battery DataFrame
        house_id: House ID
        connector: Supabase connector
        batch_size: Records per batch
    
    Returns:
        True if successful
    """
    table_name = f"house{house_id}_battery"
    
    print(f"\n  Uploading to {table_name}...")
    
    try:
        # Clean data
        battery_df = battery_df.replace([np.inf, -np.inf], 0)
        battery_df = battery_df.fillna(0)
        
        records = battery_df.to_dict('records')
        
        # Convert timestamps and ensure proper types
        for record in records:
            if isinstance(record['timestamp'], pd.Timestamp):
                record['timestamp'] = record['timestamp'].isoformat()
            
            # Ensure float fields
            for key in ['battery_soc_percent', 'battery_charge_kwh', 'battery_available_kwh',
                       'battery_capacity_remaining_kwh', 'battery_soh_percent', 
                       'total_capacity_kwh', 'max_charge_rate_kw', 'max_discharge_rate_kw',
                       'trade_amount_kwh', 'price_per_kwh', 'consumption_kwh']:
                if key in record:
                    record[key] = float(record[key])
            
            # Ensure int fields
            for key in ['house_id', 'battery_count']:
                if key in record:
                    record[key] = int(record[key])
        
        # Upload in batches
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            connector.client.table(table_name).insert(batch).execute()
            total += len(batch)
            print(f"    Uploaded {total}/{len(records)} records...", end='\r')
        
        print(f"\n    ✓ Successfully uploaded {total:,} records to {table_name}")
        return True
        
    except Exception as e:
        print(f"\n    ⚠ Error uploading to {table_name}: {e}")
        return False


def main():
    """Main execution."""
    print("="*70)
    print("BATTERY TRADING DATA GENERATOR")
    print("="*70)
    print("\nGenerates intelligent battery trading decisions from existing")
    print("consumption and pricing data to train the Trading Transformer.")
    
    # Connect to Supabase
    print("\n1. Connecting to Supabase...")
    connector = SupabaseConnector()
    print("   ✓ Connected")
    
    # Load existing data
    print("\n2. Loading existing data from Supabase...")
    print("   Loading consumption data...")
    consumption_df = load_consumption_data(connector=connector)
    print(f"   ✓ Loaded {len(consumption_df):,} consumption records")
    
    print("   Loading pricing data...")
    pricing_df = load_pricing_data(connector=connector)
    print(f"   ✓ Loaded {len(pricing_df):,} pricing records")
    
    # Get unique houses
    houses = sorted(consumption_df['house_id'].unique())
    print(f"\n   Found data for {len(houses)} houses: {houses}")
    
    # Ask which houses to generate for
    print(f"\n3. Select houses to generate battery data for:")
    print(f"   Available houses: {houses}")
    
    house_input = input("   Enter house IDs (comma-separated, or 'all'): ").strip()
    
    if house_input.lower() == 'all':
        selected_houses = houses
    else:
        selected_houses = [int(h.strip()) for h in house_input.split(',') if h.strip()]
    
    print(f"   Selected: {selected_houses}")
    
    # Ask for battery capacity
    print(f"\n4. Battery capacity configuration:")
    print(f"   Houses 1-9: 40 kWh (standard)")
    print(f"   Houses 10-11: 80 kWh (double capacity)")
    use_custom = input("   Use custom capacities? (y/n): ").strip().lower()
    
    capacities = {}
    if use_custom == 'y':
        for house_id in selected_houses:
            cap = input(f"   Capacity for House {house_id} (kWh, default=40): ").strip()
            capacities[house_id] = float(cap) if cap else 40.0
    else:
        for house_id in selected_houses:
            capacities[house_id] = 80.0 if house_id >= 10 else 40.0
    
    # Generate battery data
    print("\n" + "="*70)
    print("GENERATING BATTERY TRADING DATA")
    print("="*70)
    
    battery_dfs = {}
    for house_id in selected_houses:
        battery_df = generate_battery_trading_for_house(
            consumption_df,
            pricing_df,
            house_id,
            capacities[house_id]
        )
        
        if len(battery_df) > 0:
            battery_dfs[house_id] = battery_df
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    total_records = sum(len(df) for df in battery_dfs.values())
    print(f"\nGenerated {total_records:,} battery trading records for {len(battery_dfs)} houses")
    
    for house_id, df in battery_dfs.items():
        buy_pct = len(df[df['action'] == 'buy']) / len(df) * 100
        sell_pct = len(df[df['action'] == 'sell']) / len(df) * 100
        print(f"  House {house_id}: {len(df):,} records (Buy: {buy_pct:.1f}%, Sell: {sell_pct:.1f}%)")
    
    # Ask to upload
    upload = input("\n  Upload to Supabase? (y/n): ").strip().lower()
    if upload != 'y':
        print("  Skipping upload.")
        return
    
    # Upload
    print("\n" + "="*70)
    print("UPLOADING TO SUPABASE")
    print("="*70)
    
    success_count = 0
    for house_id, battery_df in battery_dfs.items():
        if upload_battery_data(battery_df, house_id, connector):
            success_count += 1
    
    print("\n" + "="*70)
    print("✅ BATTERY TRADING DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nSuccessfully uploaded data for {success_count}/{len(battery_dfs)} houses")
    print("\nThis data can now be used to train the Trading Transformer!")


if __name__ == "__main__":
    main()
