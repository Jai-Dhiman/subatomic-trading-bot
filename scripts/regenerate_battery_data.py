"""
Regenerate Battery Trading Data with Corrected Rules.

This script:
1. Loads existing consumption and pricing data from Supabase
2. Clears old battery data (generated with incorrect $270/MWh threshold)
3. Regenerates battery data using corrected optimizer ($27/MWh threshold)
4. Uploads new battery data to Supabase

This gives us maximum training data using the real pricing and consumption records.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import load_consumption_data, load_pricing_data
from src.models.trading_optimizer import calculate_optimal_trading_decisions


def clear_battery_tables(connector: SupabaseConnector, house_ids: list):
    """Clear existing battery data from Supabase."""
    print("\n" + "="*70)
    print("CLEARING OLD BATTERY DATA")
    print("="*70)
    
    for house_id in house_ids:
        table_name = f"house{house_id}_battery"
        print(f"\nClearing {table_name}...")
        try:
            # Delete all records
            connector.client.table(table_name).delete().neq('house_id', 0).execute()
            print(f"  ✓ Cleared {table_name}")
        except Exception as e:
            print(f"  ⚠ Error clearing {table_name}: {e}")


def generate_battery_data_for_house(
    house_id: int,
    consumption_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    battery_capacity_kwh: float = 40.0
) -> pd.DataFrame:
    """
    Generate battery trading data for a single house using corrected optimizer.
    
    Args:
        house_id: House ID
        consumption_df: Consumption data (hourly)
        pricing_df: Pricing data (30-min intervals)
        battery_capacity_kwh: Battery capacity in kWh
    
    Returns:
        DataFrame with battery state data
    """
    print(f"\nGenerating battery data for House {house_id} ({battery_capacity_kwh} kWh)...")
    
    # Filter consumption for this house
    house_data = consumption_df[consumption_df['house_id'] == house_id].copy()
    house_data = house_data.sort_values('timestamp').reset_index(drop=True)
    
    if len(house_data) == 0:
        print(f"  ⚠ No consumption data for house {house_id}")
        return pd.DataFrame()
    
    # Match pricing to consumption by hour/dayofweek pattern
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
    
    # Calculate hourly consumption from appliances
    appliance_cols = [col for col in house_data.columns if col.startswith('appliance_')]
    if appliance_cols:
        house_data['hourly_consumption_kwh'] = house_data[appliance_cols].sum(axis=1)
    else:
        # Fallback: use daily total / 24
        house_data['hourly_consumption_kwh'] = house_data['total_consumption_kwh'] / 24.0
    
    # Extract arrays for optimizer
    # Consumption is HOURLY, so divide by 2 for 30-min intervals
    consumption = house_data['hourly_consumption_kwh'].values / 2.0
    prices = house_data['price_per_kwh'].values
    
    print(f"  Processing {len(consumption):,} intervals...")
    print(f"  Consumption range: {consumption.min():.2f} to {consumption.max():.2f} kWh per 30-min")
    print(f"  Price range: ${prices.min()*1000:.1f} to ${prices.max()*1000:.1f} per MWh")
    
    # Battery configuration
    battery_state = {
        'current_charge_kwh': battery_capacity_kwh * 0.50,  # Start at 50%
        'capacity_kwh': battery_capacity_kwh,
        'min_soc': 0.20,
        'max_soc': 1.0,
        'max_charge_rate_kw': 10.0,
        'max_discharge_rate_kw': 8.0,
        'efficiency': 0.95
    }
    
    # Calculate optimal trading decisions with CORRECTED rules
    result = calculate_optimal_trading_decisions(
        predicted_consumption=consumption,
        actual_prices=prices,
        battery_state=battery_state,
        household_price_kwh=0.027  # CORRECTED: $27/MWh, not $270/MWh!
    )
    
    # Create battery records
    battery_records = []
    soh = 98.0  # State of health
    
    for i in range(len(house_data)):
        decision = result['optimal_decisions'][i]
        quantity = result['optimal_quantities'][i]
        
        # Map decision to action
        action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        action = action_map[decision]
        
        # Get SoC from optimizer's trajectory
        current_soc = result['battery_trajectory'][i] / 100.0
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
    
    print(f"  ✓ Generated {len(df):,} battery states")
    print(f"  ✓ Actions: Buy={buy_count} ({buy_count/len(df)*100:.1f}%), " +
          f"Hold={hold_count} ({hold_count/len(df)*100:.1f}%), " +
          f"Sell={sell_count} ({sell_count/len(df)*100:.1f}%)")
    print(f"  ✓ SoC range: {df['battery_soc_percent'].min():.1f}% to {df['battery_soc_percent'].max():.1f}%")
    print(f"  ✓ Market profit: ${result['market_profit']:.2f}")
    
    # Verify business rules
    if buy_count > 0:
        buy_prices = df[df['action'] == 'buy']['price_per_kwh'] * 1000
        print(f"  ✓ Avg buy price: ${buy_prices.mean():.1f}/MWh (should be < $27)")
    
    if sell_count > 0:
        sell_prices = df[df['action'] == 'sell']['price_per_kwh'] * 1000
        print(f"  ✓ Avg sell price: ${sell_prices.mean():.1f}/MWh (should be > $40)")
    
    return df


def upload_battery_data(
    battery_df: pd.DataFrame,
    house_id: int,
    connector: SupabaseConnector,
    batch_size: int = 500
):
    """Upload battery data to Supabase."""
    table_name = f"house{house_id}_battery"
    print(f"\n  Uploading to {table_name}...")
    
    try:
        # Clean data
        battery_df = battery_df.replace([np.inf, -np.inf], 0)
        battery_df = battery_df.fillna(0)
        
        records = battery_df.to_dict('records')
        for record in records:
            if isinstance(record['timestamp'], pd.Timestamp):
                record['timestamp'] = record['timestamp'].isoformat()
        
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            connector.client.table(table_name).insert(batch).execute()
            total += len(batch)
            print(f"    Uploaded {total}/{len(records)} records...", end='\r')
        
        print(f"\n    ✓ Uploaded {total:,} records to {table_name}")
        return True
    except Exception as e:
        print(f"\n    ⚠ Error uploading to {table_name}: {e}")
        return False


def main():
    """Main execution."""
    print("="*70)
    print("REGENERATE BATTERY DATA WITH CORRECTED OPTIMIZER")
    print("="*70)
    print("\nThis will:")
    print("  1. Load existing consumption and pricing data")
    print("  2. Clear old battery data (incorrect $270/MWh threshold)")
    print("  3. Regenerate battery data (correct $27/MWh threshold)")
    print("  4. Upload new data to Supabase")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return
    
    # Load existing data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    consumption_df = load_consumption_data(source='supabase')
    pricing_df = load_pricing_data()
    
    # Get unique houses
    house_ids = sorted(consumption_df['house_id'].unique())
    print(f"\nFound {len(house_ids)} houses: {house_ids}")
    
    # Connect to Supabase
    print("\nConnecting to Supabase...")
    connector = SupabaseConnector()
    print("  ✓ Connected")
    
    # Clear old data
    clear_battery_tables(connector, house_ids)
    
    # Generate new battery data
    print("\n" + "="*70)
    print("GENERATING BATTERY DATA")
    print("="*70)
    
    battery_dfs = {}
    for house_id in house_ids:
        # Houses 1-9: 40 kWh, Houses 10+: 80 kWh
        capacity = 80.0 if house_id >= 10 else 40.0
        
        battery_df = generate_battery_data_for_house(
            house_id, consumption_df, pricing_df, capacity
        )
        
        if len(battery_df) > 0:
            battery_dfs[house_id] = battery_df
    
    # Upload to Supabase
    print("\n" + "="*70)
    print("UPLOADING TO SUPABASE")
    print("="*70)
    
    success_count = 0
    for house_id, battery_df in battery_dfs.items():
        if upload_battery_data(battery_df, house_id, connector):
            success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nSuccessfully regenerated battery data for {success_count}/{len(battery_dfs)} houses")
    print(f"Total records generated: {sum(len(df) for df in battery_dfs.values()):,}")
    
    # Show sample statistics
    if battery_dfs:
        all_battery_data = pd.concat(battery_dfs.values())
        
        buy_count = len(all_battery_data[all_battery_data['action'] == 'buy'])
        sell_count = len(all_battery_data[all_battery_data['action'] == 'sell'])
        hold_count = len(all_battery_data[all_battery_data['action'] == 'hold'])
        
        print(f"\nOverall Statistics:")
        print(f"  Buy decisions: {buy_count} ({buy_count/len(all_battery_data)*100:.1f}%)")
        print(f"  Hold decisions: {hold_count} ({hold_count/len(all_battery_data)*100:.1f}%)")
        print(f"  Sell decisions: {sell_count} ({sell_count/len(all_battery_data)*100:.1f}%)")
        
        if buy_count > 0:
            avg_buy = (all_battery_data[all_battery_data['action'] == 'buy']['price_per_kwh'] * 1000).mean()
            print(f"  Avg buy price: ${avg_buy:.2f}/MWh")
        
        if sell_count > 0:
            avg_sell = (all_battery_data[all_battery_data['action'] == 'sell']['price_per_kwh'] * 1000).mean()
            print(f"  Avg sell price: ${avg_sell:.2f}/MWh")
    
    print("\n✅ Ready for training with corrected data!")


if __name__ == "__main__":
    main()
