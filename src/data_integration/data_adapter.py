"""
Data Adapter for Dual-Transformer System.

Loads real data from Supabase and generates synthetic battery data:
- Consumption: august11homeconsumption table (8,184 records, 11 houses, 9 appliances)
- Pricing: cabuyingpricehistoryseptember2025 table (40,200 records, CA market)
- Battery: Synthetic Subatomic Battery data (40/80 kWh capacity)
"""

import pandas as pd
import numpy as np
import json
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_integration.supabase_connector import SupabaseConnector


def load_consumption_data(
    house_id: Optional[int] = None,
    source: str = 'supabase',
    connector: Optional[SupabaseConnector] = None
) -> pd.DataFrame:
    """
    Load consumption data from Supabase.
    
    Args:
        house_id: Specific house (1-11) or None for all houses
        source: 'supabase' (loads from august11homeconsumption)
        connector: Optional existing SupabaseConnector
        
    Returns:
        DataFrame with columns:
        - timestamp
        - house_id
        - total_consumption_kwh
        - appliance_ac
        - appliance_washing_drying
        - appliance_fridge
        - appliance_ev_charging
        - appliance_dishwasher
        - appliance_computers
        - appliance_stove
        - appliance_water_heater
        - appliance_misc
    """
    if connector is None:
        connector = SupabaseConnector()
    
    print(f"Loading consumption data from Supabase...")
    
    # Load from august11homeconsumption table
    response = connector.client.table('august11homeconsumption').select('*').execute()
    
    if not response.data:
        raise ValueError("No consumption data found in august11homeconsumption table")
    
    df = pd.DataFrame(response.data)
    print(f"  ✓ Loaded {len(df):,} records")
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'House': 'house_id',
        'Timestamp': 'timestamp',
        'Total kWh': 'total_consumption_kwh'
    })
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Parse appliance JSON and extract features
    appliance_data = []
    for idx, row in df.iterrows():
        try:
            appliances = json.loads(row['Appliance_Breakdown_JSON'])
            appliance_data.append({
                'appliance_ac': appliances.get('A/C', 0),
                'appliance_washing_drying': appliances.get('Washing/Drying', 0),
                'appliance_fridge': appliances.get('Refrig.', 0),
                'appliance_ev_charging': appliances.get('EV Charging', 0),
                'appliance_dishwasher': appliances.get('DishWasher', 0),
                'appliance_computers': appliances.get('Computers', 0),
                'appliance_stove': appliances.get('Stovetop', 0),
                'appliance_water_heater': appliances.get('Water Heater', 0),
                'appliance_misc': appliances.get('Standby/ Misc.', 0)
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠ Warning: Could not parse appliances for row {idx}: {e}")
            appliance_data.append({
                'appliance_ac': 0, 'appliance_washing_drying': 0, 'appliance_fridge': 0,
                'appliance_ev_charging': 0, 'appliance_dishwasher': 0, 'appliance_computers': 0,
                'appliance_stove': 0, 'appliance_water_heater': 0, 'appliance_misc': 0
            })
    
    # Add appliance columns
    appliance_df = pd.DataFrame(appliance_data)
    df = pd.concat([df.drop('Appliance_Breakdown_JSON', axis=1), appliance_df], axis=1)
    
    # Filter by house_id if specified
    if house_id is not None:
        df = df[df['house_id'] == house_id].copy()
        print(f"  ✓ Filtered to house {house_id}: {len(df):,} records")
    
    # Sort by timestamp
    df = df.sort_values(['house_id', 'timestamp']).reset_index(drop=True)
    
    print(f"  ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  ✓ Houses: {sorted(df['house_id'].unique().tolist())}")
    
    return df


def load_pricing_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    connector: Optional[SupabaseConnector] = None,
    interpolate_to_30min: bool = True
) -> pd.DataFrame:
    """
    Load pricing data from Supabase.
    
    IMPORTANT: Data is filtered for LMP_TYPE='LMP' only in SupabaseConnector.get_pricing_data().
    Pricing data comes in 1-hour intervals and is converted from $/MWh to $/kWh.
    
    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        connector: Optional existing SupabaseConnector
        interpolate_to_30min: If True, interpolate hourly prices to 30-min intervals (default: True)
        
    Returns:
        DataFrame with columns:
        - timestamp
        - price_per_kwh
    """
    if connector is None:
        connector = SupabaseConnector()
    
    print(f"Loading pricing data from Supabase...")
    
    # Load from cabuyingpricehistoryseptember2025 table (already filtered for LMP_TYPE='LMP')
    df = connector.get_pricing_data(start_date, end_date)
    
    if df.empty:
        raise ValueError("No pricing data found in cabuyingpricehistoryseptember2025 table")
    
    print(f"  ✓ Loaded {len(df):,} hourly pricing records (LMP_TYPE='LMP')")
    
    # Rename and select relevant columns
    df = df.rename(columns={
        'INTERVALSTARTTIME_GMT': 'timestamp',
        'Price KWH': 'price_per_kwh'
    })
    
    # Keep only needed columns
    df = df[['timestamp', 'price_per_kwh']].copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove any zero or null prices (data quality check)
    original_len = len(df)
    df = df[df['price_per_kwh'].notna()].copy()
    df = df[df['price_per_kwh'] != 0].copy()
    
    if len(df) < original_len:
        print(f"  ⚠ Removed {original_len - len(df)} records with null/zero prices")
    
    if df.empty:
        raise ValueError("No valid pricing data after filtering null/zero prices")
    
    # Interpolate to 30-min intervals if requested
    if interpolate_to_30min:
        print(f"  → Interpolating from 1-hour to 30-min intervals...")
        
        # Create 30-min timestamp range
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        new_timestamps = pd.date_range(start=start_time, end=end_time, freq='30T')
        
        # Create new dataframe with 30-min intervals
        df_30min = pd.DataFrame({'timestamp': new_timestamps})
        
        # Merge with original hourly data
        df_30min = df_30min.merge(df, on='timestamp', how='left')
        
        # Forward-fill prices (each hourly price applies to both 30-min intervals within that hour)
        df_30min['price_per_kwh'] = df_30min['price_per_kwh'].fillna(method='ffill')
        
        # Handle any remaining NaNs at the beginning
        df_30min['price_per_kwh'] = df_30min['price_per_kwh'].fillna(method='bfill')
        
        df = df_30min.copy()
        print(f"  ✓ Interpolated to {len(df):,} 30-min intervals")
    
    print(f"  ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  ✓ Price range: ${df['price_per_kwh'].min():.4f} to ${df['price_per_kwh'].max():.4f}")
    print(f"  ✓ Mean: ${df['price_per_kwh'].mean():.4f}, Median: ${df['price_per_kwh'].median():.4f}")
    
    return df


def generate_battery_data(
    timestamps: pd.DatetimeIndex,
    consumption_data: pd.DataFrame,
    pricing_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate synthetic Subatomic Battery state data for all households.
    
    Specifications:
    - Houses 1-9: 1 battery (40 kWh capacity)
    - Houses 10-11: 2 batteries (80 kWh capacity)
    - Max charge rate: 10 kW
    - Max discharge rate: 8 kW
    - Operating range: 20-90% SoC
    
    Args:
        timestamps: DatetimeIndex for all timestamps
        consumption_data: DataFrame with consumption data (for realistic simulation)
        pricing_data: Optional pricing data (for price-aware charging)
        
    Returns:
        DataFrame with battery state for each house at each timestamp
    """
    print(f"Generating synthetic battery data...")
    
    battery_records = []
    house_ids = consumption_data['house_id'].unique()
    
    for house_id in sorted(house_ids):
        # Initialize battery for this house
        if house_id <= 9:
            battery_count = 1
            total_capacity = 40.0
        else:
            battery_count = 2
            total_capacity = 80.0
        
        # Start at 50% SoC
        current_soc = 50.0
        current_charge = total_capacity * (current_soc / 100)
        soh = np.random.uniform(97.0, 100.0)
        
        # Get house-specific consumption
        house_consumption = consumption_data[
            consumption_data['house_id'] == house_id
        ].set_index('timestamp')['total_consumption_kwh']
        
        # Simulate battery behavior for each timestamp
        for timestamp in timestamps:
            hour = timestamp.hour
            
            # Get consumption for this hour (if available)
            if timestamp in house_consumption.index:
                consumption = house_consumption[timestamp]
            else:
                consumption = house_consumption.mean()  # Use average if not available
            
            # Get price for this hour (if available)
            if pricing_data is not None:
                price_row = pricing_data[pricing_data['timestamp'] == timestamp]
                price = price_row['price_per_kwh'].values[0] if len(price_row) > 0 else 0.30
            else:
                price = 0.30  # Default mid-range price
            
            # Simulate battery behavior
            # Night charging (11pm-6am): Charge if price < $0.25 and SoC < 80%
            if 23 <= hour or hour < 6:
                if price < 0.25 and current_soc < 80:
                    charge_amount = min(
                        10.0,  # Max charge rate
                        total_capacity * 0.90 - current_charge  # Don't exceed 90%
                    )
                    current_charge += charge_amount * 0.95  # 95% efficiency
            
            # Peak hours (4pm-9pm): Discharge if consumption > 2 kWh and SoC > 30%
            elif 16 <= hour <= 21:
                if consumption > 2.0 and current_soc > 30:
                    discharge_amount = min(
                        8.0,  # Max discharge rate
                        consumption * 0.5,  # Cover 50% of consumption
                        current_charge - (total_capacity * 0.20)  # Don't go below 20%
                    )
                    current_charge -= discharge_amount
            
            # Off-peak: Opportunistic charging
            else:
                if price < 0.30 and current_soc < 70:
                    charge_amount = min(5.0, total_capacity * 0.90 - current_charge)
                    current_charge += charge_amount * 0.95
            
            # Calculate SoC
            current_soc = (current_charge / total_capacity) * 100
            
            # Ensure SoC stays within bounds
            current_soc = np.clip(current_soc, 20.0, 90.0)
            current_charge = total_capacity * (current_soc / 100)
            
            # Calculate available capacity
            available_for_discharge = current_charge - (total_capacity * 0.20)
            available_for_charge = (total_capacity * 0.90) - current_charge
            
            # Record state
            battery_records.append({
                'timestamp': timestamp,
                'house_id': house_id,
                'battery_count': battery_count,
                'total_capacity_kwh': total_capacity,
                'battery_soc_percent': current_soc,
                'battery_charge_kwh': current_charge,
                'battery_available_kwh': max(0, available_for_discharge),
                'battery_capacity_remaining_kwh': max(0, available_for_charge),
                'battery_soh_percent': soh,
                'max_charge_rate_kw': 10.0,
                'max_discharge_rate_kw': 8.0
            })
    
    df = pd.DataFrame(battery_records)
    print(f"  ✓ Generated {len(df):,} battery state records")
    print(f"  ✓ Houses with 1 battery (40 kWh): {len([h for h in house_ids if h <= 9])}")
    print(f"  ✓ Houses with 2 batteries (80 kWh): {len([h for h in house_ids if h > 9])}")
    
    return df


def merge_all_data(
    consumption_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    battery_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge consumption, pricing, and battery data on timestamp and house_id.
    
    Handles:
    - Date alignment (consumption is Aug 2025, pricing is Oct 2024)
    - Missing value interpolation
    - Timezone handling
    
    Args:
        consumption_df: Consumption data from Supabase
        pricing_df: Pricing data from Supabase
        battery_df: Synthetic battery data
        
    Returns:
        Complete DataFrame ready for feature engineering
    """
    print(f"\nMerging all data sources...")
    
    # Date alignment challenge:
    # - Consumption: August 2025
    # - Pricing: October 2024
    # Solution: Use pricing patterns (hour-of-day, day-of-week) mapped to consumption dates
    
    # Extract time-of-day and day-of-week from pricing
    pricing_df = pricing_df.copy()
    pricing_df['hour'] = pricing_df['timestamp'].dt.hour
    pricing_df['dayofweek'] = pricing_df['timestamp'].dt.dayofweek
    
    # Calculate average price per hour and day-of-week
    price_patterns = pricing_df.groupby(['hour', 'dayofweek'])['price_per_kwh'].mean().reset_index()
    price_patterns.columns = ['hour', 'dayofweek', 'avg_price_per_kwh']
    
    # Add time features to consumption data
    consumption_df = consumption_df.copy()
    consumption_df['hour'] = consumption_df['timestamp'].dt.hour
    consumption_df['dayofweek'] = consumption_df['timestamp'].dt.dayofweek
    
    # Merge consumption with price patterns
    df = consumption_df.merge(
        price_patterns,
        on=['hour', 'dayofweek'],
        how='left'
    )
    
    # Rename to price_per_kwh
    df['price_per_kwh'] = df['avg_price_per_kwh']
    df = df.drop('avg_price_per_kwh', axis=1)
    
    # Fill any missing prices with median
    df['price_per_kwh'] = df['price_per_kwh'].fillna(df['price_per_kwh'].median())
    
    # Merge with battery data
    df = df.merge(
        battery_df,
        on=['timestamp', 'house_id'],
        how='left'
    )
    
    # Verify no missing values in critical columns
    critical_cols = [
        'total_consumption_kwh', 'price_per_kwh',
        'battery_soc_percent', 'battery_charge_kwh'
    ]
    
    missing_counts = df[critical_cols].isnull().sum()
    if missing_counts.any():
        print(f"  ⚠ Warning: Missing values detected:")
        for col in critical_cols:
            if missing_counts[col] > 0:
                print(f"    {col}: {missing_counts[col]} missing")
        
        # Fill with forward fill then backward fill
        df[critical_cols] = df[critical_cols].fillna(method='ffill').fillna(method='bfill')
    
    print(f"  ✓ Merged dataset: {len(df):,} records")
    print(f"  ✓ Columns: {len(df.columns)}")
    print(f"  ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  ✓ No missing values in critical columns")
    
    return df


if __name__ == "__main__":
    print("="*70)
    print("DATA ADAPTER TEST")
    print("="*70)
    
    # Test 1: Load consumption data
    print("\n1. Testing consumption data loading...")
    consumption_df = load_consumption_data()
    print(f"   Consumption shape: {consumption_df.shape}")
    print(f"   Columns: {consumption_df.columns.tolist()}")
    
    # Test 2: Load pricing data
    print("\n2. Testing pricing data loading...")
    pricing_df = load_pricing_data()
    print(f"   Pricing shape: {pricing_df.shape}")
    print(f"   Columns: {pricing_df.columns.tolist()}")
    
    # Test 3: Generate battery data
    print("\n3. Testing battery data generation...")
    timestamps = consumption_df['timestamp'].unique()
    battery_df = generate_battery_data(timestamps, consumption_df, pricing_df)
    print(f"   Battery shape: {battery_df.shape}")
    print(f"   Columns: {battery_df.columns.tolist()}")
    
    # Test 4: Merge all data
    print("\n4. Testing data merging...")
    df_complete = merge_all_data(consumption_df, pricing_df, battery_df)
    print(f"   Complete dataset shape: {df_complete.shape}")
    print(f"   Sample:")
    print(df_complete.head(3))
    
    print("\n" + "="*70)
    print("✅ DATA ADAPTER TEST COMPLETE")
    print("="*70)
