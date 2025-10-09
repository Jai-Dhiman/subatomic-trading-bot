#!/usr/bin/env python3
"""
Align house appliance data timestamps with pricing data for demo.

Shifts house1 and house2 data from August 2025 to October/November 2024
to align with available pricing data for backtesting demonstration.

Target date ranges:
- Input week: October 28 - November 4, 2024 (for model input)
- Prediction week: November 4-11, 2024 (for validation against actuals)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_integration.supabase_connector import SupabaseConnector


def fetch_house_data(connector: SupabaseConnector, house_num: int) -> pd.DataFrame:
    """Fetch appliance data for a house."""
    table_name = f'house{house_num}'
    print(f"Fetching data from {table_name}...")
    
    response = connector.client.table(table_name).select('*').execute()
    df = pd.DataFrame(response.data)
    
    # Parse timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Parse appliance JSON
    df['appliances'] = df['Appliance_Breakdown_JSON'].apply(json.loads)
    
    print(f"  Loaded {len(df)} records from {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    return df


def shift_timestamps(df: pd.DataFrame, target_start_date: datetime) -> pd.DataFrame:
    """
    Shift DataFrame timestamps to align with target date.
    
    Args:
        df: DataFrame with 'Timestamp' column
        target_start_date: Desired start date
        
    Returns:
        DataFrame with shifted timestamps
    """
    # Calculate offset
    original_start = df['Timestamp'].min()
    offset = target_start_date - original_start
    
    # Shift timestamps
    df = df.copy()
    df['Timestamp'] = df['Timestamp'] + offset
    
    print(f"  Shifted timestamps by {offset.days} days")
    print(f"  New range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    return df


def expand_to_30min_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand hourly data to 30-minute intervals through linear interpolation.
    
    Args:
        df: DataFrame with hourly data
        
    Returns:
        DataFrame with 30-minute intervals
    """
    print("  Expanding to 30-minute intervals...")
    
    # Create 30-minute intervals
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()
    
    # Generate 30-minute time range
    time_range = pd.date_range(start=start_time, end=end_time, freq='30min')
    
    # Create new DataFrame with 30-min intervals
    df_30min = pd.DataFrame({'Timestamp': time_range})
    
    # For each numeric column, interpolate
    df_hourly = df.set_index('Timestamp')
    
    # Resample to 30 minutes and interpolate
    df_resampled = df_hourly.resample('30min').interpolate(method='linear')
    
    # Merge back
    df_30min = df_30min.merge(
        df_resampled.reset_index(),
        on='Timestamp',
        how='left'
    )
    
    print(f"  Expanded from {len(df)} hourly to {len(df_30min)} 30-minute records")
    
    return df_30min


def parse_appliance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse appliance JSON into separate columns with standardized names.
    
    Maps from appliance display names to model feature names:
    - 'A/C' → 'ac'
    - 'Refrig.' → 'fridge'
    - 'Washing/Drying' → 'washing_machine'
    - etc.
    """
    print("  Parsing appliance features...")
    
    # Mapping from JSON keys to model feature names
    appliance_mapping = {
        'A/C': 'ac',
        'Refrig.': 'fridge',
        'Washing/Drying': 'washing_machine',
        'EV Charging': 'ev_charging',
        'DishWasher': 'dishwasher',
        'Computers': 'computers',
        'Stovetop': 'stove',
        'Water Heater': 'water_heater',
        'Standby/ Misc.': 'misc'
    }
    
    # Parse appliances if not already done
    if 'appliances' in df.columns and isinstance(df['appliances'].iloc[0], str):
        df['appliances'] = df['appliances'].apply(json.loads)
    elif 'Appliance_Breakdown_JSON' in df.columns:
        df['appliances'] = df['Appliance_Breakdown_JSON'].apply(json.loads)
    
    # Extract each appliance into its own column
    # Scale by 0.5 to get realistic consumption (~30-35 kWh per day)
    for json_key, feature_name in appliance_mapping.items():
        df[feature_name] = df['appliances'].apply(
            lambda x: float(x.get(json_key, 0)) * 0.5 if isinstance(x, dict) else 0
        )
    
    print(f"  Extracted {len(appliance_mapping)} appliance features")
    
    return df


def fetch_pricing_data(connector: SupabaseConnector, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch pricing data for date range."""
    print(f"\nFetching pricing data from {start_date.date()} to {end_date.date()}...")
    
    df = connector.get_pricing_data(start_date=start_date, end_date=end_date)
    
    if df.empty:
        raise ValueError(f"No pricing data found for range {start_date} to {end_date}")
    
    # Rename columns for consistency
    df = df.rename(columns={
        'INTERVALSTARTTIME_GMT': 'timestamp',
        'Price KWH': 'price_kwh'
    })
    
    print(f"  Loaded {len(df)} pricing records")
    print(f"  Price range: ${df['price_kwh'].min():.4f} to ${df['price_kwh'].max():.4f} per kWh")
    
    return df[['timestamp', 'price_kwh']]


def main():
    """Main execution function."""
    print("="*70)
    print("DATA ALIGNMENT FOR DEMO - Shifting House Data to Match Pricing Dates")
    print("="*70)
    
    # Initialize connector
    connector = SupabaseConnector()
    
    # Define target date ranges
    # We'll use October 21-November 11, 2024 (3 weeks total)
    # - Week 1 (Oct 21-28): Historical context
    # - Week 2 (Oct 28-Nov 4): Input week for predictions
    # - Week 3 (Nov 4-11): Prediction week to validate against actuals
    
    import pytz
    target_start = datetime(2024, 10, 21, 0, 0, 0, tzinfo=pytz.UTC)  # 3 weeks of data
    
    # Step 1: Fetch and align house data
    print("\n" + "="*70)
    print("STEP 1: Fetch and Align House Data")
    print("="*70)
    
    house1_df = fetch_house_data(connector, 1)
    house1_aligned = shift_timestamps(house1_df, target_start)
    house1_aligned = parse_appliance_features(house1_aligned)
    
    house2_df = fetch_house_data(connector, 2)
    house2_aligned = shift_timestamps(house2_df, target_start)
    house2_aligned = parse_appliance_features(house2_aligned)
    
    # Step 2: Expand to 30-minute intervals
    print("\n" + "="*70)
    print("STEP 2: Expand to 30-Minute Intervals")
    print("="*70)
    
    house1_30min = expand_to_30min_intervals(house1_aligned)
    house2_30min = expand_to_30min_intervals(house2_aligned)
    
    # Step 3: Fetch pricing data
    print("\n" + "="*70)
    print("STEP 3: Fetch Pricing Data")
    print("="*70)
    
    pricing_start = datetime(2024, 10, 21, 0, 0, 0, tzinfo=pytz.UTC)
    pricing_end = datetime(2024, 11, 11, 23, 59, 59, tzinfo=pytz.UTC)
    pricing_df = fetch_pricing_data(connector, pricing_start, pricing_end)
    
    # Resample pricing to 30-minute intervals
    pricing_df['timestamp'] = pd.to_datetime(pricing_df['timestamp'])
    pricing_df = pricing_df.set_index('timestamp')
    pricing_30min = pricing_df.resample('30min').interpolate(method='linear')
    pricing_30min = pricing_30min.reset_index()
    
    print(f"  Resampled to {len(pricing_30min)} 30-minute pricing records")
    
    # Step 4: Save aligned data to files
    print("\n" + "="*70)
    print("STEP 4: Save Aligned Data")
    print("="*70)
    
    data_dir = project_root / 'data' / 'demo'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define feature columns
    appliance_cols = ['ac', 'fridge', 'washing_machine', 'ev_charging', 'dishwasher',
                      'computers', 'stove', 'water_heater', 'misc']
    
    # Save house data with only necessary columns
    # Calculate actual consumption per interval by summing appliances (NOT using Total kWh which is daily)
    house1_export = house1_30min[['Timestamp'] + appliance_cols].copy()
    house1_export['total_kwh'] = house1_30min[appliance_cols].sum(axis=1)
    house1_export.columns = ['timestamp'] + appliance_cols + ['total_kwh']
    house1_export.to_csv(data_dir / 'house1_aligned_30min.csv', index=False)
    print(f"  ✓ Saved house1 data: {data_dir / 'house1_aligned_30min.csv'}")
    
    house2_export = house2_30min[['Timestamp'] + appliance_cols].copy()
    house2_export['total_kwh'] = house2_30min[appliance_cols].sum(axis=1)
    house2_export.columns = ['timestamp'] + appliance_cols + ['total_kwh']
    house2_export.to_csv(data_dir / 'house2_aligned_30min.csv', index=False)
    print(f"  ✓ Saved house2 data: {data_dir / 'house2_aligned_30min.csv'}")
    
    # Save pricing data
    pricing_30min.to_csv(data_dir / 'pricing_aligned_30min.csv', index=False)
    print(f"  ✓ Saved pricing data: {data_dir / 'pricing_aligned_30min.csv'}")
    
    # Step 5: Create combined dataset
    print("\n" + "="*70)
    print("STEP 5: Create Combined Dataset")
    print("="*70)
    
    # Merge house1 with pricing (using house1 as primary)
    combined = house1_export.copy()
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    
    # Merge pricing
    pricing_merge = pricing_30min.copy()
    pricing_merge['timestamp'] = pd.to_datetime(pricing_merge['timestamp'])
    
    combined = combined.merge(
        pricing_merge,
        on='timestamp',
        how='left'
    )
    
    # Forward fill any missing prices
    combined['price_kwh'] = combined['price_kwh'].ffill()
    
    combined.to_csv(data_dir / 'combined_demo_data.csv', index=False)
    print(f"  ✓ Saved combined data: {data_dir / 'combined_demo_data.csv'}")
    print(f"  Records: {len(combined)}")
    print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    
    # Summary
    print("\n" + "="*70)
    print("DATA ALIGNMENT COMPLETE!")
    print("="*70)
    print(f"\nAligned data saved to: {data_dir}")
    print(f"\nDate ranges:")
    print(f"  Historical context: Oct 21-28, 2024 (7 days)")
    print(f"  Input week:         Oct 28-Nov 4, 2024 (7 days)")
    print(f"  Prediction week:    Nov 4-11, 2024 (7 days)")
    print(f"\nTotal records per dataset:")
    print(f"  House 1: {len(house1_export)} (30-min intervals)")
    print(f"  House 2: {len(house2_export)} (30-min intervals)")
    print(f"  Pricing: {len(pricing_30min)} (30-min intervals)")
    print(f"  Combined: {len(combined)} (30-min intervals)")
    print(f"\nFeatures available:")
    print(f"  Appliances: {', '.join(appliance_cols)}")
    print(f"  Pricing: price_kwh")
    print(f"\n✓ Ready for transformer model inference!")


if __name__ == "__main__":
    main()
