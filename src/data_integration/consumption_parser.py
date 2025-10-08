"""
Consumption data parser for per-house data tables.

This module parses house-specific data tables with appliance-level consumption,
weather data, and battery sensor readings. NO synthetic data - all values
come from real sensors in the database.
"""

import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime


# Define the 9 appliances we expect in each house table
APPLIANCE_COLUMNS = [
    "appliance_washing_machine_kwh",
    "appliance_dishwasher_kwh",
    "appliance_ev_charging_kwh",
    "appliance_fridge_kwh",
    "appliance_ac_kwh",
    "appliance_stove_kwh",
    "appliance_water_heater_kwh",
    "appliance_computers_kwh",
    "appliance_misc_kwh"
]


def validate_appliance_data(df: pd.DataFrame, house_id: int) -> None:
    """
    Validate appliance consumption data from house table.
    
    Args:
        df: DataFrame with appliance columns
        house_id: House ID for error messages
        
    Raises:
        ValueError: If data validation fails
    """
    # Check all appliance columns present
    missing_appliances = [col for col in APPLIANCE_COLUMNS if col not in df.columns]
    if missing_appliances:
        raise ValueError(
            f"Missing appliance columns in house_{house_id}_data table: {missing_appliances}. "
            f"Expected all 9 appliances: {APPLIANCE_COLUMNS}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Check for negative values
    for col in APPLIANCE_COLUMNS:
        negative_mask = df[col] < 0
        if negative_mask.any():
            negative_count = negative_mask.sum()
            first_negative = df[negative_mask].head(1)
            raise ValueError(
                f"Negative consumption detected in house_{house_id}_data.{col}: "
                f"{negative_count} records with negative values. "
                f"First occurrence: timestamp={first_negative['timestamp'].values[0]}, "
                f"value={first_negative[col].values[0]}. "
                f"Check appliance sensor calibration."
            )
    
    # Validate total_consumption_kwh matches sum
    if 'total_consumption_kwh' in df.columns:
        df['computed_total'] = df[APPLIANCE_COLUMNS].sum(axis=1)
        tolerance = 0.01  # 0.01 kWh tolerance for floating point
        mismatch_mask = abs(df['total_consumption_kwh'] - df['computed_total']) > tolerance
        
        if mismatch_mask.any():
            mismatch_count = mismatch_mask.sum()
            first_mismatch = df[mismatch_mask].head(1)
            raise ValueError(
                f"Total consumption mismatch in house_{house_id}_data: "
                f"{mismatch_count} records where total_consumption_kwh doesn't match sum of appliances. "
                f"First occurrence: timestamp={first_mismatch['timestamp'].values[0]}, "
                f"total={first_mismatch['total_consumption_kwh'].values[0]:.3f}, "
                f"sum={first_mismatch['computed_total'].values[0]:.3f}. "
                f"Check data pipeline or sensor aggregation logic."
            )


def parse_house_data(
    house_id: int,
    connector,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Parse consumption data from house-specific table.
    
    Reads from house_{id}_data table which contains:
    - 9 appliance consumption columns
    - Total consumption
    - Weather data (temperature, solar_irradiance)
    - Battery sensor data (SoC, SoH, charge, cycles)
    - Time features
    
    Args:
        house_id: Household ID
        connector: SupabaseConnector instance
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame with all columns needed for LSTM training/prediction
        
    Raises:
        ValueError: If table doesn't exist or data is malformed
    """
    table_name = f"house_{house_id}_data"
    
    # Query the house-specific data table
    try:
        query = connector.client.table(table_name).select('*').order('timestamp', desc=False)
        
        if start_date:
            query = query.gte('timestamp', start_date.isoformat())
        
        if end_date:
            query = query.lte('timestamp', end_date.isoformat())
        
        response = query.execute()
        
        if not response.data:
            raise ValueError(
                f"No data found in {table_name} table. "
                f"Date range: {start_date} to {end_date}. "
                f"Ensure sensors are writing data to this table."
            )
        
        df = pd.DataFrame(response.data)
        
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            raise ValueError(
                f"Table {table_name} does not exist in database. "
                f"Create this table with the per-house schema before loading data. "
                f"Expected columns: timestamp, 9 appliances, weather, battery sensors."
            )
        else:
            raise ValueError(
                f"Error querying {table_name}: {str(e)}. "
                f"Check database permissions and table schema."
            )
    
    # Validate required columns
    required_columns = [
        'timestamp',
        'total_consumption_kwh',
        'temperature',
        'solar_irradiance',
        'battery_soc_percent',
        'battery_soh_percent',
        'battery_charge_kwh',
        'battery_cycle_count'
    ] + APPLIANCE_COLUMNS
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {table_name}: {missing}. "
            f"Expected columns: {required_columns}. "
            f"Available columns: {df.columns.tolist()}. "
            f"Update table schema or check sensor data pipeline."
        )
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Validate appliance data
    validate_appliance_data(df, house_id)
    
    # Add/validate time features
    if 'hour_of_day' not in df.columns:
        df['hour_of_day'] = df['timestamp'].dt.hour
    
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'] >= 5
    
    # Rename consumption column for consistency with LSTM model
    df['consumption_kwh'] = df['total_consumption_kwh']
    
    return df


def load_battery_config(house_id: int, connector) -> Dict:
    """
    Load static battery configuration from database.
    
    Args:
        house_id: Household ID
        connector: SupabaseConnector instance
        
    Returns:
        Dictionary with battery specs
        
    Raises:
        ValueError: If config not found or incomplete
    """
    table_name = f"house_{house_id}_battery_config"
    
    try:
        response = (
            connector.client
            .table(table_name)
            .select('*')
            .eq('household_id', house_id)
            .single()
            .execute()
        )
        
        if not response.data:
            raise ValueError(
                f"No battery configuration found in {table_name} for household {house_id}. "
                f"Insert battery specs into this table before initializing battery manager."
            )
        
        config = response.data
        
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            raise ValueError(
                f"Table {table_name} does not exist in database. "
                f"Create battery configuration table for house {house_id}."
            )
        else:
            raise ValueError(
                f"Error loading battery config from {table_name}: {str(e)}"
            )
    
    # Validate required fields
    required_fields = [
        'total_capacity_kwh',
        'max_charge_rate_kw',
        'max_discharge_rate_kw',
        'efficiency',
        'min_reserve_percent',
        'max_charge_percent'
    ]
    
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(
            f"Missing required battery config fields in {table_name}: {missing}. "
            f"Required fields: {required_fields}"
        )
    
    # Map to expected parameter names
    return {
        'capacity_kwh': config['total_capacity_kwh'],
        'max_charge_rate_kw': config['max_charge_rate_kw'],
        'max_discharge_rate_kw': config['max_discharge_rate_kw'],
        'efficiency': config['efficiency'],
        'min_reserve_percent': config['min_reserve_percent'],
        'max_charge_percent': config['max_charge_percent'],
        'usable_capacity_kwh': config.get('usable_capacity_kwh'),
    }


def load_grid_config(house_id: int, connector) -> Dict:
    """
    Load grid connection configuration from database.
    
    Args:
        house_id: Household ID
        connector: SupabaseConnector instance
        
    Returns:
        Dictionary with grid constraints (max import/export)
        
    Raises:
        ValueError: If config not found
    """
    table_name = f"house_{house_id}_grid_config"
    
    try:
        response = (
            connector.client
            .table(table_name)
            .select('*')
            .eq('household_id', house_id)
            .single()
            .execute()
        )
        
        if not response.data:
            raise ValueError(
                f"No grid configuration found in {table_name} for household {house_id}. "
                f"Insert grid connection specs into this table."
            )
        
        config = response.data
        
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            raise ValueError(
                f"Table {table_name} does not exist in database. "
                f"Create grid configuration table for house {house_id}."
            )
        else:
            raise ValueError(
                f"Error loading grid config from {table_name}: {str(e)}"
            )
    
    required_fields = ['max_import_kw', 'max_export_kw']
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(
            f"Missing required grid config fields in {table_name}: {missing}. "
            f"Required fields: {required_fields}"
        )
    
    return {
        'max_import_kw': config['max_import_kw'],
        'max_export_kw': config['max_export_kw'],
    }


def load_trading_config(house_id: int, connector) -> Dict:
    """
    Load trading configuration from database.
    
    Args:
        house_id: Household ID
        connector: SupabaseConnector instance
        
    Returns:
        Dictionary with trading rules
        
    Raises:
        ValueError: If config not found
    """
    table_name = f"house_{house_id}_trading_config"
    
    try:
        response = (
            connector.client
            .table(table_name)
            .select('*')
            .eq('household_id', house_id)
            .single()
            .execute()
        )
        
        if not response.data:
            raise ValueError(
                f"No trading configuration found in {table_name} for household {house_id}. "
                f"Insert trading rules into this table."
            )
        
        config = response.data
        
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            raise ValueError(
                f"Table {table_name} does not exist in database. "
                f"Create trading configuration table for house {house_id}."
            )
        else:
            raise ValueError(
                f"Error loading trading config from {table_name}: {str(e)}"
            )
    
    required_fields = [
        'max_sell_per_day_kwh',
        'max_single_trade_kwh',
        'min_profit_margin'
    ]
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(
            f"Missing required trading config fields in {table_name}: {missing}. "
            f"Required fields: {required_fields}"
        )
    
    return {
        'max_sell_per_day_kwh': config['max_sell_per_day_kwh'],
        'max_single_trade_kwh': config['max_single_trade_kwh'],
        'min_profit_margin': config['min_profit_margin'],
        'enabled': config.get('enabled', True),
    }


def get_battery_sensor_data(df: pd.DataFrame, interval_index: int) -> Dict:
    """
    Extract battery sensor data for a specific interval.
    
    Args:
        df: DataFrame with battery sensor columns
        interval_index: Index of the interval to extract
        
    Returns:
        Dictionary with sensor readings for BatteryManager.update_state_from_sensors()
        
    Raises:
        ValueError: If sensor data is missing or invalid
    """
    if interval_index >= len(df):
        raise ValueError(
            f"Interval index {interval_index} out of range (data has {len(df)} intervals)"
        )
    
    row = df.iloc[interval_index]
    
    required_sensors = ['battery_soc_percent', 'battery_soh_percent', 
                       'battery_charge_kwh', 'battery_cycle_count']
    missing = [s for s in required_sensors if pd.isna(row[s])]
    
    if missing:
        raise ValueError(
            f"Missing battery sensor data at interval {interval_index} "
            f"(timestamp={row['timestamp']}): {missing}. "
            f"All battery sensors must report values. Check sensor connectivity."
        )
    
    return {
        'current_charge_kwh': float(row['battery_charge_kwh']),
        'soh_percent': float(row['battery_soh_percent']),
        'cycle_count': float(row['battery_cycle_count']),
        'soc_percent': float(row['battery_soc_percent']),  # For reference
    }


if __name__ == "__main__":
    print("Consumption Parser Module")
    print("=" * 60)
    print("\nThis module parses per-house data tables with:")
    print("  - 9 appliance consumption columns")
    print("  - Weather data (temperature, solar_irradiance)")
    print("  - Battery sensor data (SoC, SoH, charge, cycles)")
    print("  - Time features")
    print("\nAll data comes from real sensors - NO synthetic data!")
    print("\nExpected table schema:")
    print(f"  house_{{id}}_data:")
    print(f"    - timestamp")
    for col in APPLIANCE_COLUMNS:
        print(f"    - {col}")
    print(f"    - total_consumption_kwh")
    print(f"    - temperature")
    print(f"    - solar_irradiance")
    print(f"    - battery_soc_percent")
    print(f"    - battery_soh_percent")
    print(f"    - battery_charge_kwh")
    print(f"    - battery_cycle_count")
    print("\nConfiguration tables:")
    print(f"  - house_{{id}}_battery_config")
    print(f"  - house_{{id}}_grid_config")
    print(f"  - house_{{id}}_trading_config")
    print("=" * 60)
