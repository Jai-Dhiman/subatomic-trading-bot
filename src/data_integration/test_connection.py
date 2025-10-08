"""
Test Supabase connection and data structure.
Run this script to verify your database is properly connected.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import DataAdapter


def test_connection():
    """Test basic Supabase connection."""
    print("\n" + "="*60)
    print("SUPABASE CONNECTION TEST")
    print("="*60)
    
    print("\n1. Testing connection...")
    connector = SupabaseConnector()
        
    print("   ✓ Connection established")
    
    return connector


def test_data_summary(connector):
    """Test data availability and structure."""
    print("\n2. Fetching data summary...")
    
    summary = connector.get_data_summary()
        
    print(f"\n   Data Summary:")
    print(f"   - Households: {summary.get('num_households', 0)}")
    
    if summary.get('household_ids'):
        print(f"   - Household IDs: {summary['household_ids']}")
    
    print(f"   - Consumption records: {summary.get('total_consumption_records', 0):,}")
    print(f"   - Weather records: {summary.get('total_weather_records', 0):,}")
    
    if 'consumption_date_range' in summary:
        date_range = summary['consumption_date_range']
        print(f"   - Date range: {date_range['start']} to {date_range['end']}")
    
    return summary


def test_household_data(connector, household_id):
    """Test fetching and adapting household data."""
    print(f"\n3. Testing household {household_id} data...")
    
    print(f"   Fetching consumption data...")
    raw_consumption = connector.get_consumption_data(household_id, limit=100)
        
    if raw_consumption.empty:
        print(f"   ✗ No consumption data found for household {household_id}")
        return None
    
    print(f"   ✓ Found {len(raw_consumption)} consumption records")
    print(f"   Columns: {raw_consumption.columns.tolist()}")
    
    print(f"\n   Adapting consumption data...")
    adapted_consumption = DataAdapter.adapt_consumption_data(raw_consumption)
        
    print(f"   ✓ Adapted successfully")
    print(f"   Final columns: {adapted_consumption.columns.tolist()}")
    
    return adapted_consumption


def test_weather_data(connector):
    """Test fetching and adapting weather data."""
    print(f"\n4. Testing weather data...")
    
    print(f"   Fetching weather data...")
    raw_weather = connector.get_weather_data(limit=100)
        
    if raw_weather.empty:
        print(f"   ⚠ No weather data found (will use defaults)")
        return None
    
    print(f"   ✓ Found {len(raw_weather)} weather records")
    print(f"   Columns: {raw_weather.columns.tolist()}")
    
    print(f"\n   Adapting weather data...")
    adapted_weather = DataAdapter.adapt_weather_data(raw_weather)
        
    print(f"   ✓ Adapted successfully")
    print(f"   Final columns: {adapted_weather.columns.tolist()}")
    
    return adapted_weather


def test_data_merge(consumption_df, weather_df):
    """Test merging consumption with weather."""
    print(f"\n5. Testing data merge...")
    
    if consumption_df is None:
        print(f"   ✗ Cannot merge - no consumption data")
        return None
    
    merged_df = DataAdapter.adapt_household_consumption_with_weather(
        consumption_df, weather_df or consumption_df
    )
        
    print(f"   ✓ Merged successfully")
    print(f"   Final shape: {merged_df.shape}")
    print(f"   Columns: {merged_df.columns.tolist()}")
    
    info = DataAdapter.get_data_info(merged_df)
    print(f"\n   Data Info:")
    print(f"   - Records: {info['num_records']}")
    if info.get('date_range'):
        print(f"   - Date range: {info['date_range']['start']} to {info['date_range']['end']}")
    
    null_counts = info.get('null_counts', {})
    if any(null_counts.values()):
        print(f"   - Null values: {null_counts}")
    
    return merged_df


def validate_for_simulation(merged_df):
    """Validate data is ready for simulation."""
    print(f"\n6. Validating for simulation...")
    
    if merged_df is None or merged_df.empty:
        print(f"   ✗ No data available")
        return False
    
    required_columns = [
        'timestamp', 'household_id', 'consumption_kwh',
        'temperature', 'solar_irradiance',
        'hour_of_day', 'day_of_week', 'is_weekend'
    ]
    
    DataAdapter.validate_data_format(merged_df, required_columns)
        
    print(f"   ✓ All required columns present")
    
    num_records = len(merged_df)
    min_required = 48
    recommended = 17520
    
    print(f"\n   Data sufficiency:")
    print(f"   - Current records: {num_records:,}")
    print(f"   - Minimum required: {min_required:,} (1 day)")
    print(f"   - Recommended: {recommended:,} (1 year)")
    
    if num_records < min_required:
        print(f"   ✗ Insufficient data for simulation")
        return False
    elif num_records < recommended:
        print(f"   ⚠ Below recommended amount (model may underperform)")
    else:
        print(f"   ✓ Sufficient data for training")
    
    return True


def main():
    """Run all tests."""
    connector = test_connection()
        
    summary = test_data_summary(connector)
        
    household_ids = summary.get('household_ids', [])
    if not household_ids:
        print(f"\n✗ No households found in database")
        print(f"\nPlease ensure your Supabase database has:")
        print(f"  1. A 'households' table with data")
        print(f"  2. A 'consumption' table with household data")
        print(f"  3. Optional: 'weather' table with weather data")
        return
    
    test_household_id = household_ids[0]
    
    consumption_df = test_household_data(connector, test_household_id)
        
    weather_df = test_weather_data(connector)
        
    if consumption_df is not None:
        merged_df = test_data_merge(consumption_df, weather_df)
            
        is_valid = validate_for_simulation(merged_df)
            
        print("\n" + "="*60)
        if is_valid:
            print("✓ ALL TESTS PASSED - DATA READY FOR SIMULATION")
        else:
            print("⚠ TESTS COMPLETED WITH WARNINGS")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Update config/config.yaml to use 'supabase' data source")
        print("2. Run simulation with: python -m src.simulation.run_demo")
        print("3. Check DATA_SCHEMA.md if you encounter schema issues")
    else:
        print("\n" + "="*60)
        print("✗ TESTS FAILED - CHECK DATA AVAILABILITY")
        print("="*60)


if __name__ == "__main__":
    main()
