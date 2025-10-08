"""
Test CA pricing data integration end-to-end.
Validates that pricing data can be fetched and adapted correctly.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import DataAdapter


def test_ca_pricing_integration():
    """Test complete pricing data flow."""
    print("\n" + "="*60)
    print("CA PRICING DATA INTEGRATION TEST")
    print("="*60)
    
    print("\n1. Connecting to Supabase...")
    connector = SupabaseConnector()
    print("   ✓ Connected")
    
    print("\n2. Fetching pricing data...")
    pricing_df = connector.get_pricing_data()
    
    if pricing_df.empty:
        print("   ✗ No pricing data returned")
        return False
    
    print(f"   ✓ Fetched {len(pricing_df):,} records")
    print(f"   Raw columns: {pricing_df.columns.tolist()}")
    
    print("\n3. Sample raw data:")
    sample = pricing_df.head(3)
    for idx, row in sample.iterrows():
        print(f"   Record {idx + 1}:")
        print(f"     Time: {row['INTERVALSTARTTIME_GMT']}")
        print(f"     Price: ${row['Price KWH']:.4f}/kWh")
        if idx == 0:
            print(f"     Node: {row.get('NODE', 'N/A')}")
    
    print("\n4. Adapting pricing data to simulation format...")
    adapted_df = DataAdapter.adapt_pricing_data(pricing_df)
    
    if adapted_df.empty:
        print("   ✗ Adapter returned empty DataFrame")
        return False
    
    print(f"   ✓ Adapted successfully")
    print(f"   Final columns: {adapted_df.columns.tolist()}")
    print(f"   Final records: {len(adapted_df):,}")
    
    print("\n5. Validating adapted data...")
    
    # Check required columns
    required = ['timestamp', 'price_per_kwh']
    missing = set(required) - set(adapted_df.columns)
    if missing:
        print(f"   ✗ Missing columns: {missing}")
        return False
    print(f"   ✓ All required columns present")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(adapted_df['timestamp']):
        print(f"   ✗ timestamp is not datetime")
        return False
    print(f"   ✓ timestamp is datetime")
    
    if not pd.api.types.is_numeric_dtype(adapted_df['price_per_kwh']):
        print(f"   ✗ price_per_kwh is not numeric")
        return False
    print(f"   ✓ price_per_kwh is numeric")
    
    # Check for nulls
    null_counts = adapted_df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"   ⚠ Warning: Found null values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"     - {col}: {count} nulls")
    else:
        print(f"   ✓ No null values")
    
    # Price statistics
    print("\n6. Price statistics:")
    print(f"   - Mean: ${adapted_df['price_per_kwh'].mean():.4f}/kWh")
    print(f"   - Min:  ${adapted_df['price_per_kwh'].min():.4f}/kWh")
    print(f"   - Max:  ${adapted_df['price_per_kwh'].max():.4f}/kWh")
    print(f"   - Std:  ${adapted_df['price_per_kwh'].std():.4f}/kWh")
    
    # Date range
    print("\n7. Date range:")
    print(f"   - Start: {adapted_df['timestamp'].min()}")
    print(f"   - End:   {adapted_df['timestamp'].max()}")
    duration = adapted_df['timestamp'].max() - adapted_df['timestamp'].min()
    print(f"   - Duration: {duration.days} days, {duration.seconds // 3600} hours")
    
    # Sample adapted data
    print("\n8. Sample adapted data:")
    sample_adapted = adapted_df.head(3)
    for idx, row in sample_adapted.iterrows():
        print(f"   {row['timestamp']}: ${row['price_per_kwh']:.4f}/kWh")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    
    print("\nThe pricing data is ready for simulation!")
    print("\nNext steps:")
    print("1. Generate or load household consumption data")
    print("2. Generate or load weather data")
    print("3. Run simulation: python -m src.simulation.run_demo")
    
    return True


if __name__ == "__main__":
    import pandas as pd
    test_ca_pricing_integration()
