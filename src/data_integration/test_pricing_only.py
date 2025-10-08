"""
Test Supabase connection for pricing data only.
Run this to verify pricing table is accessible.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_integration.supabase_connector import SupabaseConnector
from datetime import datetime


def test_pricing_connection():
    """Test connection and pricing data availability."""
    print("\n" + "="*60)
    print("PRICING DATA TEST")
    print("="*60)
    
    print("\n1. Establishing connection...")
    connector = SupabaseConnector()
    print("   ✓ Connection established")
    
    print("\n2. Fetching pricing data from 'cabuyingpricehistoryseptember2025'...")
    
    # Test with the actual table name from your context
    try:
        response = (
            connector.client
            .table('cabuyingpricehistoryseptember2025')
            .select('*')
            .limit(10)
            .execute()
        )
        
        if response.data:
            print(f"   ✓ Found pricing data!")
            print(f"   - Records fetched: {len(response.data)}")
            print(f"\n3. Sample record structure:")
            
            # Show first record structure
            if response.data:
                sample = response.data[0]
                print(f"   Available columns:")
                for key in sample.keys():
                    value = sample[key]
                    print(f"     - {key}: {value} (type: {type(value).__name__})")
            
            # Get total count
            count_response = (
                connector.client
                .table('cabuyingpricehistoryseptember2025')
                .select('*', count='exact')
                .execute()
            )
            
            total_records = count_response.count if count_response.count else 0
            print(f"\n4. Data summary:")
            print(f"   - Total pricing records: {total_records:,}")
            
            # Try to get date range if there's a timestamp column
            # We'll check common column names
            timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'datetime_beginning_utc']
            
            for ts_col in timestamp_cols:
                if ts_col in sample:
                    print(f"\n5. Date range (using '{ts_col}' column):")
                    
                    first = (
                        connector.client
                        .table('cabuyingpricehistoryseptember2025')
                        .select(ts_col)
                        .order(ts_col, desc=False)
                        .limit(1)
                        .execute()
                    )
                    
                    last = (
                        connector.client
                        .table('cabuyingpricehistoryseptember2025')
                        .select(ts_col)
                        .order(ts_col, desc=True)
                        .limit(1)
                        .execute()
                    )
                    
                    if first.data and last.data:
                        print(f"   - Start: {first.data[0][ts_col]}")
                        print(f"   - End: {last.data[0][ts_col]}")
                    break
            
            print("\n" + "="*60)
            print("✓ PRICING DATA ACCESSIBLE")
            print("="*60)
            
            print("\nNext steps:")
            print("1. Update supabase_connector.py to query 'cabuyingpricehistoryseptember2025'")
            print("2. Update data_adapter.py to map CA pricing columns to simulation format")
            print("3. Generate synthetic household/weather data to match pricing dates")
            
            return True
            
        else:
            print("   ✗ No data returned from table")
            return False
            
    except Exception as e:
        print(f"   ✗ Error accessing table: {e}")
        print(f"\n   Please verify:")
        print(f"   - Table name is exactly: 'cabuyingpricehistoryseptember2025'")
        print(f"   - Table exists in Supabase")
        print(f"   - API credentials have read access")
        return False


if __name__ == "__main__":
    test_pricing_connection()
