"""
List all tables accessible in Supabase.
This helps identify the correct table names.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_integration.supabase_connector import SupabaseConnector


def list_tables():
    """Try to discover available tables."""
    print("\n" + "="*60)
    print("DISCOVERING SUPABASE TABLES")
    print("="*60)
    
    print("\n1. Establishing connection...")
    connector = SupabaseConnector()
    print("   ✓ Connection established")
    
    print("\n2. Testing common table names...")
    
    # List of possible table names to test
    test_names = [
        'cabuyingpricehistoryseptember2025',
        'ca_buying_price_history_september_2025',
        'CaBuyingPriceHistorySeptember2025',
        'pricing',
        'prices',
        'electricity_prices',
        'households',
        'consumption',
        'weather'
    ]
    
    accessible_tables = []
    
    for table_name in test_names:
        try:
            response = (
                connector.client
                .table(table_name)
                .select('*', count='exact')
                .limit(1)
                .execute()
            )
            
            if response.count is not None:
                accessible_tables.append({
                    'name': table_name,
                    'count': response.count,
                    'sample': response.data[0] if response.data else None
                })
                print(f"   ✓ Found: '{table_name}' ({response.count:,} records)")
            
        except Exception as e:
            error_msg = str(e)
            if "PGRST205" in error_msg or "Could not find" in error_msg:
                print(f"   ✗ Not found: '{table_name}'")
            else:
                print(f"   ⚠ Error testing '{table_name}': {error_msg}")
    
    if accessible_tables:
        print("\n" + "="*60)
        print("ACCESSIBLE TABLES")
        print("="*60)
        
        for table in accessible_tables:
            print(f"\nTable: '{table['name']}'")
            print(f"  Records: {table['count']:,}")
            
            if table['sample']:
                print(f"  Columns: {list(table['sample'].keys())}")
        
        return accessible_tables
    else:
        print("\n" + "="*60)
        print("NO TABLES FOUND")
        print("="*60)
        
        print("\nPossible issues:")
        print("1. RLS (Row Level Security) is enabled - check Supabase dashboard")
        print("2. Table names are different - check actual table names in Supabase")
        print("3. API key doesn't have read permissions")
        print("\nTo check in Supabase:")
        print("- Go to Table Editor to see actual table names")
        print("- Go to Authentication > Policies to check RLS settings")
        
        return []


if __name__ == "__main__":
    list_tables()
