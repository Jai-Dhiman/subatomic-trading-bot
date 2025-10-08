"""
Clear Battery Data from Supabase.

Removes all existing battery trading data from specified house battery tables.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_integration.supabase_connector import SupabaseConnector


def clear_battery_table(connector: SupabaseConnector, house_id: int) -> bool:
    """
    Clear all data from a house battery table.
    
    Args:
        connector: Supabase connector
        house_id: House ID
    
    Returns:
        True if successful
    """
    table_name = f"house{house_id}_battery"
    
    try:
        # Delete all records
        result = connector.client.table(table_name).delete().neq('house_id', -999).execute()
        print(f"  ✓ Cleared {table_name}")
        return True
    except Exception as e:
        print(f"  ⚠ Error clearing {table_name}: {e}")
        return False


def main():
    """Main execution."""
    print("="*70)
    print("CLEAR BATTERY DATA")
    print("="*70)
    print("\nThis will delete all existing battery trading data.")
    
    # Connect
    print("\n1. Connecting to Supabase...")
    connector = SupabaseConnector()
    print("   ✓ Connected")
    
    # Ask which houses
    print("\n2. Which houses to clear?")
    house_input = input("   Enter house IDs (comma-separated, or 'all' for 1,2): ").strip()
    
    if house_input.lower() == 'all':
        houses = [1, 2]
    else:
        houses = [int(h.strip()) for h in house_input.split(',') if h.strip()]
    
    print(f"   Will clear data for houses: {houses}")
    
    # Confirm
    confirm = input("\n   Are you sure? This cannot be undone (y/n): ").strip().lower()
    if confirm != 'y':
        print("   Cancelled.")
        return
    
    # Clear
    print("\n3. Clearing battery data...")
    success_count = 0
    for house_id in houses:
        if clear_battery_table(connector, house_id):
            success_count += 1
    
    print("\n" + "="*70)
    print("✅ CLEAR COMPLETE")
    print("="*70)
    print(f"\nCleared {success_count}/{len(houses)} tables")


if __name__ == "__main__":
    main()
