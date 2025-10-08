"""
Diagnose RLS (Row Level Security) and permission issues.
"""

import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_integration.supabase_connector import SupabaseConnector


def diagnose_rls():
    """Check for RLS issues."""
    print("\n" + "="*60)
    print("SUPABASE RLS DIAGNOSTIC")
    print("="*60)
    
    print("\n1. Checking API key type...")
    
    # Check which key is being used
    key = os.getenv('SUPABASE_KEY', '')
    
    if not key:
        print("   ✗ No SUPABASE_KEY found in .env")
        return
    
    # Anon keys are typically shorter and service_role keys are longer
    # But both are JWT tokens, so we'll just check if it starts with 'eyJ'
    if key.startswith('eyJ'):
        print("   ✓ Valid JWT token detected")
        # We can't easily determine if it's anon or service_role without decoding
        # But the length can give us a hint
        if len(key) < 200:
            print("   → Likely using 'anon' (public) key")
            print("   → This key is subject to RLS policies")
        else:
            print("   → Likely using 'service_role' key")
            print("   → This key bypasses RLS")
    else:
        print("   ⚠ Key format unexpected")
    
    print("\n2. Testing table access...")
    
    connector = SupabaseConnector()
    table_name = 'cabuyingpricehistoryseptember2025'
    
    try:
        # Try to count records
        response = (
            connector.client
            .table(table_name)
            .select('*', count='exact')
            .limit(0)
            .execute()
        )
        
        count = response.count if response.count is not None else 0
        
        print(f"   API returned count: {count:,}")
        
        if count == 0:
            print("\n" + "="*60)
            print("ISSUE IDENTIFIED: RLS BLOCKING ACCESS")
            print("="*60)
            
            print("\nThe table exists but returns 0 records.")
            print("This is almost certainly due to Row Level Security (RLS).")
            
            print("\n📋 SOLUTION OPTIONS:")
            print("\n--- Option 1: Disable RLS (Easiest for Development) ---")
            print("1. Go to Supabase Dashboard")
            print("2. Navigate to: Table Editor → cabuyingpricehistoryseptember2025")
            print("3. Click the table settings (⚙️ icon)")
            print("4. Toggle OFF 'Enable Row Level Security'")
            print("5. Save and test again")
            
            print("\n--- Option 2: Add RLS Policy (More Secure) ---")
            print("1. Go to Supabase Dashboard")
            print("2. Navigate to: Authentication → Policies")
            print("3. Find 'cabuyingpricehistoryseptember2025'")
            print("4. Click 'New Policy'")
            print("5. Choose 'Enable read access for all users'")
            print("6. Set policy to: true")
            print("7. Save and test again")
            
            print("\n--- Option 3: Use Service Role Key (Bypasses RLS) ---")
            print("1. Go to Supabase Dashboard")
            print("2. Navigate to: Project Settings → API")
            print("3. Copy the 'service_role' key (not anon key)")
            print("4. Update .env file:")
            print("   SUPABASE_KEY=<paste-service-role-key-here>")
            print("5. Test again")
            print("\n⚠️  Warning: Service role key bypasses ALL security.")
            print("   Only use in development or with proper safeguards!")
            
        else:
            print(f"\n✓ Access working! Found {count:,} records")
            
            # Try to fetch one record
            sample = (
                connector.client
                .table(table_name)
                .select('*')
                .limit(1)
                .execute()
            )
            
            if sample.data:
                print(f"\n3. Sample record structure:")
                print(f"   Columns: {list(sample.data[0].keys())}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        
        if "PGRST301" in str(e) or "JWT" in str(e):
            print("\n   → This looks like an authentication/RLS issue")
        
        print("\n   Check:")
        print("   1. SUPABASE_KEY is correct in .env")
        print("   2. RLS policies allow access")
        print("   3. API key has necessary permissions")


if __name__ == "__main__":
    diagnose_rls()
