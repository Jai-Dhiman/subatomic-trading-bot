"""
Generate Realistic Battery Data for Supabase.

Simulates Subatomic Battery behavior with intelligent trading:
- Buy power when prices are low (bottom 25%)
- Sell power when prices are high (top 25%)
- Maintain 10% buffer above consumption
- Respect battery constraints (20-90% SoC, 10kW charge, 8kW discharge)

Creates per-house battery tables in Supabase with 30-min interval data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import load_consumption_data, load_pricing_data


class BatteryDataGenerator:
    """Generate realistic battery data with intelligent trading behavior."""
    
    def __init__(self, capacity_kwh: float = 40.0):
        """
        Initialize battery data generator.
        
        Args:
            capacity_kwh: Battery capacity (40 kWh for houses 1-9, 80 kWh for houses 10-11)
        """
        self.capacity_kwh = capacity_kwh
        self.max_charge_rate_kw = 10.0
        self.max_discharge_rate_kw = 8.0
        self.min_soc = 0.20
        self.max_soc = 0.90
        self.efficiency = 0.95
        self.buffer_percentage = 0.10  # 10% buffer above consumption
        
    def simulate_intelligent_trading(
        self,
        consumption_data: pd.DataFrame,
        pricing_data: pd.DataFrame,
        house_id: int
    ) -> pd.DataFrame:
        """
        Simulate battery behavior with intelligent trading.
        
        Strategy:
        1. Calculate price percentiles (25th and 75th)
        2. Buy when price is low (< 25th percentile) and battery has capacity
        3. Sell when price is high (> 75th percentile) and battery has excess
        4. Always maintain 10% buffer above predicted consumption
        5. Respect battery constraints
        
        Args:
            consumption_data: DataFrame with consumption for this house
            pricing_data: DataFrame with pricing patterns
            house_id: House ID (1-11)
            
        Returns:
            DataFrame with battery state for each timestamp
        """
        print(f"\n  Simulating battery for House {house_id}...")
        
        # Filter to this house only
        house_data = consumption_data[consumption_data['house_id'] == house_id].copy()
        house_data = house_data.sort_values('timestamp').reset_index(drop=True)
        
        if len(house_data) == 0:
            print(f"    ⚠ No data for house {house_id}")
            return pd.DataFrame()
        
        # Merge with pricing data (using hour/dayofweek patterns)
        house_data['hour'] = house_data['timestamp'].dt.hour
        house_data['dayofweek'] = house_data['timestamp'].dt.dayofweek
        
        # Get pricing patterns
        pricing_data = pricing_data.copy()
        pricing_data['hour'] = pricing_data['timestamp'].dt.hour
        pricing_data['dayofweek'] = pricing_data['timestamp'].dt.dayofweek
        price_patterns = pricing_data.groupby(['hour', 'dayofweek'])['price_per_kwh'].mean().reset_index()
        
        house_data = house_data.merge(price_patterns, on=['hour', 'dayofweek'], how='left')
        house_data['price_per_kwh'] = house_data['price_per_kwh'].fillna(
            pricing_data['price_per_kwh'].median()
        )
        
        # Calculate price percentiles for trading decisions
        price_25th = house_data['price_per_kwh'].quantile(0.25)
        price_75th = house_data['price_per_kwh'].quantile(0.75)
        
        print(f"    Price range: ${house_data['price_per_kwh'].min():.4f} to ${house_data['price_per_kwh'].max():.4f}")
        print(f"    25th percentile: ${price_25th:.4f}, 75th percentile: ${price_75th:.4f}")
        
        # Initialize battery state
        current_charge = self.capacity_kwh * 0.50  # Start at 50%
        soh = np.random.uniform(97.0, 100.0)  # State of health
        
        battery_records = []
        
        for idx, row in house_data.iterrows():
            timestamp = row['timestamp']
            consumption = row['total_consumption_kwh']
            price = row['price_per_kwh']
            hour = row['hour']
            
            # Calculate required energy with 10% buffer
            required_energy = consumption * (1 + self.buffer_percentage)
            
            # Current state
            current_soc = current_charge / self.capacity_kwh
            available_for_discharge = max(0, current_charge - (self.capacity_kwh * self.min_soc))
            available_for_charge = max(0, (self.capacity_kwh * self.max_soc) - current_charge)
            
            # Trading decision
            action = "hold"
            trade_amount = 0.0
            
            # Decision 1: Buy when price is low and battery has capacity
            if price < price_25th and current_soc < self.max_soc:
                # Good price - buy energy
                max_buy = min(
                    self.max_charge_rate_kw * 0.5,  # 0.5 hours = 30 min interval
                    available_for_charge
                )
                
                # Buy enough to reach 80% SoC or max charge rate
                target_charge = self.capacity_kwh * 0.80
                desired_buy = max(0, target_charge - current_charge)
                
                trade_amount = min(max_buy, desired_buy)
                
                if trade_amount > 0.1:  # Only trade if significant
                    current_charge += trade_amount * self.efficiency
                    action = "buy"
            
            # Decision 2: Sell when price is high and battery has excess
            elif price > price_75th and current_soc > 0.40:
                # Good price - sell excess energy
                max_sell = min(
                    self.max_discharge_rate_kw * 0.5,  # 0.5 hours = 30 min interval
                    available_for_discharge
                )
                
                # Keep buffer above consumption
                safe_discharge = max(0, current_charge - required_energy)
                
                trade_amount = min(max_sell, safe_discharge)
                
                if trade_amount > 0.1:  # Only trade if significant
                    current_charge -= trade_amount
                    action = "sell"
            
            # Decision 3: Night charging (cheap electricity, even if not in bottom 25%)
            elif (23 <= hour or hour < 6) and price < 0.30 and current_soc < 0.80:
                # Charge at night when typically cheaper
                max_charge = min(
                    self.max_charge_rate_kw * 0.5,
                    available_for_charge
                )
                trade_amount = max_charge
                
                if trade_amount > 0.1:
                    current_charge += trade_amount * self.efficiency
                    action = "night_charge"
            
            # Decision 4: Peak support (discharge during peak hours if needed)
            elif (16 <= hour <= 21) and consumption > 2.0 and current_soc > 0.40:
                # Discharge to support consumption during peak hours
                support_amount = min(
                    consumption * 0.3,  # Support 30% of consumption
                    available_for_discharge
                )
                
                if support_amount > 0.1:
                    current_charge -= support_amount
                    action = "peak_support"
            
            # Ensure SoC stays within bounds
            current_soc = current_charge / self.capacity_kwh
            current_soc = np.clip(current_soc, self.min_soc, self.max_soc)
            current_charge = self.capacity_kwh * current_soc
            
            # Calculate final state
            available_for_discharge = max(0, current_charge - (self.capacity_kwh * self.min_soc))
            available_for_charge = max(0, (self.capacity_kwh * self.max_soc) - current_charge)
            
            # Record state
            battery_records.append({
                'timestamp': timestamp,
                'house_id': house_id,
                'battery_soc_percent': current_soc * 100,
                'battery_charge_kwh': current_charge,
                'battery_available_kwh': available_for_discharge,
                'battery_capacity_remaining_kwh': available_for_charge,
                'battery_soh_percent': soh,
                'battery_count': 1 if house_id <= 9 else 2,
                'total_capacity_kwh': self.capacity_kwh,
                'max_charge_rate_kw': self.max_charge_rate_kw,
                'max_discharge_rate_kw': self.max_discharge_rate_kw,
                'action': action,
                'trade_amount_kwh': trade_amount,
                'price_per_kwh': price,
                'consumption_kwh': consumption
            })
        
        df = pd.DataFrame(battery_records)
        
        # Print statistics
        print(f"    ✓ Generated {len(df):,} battery states")
        print(f"    ✓ SoC range: {df['battery_soc_percent'].min():.1f}% to {df['battery_soc_percent'].max():.1f}%")
        print(f"    ✓ Actions: Buy={len(df[df['action']=='buy'])}, Sell={len(df[df['action']=='sell'])}, "
              f"Night={len(df[df['action']=='night_charge'])}, Peak={len(df[df['action']=='peak_support'])}")
        
        return df


def create_battery_table_schema(connector: SupabaseConnector, house_id: int):
    """
    Create battery data table in Supabase for a house.
    
    Note: This uses SQL to create the table. Run this once per house.
    """
    table_name = f"house{house_id}_battery"
    
    # SQL to create table (run this in Supabase SQL editor)
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        house_id INTEGER NOT NULL,
        battery_soc_percent REAL NOT NULL,
        battery_charge_kwh REAL NOT NULL,
        battery_available_kwh REAL NOT NULL,
        battery_capacity_remaining_kwh REAL NOT NULL,
        battery_soh_percent REAL NOT NULL,
        battery_count INTEGER NOT NULL,
        total_capacity_kwh REAL NOT NULL,
        max_charge_rate_kw REAL NOT NULL,
        max_discharge_rate_kw REAL NOT NULL,
        action TEXT,
        trade_amount_kwh REAL,
        price_per_kwh REAL,
        consumption_kwh REAL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_house_id ON {table_name}(house_id);
    """
    
    print(f"\n  SQL to create table {table_name}:")
    print("  " + "-" * 68)
    print(create_table_sql)
    print("  " + "-" * 68)
    print(f"  ⚠ Please run this SQL in Supabase SQL Editor to create the table")


def upload_battery_data_to_supabase(
    df: pd.DataFrame,
    house_id: int,
    connector: SupabaseConnector,
    batch_size: int = 100
):
    """
    Upload battery data to Supabase.
    
    Args:
        df: DataFrame with battery data
        house_id: House ID
        connector: SupabaseConnector instance
        batch_size: Number of records to upload per batch
    """
    table_name = f"house{house_id}_battery"
    
    print(f"\n  Uploading to {table_name}...")
    
    # Convert DataFrame to list of dicts
    records = df.to_dict('records')
    
    # Convert timestamps to ISO format strings
    for record in records:
        if isinstance(record['timestamp'], pd.Timestamp):
            record['timestamp'] = record['timestamp'].isoformat()
    
    # Upload in batches
    total_uploaded = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        
        try:
            response = connector.client.table(table_name).insert(batch).execute()
            total_uploaded += len(batch)
            print(f"    Uploaded {total_uploaded}/{len(records)} records...", end='\r')
        except Exception as e:
            print(f"\n    ⚠ Error uploading batch {i//batch_size + 1}: {str(e)}")
            if "does not exist" in str(e):
                print(f"    ⚠ Table {table_name} does not exist. Please create it first.")
                create_battery_table_schema(connector, house_id)
                return False
    
    print(f"\n    ✓ Successfully uploaded {total_uploaded} records to {table_name}")
    return True


def main():
    """Generate and upload battery data for all houses."""
    print("="*70)
    print("GENERATE REALISTIC BATTERY DATA FOR SUPABASE")
    print("="*70)
    
    # Initialize connector
    print("\n1. Connecting to Supabase...")
    connector = SupabaseConnector()
    print("   ✓ Connected")
    
    # Load consumption and pricing data
    print("\n2. Loading consumption and pricing data...")
    consumption_df = load_consumption_data()
    pricing_df = load_pricing_data()
    print(f"   ✓ Consumption: {len(consumption_df):,} records")
    print(f"   ✓ Pricing: {len(pricing_df):,} records")
    
    # Get unique houses
    house_ids = sorted(consumption_df['house_id'].unique())
    print(f"   ✓ Houses: {house_ids}")
    
    # Generate battery data for each house
    print("\n3. Generating intelligent battery data...")
    
    for house_id in house_ids:
        # Determine battery capacity
        if house_id <= 9:
            capacity = 40.0  # 1 battery
        else:
            capacity = 80.0  # 2 batteries
        
        print(f"\n  House {house_id}: {capacity} kWh capacity")
        
        # Generate battery data
        generator = BatteryDataGenerator(capacity_kwh=capacity)
        battery_df = generator.simulate_intelligent_trading(
            consumption_df,
            pricing_df,
            house_id
        )
        
        if len(battery_df) == 0:
            print(f"    ⚠ Skipping house {house_id} - no data")
            continue
        
        # Ask user if they want to upload
        print(f"\n  Ready to upload {len(battery_df):,} records for house {house_id}")
        response = input(f"  Upload to Supabase? (y/n/skip remaining): ").strip().lower()
        
        if response == 'skip remaining':
            print("\n  Skipping remaining houses...")
            break
        elif response != 'y':
            print(f"  Skipped house {house_id}")
            continue
        
        # Upload to Supabase
        success = upload_battery_data_to_supabase(
            battery_df,
            house_id,
            connector,
            batch_size=100
        )
        
        if not success:
            print(f"\n  ⚠ Upload failed for house {house_id}")
            print(f"  Please create the table first using the SQL above")
            break
    
    print("\n" + "="*70)
    print("✅ BATTERY DATA GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
