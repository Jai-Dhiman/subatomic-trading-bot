"""
Comprehensive Synthetic Data Generator for Supabase.

Generates realistic data following business rules:
- Household consumption patterns (realistic appliance usage)
- Market pricing with volatility
- Intelligent battery trading (buy <$20/MWh, sell >$40/MWh)
- Multiple months of historical data

Business Model:
- Company charges households $270/MWh ($0.27/kWh)
- Buy from market when < $20/MWh
- Sell to market when > $40/MWh
- Maximize profit while meeting household demand
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_integration.supabase_connector import SupabaseConnector
from src.models.trading_optimizer import calculate_optimal_trading_decisions


class SyntheticDataGenerator:
    """Generate realistic synthetic data for Supabase."""
    
    def __init__(self, num_houses: int = 11, num_days: int = 90):
        """
        Initialize data generator.
        
        Args:
            num_houses: Number of houses to generate data for (default 11)
            num_days: Number of days of historical data (default 90)
        """
        self.num_houses = num_houses
        self.num_days = num_days
        self.intervals_per_day = 24 * 2  # 30-min intervals
        self.total_intervals = num_days * self.intervals_per_day
        
        # House configurations
        self.house_configs = self._create_house_configs()
        
    def _create_house_configs(self) -> Dict:
        """Create configuration for each house (occupancy, appliances, etc.)."""
        configs = {}
        
        for house_id in range(1, self.num_houses + 1):
            # Battery capacity: 40 kWh for houses 1-9, 80 kWh for houses 10-11
            battery_capacity = 80.0 if house_id >= 10 else 40.0
            
            # Random household characteristics
            occupancy = np.random.randint(1, 5)  # 1-4 people
            has_ev = np.random.random() > 0.5
            ac_usage = np.random.choice(['high', 'medium', 'low'])
            
            configs[house_id] = {
                'occupancy': occupancy,
                'has_ev': has_ev,
                'ac_usage': ac_usage,
                'battery_capacity_kwh': battery_capacity,
                'base_consumption': 0.5 + (occupancy * 0.3),  # Base load per 30-min
            }
        
        return configs
    
    def generate_consumption_data(self) -> pd.DataFrame:
        """
        Generate realistic household consumption data.
        
        Returns:
            DataFrame with consumption data for all houses
        """
        print(f"\nGenerating consumption data for {self.num_houses} houses, {self.num_days} days...")
        
        # Create timestamps
        start_date = datetime.now() - timedelta(days=self.num_days)
        timestamps = pd.date_range(start=start_date, periods=self.total_intervals, freq='30min')
        
        all_data = []
        
        for house_id in range(1, self.num_houses + 1):
            config = self.house_configs[house_id]
            
            print(f"  House {house_id}: {config['occupancy']} occupants, " +
                  f"EV={'Yes' if config['has_ev'] else 'No'}, " +
                  f"AC={config['ac_usage']}")
            
            for i, timestamp in enumerate(timestamps):
                hour = timestamp.hour
                day_of_week = timestamp.dayofweek
                is_weekend = day_of_week >= 5
                
                # Generate appliance consumption based on time of day
                appliances = self._generate_appliance_consumption(
                    hour, is_weekend, config
                )
                
                all_data.append({
                    'House': house_id,
                    'Timestamp': timestamp,
                    'Total kWh': sum(appliances.values()),
                    'Appliance_Breakdown_JSON': str(appliances).replace("'", '"')
                })
        
        df = pd.DataFrame(all_data)
        print(f"  ✓ Generated {len(df):,} consumption records")
        return df
    
    def _generate_appliance_consumption(
        self, hour: int, is_weekend: bool, config: Dict
    ) -> Dict[str, float]:
        """Generate realistic appliance consumption for a 30-min interval."""
        
        base = config['base_consumption']
        occupancy = config['occupancy']
        
        # Time-based patterns
        is_morning = 6 <= hour < 9
        is_evening = 17 <= hour < 22
        is_night = hour < 6 or hour >= 23
        is_midday = 11 <= hour < 14
        
        # Base consumption (always on)
        standby = 0.1 + np.random.uniform(0, 0.05)
        
        # Refrigerator (always on with cycles)
        refrig = 0.08 + np.random.uniform(0, 0.04)
        
        # A/C (seasonal and time-based)
        ac_map = {'high': 1.5, 'medium': 0.8, 'low': 0.3}
        ac_factor = ac_map[config['ac_usage']]
        if 10 <= hour <= 18:  # Peak cooling
            ac = ac_factor * (0.5 + np.random.uniform(0, 0.5))
        else:
            ac = ac_factor * (0.1 + np.random.uniform(0, 0.2))
        
        # Water heater (morning and evening peaks)
        if is_morning or is_evening:
            water_heater = 0.4 + np.random.uniform(0, 0.3)
        else:
            water_heater = 0.1 + np.random.uniform(0, 0.1)
        
        # Cooking (stovetop)
        if is_morning or is_evening or (is_midday and is_weekend):
            stove = np.random.uniform(0, 0.6) if np.random.random() > 0.5 else 0
        else:
            stove = 0
        
        # Washing/Drying (mainly daytime)
        if (10 <= hour <= 20) and (is_weekend or np.random.random() > 0.7):
            washing = np.random.uniform(0, 1.0) if np.random.random() > 0.8 else 0
        else:
            washing = 0
        
        # Dishwasher (after meals)
        if (hour in [8, 9, 19, 20, 21]) and np.random.random() > 0.6:
            dishwasher = np.random.uniform(0, 0.8)
        else:
            dishwasher = 0
        
        # Computers (work hours + evening)
        if (8 <= hour <= 23) and not is_night:
            computers = 0.15 * occupancy + np.random.uniform(0, 0.2)
        else:
            computers = 0.05 + np.random.uniform(0, 0.05)
        
        # EV Charging (overnight, if has EV)
        if config['has_ev'] and (0 <= hour <= 6):
            ev_charging = np.random.uniform(2.0, 4.0) if np.random.random() > 0.5 else 0
        else:
            ev_charging = 0
        
        return {
            'A/C': round(ac, 3),
            'Washing/Drying': round(washing, 3),
            'Refrig.': round(refrig, 3),
            'EV Charging': round(ev_charging, 3),
            'DishWasher': round(dishwasher, 3),
            'Computers': round(computers, 3),
            'Stovetop': round(stove, 3),
            'Water Heater': round(water_heater, 3),
            'Standby/ Misc.': round(standby, 3)
        }
    
    def generate_pricing_data(self) -> pd.DataFrame:
        """
        Generate realistic market pricing data with volatility.
        
        Prices range from $10-$100/MWh ($0.010-$0.100/kWh)
        Following typical electricity market patterns
        
        Returns:
            DataFrame with pricing data
        """
        print(f"\nGenerating market pricing data for {self.num_days} days...")
        
        start_date = datetime.now() - timedelta(days=self.num_days)
        timestamps = pd.date_range(start=start_date, periods=self.total_intervals, freq='30min')
        
        prices = []
        base_price = 35.0  # $35/MWh base
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            
            # Time-based patterns
            # Peak hours (high prices): 16-21
            # Off-peak (low prices): 0-6
            # Mid-range: rest of day
            
            if 16 <= hour <= 21:  # Peak hours
                hourly_factor = 1.5 + np.random.uniform(-0.2, 0.5)
            elif 0 <= hour <= 6:  # Off-peak
                hourly_factor = 0.5 + np.random.uniform(-0.2, 0.2)
            elif 11 <= hour <= 15:  # Mid-day (solar, lower prices)
                hourly_factor = 0.7 + np.random.uniform(-0.3, 0.2)
            else:
                hourly_factor = 1.0 + np.random.uniform(-0.2, 0.2)
            
            # Weekend factor (slightly lower)
            if day_of_week >= 5:
                hourly_factor *= 0.9
            
            # Add random market volatility
            volatility = np.random.normal(0, 0.15)
            
            # Calculate price (in $/MWh)
            price_mwh = base_price * hourly_factor * (1 + volatility)
            
            # Ensure within reasonable bounds: $10-$100/MWh
            price_mwh = np.clip(price_mwh, 10, 100)
            
            # Convert to $/kWh
            price_kwh = price_mwh / 1000.0
            
            prices.append({
                'INTERVALSTARTTIME_GMT': timestamp,
                'INTERVALENDTIME_GMT': timestamp + timedelta(minutes=30),
                'Price KWH': round(price_kwh, 6)
            })
        
        df = pd.DataFrame(prices)
        print(f"  ✓ Generated {len(df):,} pricing records")
        print(f"  ✓ Price range: ${df['Price KWH'].min():.4f} to ${df['Price KWH'].max():.4f} per kWh")
        print(f"  ✓ Price range: ${df['Price KWH'].min()*1000:.1f} to ${df['Price KWH'].max()*1000:.1f} per MWh")
        
        return df
    
    def generate_battery_data(
        self,
        consumption_df: pd.DataFrame,
        pricing_df: pd.DataFrame,
        house_id: int
    ) -> pd.DataFrame:
        """
        Generate intelligent battery trading data following business rules.
        
        Uses the trading optimizer with actual business rules:
        - Buy when price < $20/MWh
        - Sell when price > $40/MWh
        - Prioritize household demand
        - Maximize profit
        
        Args:
            consumption_df: Consumption data
            pricing_df: Pricing data
            house_id: House ID
        
        Returns:
            DataFrame with battery state data
        """
        config = self.house_configs[house_id]
        capacity_kwh = config['battery_capacity_kwh']
        
        print(f"  Simulating battery for House {house_id} ({capacity_kwh} kWh)...")
        
        # Filter to this house
        house_data = consumption_df[consumption_df['House'] == house_id].copy()
        house_data = house_data.sort_values('Timestamp').reset_index(drop=True)
        
        # Merge with pricing
        house_data = house_data.merge(
            pricing_df[['INTERVALSTARTTIME_GMT', 'Price KWH']],
            left_on='Timestamp',
            right_on='INTERVALSTARTTIME_GMT',
            how='left'
        )
        
        # Use trading optimizer to calculate optimal decisions
        consumption = house_data['Total kWh'].values
        prices = house_data['Price KWH'].values
        
        battery_state = {
            'current_charge_kwh': capacity_kwh * 0.50,  # Start at 50%
            'capacity_kwh': capacity_kwh,
            'min_soc': 0.20,
            'max_soc': 1.0,  # 100% per business rules
            'max_charge_rate_kw': 10.0,
            'max_discharge_rate_kw': 8.0,
            'efficiency': 0.95
        }
        
        # Calculate optimal trading strategy
        result = calculate_optimal_trading_decisions(
            predicted_consumption=consumption,
            actual_prices=prices,
            battery_state=battery_state,
            household_price_kwh=0.027,  # $27/MWh threshold for conditional trades
            buy_threshold_mwh=20.0,
            sell_threshold_mwh=40.0
        )
        
        # Create battery records
        battery_records = []
        current_charge = battery_state['current_charge_kwh']
        soh = 98.0  # State of health
        
        for i, row in house_data.iterrows():
            decision = result['optimal_decisions'][i]
            quantity = result['optimal_quantities'][i]
            
            # Map decision to action string
            action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
            action = action_map[decision]
            
            # Update charge based on decision
            if decision == 0:  # Buy
                current_charge += quantity * battery_state['efficiency']
            elif decision == 2:  # Sell
                current_charge -= quantity
            
            # Ensure within bounds
            current_soc = current_charge / capacity_kwh
            current_soc = np.clip(current_soc, battery_state['min_soc'], battery_state['max_soc'])
            current_charge = capacity_kwh * current_soc
            
            available_for_discharge = max(0, current_charge - (capacity_kwh * battery_state['min_soc']))
            available_for_charge = max(0, (capacity_kwh * battery_state['max_soc']) - current_charge)
            
            battery_records.append({
                'timestamp': row['Timestamp'],
                'house_id': house_id,
                'battery_soc_percent': current_soc * 100,
                'battery_charge_kwh': current_charge,
                'battery_available_kwh': available_for_discharge,
                'battery_capacity_remaining_kwh': available_for_charge,
                'battery_soh_percent': soh,
                'battery_count': 2 if house_id >= 10 else 1,
                'total_capacity_kwh': capacity_kwh,
                'max_charge_rate_kw': 10.0,
                'max_discharge_rate_kw': 8.0,
                'action': action,
                'trade_amount_kwh': quantity,
                'price_per_kwh': row['Price KWH'],
                'consumption_kwh': row['Total kWh']
            })
        
        df = pd.DataFrame(battery_records)
        
        # Print statistics
        buy_count = len(df[df['action'] == 'buy'])
        sell_count = len(df[df['action'] == 'sell'])
        hold_count = len(df[df['action'] == 'hold'])
        
        print(f"    ✓ Generated {len(df):,} battery states")
        print(f"    ✓ Actions: Buy={buy_count}, Sell={sell_count}, Hold={hold_count}")
        print(f"    ✓ SoC range: {df['battery_soc_percent'].min():.1f}% to {df['battery_soc_percent'].max():.1f}%")
        print(f"    ✓ Expected profit: ${result['expected_profit']:.2f}")
        
        return df


def upload_to_supabase(
    consumption_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    battery_dfs: Dict[int, pd.DataFrame],
    connector: SupabaseConnector,
    batch_size: int = 500
):
    """
    Upload all data to Supabase.
    
    Args:
        consumption_df: Consumption data
        pricing_df: Pricing data
        battery_dfs: Dict of battery dataframes by house_id
        connector: Supabase connector
        batch_size: Records per batch
    """
    print("\n" + "="*70)
    print("UPLOADING DATA TO SUPABASE")
    print("="*70)
    
    # 1. Upload consumption data
    print("\n1. Uploading consumption data...")
    try:
        # Clean data - replace NaN with None
        consumption_df = consumption_df.fillna(0)
        records = consumption_df.to_dict('records')
        for record in records:
            if isinstance(record['Timestamp'], pd.Timestamp):
                record['Timestamp'] = record['Timestamp'].isoformat()
            # Ensure numeric fields are proper floats
            if 'Total kWh' in record:
                record['Total kWh'] = float(record['Total kWh'])
        
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            connector.client.table('august11homeconsumption').insert(batch).execute()
            total += len(batch)
            print(f"   Uploaded {total}/{len(records)} records...", end='\r')
        
        print(f"\n   ✓ Uploaded {total:,} consumption records")
    except Exception as e:
        print(f"\n   ⚠ Error uploading consumption: {e}")
    
    # 2. Upload pricing data
    print("\n2. Uploading pricing data...")
    try:
        records = pricing_df.to_dict('records')
        for record in records:
            if isinstance(record['INTERVALSTARTTIME_GMT'], pd.Timestamp):
                record['INTERVALSTARTTIME_GMT'] = record['INTERVALSTARTTIME_GMT'].isoformat()
            if isinstance(record['INTERVALENDTIME_GMT'], pd.Timestamp):
                record['INTERVALENDTIME_GMT'] = record['INTERVALENDTIME_GMT'].isoformat()
        
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            connector.client.table('cabuyingpricehistoryseptember2025').insert(batch).execute()
            total += len(batch)
            print(f"   Uploaded {total}/{len(records)} records...", end='\r')
        
        print(f"\n   ✓ Uploaded {total:,} pricing records")
    except Exception as e:
        print(f"\n   ⚠ Error uploading pricing: {e}")
    
    # 3. Upload battery data for each house
    print("\n3. Uploading battery data...")
    for house_id, battery_df in battery_dfs.items():
        table_name = f"house{house_id}_battery"
        print(f"\n   House {house_id} -> {table_name}...")
        
        try:
            # Clean data - replace NaN/inf with 0
            battery_df = battery_df.replace([np.inf, -np.inf], 0)
            battery_df = battery_df.fillna(0)
            
            records = battery_df.to_dict('records')
            for record in records:
                if isinstance(record['timestamp'], pd.Timestamp):
                    record['timestamp'] = record['timestamp'].isoformat()
                
                # Ensure all numeric fields are proper floats/ints
                for key in ['battery_soc_percent', 'battery_charge_kwh', 'battery_available_kwh',
                           'battery_capacity_remaining_kwh', 'battery_soh_percent', 
                           'total_capacity_kwh', 'max_charge_rate_kw', 'max_discharge_rate_kw',
                           'trade_amount_kwh', 'price_per_kwh', 'consumption_kwh']:
                    if key in record:
                        record[key] = float(record[key]) if not pd.isna(record[key]) else 0.0
                
                # Ensure integer fields
                for key in ['house_id', 'battery_count']:
                    if key in record:
                        record[key] = int(record[key])
            
            total = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                connector.client.table(table_name).insert(batch).execute()
                total += len(batch)
                print(f"     Uploaded {total}/{len(records)} records...", end='\r')
            
            print(f"\n     ✓ Uploaded {total:,} records to {table_name}")
        except Exception as e:
            print(f"\n     ⚠ Error uploading to {table_name}: {e}")


def main():
    """Main execution function."""
    print("="*70)
    print("SYNTHETIC DATA GENERATOR FOR SUPABASE")
    print("="*70)
    
    # Configuration
    print("\nConfiguration:")
    num_houses = int(input("  Number of houses (default 11): ") or "11")
    num_days = int(input("  Number of days of historical data (default 90): ") or "90")
    
    print(f"\n  Houses: {num_houses}")
    print(f"  Days: {num_days}")
    print(f"  Total intervals: {num_days * 48:,} per house")
    print(f"  Total records: ~{num_houses * num_days * 48:,} across all tables")
    
    confirm = input("\n  Proceed with generation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Aborted.")
        return
    
    # Initialize generator
    generator = SyntheticDataGenerator(num_houses=num_houses, num_days=num_days)
    
    # Generate data
    print("\n" + "="*70)
    print("GENERATING DATA")
    print("="*70)
    
    # 1. Consumption
    consumption_df = generator.generate_consumption_data()
    
    # 2. Pricing
    pricing_df = generator.generate_pricing_data()
    
    # 3. Battery data for each house
    print(f"\nGenerating battery data for {num_houses} houses...")
    battery_dfs = {}
    for house_id in range(1, num_houses + 1):
        battery_dfs[house_id] = generator.generate_battery_data(
            consumption_df, pricing_df, house_id
        )
    
    # Ask to upload
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated:")
    print(f"  - {len(consumption_df):,} consumption records")
    print(f"  - {len(pricing_df):,} pricing records")
    print(f"  - {sum(len(df) for df in battery_dfs.values()):,} battery records ({num_houses} houses)")
    
    upload = input("\n  Upload to Supabase? (y/n): ").strip().lower()
    if upload != 'y':
        print("  Skipping upload. Data generated but not uploaded.")
        return
    
    # Connect and upload
    print("\nConnecting to Supabase...")
    connector = SupabaseConnector()
    print("  ✓ Connected")
    
    upload_to_supabase(consumption_df, pricing_df, battery_dfs, connector)
    
    print("\n" + "="*70)
    print("✅ DATA SEEDING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
