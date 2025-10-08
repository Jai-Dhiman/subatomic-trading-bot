"""
Generate synthetic training data for multiple households.

Creates varied consumption patterns, battery configurations, and behaviors
across 10 households for realistic demo.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_integration.synthetic_data_generator import (
    generate_appliance_fridge,
    generate_appliance_washing_machine,
    generate_appliance_dishwasher,
    generate_appliance_ev_charging,
    generate_appliance_ac,
    generate_appliance_stove,
    generate_appliance_water_heater,
    generate_appliance_computers,
    generate_appliance_misc,
    generate_battery_sensors,
    generate_weather_data,
    generate_pricing_data
)


def generate_household_data(household_id: int, days: int, start_date: str, base_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data for a single household with unique characteristics.
    
    Args:
        household_id: Unique household identifier (1-10)
        days: Number of days to generate
        start_date: Starting date
        base_seed: Base random seed for reproducibility
        
    Returns:
        DataFrame with all features for this household
    """
    # Set seed based on household ID for reproducible but varied patterns
    np.random.seed(base_seed + household_id)
    
    n_samples = days * 48
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='30min')
    
    # Generate weather (same for all households in same location)
    np.random.seed(base_seed)  # Same weather for everyone
    temperature, solar_irradiance = generate_weather_data(timestamps)
    
    # Generate pricing (same market for all)
    prices = generate_pricing_data(timestamps)
    
    # Reset to household-specific seed
    np.random.seed(base_seed + household_id)
    
    # Household-specific scaling factors (20% variation)
    consumption_scale = 0.9 + (household_id % 5) * 0.05  # 0.9 to 1.1
    ev_ownership = household_id <= 7  # 70% have EVs
    
    # Generate appliance consumption with household variation
    fridge = generate_appliance_fridge(timestamps) * consumption_scale
    washing_machine = generate_appliance_washing_machine(timestamps) * consumption_scale
    dishwasher = generate_appliance_dishwasher(timestamps) * consumption_scale
    
    if ev_ownership:
        ev_charging = generate_appliance_ev_charging(timestamps) * (0.8 + household_id * 0.03)
    else:
        ev_charging = np.zeros(n_samples)
    
    ac = generate_appliance_ac(timestamps, temperature) * consumption_scale
    stove = generate_appliance_stove(timestamps) * consumption_scale
    water_heater = generate_appliance_water_heater(timestamps) * consumption_scale
    computers = generate_appliance_computers(timestamps) * (0.5 + household_id * 0.1)
    misc = generate_appliance_misc(timestamps) * consumption_scale
    
    # Total consumption for battery simulation
    total_consumption = (fridge + washing_machine + dishwasher + ev_charging + 
                        ac + stove + water_heater + computers + misc)
    
    # Generate battery sensors
    soc, soh, charge_kwh, cycle_count = generate_battery_sensors(timestamps, total_consumption)
    
    # Create DataFrame
    df = pd.DataFrame({
        'household_id': household_id,
        'timestamp': timestamps,
        
        # Appliances (9)
        'appliance_fridge': fridge,
        'appliance_washing_machine': washing_machine,
        'appliance_dishwasher': dishwasher,
        'appliance_ev_charging': ev_charging,
        'appliance_ac': ac,
        'appliance_stove': stove,
        'appliance_water_heater': water_heater,
        'appliance_computers': computers,
        'appliance_misc': misc,
        
        # Battery sensors (4)
        'battery_soc_percent': soc,
        'battery_soh_percent': soh,
        'battery_charge_kwh': charge_kwh,
        'battery_cycle_count': cycle_count,
        
        # Weather (2)
        'temperature': temperature,
        'solar_irradiance': solar_irradiance,
        
        # Pricing (1)
        'price_per_kwh': prices,
        
        # Total consumption for reference
        'total_consumption_kwh': total_consumption
    })
    
    return df


def generate_multi_household_dataset(
    num_households: int = 10,
    days: int = 10,
    start_date: str = "2024-10-01"
) -> dict:
    """
    Generate complete dataset for multiple households.
    
    Args:
        num_households: Number of households to generate
        days: Days of historical data per household
        start_date: Starting date
        
    Returns:
        Dictionary with household data and metadata
    """
    print("=" * 60)
    print(f"Generating Multi-Household Training Data")
    print("=" * 60)
    print(f"  Households: {num_households}")
    print(f"  Days per household: {days}")
    print(f"  Samples per household: {days * 48}")
    print(f"  Start date: {start_date}")
    print()
    
    household_data = []
    
    for hh_id in range(1, num_households + 1):
        print(f"  Generating household {hh_id}/{num_households}...", end=" ")
        df = generate_household_data(hh_id, days, start_date)
        household_data.append(df)
        print(f"✓ ({len(df)} samples)")
    
    # Combine all households
    combined_df = pd.concat(household_data, ignore_index=True)
    
    print(f"\n✓ Generated {len(combined_df)} total samples")
    print(f"  ({num_households} households × {days * 48} samples)")
    
    # Calculate statistics
    print("\nDataset Statistics:")
    for hh_id in range(1, num_households + 1):
        hh_data = combined_df[combined_df['household_id'] == hh_id]
        avg_consumption = hh_data['total_consumption_kwh'].mean()
        has_ev = hh_data['appliance_ev_charging'].max() > 0
        print(f"  Household {hh_id}: avg={avg_consumption:.2f} kW, EV={'Yes' if has_ev else 'No'}")
    
    dataset = {
        'data': combined_df,
        'metadata': {
            'num_households': num_households,
            'days': days,
            'samples_per_household': days * 48,
            'total_samples': len(combined_df),
            'start_date': start_date,
            'features': [col for col in combined_df.columns if col not in ['household_id', 'timestamp', 'total_consumption_kwh']]
        }
    }
    
    return dataset


def save_dataset(dataset: dict, output_path: Path):
    """Save dataset to disk."""
    print(f"\nSaving dataset to {output_path}...")
    
    df = dataset['data']
    metadata = dataset['metadata']
    
    # Save as CSV for easy inspection
    csv_path = output_path.parent / f"{output_path.stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV: {csv_path}")
    
    # Save as compressed npz for fast loading
    np.savez_compressed(
        output_path,
        data=df.values,
        columns=df.columns.tolist(),
        **metadata
    )
    print(f"  ✓ Saved NPZ: {output_path}")
    
    print(f"\n✅ Dataset generation complete!")
    

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_multi_household_dataset(
        num_households=10,
        days=10,
        start_date="2024-10-01"
    )
    
    # Save to disk
    output_path = Path(__file__).parent.parent.parent / "data" / "demo" / "multi_household_training_data.npz"
    save_dataset(dataset, output_path)
