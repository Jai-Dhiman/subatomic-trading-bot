"""
Synthetic Data Generator for Transformer Pre-Flight Testing.

Generates realistic household energy consumption data with:
- 9 appliance consumption patterns
- 4 battery sensor readings
- 2 weather features
- Pricing data with daily cycles

This synthetic data mimics real patterns for architecture verification
before real data becomes available from Supabase.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


def generate_pricing_data(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate realistic electricity pricing with daily cycles.
    
    Pattern:
    - Low: 2-5am ($0.25-0.30/kWh)
    - Peak morning: 8-10am ($0.45-0.55/kWh)
    - Peak evening: 6-9pm ($0.48-0.58/kWh)
    - Mid-day: moderate ($0.35-0.40/kWh)
    
    Args:
        timestamps: DatetimeIndex for the data
        
    Returns:
        Array of prices per kWh
    """
    n_samples = len(timestamps)
    hours = timestamps.hour + timestamps.minute / 60.0
    
    # Base sinusoidal pattern (2 peaks per day)
    base_pattern = 0.35 + 0.15 * np.sin(2 * np.pi * hours / 24)
    
    # Morning peak (8-10am)
    morning_peak = 0.15 * np.exp(-((hours - 9) ** 2) / 2)
    
    # Evening peak (6-9pm) - stronger than morning
    evening_peak = 0.20 * np.exp(-((hours - 19.5) ** 2) / 2)
    
    # Night discount (2-5am)
    night_discount = -0.10 * np.exp(-((hours - 3.5) ** 2) / 2)
    
    # Combine patterns
    prices = base_pattern + morning_peak + evening_peak + night_discount
    
    # Add small random noise
    noise = np.random.randn(n_samples) * 0.02
    prices = prices + noise
    
    # Ensure prices stay positive and realistic
    prices = np.clip(prices, 0.10, 0.65)
    
    return prices


def generate_appliance_fridge(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Fridge: constant ~0.15 kW with small variations."""
    n_samples = len(timestamps)
    base = 0.15
    noise = np.random.randn(n_samples) * 0.02
    return np.clip(base + noise, 0.10, 0.20)


def generate_appliance_washing_machine(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Washing machine: 2-3 cycles per week, 1.5 kW for 90 min."""
    n_samples = len(timestamps)
    consumption = np.zeros(n_samples)
    
    # Random cycles (2-3 per week)
    n_cycles = np.random.randint(2, 4)
    cycle_duration = 3  # 3 intervals = 90 min
    
    for _ in range(n_cycles):
        start_idx = np.random.randint(0, n_samples - cycle_duration)
        consumption[start_idx:start_idx + cycle_duration] = 1.5
    
    return consumption


def generate_appliance_dishwasher(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Dishwasher: daily at 7pm, 1.8 kW for 120 min."""
    n_samples = len(timestamps)
    consumption = np.zeros(n_samples)
    
    # Find 7pm slots (19:00)
    hours = timestamps.hour
    for i in range(n_samples):
        if hours[i] == 19:  # 7pm
            # 4 intervals = 120 min
            end_idx = min(i + 4, n_samples)
            consumption[i:end_idx] = 1.8
    
    return consumption


def generate_appliance_ev_charging(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """EV charging: nightly 11pm-6am, 7.4 kW."""
    n_samples = len(timestamps)
    consumption = np.zeros(n_samples)
    
    hours = timestamps.hour
    # Charge between 11pm (23) and 6am (6)
    night_hours = (hours >= 23) | (hours < 6)
    consumption[night_hours] = 7.4
    
    return consumption


def generate_appliance_ac(timestamps: pd.DatetimeIndex, temperature: np.ndarray) -> np.ndarray:
    """AC: peaks 12pm-6pm, 2-3.5 kW (temperature dependent)."""
    n_samples = len(timestamps)
    consumption = np.zeros(n_samples)
    
    hours = timestamps.hour
    
    # AC runs when hot (temp > 24°C) and during day
    hot_hours = (hours >= 12) & (hours < 18)
    
    # Base AC consumption proportional to temperature
    for i in range(n_samples):
        if hot_hours[i] and temperature[i] > 24:
            # Scale with temperature: hotter = more AC
            intensity = (temperature[i] - 24) / 6  # normalize to 0-1
            consumption[i] = 2.0 + intensity * 1.5  # 2-3.5 kW range
    
    return consumption


def generate_appliance_stove(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Stove: peaks at meal times, 2.5 kW."""
    n_samples = len(timestamps)
    consumption = np.zeros(n_samples)
    
    hours = timestamps.hour
    
    # Breakfast (7-8am), Lunch (12-1pm), Dinner (6-7pm)
    meal_times = ((hours >= 7) & (hours < 8)) | \
                 ((hours >= 12) & (hours < 13)) | \
                 ((hours >= 18) & (hours < 19))
    
    # Random usage during meal times (not every interval)
    for i in range(n_samples):
        if meal_times[i] and np.random.rand() > 0.3:
            consumption[i] = 2.5
    
    return consumption


def generate_appliance_water_heater(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Water heater: morning (6-8am) and evening (6-9pm) peaks, 3 kW."""
    n_samples = len(timestamps)
    consumption = np.zeros(n_samples)
    
    hours = timestamps.hour
    
    # Morning peak (6-8am)
    morning = (hours >= 6) & (hours < 8)
    # Evening peak (6-9pm)
    evening = (hours >= 18) & (hours < 21)
    
    peak_times = morning | evening
    consumption[peak_times] = 3.0
    
    return consumption


def generate_appliance_computers(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Computers: business hours 9am-6pm, 0.3 kW."""
    n_samples = len(timestamps)
    consumption = np.zeros(n_samples)
    
    hours = timestamps.hour
    day_of_week = timestamps.dayofweek  # 0=Monday, 6=Sunday
    
    # Business hours on weekdays
    business_hours = (hours >= 9) & (hours < 18) & (day_of_week < 5)
    consumption[business_hours] = 0.3
    
    return consumption


def generate_appliance_misc(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Misc appliances: baseline 0.1-0.2 kW + noise."""
    n_samples = len(timestamps)
    base = 0.15
    noise = np.random.randn(n_samples) * 0.03
    return np.clip(base + noise, 0.10, 0.25)


def generate_battery_sensors(timestamps: pd.DatetimeIndex, total_consumption: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic battery sensor readings.
    
    Battery cycles between 20-80% SoC, charges/discharges based on consumption patterns.
    
    Returns:
        Tuple of (soc_percent, soh_percent, charge_kwh, cycle_count)
    """
    n_samples = len(timestamps)
    capacity = 13.5  # kWh (Tesla Powerwall equivalent)
    
    # Initialize
    soc = np.zeros(n_samples)
    soh = np.zeros(n_samples)
    charge_kwh = np.zeros(n_samples)
    cycle_count = np.zeros(n_samples)
    
    # Starting values
    current_soc = 50.0  # Start at 50%
    current_soh = 100.0
    current_cycles = 0
    
    hours = timestamps.hour
    
    for i in range(n_samples):
        # Charge during cheap hours (11pm-6am)
        if (hours[i] >= 23) or (hours[i] < 6):
            # Charge towards 80%
            if current_soc < 80:
                current_soc = min(80, current_soc + 2.0)  # Charge 2% per interval
        # Discharge during peak hours (6-9pm) or high consumption
        elif (hours[i] >= 18) and (hours[i] < 21):
            # Discharge towards 20%
            if current_soc > 20:
                discharge_rate = min(3.0, current_soc - 20)
                current_soc -= discharge_rate
        
        # Update cycle count (increment when crossing 50% threshold)
        if i > 0 and ((soc[i-1] < 50 and current_soc >= 50) or (soc[i-1] > 50 and current_soc <= 50)):
            current_cycles += 0.5  # Half cycle
        
        # Degrade health slowly (0.001% per day)
        current_soh -= 0.001 / 48  # Per interval
        
        # Store values
        soc[i] = current_soc
        soh[i] = current_soh
        charge_kwh[i] = (current_soc / 100) * capacity
        cycle_count[i] = current_cycles
    
    return soc, soh, charge_kwh, cycle_count


def generate_weather_data(timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic weather data.
    
    Returns:
        Tuple of (temperature in °C, solar_irradiance in W/m²)
    """
    n_samples = len(timestamps)
    hours = timestamps.hour + timestamps.minute / 60.0
    
    # Temperature: daily cycle 15-30°C, peaks at 2pm
    temp_base = 22.5  # Average temp
    temp_amplitude = 7.5  # +/- range
    # Peak at 2pm (14:00)
    temperature = temp_base + temp_amplitude * np.sin(2 * np.pi * (hours - 6) / 24)
    
    # Add some random variation
    temperature += np.random.randn(n_samples) * 1.5
    temperature = np.clip(temperature, 10, 35)
    
    # Solar irradiance: 0 at night, peaks at noon (800 W/m²)
    solar = np.zeros(n_samples)
    # Sun is up roughly 6am-6pm
    day_hours = (hours >= 6) & (hours < 18)
    
    for i in range(n_samples):
        if day_hours[i]:
            # Bell curve peaking at noon
            time_from_noon = abs(hours[i] - 12)
            solar[i] = 800 * np.exp(-(time_from_noon ** 2) / 8)
    
    # Add small random variation
    solar += np.random.randn(n_samples) * 20
    solar = np.clip(solar, 0, 900)
    
    return temperature, solar


def generate_synthetic_household_data(days: int = 10, start_date: str = "2024-10-01") -> pd.DataFrame:
    """
    Generate complete synthetic household data for Transformer pre-flight testing.
    
    Args:
        days: Number of days to generate (default 10)
        start_date: Starting date for data generation
        
    Returns:
        DataFrame with all 24 features:
        - timestamp
        - 9 appliance consumption columns
        - 4 battery sensor columns
        - 2 weather columns
        - 1 pricing column
    """
    # Generate timestamps (30-minute intervals)
    n_samples = days * 48  # 48 intervals per day
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='30min')
    
    print(f"Generating {days} days of synthetic data ({n_samples} samples)...")
    
    # Generate weather data first (needed for AC)
    temperature, solar_irradiance = generate_weather_data(timestamps)
    
    # Generate appliance consumption
    print("  Generating appliance patterns...")
    fridge = generate_appliance_fridge(timestamps)
    washing_machine = generate_appliance_washing_machine(timestamps)
    dishwasher = generate_appliance_dishwasher(timestamps)
    ev_charging = generate_appliance_ev_charging(timestamps)
    ac = generate_appliance_ac(timestamps, temperature)
    stove = generate_appliance_stove(timestamps)
    water_heater = generate_appliance_water_heater(timestamps)
    computers = generate_appliance_computers(timestamps)
    misc = generate_appliance_misc(timestamps)
    
    # Total consumption for battery simulation
    total_consumption = fridge + washing_machine + dishwasher + ev_charging + \
                       ac + stove + water_heater + computers + misc
    
    # Generate battery sensors
    print("  Generating battery sensor data...")
    soc, soh, charge_kwh, cycle_count = generate_battery_sensors(timestamps, total_consumption)
    
    # Generate pricing
    print("  Generating pricing data...")
    prices = generate_pricing_data(timestamps)
    
    # Create DataFrame
    df = pd.DataFrame({
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
    })
    
    # Add total consumption for reference
    df['total_consumption_kwh'] = total_consumption
    
    print(f"  Generated {len(df.columns)} columns")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def validate_synthetic_data(df: pd.DataFrame) -> bool:
    """
    Validate that synthetic data has expected properties.
    
    Returns:
        True if validation passes
    """
    print("\nValidating synthetic data...")
    
    # Check columns
    expected_appliances = [
        'appliance_fridge', 'appliance_washing_machine', 'appliance_dishwasher',
        'appliance_ev_charging', 'appliance_ac', 'appliance_stove',
        'appliance_water_heater', 'appliance_computers', 'appliance_misc'
    ]
    expected_battery = [
        'battery_soc_percent', 'battery_soh_percent', 
        'battery_charge_kwh', 'battery_cycle_count'
    ]
    expected_weather = ['temperature', 'solar_irradiance']
    expected_pricing = ['price_per_kwh']
    
    all_expected = ['timestamp'] + expected_appliances + expected_battery + expected_weather + expected_pricing
    
    missing = [col for col in all_expected if col not in df.columns]
    if missing:
        print(f"  ❌ Missing columns: {missing}")
        return False
    
    print(f"  ✓ All {len(all_expected)} expected columns present")
    
    # Check for NaNs
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"  ❌ NaN values found in: {nan_cols}")
        return False
    
    print("  ✓ No NaN values")
    
    # Check value ranges
    checks = [
        ('battery_soc_percent', 0, 100),
        ('battery_soh_percent', 99, 101),
        ('temperature', 10, 35),
        ('solar_irradiance', 0, 900),
        ('price_per_kwh', 0.10, 0.70),
    ]
    
    for col, min_val, max_val in checks:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min < min_val or col_max > max_val:
            print(f"  ❌ {col}: range [{col_min:.2f}, {col_max:.2f}] outside expected [{min_val}, {max_val}]")
            return False
    
    print("  ✓ All value ranges realistic")
    
    # Check timestamp continuity
    time_diffs = df['timestamp'].diff()[1:]
    expected_diff = pd.Timedelta('30min')
    if not all(time_diffs == expected_diff):
        print(f"  ❌ Timestamps not continuous (30-min intervals)")
        return False
    
    print("  ✓ Timestamps are continuous (30-min intervals)")
    
    # Check patterns exist
    ev_charging_night = df[(df['timestamp'].dt.hour >= 23) | (df['timestamp'].dt.hour < 6)]['appliance_ev_charging']
    if ev_charging_night.mean() < 5.0:
        print(f"  ⚠️  Warning: EV charging pattern weak (night avg: {ev_charging_night.mean():.2f} kW)")
    else:
        print(f"  ✓ EV charging pattern visible (night avg: {ev_charging_night.mean():.2f} kW)")
    
    print("\n✅ Validation passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Synthetic Data Generator - Transformer Pre-Flight")
    print("=" * 60)
    
    # Generate data
    df = generate_synthetic_household_data(days=10)
    
    # Validate
    valid = validate_synthetic_data(df)
    
    if valid:
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 60)
        
        print("\nAppliance Consumption (kW):")
        appliance_cols = [col for col in df.columns if col.startswith('appliance_')]
        for col in appliance_cols:
            mean_val = df[col].mean()
            max_val = df[col].max()
            print(f"  {col:30s}: mean={mean_val:.3f}, max={max_val:.3f}")
        
        print(f"\nTotal Consumption:")
        print(f"  Mean: {df['total_consumption_kwh'].mean():.2f} kW")
        print(f"  Max:  {df['total_consumption_kwh'].max():.2f} kW")
        print(f"  Min:  {df['total_consumption_kwh'].min():.2f} kW")
        
        print(f"\nBattery:")
        print(f"  SoC range: {df['battery_soc_percent'].min():.1f}% - {df['battery_soc_percent'].max():.1f}%")
        print(f"  SoH: {df['battery_soh_percent'].iloc[-1]:.3f}%")
        print(f"  Cycles: {df['battery_cycle_count'].iloc[-1]:.1f}")
        
        print(f"\nWeather:")
        print(f"  Temperature: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
        print(f"  Solar: {df['solar_irradiance'].min():.1f} - {df['solar_irradiance'].max():.1f} W/m²")
        
        print(f"\nPricing:")
        print(f"  Range: ${df['price_per_kwh'].min():.3f} - ${df['price_per_kwh'].max():.3f}/kWh")
        print(f"  Mean: ${df['price_per_kwh'].mean():.3f}/kWh")
        
        print("\n" + "=" * 60)
        print("✅ Synthetic data generator working correctly!")
        print(f"✅ Generated DataFrame with shape: {df.shape}")
        print(f"✅ Ready for feature engineering (Task 2)")
        print("=" * 60)
    else:
        print("\n❌ Validation failed - check the issues above")
