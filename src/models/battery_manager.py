"""
Battery management system for household energy storage.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class BatteryState:
    """Represents the state of a household battery from sensor readings.
    
    All state values (SoC, SoH, current_charge, cycle_count) are READ from
    the database as reported by battery sensors. This class does NOT calculate
    or synthesize these values.
    """

    capacity_kwh: float
    usable_capacity_kwh: float
    current_charge_kwh: float
    efficiency: float
    max_charge_rate_kw: float
    max_discharge_rate_kw: float
    min_reserve_percent: float
    max_charge_percent: float
    soh_percent: float
    cycle_count: float

    def state_of_charge(self) -> float:
        """Return current state of charge as percentage (0-1) from sensor data."""
        return self.current_charge_kwh / self.usable_capacity_kwh
    
    def state_of_health(self) -> float:
        """Return current state of health as percentage (0-100) from sensor data."""
        return self.soh_percent
    
    def get_effective_capacity(self) -> float:
        """Return effective capacity adjusted for SoH (from sensor)."""
        return self.usable_capacity_kwh * (self.soh_percent / 100.0)
    
    def get_max_charge_kwh(self) -> float:
        """Return maximum allowed charge level (80% cap)."""
        return self.usable_capacity_kwh * self.max_charge_percent
    
    def validate_charge_cap(self):
        """Validate that current charge doesn't exceed 80% cap.
        
        Raises:
            ValueError: If sensor reports charge above 80% cap (sensor error)
        """
        max_charge = self.get_max_charge_kwh()
        if self.current_charge_kwh > max_charge:
            raise ValueError(
                f"Battery charge cap violation detected in sensor data: "
                f"current charge {self.current_charge_kwh:.2f} kWh "
                f"exceeds maximum allowed {max_charge:.2f} kWh (80% of {self.usable_capacity_kwh:.2f} kWh). "
                f"Check battery sensor readings or BMS configuration."
            )

    def available_capacity(self) -> float:
        """Return available space for charging in kWh (respecting 80% cap)."""
        max_charge = self.get_max_charge_kwh()
        return max(0.0, max_charge - self.current_charge_kwh)

    def available_energy(self) -> float:
        """Return available energy for discharging (above reserve) in kWh."""
        reserve = self.usable_capacity_kwh * self.min_reserve_percent
        return max(0, self.current_charge_kwh - reserve)

    def calculate_charge_limit(self, energy_kwh: float, interval_hours: float = 0.5) -> float:
        """
        Calculate how much energy can be charged respecting constraints.
        
        NOTE: This is for planning/validation only. Actual charge/SoC/SoH 
        comes from battery sensor readings in the database.

        Args:
            energy_kwh: Requested energy to store
            interval_hours: Time interval (default 0.5 for 30-min)

        Returns:
            Maximum energy that can be charged respecting rate and capacity limits
        """
        max_charge_this_interval = self.max_charge_rate_kw * interval_hours

        energy_to_charge = min(
            energy_kwh, 
            self.available_capacity(), 
            max_charge_this_interval
        )

        actual_stored = energy_to_charge * self.efficiency
        return actual_stored

    def calculate_discharge_limit(self, energy_kwh: float, interval_hours: float = 0.5) -> float:
        """
        Calculate how much energy can be discharged respecting constraints.
        
        NOTE: This is for planning/validation only. Actual discharge/SoC/SoH 
        comes from battery sensor readings in the database.

        Args:
            energy_kwh: Requested energy to deliver
            interval_hours: Time interval (default 0.5 for 30-min)

        Returns:
            Maximum energy that can be discharged respecting rate and reserve limits
        """
        max_discharge_this_interval = self.max_discharge_rate_kw * interval_hours
        energy_available = self.available_energy()

        energy_to_discharge = min(energy_kwh, energy_available, max_discharge_this_interval)
        actual_delivered = energy_to_discharge * self.efficiency
        
        return actual_delivered

    def get_stats(self) -> dict:
        """Return comprehensive battery statistics."""
        return {
            "capacity_kwh": self.capacity_kwh,
            "usable_capacity_kwh": self.usable_capacity_kwh,
            "current_charge_kwh": self.current_charge_kwh,
            "state_of_charge_percent": self.state_of_charge() * 100,
            "state_of_health_percent": self.state_of_health(),
            "cycle_count": self.cycle_count,
            "effective_capacity_kwh": self.get_effective_capacity(),
            "max_charge_allowed_kwh": self.get_max_charge_kwh(),
            "available_capacity_kwh": self.available_capacity(),
            "available_energy_kwh": self.available_energy(),
        }

    def copy(self):
        """Create a copy of the battery state."""
        return BatteryState(
            capacity_kwh=self.capacity_kwh,
            usable_capacity_kwh=self.usable_capacity_kwh,
            current_charge_kwh=self.current_charge_kwh,
            efficiency=self.efficiency,
            max_charge_rate_kw=self.max_charge_rate_kw,
            max_discharge_rate_kw=self.max_discharge_rate_kw,
            min_reserve_percent=self.min_reserve_percent,
            max_charge_percent=self.max_charge_percent,
            soh_percent=self.soh_percent,
            cycle_count=self.cycle_count,
        )


class BatteryManager:
    """Manages battery operations and validation for a household.
    
    This manager works with REAL sensor data from the database.
    All battery state (SoC, SoH, cycle_count) is READ from sensors,
    NOT calculated or synthesized.
    """

    def __init__(self, config: dict, household_id: Optional[int] = None):
        """
        Initialize battery manager with static configuration from database.

        Args:
            config: Dictionary with battery specs (capacity, rates, efficiency)
            household_id: Household ID for reference
            
        Raises:
            ValueError: If required battery parameters are missing
            
        Note:
            This loads STATIC battery specs (capacity, rates, limits).
            Dynamic state (SoC, SoH, charge level) must be loaded from
            house_{id}_data table using update_state_from_sensors().
        """
        required_params = [
            "capacity_kwh", "efficiency", "max_charge_rate_kw", 
            "max_discharge_rate_kw", "min_reserve_percent", "max_charge_percent"
        ]
        missing = [p for p in required_params if p not in config]
        if missing:
            raise ValueError(
                f"Missing required battery parameters: {missing}. "
                f"Battery configuration must include all of: {required_params}. "
                f"Load configuration from database using consumption_parser.load_battery_config()."
            )
        
        self.household_id = household_id
        self.config = config
        self.battery = None  # Will be set by update_state_from_sensors()
        
    def update_state_from_sensors(self, sensor_data: dict):
        """
        Update battery state from sensor readings in database.
        
        Args:
            sensor_data: Dictionary with sensor readings:
                - current_charge_kwh: Current battery charge from BMS
                - soh_percent: State of Health from BMS (0-100)
                - cycle_count: Total cycles from BMS
                
        Raises:
            ValueError: If required sensor data is missing
        """
        required_sensors = ["current_charge_kwh", "soh_percent", "cycle_count"]
        missing = [s for s in required_sensors if s not in sensor_data]
        if missing:
            raise ValueError(
                f"Missing required battery sensor data: {missing}. "
                f"Sensor data must include: {required_sensors}. "
                f"Load from house_{{id}}_data table battery columns."
            )
        
        usable_capacity = self.config["capacity_kwh"] * 0.80
        
        self.battery = BatteryState(
            capacity_kwh=self.config["capacity_kwh"],
            usable_capacity_kwh=usable_capacity,
            current_charge_kwh=sensor_data["current_charge_kwh"],
            efficiency=self.config["efficiency"],
            max_charge_rate_kw=self.config["max_charge_rate_kw"],
            max_discharge_rate_kw=self.config["max_discharge_rate_kw"],
            min_reserve_percent=self.config["min_reserve_percent"],
            max_charge_percent=self.config["max_charge_percent"],
            soh_percent=sensor_data["soh_percent"],
            cycle_count=sensor_data["cycle_count"],
        )
        
        # Validate sensor data
        self.battery.validate_charge_cap()

    def can_charge(self, energy_kwh: float, interval_hours: float = 0.5) -> Tuple[bool, float]:
        """
        Check if battery can accept charge and calculate actual energy.
        
        NOTE: This is for validation/planning only. Actual charge operation
        happens in hardware, and new state is read from sensors.

        Args:
            energy_kwh: Requested energy to charge
            interval_hours: Time interval

        Returns:
            Tuple of (can_charge, max_energy_accepted)
        """
        if self.battery is None:
            raise RuntimeError(
                "Battery state not initialized. "
                "Call update_state_from_sensors() with sensor data first."
            )
        
        max_energy = self.battery.calculate_charge_limit(energy_kwh, interval_hours)
        return (max_energy > 0, max_energy)

    def can_discharge(self, energy_kwh: float, interval_hours: float = 0.5) -> Tuple[bool, float]:
        """
        Check if battery can deliver energy and calculate actual energy.
        
        NOTE: This is for validation/planning only. Actual discharge operation
        happens in hardware, and new state is read from sensors.

        Args:
            energy_kwh: Requested energy to discharge
            interval_hours: Time interval

        Returns:
            Tuple of (can_discharge, max_energy_delivered)
        """
        if self.battery is None:
            raise RuntimeError(
                "Battery state not initialized. "
                "Call update_state_from_sensors() with sensor data first."
            )
        
        max_energy = self.battery.calculate_discharge_limit(energy_kwh, interval_hours)
        return (max_energy > 0, max_energy)

    def get_state(self) -> BatteryState:
        """Get current battery state from sensors.
        
        Raises:
            RuntimeError: If state hasn't been loaded from sensors yet
        """
        if self.battery is None:
            raise RuntimeError(
                "Battery state not initialized. "
                "Call update_state_from_sensors() with sensor data first."
            )
        return self.battery

    def get_statistics(self) -> dict:
        """Get comprehensive battery statistics from sensor data.
        
        Raises:
            RuntimeError: If state hasn't been loaded from sensors yet
        """
        if self.battery is None:
            raise RuntimeError(
                "Battery state not initialized. "
                "Call update_state_from_sensors() with sensor data first."
            )
        
        return {
            "current_charge_kwh": self.battery.current_charge_kwh,
            "state_of_charge_percent": self.battery.state_of_charge() * 100,
            "state_of_health_percent": self.battery.state_of_health(),
            "cycle_count": self.battery.cycle_count,
            "efficiency": self.battery.efficiency,
            "effective_capacity_kwh": self.battery.get_effective_capacity(),
            "max_charge_allowed_kwh": self.battery.get_max_charge_kwh(),
            "available_capacity_kwh": self.battery.available_capacity(),
            "available_energy_kwh": self.battery.available_energy(),
        }


if __name__ == "__main__":
    print("Battery Manager - Reading Real Sensor Data")
    print("="*60)
    
    # Static battery configuration (from database battery_config table)
    config = {
        "capacity_kwh": 13.5,
        "efficiency": 0.90,
        "max_charge_rate_kw": 5.0,
        "max_discharge_rate_kw": 5.0,
        "min_reserve_percent": 0.10,
        "max_charge_percent": 0.80,
    }

    manager = BatteryManager(config, household_id=6)
    
    # Simulate loading sensor data from database (house_6_data table)
    print("\nLoading sensor data from database...")
    sensor_data = {
        "current_charge_kwh": 5.4,  # From battery sensor
        "soh_percent": 98.5,         # From battery BMS
        "cycle_count": 75.3,         # From battery BMS
    }
    
    manager.update_state_from_sensors(sensor_data)
    print("  Battery state loaded from sensors ✓")

    print("\nCurrent battery state (from sensors):")
    stats = manager.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nChecking if can charge 2 kWh:")
    can_charge, max_energy = manager.can_charge(2.0)
    print(f"  Can charge: {can_charge}")
    print(f"  Max energy accepted: {max_energy:.2f} kWh")

    print("\nChecking if can discharge 3 kWh:")
    can_discharge, max_energy = manager.can_discharge(3.0)
    print(f"  Can discharge: {can_discharge}")
    print(f"  Max energy delivered: {max_energy:.2f} kWh")
    
    print("\nSimulating sensor update after charging...")
    new_sensor_data = {
        "current_charge_kwh": 7.2,  # Updated by battery sensor
        "soh_percent": 98.5,         # Still same
        "cycle_count": 75.5,         # Incremented by BMS
    }
    manager.update_state_from_sensors(new_sensor_data)
    print(f"  New SoC: {manager.get_state().state_of_charge() * 100:.1f}%")
    print(f"  New charge: {manager.get_state().current_charge_kwh:.2f} kWh")
    
    print("\nTesting 80% charge cap validation:")
    try:
        bad_sensor_data = {
            "current_charge_kwh": 9.5,  # Exceeds 80% cap!
            "soh_percent": 98.5,
            "cycle_count": 76.0,
        }
        manager.update_state_from_sensors(bad_sensor_data)
        print("  ERROR: Should have raised exception!")
    except ValueError as e:
        print(f"  ✓ Caught sensor error: {str(e)[:80]}...")
    
    print("\n" + "="*60)
    print("Battery Manager uses REAL sensor data - no synthetic values!")
    print("="*60)
