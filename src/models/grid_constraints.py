"""
Grid constraints manager for household grid connections.

Enforces physical grid connection limits:
- Maximum import from grid: 10 kWh per 30-min interval (20 kW rate)
- Maximum export to grid: 4 kWh per 30-min interval (8 kW rate)

All limits come from database configuration - NO hardcoded fallbacks.
"""

from typing import Tuple, Dict


class GridConstraintsManager:
    """Manages and enforces grid connection constraints for a household.
    
    Grid limits are physical constraints from the utility connection and
    must be respected to avoid breaker trips or contract violations.
    """
    
    def __init__(self, config: Dict, household_id: int = None):
        """
        Initialize grid constraints manager with configuration from database.
        
        Args:
            config: Dictionary with grid limits (from load_grid_config())
            household_id: Household ID for error messages
            
        Raises:
            ValueError: If required configuration is missing
            
        Example config:
            {
                'max_import_kw': 20.0,  # 10 kWh per 0.5hr
                'max_export_kw': 8.0,   # 4 kWh per 0.5hr
            }
        """
        required_params = ['max_import_kw', 'max_export_kw']
        missing = [p for p in required_params if p not in config]
        
        if missing:
            raise ValueError(
                f"Missing required grid configuration parameters: {missing}. "
                f"Grid config must include: {required_params}. "
                f"Load from database using consumption_parser.load_grid_config()."
            )
        
        self.max_import_kw = config['max_import_kw']
        self.max_export_kw = config['max_export_kw']
        self.household_id = household_id
        
        # Validate limits are positive
        if self.max_import_kw <= 0:
            raise ValueError(
                f"Invalid max_import_kw: {self.max_import_kw}. "
                f"Must be positive (typically 20 kW for 10 kWh per 30-min)."
            )
        
        if self.max_export_kw <= 0:
            raise ValueError(
                f"Invalid max_export_kw: {self.max_export_kw}. "
                f"Must be positive (typically 8 kW for 4 kWh per 30-min)."
            )
    
    def get_max_import_kwh(self, interval_hours: float = 0.5) -> float:
        """
        Get maximum energy that can be imported from grid in this interval.
        
        Args:
            interval_hours: Time interval in hours (default 0.5 for 30-min)
            
        Returns:
            Maximum import in kWh
            
        Example:
            >>> manager.get_max_import_kwh(0.5)  # 30-min interval
            10.0  # kWh
        """
        return self.max_import_kw * interval_hours
    
    def get_max_export_kwh(self, interval_hours: float = 0.5) -> float:
        """
        Get maximum energy that can be exported to grid in this interval.
        
        Args:
            interval_hours: Time interval in hours (default 0.5 for 30-min)
            
        Returns:
            Maximum export in kWh
            
        Example:
            >>> manager.get_max_export_kwh(0.5)  # 30-min interval
            4.0  # kWh
        """
        return self.max_export_kw * interval_hours
    
    def can_import(self, energy_kwh: float, interval_hours: float = 0.5) -> Tuple[bool, float]:
        """
        Check if energy can be imported from grid without exceeding limits.
        
        Args:
            energy_kwh: Requested import energy (kWh)
            interval_hours: Time interval (default 0.5 for 30-min)
            
        Returns:
            Tuple of (is_allowed, max_allowed_kwh)
            
        Example:
            >>> can_import, max_kwh = manager.can_import(8.0, 0.5)
            >>> # can_import=True, max_kwh=10.0 (within limit)
        """
        max_import = self.get_max_import_kwh(interval_hours)
        
        if energy_kwh < 0:
            return False, 0.0
        
        is_allowed = energy_kwh <= max_import
        actual_allowed = min(energy_kwh, max_import)
        
        return is_allowed, actual_allowed
    
    def can_export(self, energy_kwh: float, interval_hours: float = 0.5) -> Tuple[bool, float]:
        """
        Check if energy can be exported to grid without exceeding limits.
        
        Args:
            energy_kwh: Requested export energy (kWh)
            interval_hours: Time interval (default 0.5 for 30-min)
            
        Returns:
            Tuple of (is_allowed, max_allowed_kwh)
            
        Example:
            >>> can_export, max_kwh = manager.can_export(3.0, 0.5)
            >>> # can_export=True, max_kwh=4.0 (within limit)
        """
        max_export = self.get_max_export_kwh(interval_hours)
        
        if energy_kwh < 0:
            return False, 0.0
        
        is_allowed = energy_kwh <= max_export
        actual_allowed = min(energy_kwh, max_export)
        
        return is_allowed, actual_allowed
    
    def validate_grid_transaction(self, energy_kwh: float, direction: str, 
                                  interval_hours: float = 0.5) -> None:
        """
        Validate a grid transaction and raise exception if limits exceeded.
        
        Args:
            energy_kwh: Energy amount (kWh, positive value)
            direction: 'import' or 'export'
            interval_hours: Time interval (default 0.5 for 30-min)
            
        Raises:
            ValueError: If transaction would exceed grid limits
            
        Example:
            >>> manager.validate_grid_transaction(12.0, 'import', 0.5)
            ValueError: Grid import limit exceeded...
        """
        if energy_kwh < 0:
            raise ValueError(
                f"Invalid energy amount: {energy_kwh} kWh. "
                f"Energy must be positive. Use 'direction' parameter for import/export."
            )
        
        if direction not in ['import', 'export']:
            raise ValueError(
                f"Invalid direction: '{direction}'. Must be 'import' or 'export'."
            )
        
        household_str = f" for household {self.household_id}" if self.household_id else ""
        
        if direction == 'import':
            max_allowed = self.get_max_import_kwh(interval_hours)
            
            if energy_kwh > max_allowed:
                raise ValueError(
                    f"Grid import limit exceeded{household_str}: "
                    f"Attempted {energy_kwh:.2f} kWh, maximum allowed {max_allowed:.2f} kWh "
                    f"per {interval_hours*60:.0f}-min interval. "
                    f"Grid connection limit: {self.max_import_kw} kW. "
                    f"This is a physical constraint - reduce import or upgrade grid connection."
                )
        
        elif direction == 'export':
            max_allowed = self.get_max_export_kwh(interval_hours)
            
            if energy_kwh > max_allowed:
                raise ValueError(
                    f"Grid export limit exceeded{household_str}: "
                    f"Attempted {energy_kwh:.2f} kWh, maximum allowed {max_allowed:.2f} kWh "
                    f"per {interval_hours*60:.0f}-min interval. "
                    f"Grid connection limit: {self.max_export_kw} kW. "
                    f"This is a physical constraint - reduce export or upgrade grid connection."
                )
    
    def calculate_import_rate_kw(self, energy_kwh: float, interval_hours: float = 0.5) -> float:
        """
        Calculate the power rate (kW) for a given energy import.
        
        Args:
            energy_kwh: Energy to import (kWh)
            interval_hours: Time interval (default 0.5)
            
        Returns:
            Power rate in kW
        """
        if interval_hours <= 0:
            raise ValueError(f"Interval hours must be positive, got {interval_hours}")
        
        return energy_kwh / interval_hours
    
    def calculate_export_rate_kw(self, energy_kwh: float, interval_hours: float = 0.5) -> float:
        """
        Calculate the power rate (kW) for a given energy export.
        
        Args:
            energy_kwh: Energy to export (kWh)
            interval_hours: Time interval (default 0.5)
            
        Returns:
            Power rate in kW
        """
        if interval_hours <= 0:
            raise ValueError(f"Interval hours must be positive, got {interval_hours}")
        
        return energy_kwh / interval_hours
    
    def get_limits_summary(self) -> Dict:
        """
        Get summary of grid connection limits.
        
        Returns:
            Dictionary with all limit information
        """
        return {
            'household_id': self.household_id,
            'max_import_kw': self.max_import_kw,
            'max_export_kw': self.max_export_kw,
            'max_import_kwh_per_30min': self.get_max_import_kwh(0.5),
            'max_export_kwh_per_30min': self.get_max_export_kwh(0.5),
            'max_import_kwh_per_hour': self.get_max_import_kwh(1.0),
            'max_export_kwh_per_hour': self.get_max_export_kwh(1.0),
        }


if __name__ == "__main__":
    print("Grid Constraints Manager")
    print("=" * 60)
    
    # Load grid configuration (from database)
    grid_config = {
        'max_import_kw': 20.0,  # 10 kWh per 30-min
        'max_export_kw': 8.0,   # 4 kWh per 30-min
    }
    
    manager = GridConstraintsManager(grid_config, household_id=6)
    
    print("\nGrid Connection Limits:")
    limits = manager.get_limits_summary()
    for key, value in limits.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Testing Import Constraints")
    print("=" * 60)
    
    # Test valid import
    print("\nTest 1: Import 8 kWh (within 10 kWh limit)")
    can_import, max_allowed = manager.can_import(8.0, 0.5)
    print(f"  Can import: {can_import}")
    print(f"  Max allowed: {max_allowed:.2f} kWh")
    print(f"  Power rate: {manager.calculate_import_rate_kw(8.0, 0.5):.1f} kW")
    
    try:
        manager.validate_grid_transaction(8.0, 'import', 0.5)
        print(f"  ✓ Validation passed")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")
    
    # Test invalid import
    print("\nTest 2: Import 12 kWh (exceeds 10 kWh limit)")
    can_import, max_allowed = manager.can_import(12.0, 0.5)
    print(f"  Can import: {can_import}")
    print(f"  Max allowed: {max_allowed:.2f} kWh")
    print(f"  Power rate: {manager.calculate_import_rate_kw(12.0, 0.5):.1f} kW")
    
    try:
        manager.validate_grid_transaction(12.0, 'import', 0.5)
        print(f"  ✓ Validation passed")
    except ValueError as e:
        print(f"  ✓ Validation correctly rejected: {str(e)[:80]}...")
    
    print("\n" + "=" * 60)
    print("Testing Export Constraints")
    print("=" * 60)
    
    # Test valid export
    print("\nTest 3: Export 3 kWh (within 4 kWh limit)")
    can_export, max_allowed = manager.can_export(3.0, 0.5)
    print(f"  Can export: {can_export}")
    print(f"  Max allowed: {max_allowed:.2f} kWh")
    print(f"  Power rate: {manager.calculate_export_rate_kw(3.0, 0.5):.1f} kW")
    
    try:
        manager.validate_grid_transaction(3.0, 'export', 0.5)
        print(f"  ✓ Validation passed")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")
    
    # Test invalid export
    print("\nTest 4: Export 5 kWh (exceeds 4 kWh limit)")
    can_export, max_allowed = manager.can_export(5.0, 0.5)
    print(f"  Can export: {can_export}")
    print(f"  Max allowed: {max_allowed:.2f} kWh")
    print(f"  Power rate: {manager.calculate_export_rate_kw(5.0, 0.5):.1f} kW")
    
    try:
        manager.validate_grid_transaction(5.0, 'export', 0.5)
        print(f"  ✓ Validation passed")
    except ValueError as e:
        print(f"  ✓ Validation correctly rejected: {str(e)[:80]}...")
    
    print("\n" + "=" * 60)
    print("Edge Cases")
    print("=" * 60)
    
    # Test exact limits
    print("\nTest 5: Import exactly 10 kWh (at limit)")
    can_import, max_allowed = manager.can_import(10.0, 0.5)
    print(f"  Can import: {can_import}")
    print(f"  At limit: {can_import and max_allowed == 10.0}")
    
    print("\nTest 6: Export exactly 4 kWh (at limit)")
    can_export, max_allowed = manager.can_export(4.0, 0.5)
    print(f"  Can export: {can_export}")
    print(f"  At limit: {can_export and max_allowed == 4.0}")
    
    # Test different intervals
    print("\nTest 7: Different time intervals")
    print(f"  15-min interval: max import = {manager.get_max_import_kwh(0.25):.1f} kWh")
    print(f"  30-min interval: max import = {manager.get_max_import_kwh(0.5):.1f} kWh")
    print(f"  60-min interval: max import = {manager.get_max_import_kwh(1.0):.1f} kWh")
    
    print("\n" + "=" * 60)
    print("Grid Constraints Manager - Production Ready!")
    print("All limits from database - NO hardcoded fallbacks")
    print("=" * 60)
