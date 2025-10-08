"""
Feature engineering for Transformer model.

Extracts and processes:
- 9 appliance consumption features
- 4 battery state features
- 2 weather features
- 4 temporal features (cyclical encoding)
- 5 pricing features (current + 4 lags)

Total: 24 input features for EnergyTransformer
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Extract and engineer features for energy prediction."""
    
    def __init__(self, config: dict = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config or {}
        self.consumption_scaler = StandardScaler()
        self.battery_scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        
        # Feature names for tracking
        self.appliance_features = [
            'appliance_fridge',
            'appliance_washing_machine',
            'appliance_dishwasher',
            'appliance_ev_charging',
            'appliance_ac',
            'appliance_stove',
            'appliance_water_heater',
            'appliance_computers',
            'appliance_misc'
        ]
        
        self.battery_features = [
            'battery_soc_percent',
            'battery_soh_percent',
            'battery_charge_kwh',
            'battery_cycle_count'
        ]
        
        self.weather_features = [
            'temperature',
            'solar_irradiance'
        ]
        
        self._fitted = False
    
    def extract_appliance_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 9 appliance-level consumption features.
        
        Args:
            df: DataFrame with appliance columns
            
        Returns:
            Array of shape (n_samples, 9)
        """
        missing = [col for col in self.appliance_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing appliance columns: {missing}")
        
        features = df[self.appliance_features].values
        
        if not self._fitted:
            return self.consumption_scaler.fit_transform(features)
        else:
            return self.consumption_scaler.transform(features)
    
    def extract_battery_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract battery state features.
        
        Args:
            df: DataFrame with battery sensor columns
            
        Returns:
            Array of shape (n_samples, 4)
        """
        missing = [col for col in self.battery_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing battery columns: {missing}")
        
        features = df[self.battery_features].values
        
        if not self._fitted:
            return self.battery_scaler.fit_transform(features)
        else:
            return self.battery_scaler.transform(features)
    
    def extract_pricing_features(
        self, 
        df: pd.DataFrame,
        n_lags: int = 4
    ) -> np.ndarray:
        """
        Extract pricing features with lags.
        
        Args:
            df: DataFrame with 'price_per_kwh' column
            n_lags: Number of lag features to create (default 4)
            
        Returns:
            Array of shape (n_samples, n_lags + 1)
        """
        if 'price_per_kwh' not in df.columns:
            raise ValueError("Missing 'price_per_kwh' column")
        
        prices = df['price_per_kwh'].values.reshape(-1, 1)
        
        # Create lagged features
        features = [prices]
        for lag in range(1, n_lags + 1):
            lagged = np.roll(prices, lag)
            lagged[:lag] = prices[0]  # Fill initial values with first price
            features.append(lagged)
        
        features = np.hstack(features)
        
        if not self._fitted:
            return self.price_scaler.fit_transform(features)
        else:
            return self.price_scaler.transform(features)
    
    def extract_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract cyclical temporal features.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            Array of shape (n_samples, 4) - [hour_sin, hour_cos, day_sin, day_cos]
        """
        if 'timestamp' not in df.columns:
            raise ValueError("Missing 'timestamp' column")
        
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Hour of day (0-23) - include minutes for 30-min precision
        hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6)
        day = timestamps.dt.dayofweek
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        return np.column_stack([hour_sin, hour_cos, day_sin, day_cos])
    
    def extract_weather_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract weather features.
        
        Args:
            df: DataFrame with weather columns
            
        Returns:
            Array of shape (n_samples, 2)
        """
        missing = [col for col in self.weather_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing weather columns: {missing}")
        
        features = df[self.weather_features].values
        
        if not self._fitted:
            return self.weather_scaler.fit_transform(features)
        else:
            return self.weather_scaler.transform(features)
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """
        Prepare complete feature matrix.
        
        Args:
            df: Main DataFrame with all columns (appliances, battery, weather, pricing, timestamp)
            fit: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Feature array of shape (n_samples, 24)
            - 9 appliance features
            - 4 battery features
            - 2 weather features
            - 4 temporal features
            - 5 pricing features
        """
        if fit:
            self._fitted = False
        
        # Extract all feature groups
        appliance_feat = self.extract_appliance_features(df)  # 9 features
        battery_feat = self.extract_battery_features(df)       # 4 features
        weather_feat = self.extract_weather_features(df)       # 2 features
        temporal_feat = self.extract_temporal_features(df)     # 4 features (not scaled)
        pricing_feat = self.extract_pricing_features(df)       # 5 features
        
        if fit:
            self._fitted = True
        
        # Concatenate all features
        features = np.hstack([
            appliance_feat,     # 9
            battery_feat,       # 4
            weather_feat,       # 2
            temporal_feat,      # 4
            pricing_feat        # 5
        ])  # Total: 24 features
        
        return features
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets_consumption: np.ndarray,
        targets_price: np.ndarray,
        sequence_length: int = 48,
        horizons: Dict[str, int] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for training.
        
        Args:
            features: Feature array (n_samples, n_features)
            targets_consumption: Consumption targets (n_samples,)
            targets_price: Price targets (n_samples,)
            sequence_length: Input sequence length (default 48 = 24 hours)
            horizons: Prediction horizons (default: day, week, month)
            
        Returns:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Dict of targets for each horizon and task
        """
        if horizons is None:
            horizons = {'day': 48, 'week': 336, 'month': 1440}
        
        max_horizon = max(horizons.values())
        n_samples = len(features) - sequence_length - max_horizon
        
        if n_samples <= 0:
            raise ValueError(
                f"Insufficient data: need at least {sequence_length + max_horizon} samples, "
                f"got {len(features)}. Try reducing horizons or generating more data."
            )
        
        X = []
        y_consumption = {h: [] for h in horizons.keys()}
        y_price = {h: [] for h in horizons.keys()}
        
        for i in range(n_samples):
            # Input sequence
            X.append(features[i:i + sequence_length])
            
            # Target sequences for each horizon
            for horizon_name, horizon_len in horizons.items():
                start = i + sequence_length
                end = start + horizon_len
                
                y_consumption[horizon_name].append(
                    targets_consumption[start:end]
                )
                y_price[horizon_name].append(
                    targets_price[start:end]
                )
        
        X = np.array(X)
        
        # Combine consumption and price targets
        y = {}
        for horizon in horizons.keys():
            y[f'consumption_{horizon}'] = np.array(y_consumption[horizon])
            y[f'price_{horizon}'] = np.array(y_price[horizon])
        
        return X, y
    
    def inverse_transform_predictions(
        self,
        consumption_predictions: np.ndarray,
        price_predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse transform predictions back to original scale.
        
        Args:
            consumption_predictions: Scaled consumption predictions
            price_predictions: Scaled price predictions
            
        Returns:
            Tuple of (unscaled_consumption, unscaled_price)
        """
        # Note: Consumption is scaled as part of appliance features
        # For single value predictions, we need to inverse carefully
        
        # For now, return as-is since we're using separate scalers
        # In production, you'd implement proper inverse transform
        return consumption_predictions, price_predictions


if __name__ == "__main__":
    print("=" * 60)
    print("Feature Engineering Pipeline Test")
    print("=" * 60)
    
    # Import synthetic data generator
    import sys
    sys.path.append('/Users/jdhiman/Documents/energymvp')
    from src.data_integration.synthetic_data_generator import generate_synthetic_household_data
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_household_data(days=10)
    print(f"   Generated {len(df)} samples")
    
    # Initialize feature engineer
    config = {}
    engineer = FeatureEngineer(config)
    
    # Extract features
    print("\n2. Extracting features...")
    features = engineer.prepare_features(df, fit=True)
    print(f"   Feature shape: {features.shape}")
    print(f"   Expected: (480, 24)")
    
    if features.shape[1] != 24:
        print(f"   ❌ ERROR: Expected 24 features, got {features.shape[1]}")
    else:
        print(f"   ✓ Correct number of features!")
    
    # Check feature statistics
    print("\n3. Feature statistics:")
    print(f"   Mean: {features.mean(axis=0)[:5]} (first 5 features)")
    print(f"   Std:  {features.std(axis=0)[:5]} (first 5 features)")
    
    # Create sequences
    print("\n4. Creating sequences...")
    
    # Calculate total consumption as target
    consumption_target = df['total_consumption_kwh'].values
    price_target = df['price_per_kwh'].values
    
    try:
        X, y = engineer.create_sequences(
            features,
            consumption_target,
            price_target,
            sequence_length=48,
            horizons={'day': 48, 'week': 336}  # Removed month for 10-day data
        )
        
        print(f"   X shape: {X.shape}")
        print(f"   Expected: (n_sequences, 48, 24)")
        
        print(f"\n   Target shapes:")
        for key, val in y.items():
            print(f"     {key}: {val.shape}")
        
        # Verify shapes
        if X.shape[1] == 48 and X.shape[2] == 24:
            print(f"\n   ✓ Sequence shape correct!")
        else:
            print(f"\n   ❌ ERROR: Sequence shape incorrect")
        
        print("\n5. Verifying data ranges:")
        print(f"   X min: {X.min():.3f}, max: {X.max():.3f}")
        print(f"   Consumption target min: {y['consumption_day'].min():.3f}, max: {y['consumption_day'].max():.3f}")
        print(f"   Price target min: {y['price_day'].min():.3f}, max: {y['price_day'].max():.3f}")
        
        print("\n" + "=" * 60)
        print("✅ Feature engineering test complete!")
        print(f"✅ Created {len(X)} sequences with 24 features each")
        print(f"✅ Ready for Transformer training (Task 3)")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n   ❌ ERROR creating sequences: {e}")
        print(f"   Suggestion: Increase days or reduce horizons")
