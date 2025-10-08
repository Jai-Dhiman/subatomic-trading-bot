"""
Feature Engineering for Consumption Transformer.

Extracts 17 features from real Supabase data:
- 9 appliance consumption features (normalized)
- 4 temporal features (cyclical encoding)
- 4 historical pattern features
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler


class ConsumptionFeatureEngineer:
    """Feature engineering for consumption prediction."""
    
    def __init__(self):
        self.appliance_scaler = StandardScaler()
        self._fitted = False
        
        # Appliance feature names
        self.appliance_features = [
            'appliance_ac',
            'appliance_washing_drying',
            'appliance_fridge',
            'appliance_ev_charging',
            'appliance_dishwasher',
            'appliance_computers',
            'appliance_stove',
            'appliance_water_heater',
            'appliance_misc'
        ]
    
    def extract_appliance_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and normalize 9 appliance features from Supabase data.
        
        Args:
            df: DataFrame with appliance columns
            
        Returns:
            Array of shape (n_samples, 9) - normalized appliance consumption
        """
        missing = [col for col in self.appliance_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing appliance columns: {missing}")
        
        features = df[self.appliance_features].values
        
        if not self._fitted:
            return self.appliance_scaler.fit_transform(features)
        else:
            return self.appliance_scaler.transform(features)
    
    def extract_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract cyclical temporal features.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            Array of shape (n_samples, 4):
            - hour_sin, hour_cos
            - day_sin, day_cos
        """
        if 'timestamp' not in df.columns:
            raise ValueError("Missing 'timestamp' column")
        
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Hour of day (0-23)
        hour = timestamps.dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6)
        day = timestamps.dt.dayofweek
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        return np.column_stack([hour_sin, hour_cos, day_sin, day_cos])
    
    def extract_historical_patterns(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate historical consumption patterns.
        
        Args:
            df: DataFrame with 'timestamp' and 'total_consumption_kwh'
            
        Returns:
            Array of shape (n_samples, 4):
            - last_week_same_time
            - weekday_average
            - rolling_7day_average
            - seasonal_factor
        """
        if 'total_consumption_kwh' not in df.columns:
            raise ValueError("Missing 'total_consumption_kwh' column")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        n_samples = len(df)
        features = np.zeros((n_samples, 4))
        
        # Group by house_id if present
        if 'house_id' in df.columns:
            for house_id in df['house_id'].unique():
                mask = df['house_id'] == house_id
                house_df = df[mask].copy()
                house_features = self._calculate_patterns_for_house(house_df)
                features[mask] = house_features
        else:
            features = self._calculate_patterns_for_house(df)
        
        return features
    
    def _calculate_patterns_for_house(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate patterns for a single house."""
        n_samples = len(df)
        features = np.zeros((n_samples, 4))
        
        consumption = df['total_consumption_kwh'].values
        
        # 1. Last week same time (168 hours = 7 days)
        for i in range(n_samples):
            if i >= 168:
                features[i, 0] = consumption[i - 168]
            else:
                features[i, 0] = consumption[:i+1].mean() if i > 0 else consumption[0]
        
        # 2. Weekday average
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        for i in range(n_samples):
            dow = df.iloc[i]['dayofweek']
            hour = df.iloc[i]['hour']
            
            # Get historical average for this day-of-week and hour
            mask = (df['dayofweek'] == dow) & (df['hour'] == hour)
            mask.iloc[i:] = False  # Only look at past data
            
            if mask.sum() > 0:
                features[i, 1] = df.loc[mask, 'total_consumption_kwh'].mean()
            else:
                features[i, 1] = consumption[:i+1].mean() if i > 0 else consumption[0]
        
        # 3. Rolling 7-day average (168 hours)
        for i in range(n_samples):
            start_idx = max(0, i - 168)
            features[i, 2] = consumption[start_idx:i+1].mean()
        
        # 4. Seasonal factor (simplified: based on month)
        month = pd.to_datetime(df['timestamp']).dt.month
        # Summer (Jun-Aug): 1.0, Winter (Dec-Feb): 1.2, Spring/Fall: 1.1
        seasonal = np.where(month.isin([6, 7, 8]), 1.0,
                   np.where(month.isin([12, 1, 2]), 1.2, 1.1))
        features[:, 3] = seasonal
        
        return features
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """
        Prepare complete feature matrix.
        
        Args:
            df: DataFrame with all required columns
            fit: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Feature array of shape (n_samples, 17):
            - 9 appliance features (normalized)
            - 4 temporal features
            - 4 historical pattern features
        """
        if fit:
            self._fitted = False
        
        print(f"Extracting features from {len(df):,} samples...")
        
        # Extract all feature groups
        appliance_feat = self.extract_appliance_features(df)  # 9 features
        temporal_feat = self.extract_temporal_features(df)    # 4 features
        historical_feat = self.extract_historical_patterns(df)  # 4 features
        
        if fit:
            self._fitted = True
        
        # Concatenate all features
        features = np.hstack([
            appliance_feat,     # 9
            temporal_feat,      # 4
            historical_feat     # 4
        ])  # Total: 17 features
        
        print(f"  ✓ Feature shape: {features.shape}")
        print(f"  ✓ No NaN values: {not np.isnan(features).any()}")
        print(f"  ✓ No Inf values: {not np.isinf(features).any()}")
        
        return features
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets_consumption: np.ndarray,
        sequence_length: int = 48,
        horizons: Dict[str, int] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for training.
        
        Args:
            features: Feature array (n_samples, n_features)
            targets_consumption: Consumption targets (n_samples,)
            sequence_length: Input sequence length (default 48 = 24 hours)
            horizons: Prediction horizons (default: day=48, week=336)
            
        Returns:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Dict of consumption targets for each horizon
        """
        if horizons is None:
            horizons = {'day': 48, 'week': 336}
        
        max_horizon = max(horizons.values())
        n_samples = len(features) - sequence_length - max_horizon
        
        if n_samples <= 0:
            raise ValueError(
                f"Insufficient data: need at least {sequence_length + max_horizon} samples, "
                f"got {len(features)}. Try reducing horizons or using more data."
            )
        
        print(f"Creating sequences...")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Horizons: {horizons}")
        
        X = []
        y_consumption = {h: [] for h in horizons.keys()}
        
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
        
        X = np.array(X)
        
        # Create targets dict
        y = {}
        for horizon in horizons.keys():
            y[f'consumption_{horizon}'] = np.array(y_consumption[horizon])
        
        print(f"  ✓ X shape: {X.shape}")
        for key, val in y.items():
            print(f"  ✓ {key} shape: {val.shape}")
        
        return X, y


if __name__ == "__main__":
    print("="*70)
    print("CONSUMPTION FEATURE ENGINEERING TEST")
    print("="*70)
    
    # Import data adapter
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.data_integration.data_adapter import load_consumption_data
    
    # Load data
    print("\n1. Loading consumption data...")
    df = load_consumption_data()
    print(f"   Loaded {len(df):,} records")
    
    # Initialize feature engineer
    print("\n2. Initializing feature engineer...")
    engineer = ConsumptionFeatureEngineer()
    
    # Extract features
    print("\n3. Extracting features...")
    features = engineer.prepare_features(df, fit=True)
    print(f"   ✓ Features extracted: {features.shape}")
    
    # Create sequences
    print("\n4. Creating sequences...")
    X, y = engineer.create_sequences(
        features,
        df['total_consumption_kwh'].values,
        sequence_length=48,
        horizons={'day': 48, 'week': 336}
    )
    
    print(f"\n5. Summary:")
    print(f"   Input sequences: {X.shape}")
    print(f"   Day targets: {y['consumption_day'].shape}")
    print(f"   Week targets: {y['consumption_week'].shape}")
    print(f"   Ready for training!")
    
    print("\n" + "="*70)
    print("✅ CONSUMPTION FEATURE ENGINEERING TEST COMPLETE")
    print("="*70)
