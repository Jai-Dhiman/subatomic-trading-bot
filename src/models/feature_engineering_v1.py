"""
Feature engineering - Phase 1A: Pricing data only.

This is a simplified version that works with available data.
Will expand as appliance and battery data becomes available.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler


class FeatureEngineerV1:
    """
    Phase 1A: Extract features from pricing data only.
    
    Input: Historical pricing data from Supabase
    Output: Feature matrix with price lags and temporal features
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.price_scaler = StandardScaler()
        
    def extract_pricing_features(
        self, 
        pricing_df: pd.DataFrame,
        n_lags: int = 4
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract pricing features with lags.
        
        Args:
            pricing_df: DataFrame with 'timestamp' and 'price_per_kwh'
            n_lags: Number of lag features (default 4: 30min, 1h, 1.5h, 2h)
            
        Returns:
            features: Array of shape (n_samples, n_lags + 1)
            df: DataFrame with added features for reference
        """
        df = pricing_df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Current price
        prices = df['price_per_kwh'].values.reshape(-1, 1)
        
        # Create lagged features
        features = [prices]
        for lag in range(1, n_lags + 1):
            lagged = df['price_per_kwh'].shift(lag).bfill().values.reshape(-1, 1)
            features.append(lagged)
            df[f'price_lag_{lag}'] = lagged.flatten()
        
        features = np.hstack(features)
        
        # Normalize
        features_scaled = self.price_scaler.fit_transform(features)
        
        return features_scaled, df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract cyclical temporal features.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            Array of shape (n_samples, 4) - [hour_sin, hour_cos, day_sin, day_cos]
        """
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Hour of day (0-23) - captures daily patterns
        hour = timestamps.dt.hour + timestamps.dt.minute / 60.0  # Include minutes
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6) - captures weekly patterns
        day = timestamps.dt.dayofweek
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        return np.column_stack([hour_sin, hour_cos, day_sin, day_cos])
    
    def prepare_features(self, pricing_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Prepare complete feature matrix from pricing data.
        
        Args:
            pricing_df: Pricing DataFrame with 'timestamp' and 'price_per_kwh'
            
        Returns:
            features: Array of shape (n_samples, 9) - 5 price + 4 temporal
            df: Enhanced DataFrame with all features
        """
        # Extract pricing features (current + 4 lags = 5 features)
        price_features, df = self.extract_pricing_features(pricing_df)
        
        # Extract temporal features (4 features)
        temporal_features = self.extract_temporal_features(df)
        
        # Combine (5 + 4 = 9 features)
        features = np.hstack([price_features, temporal_features])
        
        # Add to dataframe for tracking
        df['hour_sin'] = temporal_features[:, 0]
        df['hour_cos'] = temporal_features[:, 1]
        df['day_sin'] = temporal_features[:, 2]
        df['day_cos'] = temporal_features[:, 3]
        
        return features, df
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets_price: np.ndarray,
        sequence_length: int = 48,
        horizons: Dict[str, int] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for training.
        
        Args:
            features: Feature array (n_samples, n_features)
            targets_price: Price targets (n_samples,)
            sequence_length: Input sequence length (default 48 = 24 hours)
            horizons: Prediction horizons (default: day=48, week=336)
            
        Returns:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Dict of targets for each horizon
        """
        if horizons is None:
            horizons = {'day': 48, 'week': 336}
        
        max_horizon = max(horizons.values())
        n_samples = len(features) - sequence_length - max_horizon
        
        if n_samples <= 0:
            raise ValueError(
                f"Insufficient data: need at least {sequence_length + max_horizon} samples, "
                f"got {len(features)}"
            )
        
        X = []
        y_price = {h: [] for h in horizons.keys()}
        
        for i in range(n_samples):
            # Input sequence
            X.append(features[i:i + sequence_length])
            
            # Target sequences for each horizon
            for horizon_name, horizon_len in horizons.items():
                start = i + sequence_length
                end = start + horizon_len
                y_price[horizon_name].append(targets_price[start:end])
        
        X = np.array(X)
        y = {f'price_{h}': np.array(y_price[h]) for h in horizons.keys()}
        
        return X, y
    
    def create_walk_forward_split(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray],
        test_days: int = 7,
        val_days: int = 3
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Create walk-forward validation split.
        
        Train on oldest data, validate on middle, test on most recent.
        This mimics production: train on past, predict future.
        
        Args:
            X: Input sequences
            y: Target dict
            test_days: Days to reserve for final testing (default 7 = last week)
            val_days: Days to reserve for validation (default 3)
            
        Returns:
            train_data: Dict with X and y for training
            val_data: Dict with X and y for validation
            test_data: Dict with X and y for testing
        """
        # Calculate split indices (48 intervals = 1 day)
        intervals_per_day = 48
        test_size = test_days * intervals_per_day
        val_size = val_days * intervals_per_day
        
        n_total = len(X)
        test_start = n_total - test_size
        val_start = test_start - val_size
        
        print(f"Walk-forward split:")
        print(f"  Total sequences: {n_total}")
        print(f"  Train: 0 to {val_start} ({val_start} sequences, {val_start/48:.1f} days)")
        print(f"  Val: {val_start} to {test_start} ({val_size} sequences, {val_days} days)")
        print(f"  Test: {test_start} to {n_total} ({test_size} sequences, {test_days} days)")
        
        # Create splits
        train_data = {
            'X': X[:val_start],
            'y': {key: val[:val_start] for key, val in y.items()}
        }
        
        val_data = {
            'X': X[val_start:test_start],
            'y': {key: val[val_start:test_start] for key, val in y.items()}
        }
        
        test_data = {
            'X': X[test_start:],
            'y': {key: val[test_start:] for key, val in y.items()}
        }
        
        return train_data, val_data, test_data


if __name__ == "__main__":
    print("Feature Engineering V1 - Pricing Only")
    print("=" * 60)
    
    # Create sample pricing data (31 days)
    dates = pd.date_range('2024-10-01', periods=1488, freq='30min')
    
    pricing_df = pd.DataFrame({
        'timestamp': dates,
        'price_per_kwh': 0.35 + 0.15 * np.sin(2 * np.pi * np.arange(1488) / 48) + np.random.randn(1488) * 0.02
    })
    
    print(f"\nSample data: {len(pricing_df)} records ({len(pricing_df)/48:.1f} days)")
    print(f"Date range: {pricing_df['timestamp'].min()} to {pricing_df['timestamp'].max()}")
    
    # Initialize feature engineer
    config = {}
    engineer = FeatureEngineerV1(config)
    
    # Extract features
    print("\n1. Extracting features...")
    features, df_enhanced = engineer.prepare_features(pricing_df)
    print(f"   Feature shape: {features.shape}")
    print(f"   Features: 5 price (current + 4 lags) + 4 temporal = 9 total")
    
    # Create sequences
    print("\n2. Creating sequences...")
    price_target = pricing_df['price_per_kwh'].values
    
    X, y = engineer.create_sequences(
        features,
        price_target,
        sequence_length=48,
        horizons={'day': 48, 'week': 336}
    )
    
    print(f"   X shape: {X.shape}")
    print(f"   y keys: {list(y.keys())}")
    for key, val in y.items():
        print(f"   {key} shape: {val.shape}")
    
    # Create walk-forward split
    print("\n3. Creating walk-forward split...")
    train_data, val_data, test_data = engineer.create_walk_forward_split(
        X, y, test_days=7, val_days=3
    )
    
    print(f"\n   Train X: {train_data['X'].shape}")
    print(f"   Val X: {val_data['X'].shape}")
    print(f"   Test X: {test_data['X'].shape}")
    
    print("\n✓ Feature engineering V1 test complete!")
    print("\nNext step: Test with real Supabase data or create transformer model")
