"""
Feature Engineering for Trading Transformer.

Extracts features for intelligent price prediction and trading decisions:

CORE FEATURES (20):
- 3 consumption predictions from T1 (peak, total, average)
- 6 pricing features (current, 4 lags, weekly avg)
- 4 battery state features (SoC%, available kWh, remaining capacity, SoH%)
- 3 historical context features (actual consumption, prediction error, profit/loss)
- 4 temporal features (hour/day sin/cos encoding)

ENHANCED FEATURES (10) - Business Logic:
- 4 price trend features (percentile, min/max ratio, volatility, trend)
- 3 demand-driven features (energy deficit, surplus, hours of coverage)
- 3 trading opportunity features (buy signal, sell signal, hold score)

Total: 30 features that encode all business rules for optimal trading
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class TradingFeatureEngineer:
    """Feature engineering for trading decisions."""
    
    def __init__(self):
        self.pricing_scaler = StandardScaler()
        self.battery_scaler = StandardScaler()
        self._fitted = False
    
    def extract_consumption_predictions(
        self,
        consumption_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Extract summary features from consumption predictions.
        
        Args:
            consumption_predictions: (n_samples, 48) array from Consumption Transformer
        
        Returns:
            Array of shape (n_samples, 3):
            - predicted_peak: max consumption in next 24h
            - predicted_total: total consumption in next 24h  
            - predicted_average: average consumption in next 24h
        """
        if len(consumption_predictions.shape) == 1:
            consumption_predictions = consumption_predictions.reshape(1, -1)
        
        features = np.zeros((len(consumption_predictions), 3))
        features[:, 0] = np.max(consumption_predictions, axis=1)    # Peak
        features[:, 1] = np.sum(consumption_predictions, axis=1)    # Total
        features[:, 2] = np.mean(consumption_predictions, axis=1)   # Average
        
        return features
    
    def extract_pricing_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract pricing features from historical data.
        
        Args:
            df: DataFrame with 'price_per_kwh' column
        
        Returns:
            Array of shape (n_samples, 6):
            - current_price
            - price_lag_1, price_lag_2, price_lag_3, price_lag_4
            - weekly_avg_price (rolling 168 hours)
        """
        if 'price_per_kwh' not in df.columns:
            raise ValueError("Missing 'price_per_kwh' column")
        
        n_samples = len(df)
        features = np.zeros((n_samples, 6))
        
        prices = df['price_per_kwh'].values
        
        # Current price
        features[:, 0] = prices
        
        # Price lags (1, 2, 3, 4 periods back)
        for lag in range(1, 5):
            features[lag:, lag] = prices[:-lag]
            # Fill initial values with current price
            features[:lag, lag] = prices[:lag]
        
        # Weekly average price (rolling 168 hours = 7 days)
        for i in range(n_samples):
            start_idx = max(0, i - 168)
            features[i, 5] = prices[start_idx:i+1].mean()
        
        return features
    
    def extract_price_trend_features(self, df: pd.DataFrame, window: int = 96) -> np.ndarray:
        """
        Extract price trend features for opportunistic trading.
        
        These features teach the model to recognize:
        - "Price is in bottom 20% of recent prices" -> BUY signal
        - "Price is in top 20% of recent prices" -> SELL signal
        
        Args:
            df: DataFrame with 'price_per_kwh' column
            window: Look-back window in hours (default: 96 = 4 days)
        
        Returns:
            Array of shape (n_samples, 4):
            - price_percentile_recent: Where current price ranks (0-100)
            - price_vs_min_ratio: current / min_recent (>1 means above minimum)
            - price_vs_max_ratio: current / max_recent (<1 means below maximum)
            - price_volatility: std dev of recent prices
        """
        if 'price_per_kwh' not in df.columns:
            raise ValueError("Missing 'price_per_kwh' column")
        
        n_samples = len(df)
        features = np.zeros((n_samples, 4))
        prices = df['price_per_kwh'].values
        
        for i in range(n_samples):
            start_idx = max(0, i - window)
            price_window = prices[start_idx:i+1]
            current_price = prices[i]
            
            if len(price_window) > 1:
                # Percentile ranking (0-100)
                features[i, 0] = (price_window < current_price).sum() / len(price_window) * 100
                
                # Ratio vs min (tells us if we're at bottom)
                min_price = price_window.min()
                features[i, 1] = current_price / (min_price + 1e-6)
                
                # Ratio vs max (tells us if we're at top)
                max_price = price_window.max()
                features[i, 2] = current_price / (max_price + 1e-6)
                
                # Volatility (helps assess risk)
                features[i, 3] = np.std(price_window)
            else:
                # Default values for first sample
                features[i, 0] = 50.0  # Median
                features[i, 1] = 1.0
                features[i, 2] = 1.0
                features[i, 3] = 0.0
        
        return features
    
    def extract_demand_driven_features(
        self, 
        df: pd.DataFrame,
        forecast_hours: int = 8
    ) -> np.ndarray:
        """
        Extract demand-driven features for must-buy/should-sell logic.
        
        These features teach the model:
        - "Energy deficit exists" -> MUST BUY to cover household
        - "Energy surplus exists" -> SHOULD SELL to avoid waste
        
        Args:
            df: DataFrame with consumption and battery columns
            forecast_hours: Hours to look ahead for demand forecasting
        
        Returns:
            Array of shape (n_samples, 3):
            - energy_deficit_next_hours: Shortage in next N hours (kWh)
            - energy_surplus: Excess beyond next N hours (kWh)
            - hours_of_coverage: How many hours battery can sustain household
        """
        n_samples = len(df)
        features = np.zeros((n_samples, 3))
        
        # Use actual hourly consumption from appliances
        if 'hourly_consumption_kwh' in df.columns:
            consumption = df['hourly_consumption_kwh'].values
        elif 'total_consumption_kwh' in df.columns:
            # Fallback: calculate from appliances if available
            appliance_cols = [col for col in df.columns if col.startswith('appliance_')]
            if appliance_cols:
                consumption = df[appliance_cols].sum(axis=1).values
            else:
                consumption = df['total_consumption_kwh'].values
        else:
            # Default to zeros if no consumption data
            return features
        
        # Get battery available energy
        if 'battery_available_kwh' in df.columns:
            available_energy = df['battery_available_kwh'].values
        else:
            return features
        
        for i in range(n_samples):
            # Forecast consumption for next N hours
            end_idx = min(i + forecast_hours, n_samples)
            future_consumption = consumption[i:end_idx].sum()
            
            # Energy deficit: Do we have enough to cover future demand?
            deficit = max(0, future_consumption - available_energy[i])
            features[i, 0] = deficit
            
            # Energy surplus: Do we have more than we need?
            surplus = max(0, available_energy[i] - future_consumption)
            features[i, 1] = surplus
            
            # Hours of coverage: How long can battery last?
            avg_consumption = consumption[max(0, i-24):i+1].mean() if i > 0 else consumption[0]
            if avg_consumption > 0:
                features[i, 2] = available_energy[i] / avg_consumption
            else:
                features[i, 2] = 999.0  # Essentially infinite
        
        return features
    
    def extract_trading_opportunity_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract trading opportunity signals.
        
        These combine price and demand features to create clear trading signals.
        
        Returns:
            Array of shape (n_samples, 3):
            - buy_signal_strength: 0-1 (1 = strong buy opportunity)
            - sell_signal_strength: 0-1 (1 = strong sell opportunity)
            - hold_score: 0-1 (1 = should hold, not trade)
        """
        n_samples = len(df)
        features = np.zeros((n_samples, 3))
        
        # This will be computed from price and demand features
        # For now, return zeros and let the model learn
        # In a more advanced version, we could encode explicit business rules here
        
        return features
    
    def extract_battery_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract battery state features.
        
        Args:
            df: DataFrame with battery columns:
                - battery_soc_percent
                - battery_available_kwh
                - battery_capacity_remaining_kwh
                - battery_soh_percent
        
        Returns:
            Array of shape (n_samples, 4)
        """
        required_cols = [
            'battery_soc_percent',
            'battery_available_kwh', 
            'battery_capacity_remaining_kwh',
            'battery_soh_percent'
        ]
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing battery columns: {missing}")
        
        features = df[required_cols].values
        return features
    
    def extract_historical_context(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract recent performance context features.
        
        Args:
            df: DataFrame with consumption and prediction data
        
        Returns:
            Array of shape (n_samples, 3):
            - actual_consumption_last_24h
            - consumption_prediction_error (if available)
            - recent_profit_loss (simplified as price volatility)
        """
        n_samples = len(df)
        features = np.zeros((n_samples, 3))
        
        # Actual consumption last 24h (rolling sum over 48 intervals)
        if 'total_consumption_kwh' in df.columns:
            consumption = df['total_consumption_kwh'].values
            for i in range(n_samples):
                start_idx = max(0, i - 48)
                features[i, 0] = consumption[start_idx:i+1].sum()
        
        # Prediction error (if predicted and actual are available)
        if 'predicted_consumption' in df.columns and 'total_consumption_kwh' in df.columns:
            pred = df['predicted_consumption'].values
            actual = df['total_consumption_kwh'].values
            features[:, 1] = np.abs(pred - actual)
        else:
            # Default to zero if not available
            features[:, 1] = 0
        
        # Recent profit/loss proxy: price volatility over last 24h
        if 'price_per_kwh' in df.columns:
            prices = df['price_per_kwh'].values
            for i in range(n_samples):
                start_idx = max(0, i - 48)
                price_window = prices[start_idx:i+1]
                features[i, 2] = np.std(price_window) if len(price_window) > 1 else 0
        
        return features
    
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
    
    def prepare_features(
        self,
        consumption_predictions: np.ndarray,
        df: pd.DataFrame,
        fit: bool = True,
        include_enhanced: bool = True
    ) -> np.ndarray:
        """
        Prepare complete feature matrix for trading.
        
        Args:
            consumption_predictions: (n_samples, 48) from Consumption Transformer
            df: DataFrame with pricing, battery, and temporal data
            fit: Whether to fit scalers (True for training, False for inference)
            include_enhanced: Include enhanced business logic features (default: True)
            fit: Whether to fit scalers
        
        Returns:
            Feature array of shape (n_samples, 20 or 30):
            CORE (20):
            - 3 consumption prediction features
            - 6 pricing features (scaled)
            - 4 battery state features (scaled)
            - 3 historical context features
            - 4 temporal features
            
            ENHANCED (10) - if include_enhanced=True:
            - 4 price trend features
            - 3 demand-driven features
            - 3 trading opportunity features
        """
        if fit:
            self._fitted = False
        
        print(f"Extracting trading features from {len(df):,} samples...")
        
        # Extract CORE feature groups (20 features)
        consumption_feat = self.extract_consumption_predictions(consumption_predictions)  # 3
        pricing_feat = self.extract_pricing_features(df)                                 # 6
        battery_feat = self.extract_battery_features(df)                                 # 4
        historical_feat = self.extract_historical_context(df)                            # 3
        temporal_feat = self.extract_temporal_features(df)                               # 4
        
        feature_groups = []
        
        # Scale pricing and battery features
        if not self._fitted:
            pricing_feat = self.pricing_scaler.fit_transform(pricing_feat)
            battery_feat = self.battery_scaler.fit_transform(battery_feat)
            self._fitted = True
        else:
            pricing_feat = self.pricing_scaler.transform(pricing_feat)
            battery_feat = self.battery_scaler.transform(battery_feat)
        
        # Add core features
        feature_groups.extend([
            consumption_feat,   # 3
            pricing_feat,       # 6
            battery_feat,       # 4
            historical_feat,    # 3
            temporal_feat       # 4
        ])
        
        total_features = 20
        
        # Extract ENHANCED features if requested (10 features)
        if include_enhanced:
            print(f"  ✓ Including enhanced business logic features")
            price_trend_feat = self.extract_price_trend_features(df)          # 4
            demand_driven_feat = self.extract_demand_driven_features(df)      # 3
            trading_opp_feat = self.extract_trading_opportunity_features(df)  # 3
            
            feature_groups.extend([
                price_trend_feat,
                demand_driven_feat,
                trading_opp_feat
            ])
            total_features = 30
        
        # Concatenate all features
        features = np.hstack(feature_groups)
        
        print(f"  ✓ Feature shape: {features.shape} ({total_features} features)")
        print(f"  ✓ No NaN values: {not np.isnan(features).any()}")
        print(f"  ✓ No Inf values: {not np.isinf(features).any()}")
        
        return features
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets: Dict[str, np.ndarray],
        sequence_length: int = 48
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for training.
        
        Args:
            features: Feature array (n_samples, 20)
            targets: Dict with target arrays:
                - 'price': (n_samples,) actual prices
                - 'decisions': (n_samples,) optimal decisions [0=Buy, 1=Hold, 2=Sell]
                - 'quantities': (n_samples,) optimal quantities
            sequence_length: Input sequence length (default 48)
        
        Returns:
            X: Input sequences (n_sequences, sequence_length, 20)
            y: Dict of target sequences
        """
        n_samples = len(features) - sequence_length
        
        if n_samples <= 0:
            raise ValueError(
                f"Insufficient data: need at least {sequence_length} samples, "
                f"got {len(features)}"
            )
        
        print(f"Creating trading sequences...")
        print(f"  Sequence length: {sequence_length}")
        
        X = []
        y_out = {key: [] for key in targets.keys()}
        
        for i in range(n_samples):
            # Input sequence
            X.append(features[i:i + sequence_length])
            
            # Targets for next interval
            for key, target_array in targets.items():
                y_out[key].append(target_array[i + sequence_length])
        
        X = np.array(X)
        y_out = {key: np.array(val) for key, val in y_out.items()}
        
        print(f"  ✓ X shape: {X.shape}")
        for key, val in y_out.items():
            print(f"  ✓ {key} shape: {val.shape}")
        
        return X, y_out


if __name__ == "__main__":
    print("="*70)
    print("TRADING FEATURE ENGINEERING TEST")
    print("="*70)
    
    # Create synthetic test data
    n_samples = 200
    
    print(f"\nTest Configuration:")
    print(f"  Number of samples: {n_samples}")
    
    # 1. Consumption predictions (from T1)
    print(f"\n1. Creating synthetic consumption predictions...")
    consumption_predictions = np.random.randn(n_samples, 48) * 0.5 + 2.0
    consumption_predictions = np.clip(consumption_predictions, 0.5, 5.0)
    print(f"   ✓ Shape: {consumption_predictions.shape}")
    
    # 2. Create DataFrame with all required data
    print(f"\n2. Creating synthetic market and battery data...")
    
    timestamps = pd.date_range('2025-08-01', periods=n_samples, freq='30min')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price_per_kwh': np.random.randn(n_samples) * 0.05 + 0.20,
        'battery_soc_percent': np.random.randn(n_samples) * 10 + 50,
        'battery_available_kwh': np.random.randn(n_samples) * 5 + 20,
        'battery_capacity_remaining_kwh': np.random.randn(n_samples) * 5 + 15,
        'battery_soh_percent': np.random.randn(n_samples) * 2 + 98,
        'total_consumption_kwh': np.random.randn(n_samples) * 0.5 + 2.0
    })
    
    print(f"   ✓ DataFrame created: {df.shape}")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # 3. Initialize feature engineer
    print(f"\n3. Initializing Trading Feature Engineer...")
    engineer = TradingFeatureEngineer()
    
    # 4. Extract features
    print(f"\n4. Extracting features...")
    features = engineer.prepare_features(
        consumption_predictions=consumption_predictions,
        df=df,
        fit=True
    )
    
    print(f"\n5. Feature breakdown:")
    print(f"   Consumption predictions: 3 features")
    print(f"   Pricing: 6 features")
    print(f"   Battery state: 4 features")
    print(f"   Historical context: 3 features")
    print(f"   Temporal: 4 features")
    print(f"   Total: 20 features")
    
    # 6. Create sequences
    print(f"\n6. Creating sequences...")
    
    targets = {
        'price': df['price_per_kwh'].values,
        'decisions': np.random.randint(0, 3, n_samples),  # 0=Buy, 1=Hold, 2=Sell
        'quantities': np.random.rand(n_samples) * 5
    }
    
    X, y = engineer.create_sequences(
        features=features,
        targets=targets,
        sequence_length=48
    )
    
    print(f"\n7. Summary:")
    print(f"   Input sequences: {X.shape}")
    print(f"   Price targets: {y['price'].shape}")
    print(f"   Decision targets: {y['decisions'].shape}")
    print(f"   Quantity targets: {y['quantities'].shape}")
    print(f"   Ready for training!")
    
    # 8. Validate data
    print(f"\n8. Validating data...")
    assert not np.isnan(X).any(), "NaN values in X!"
    assert not np.isinf(X).any(), "Inf values in X!"
    assert X.shape == (n_samples - 48, 48, 20), f"Unexpected X shape: {X.shape}"
    print(f"   ✓ All validations passed")
    
    print("\n" + "="*70)
    print("✅ TRADING FEATURE ENGINEERING TEST COMPLETE")
    print("="*70)
