# Phase 1 Implementation: Pricing-Only Transformer

**Status:** Adjusted for Current Data Availability  
**Goal:** Build Transformer with pricing data only, add features incrementally  
**Testing Strategy:** Train on all data except last week, predict last week  
**Timeline:** 1 week for Phase 1, then incremental additions

---

## Current Data Reality

### ✅ What You Have Now

- **Historical pricing data** from Supabase (`cabuyingpricehistoryseptember2025`)
- Access to Google Colab Pro (excellent!)

### ⏳ What's Coming Later

- 9 appliance-level consumption data
- Battery sensor data (SoC, SoH, charge_kwh, cycle_count, charge/discharge rates)
- Weather data (temperature, solar_irradiance)

### 🎯 Strategy: Incremental Feature Addition

```
Phase 1A: Pricing-Only Baseline (THIS WEEK)
    ↓
Phase 1B: Add Synthetic Consumption (for testing architecture)
    ↓
Phase 1C: Add Real Appliance Data (when available)
    ↓
Phase 1D: Add Battery Sensors (when available)
    ↓
Phase 1E: Add Weather Data (when available)
```

---

## Phase 1A: Pricing-Only Transformer (Current Focus)

### Architecture Simplified

**Input Features (7 total):**

1. Current price ($/kWh)
2. Price lag -1 (30 min ago)
3. Price lag -2 (1 hour ago)
4. Price lag -4 (2 hours ago)
5. Hour sin (cyclical encoding)
6. Hour cos (cyclical encoding)
7. Day of week sin (cyclical encoding)
8. Day of week cos (cyclical encoding)

**Output:**

- Price predictions only (3 horizons: 1 day, 1 week, 1 month)
- Consumption predictions deferred until data available

**Model Size:**

- Smaller transformer: d_model=256, n_heads=4, n_layers=4
- ~2M parameters (vs 12M in full version)
- Faster training, less memory

---

## Walk-Forward Validation Strategy

Your testing approach is excellent! Here's how we'll implement it:

```python
# Example: If you have data from Oct 1-31 (31 days)
total_data = 31 days = 1,488 intervals (30-min chunks)

# Training
train_data = Oct 1-24 (24 days) = 1,152 intervals
validation_data = Oct 25-27 (3 days) = 144 intervals

# Final Test (your "predict last week")
test_data = Oct 25-31 (7 days) = 336 intervals

# This gives us:
# - Train on ~77% of data
# - Validate on ~10%
# - Test on final ~23% (last week)
```

### Why This Works

1. **Realistic evaluation** - Models never see the future (no data leakage)
2. **Production-like** - Mimics how you'd deploy (train on past, predict future)
3. **Clear success metrics** - If last week predictions are good, model is ready

---

## Modified Feature Engineering (Pricing-Only)

### Create `src/models/feature_engineering_v1.py`

```python
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
            features: Array of shape (n_samples, n_lags)
            df: DataFrame with added features for reference
        """
        df = pricing_df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Current price
        prices = df['price_per_kwh'].values.reshape(-1, 1)
        
        # Create lagged features
        features = [prices]
        for lag in range(1, n_lags + 1):
            lagged = df['price_per_kwh'].shift(lag).fillna(method='bfill').values.reshape(-1, 1)
            features.append(lagged)
            df[f'price_lag_{lag}'] = lagged
        
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
            features: Array of shape (n_samples, 8) - 4 price + 4 temporal
            df: Enhanced DataFrame with all features
        """
        # Extract pricing features
        price_features, df = self.extract_pricing_features(pricing_df)
        
        # Extract temporal features
        temporal_features = self.extract_temporal_features(df)
        
        # Combine
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
        horizons: Dict[str, int] = {'day': 48, 'week': 336}  # Removed 'month' for now
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for training.
        
        Args:
            features: Feature array (n_samples, n_features)
            targets_price: Price targets (n_samples,)
            sequence_length: Input sequence length (default 48 = 24 hours)
            horizons: Prediction horizons (day, week only for Phase 1A)
            
        Returns:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Dict of targets for each horizon
        """
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
    
    # Create sample pricing data
    dates = pd.date_range('2024-10-01', periods=1488, freq='30min')  # 31 days
    
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
    print(f"   Features: 4 price lags + 4 temporal = 8 total")
    
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
    print("\nNext step: Create simplified transformer model")
```

---

## Simplified Transformer Model (Pricing-Only)

### Create `src/models/transformer_model_v1.py`

```python
"""
Transformer Model V1 - Pricing predictions only.

Simplified version for Phase 1A:
- Smaller model (256 dim vs 512)
- Price prediction only
- 2 horizons (day, week) - removed month for faster training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PriceTransformer(nn.Module):
    """
    Simplified transformer for price prediction only.
    
    Phase 1A: Price forecasting with limited features
    Phase 1B+: Will add consumption prediction heads
    
    Input: (batch_size, sequence_length, n_features=8)
    Output: Dict with keys:
        - price_day: (batch_size, 48)
        - price_week: (batch_size, 336)
    """
    
    def __init__(
        self,
        n_features: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        horizons: Dict[str, int] = None
    ):
        super().__init__()
        
        if horizons is None:
            horizons = {'day': 48, 'week': 336}
        
        self.horizons = horizons
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Price prediction heads (one per horizon)
        self.price_heads = nn.ModuleDict()
        
        for horizon_name, horizon_len in horizons.items():
            self.price_heads[horizon_name] = nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, horizon_len)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, n_features)
            
        Returns:
            Dictionary with price predictions for each horizon
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Use last token representation for predictions
        context = encoded[:, -1, :]  # (batch, d_model)
        
        # Generate price predictions for each horizon
        outputs = {}
        for horizon_name in self.horizons.keys():
            outputs[f'price_{horizon_name}'] = self.price_heads[horizon_name](context)
        
        return outputs
    
    def get_model_weights(self) -> dict:
        """Get model weights for federated learning."""
        return self.state_dict()
    
    def update_model_weights(self, weights: dict):
        """Update model weights from federated aggregation."""
        self.load_state_dict(weights)


class PriceLoss(nn.Module):
    """
    Loss function for price prediction.
    
    Uses MAE (Mean Absolute Error) since price prediction
    should penalize all errors equally.
    """
    
    def __init__(self, horizon_weights: Dict[str, float] = None):
        super().__init__()
        
        if horizon_weights is None:
            horizon_weights = {'day': 1.0, 'week': 0.5}
        self.horizon_weights = horizon_weights
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate weighted MAE loss.
        
        Args:
            predictions: Dict of predicted tensors
            targets: Dict of target tensors
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components for logging
        """
        total_loss = 0.0
        loss_dict = {}
        
        for horizon in self.horizon_weights.keys():
            price_key = f'price_{horizon}'
            
            if price_key in predictions and price_key in targets:
                price_loss = F.l1_loss(predictions[price_key], targets[price_key])
                weighted_loss = price_loss * self.horizon_weights[horizon]
                
                total_loss += weighted_loss
                loss_dict[f'loss_{price_key}'] = price_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing PriceTransformer (V1)")
    print("=" * 60)
    
    # Model configuration
    config = {
        'n_features': 8,
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'horizons': {'day': 48, 'week': 336}
    }
    
    model = PriceTransformer(**config)
    
    # Test input
    batch_size = 4
    seq_len = 48
    x = torch.randn(batch_size, seq_len, config['n_features'])
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    outputs = model(x)
    
    print(f"\nOutput shapes:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Test loss calculation
    targets = {key: torch.randn_like(val) for key, val in outputs.items()}
    
    criterion = PriceLoss()
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"\nLoss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    print(f"Model size: ~{n_params * 4 / 1e6:.1f} MB (FP32)")
    
    print("\n✓ Model test complete!")
```

---

## Next Steps: This Week's Tasks

### Day 1-2: Feature Engineering

```bash
# Create the simplified feature engineering
# Copy code from above into:
touch src/models/feature_engineering_v1.py

# Test it
python src/models/feature_engineering_v1.py
```

### Day 3-4: Transformer Model

```bash
# Create the simplified transformer
touch src/models/transformer_model_v1.py

# Test it
python src/models/transformer_model_v1.py
```

### Day 5-7: Colab Training

- Upload to Colab
- Train on pricing data
- Evaluate on last week
- Analyze results

---

## Success Criteria for Phase 1A

### Minimum Viable

- [ ] Model trains without errors
- [ ] Predicts last week's prices
- [ ] MAE < $0.05/kWh for 1-day predictions
- [ ] MAE < $0.08/kWh for 1-week predictions

### Stretch Goals

- [ ] Direction accuracy > 65% (up/down correct)
- [ ] Captures peak price times correctly
- [ ] Beats naive baseline (last value repeated)

---

## What Happens When Data Arrives?

When appliance/battery data becomes available:

1. **Keep Phase 1A working** - Don't break what works
2. **Create V2 feature engineering** - Add new features incrementally
3. **Expand transformer** - Add consumption prediction heads
4. **Retrain** - Use walk-forward validation again
5. **Compare** - V1 (pricing) vs V2 (pricing + consumption)

You'll have a working system at every step!

---

Ready to start? I recommend beginning with the feature engineering code above. Want me to help you create those files?
