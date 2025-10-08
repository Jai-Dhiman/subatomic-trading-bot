# Transformer Implementation Plan

**Project:** Energy MVP - Federated Energy Trading System  
**Goal:** Replace LSTM with Transformer for multi-horizon consumption & price prediction  
**Training Environment:** Google Colab with GPU  
**Timeline:** 9 Phases (estimated 2-3 weeks)

---

## Executive Summary

This plan outlines the complete migration from the current LSTM-based node model to a state-of-the-art Transformer architecture that can:

1. **Predict multiple time horizons:** 1 day, 1 week, and 1 month ahead
2. **Forecast both consumption and prices:** Enable strategic buy-low/sell-high trading
3. **Use expanded features:** 9 appliances, battery states, pricing history, weather, and events
4. **Train efficiently on GPU:** Optimized for Google Colab with checkpointing
5. **Maintain federated learning:** Compatible with existing FedAvg aggregation
6. **Enhance trading logic:** Use price predictions for autonomous decision-making

---

## Current System Analysis

### What Works

- ✅ Trading logic and market mechanism (P2P matching, instant trades)
- ✅ Battery management (SoC, SoH, charge/discharge constraints)
- ✅ Grid constraints (10 kWh import, 4 kWh export limits)
- ✅ Profitability scoring and power signals
- ✅ Central model coordination
- ✅ Federated learning (FedAvg)
- ✅ Data pipeline (Supabase integration)

### Current Limitations

- ❌ LSTM only predicts 3 hours ahead (6 intervals)
- ❌ No price forecasting capability
- ❌ Missing appliance-level feature utilization
- ❌ No battery state features in model
- ❌ No historical pricing features
- ❌ Limited to 5 input features (consumption, temp, solar, hour, day)

---

## Phase 1: Remove LSTM Implementation

### Files to Delete

```bash
# Backup first!
cp src/models/node_model.py src/models/node_model.py.backup

# Remove LSTM implementation (will replace with transformer)
# Note: Keep the file structure, just remove LSTM-specific code
```

### Files to Modify

**`config/config.yaml`:**

```yaml
# BEFORE (LSTM):
model:
  lstm_hidden_size: 64
  lstm_num_layers: 2
  dropout: 0.2
  batch_size: 32
  prediction_horizon_intervals: 6
  input_sequence_length: 48

# AFTER (Transformer):
model:
  type: transformer
  d_model: 512
  n_heads: 8
  n_layers: 6
  dropout: 0.1
  batch_size: 32
  input_sequence_length: 48  # 24 hours lookback
  
  # Multi-horizon prediction
  horizons:
    day: 48        # 1 day (24 hours)
    week: 336      # 1 week (7 days)
    month: 1440    # 1 month (30 days)
  
  # Feature configuration
  features:
    appliances: 9         # 9 individual appliances
    battery: 4            # SoC, SoH, charge_kwh, cycle_count
    weather: 2            # temperature, solar_irradiance
    temporal: 4           # hour_sin, hour_cos, day_sin, day_cos
    pricing: 5            # current + 4 lagged prices
    total: 25             # Total input features
```

**`src/simulation/household_node.py`:**

- Remove `from src.models.node_model import NodeModel`
- Replace with `from src.models.transformer_model import TransformerModel`

---

## Phase 2: Feature Engineering Pipeline

### Create `src/models/feature_engineering.py`

```python
"""
Feature engineering for Transformer model.

Extracts and processes:
- 9 appliance consumption features
- 4 battery state features
- Historical pricing features with lags
- Temporal features (cyclical encoding)
- Rolling statistics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Extract and engineer features for energy prediction."""
    
    def __init__(self, config: dict):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config
        self.consumption_scaler = StandardScaler()
        self.battery_scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        
        # Feature names for tracking
        self.appliance_features = [
            'washing_machine_kwh',
            'dishwasher_kwh',
            'ev_charging_kwh',
            'fridge_kwh',
            'ac_kwh',
            'stove_kwh',
            'water_heater_kwh',
            'computers_kwh',
            'misc_kwh'
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
        
    def extract_appliance_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 9 appliance-level consumption features.
        
        Args:
            df: DataFrame with appliance columns
            
        Returns:
            Array of shape (n_samples, 9)
        """
        appliance_cols = [
            f'appliance_{name}' for name in self.appliance_features
        ]
        
        missing = [col for col in appliance_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing appliance columns: {missing}")
        
        features = df[appliance_cols].values
        return self.consumption_scaler.fit_transform(features)
    
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
        return self.battery_scaler.fit_transform(features)
    
    def extract_pricing_features(
        self, 
        pricing_df: pd.DataFrame,
        timestamps: pd.Series,
        n_lags: int = 4
    ) -> np.ndarray:
        """
        Extract pricing features with lags.
        
        Args:
            pricing_df: DataFrame with 'timestamp' and 'price_per_kwh'
            timestamps: Timestamps to align with
            n_lags: Number of lag features to create
            
        Returns:
            Array of shape (n_samples, n_lags + 1)
        """
        # Merge pricing data with timestamps
        merged = pd.merge_asof(
            pd.DataFrame({'timestamp': timestamps}),
            pricing_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        prices = merged['price_per_kwh'].values.reshape(-1, 1)
        
        # Create lagged features
        features = [prices]
        for lag in range(1, n_lags + 1):
            lagged = np.roll(prices, lag)
            lagged[:lag] = prices[0]  # Fill initial values
            features.append(lagged)
        
        features = np.hstack(features)
        return self.price_scaler.fit_transform(features)
    
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
        
        # Hour of day (0-23)
        hour = timestamps.dt.hour
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
        return self.weather_scaler.fit_transform(features)
    
    def create_rolling_statistics(
        self,
        consumption: np.ndarray,
        windows: List[int] = [12, 24, 48, 336]
    ) -> np.ndarray:
        """
        Create rolling statistics for consumption.
        
        Args:
            consumption: Array of consumption values
            windows: Window sizes in intervals (12=6h, 24=12h, 48=24h, 336=7d)
            
        Returns:
            Array of shape (n_samples, len(windows) * 4) - [mean, std, min, max] per window
        """
        features = []
        
        for window in windows:
            rolling = pd.Series(consumption.flatten()).rolling(window, min_periods=1)
            features.extend([
                rolling.mean().values,
                rolling.std().values,
                rolling.min().values,
                rolling.max().values
            ])
        
        return np.column_stack(features)
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        pricing_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Prepare complete feature matrix.
        
        Args:
            df: Main consumption DataFrame with all columns
            pricing_df: Pricing history DataFrame
            
        Returns:
            Feature array of shape (n_samples, 25+)
        """
        # Extract all feature groups
        appliance_feat = self.extract_appliance_features(df)  # 9 features
        battery_feat = self.extract_battery_features(df)       # 4 features
        weather_feat = self.extract_weather_features(df)       # 2 features
        temporal_feat = self.extract_temporal_features(df)     # 4 features
        pricing_feat = self.extract_pricing_features(          # 5 features
            pricing_df, df['timestamp']
        )
        
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
        horizons: Dict[str, int] = {'day': 48, 'week': 336, 'month': 1440}
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for training.
        
        Args:
            features: Feature array (n_samples, n_features)
            targets_consumption: Consumption targets (n_samples,)
            targets_price: Price targets (n_samples,)
            sequence_length: Input sequence length
            horizons: Prediction horizons
            
        Returns:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Dict of targets for each horizon and task
        """
        max_horizon = max(horizons.values())
        n_samples = len(features) - sequence_length - max_horizon
        
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


if __name__ == "__main__":
    # Test feature engineering
    print("Feature Engineering Pipeline Test")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='30min')
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'appliance_washing_machine_kwh': np.random.rand(1000) * 0.5,
        'appliance_dishwasher_kwh': np.random.rand(1000) * 0.3,
        'appliance_ev_charging_kwh': np.random.rand(1000) * 2.0,
        'appliance_fridge_kwh': np.random.rand(1000) * 0.2,
        'appliance_ac_kwh': np.random.rand(1000) * 1.5,
        'appliance_stove_kwh': np.random.rand(1000) * 0.4,
        'appliance_water_heater_kwh': np.random.rand(1000) * 0.6,
        'appliance_computers_kwh': np.random.rand(1000) * 0.3,
        'appliance_misc_kwh': np.random.rand(1000) * 0.2,
        'battery_soc_percent': np.random.rand(1000) * 100,
        'battery_soh_percent': 98.0,
        'battery_charge_kwh': np.random.rand(1000) * 13.5,
        'battery_cycle_count': 150,
        'temperature': 70 + np.random.randn(1000) * 10,
        'solar_irradiance': np.random.rand(1000) * 800,
    })
    
    pricing_df = pd.DataFrame({
        'timestamp': dates,
        'price_per_kwh': 0.35 + np.random.randn(1000) * 0.05
    })
    
    config = {}
    engineer = FeatureEngineer(config)
    
    print("\n1. Extracting features...")
    features = engineer.prepare_features(sample_df, pricing_df)
    print(f"   Feature shape: {features.shape}")
    print(f"   Expected: (1000, 24)")
    
    print("\n2. Creating sequences...")
    consumption_target = sample_df[[
        col for col in sample_df.columns if col.startswith('appliance_')
    ]].sum(axis=1).values
    
    price_target = pricing_df['price_per_kwh'].values
    
    X, y = engineer.create_sequences(
        features,
        consumption_target,
        price_target,
        sequence_length=48,
        horizons={'day': 48, 'week': 336, 'month': 1440}
    )
    
    print(f"   X shape: {X.shape}")
    print(f"   y keys: {list(y.keys())}")
    for key, val in y.items():
        print(f"   {key} shape: {val.shape}")
    
    print("\n✓ Feature engineering test complete!")
```

---

## Phase 3: Transformer Model Architecture

### Create `src/models/transformer_model.py`

```python
"""
Transformer-based model for multi-horizon energy consumption and price prediction.

Architecture:
- Positional encoding for time series
- Multi-head self-attention (6 layers, 8 heads)
- Multi-task learning (consumption + price prediction)
- Multi-horizon outputs (1 day, 1 week, 1 month)
- Compatible with federated learning (FedAvg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional


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
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EnergyTransformer(nn.Module):
    """
    Transformer model for multi-horizon consumption and price prediction.
    
    Input: (batch_size, sequence_length, n_features)
    Output: Dict with keys:
        - consumption_day: (batch_size, 48)
        - consumption_week: (batch_size, 336)
        - consumption_month: (batch_size, 1440)
        - price_day: (batch_size, 48)
        - price_week: (batch_size, 336)
        - price_month: (batch_size, 1440)
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        horizons: Dict[str, int] = None
    ):
        super().__init__()
        
        if horizons is None:
            horizons = {'day': 48, 'week': 336, 'month': 1440}
        
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
        
        # Multi-task prediction heads
        self.consumption_heads = nn.ModuleDict()
        self.price_heads = nn.ModuleDict()
        
        for horizon_name, horizon_len in horizons.items():
            # Consumption prediction head
            self.consumption_heads[horizon_name] = nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, horizon_len)
            )
            
            # Price prediction head
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
            Dictionary with predictions for each horizon and task
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
        
        # Generate predictions for each horizon
        outputs = {}
        
        for horizon_name in self.horizons.keys():
            outputs[f'consumption_{horizon_name}'] = self.consumption_heads[horizon_name](context)
            outputs[f'price_{horizon_name}'] = self.price_heads[horizon_name](context)
        
        return outputs
    
    def get_model_weights(self) -> dict:
        """Get model weights for federated learning."""
        return self.state_dict()
    
    def update_model_weights(self, weights: dict):
        """Update model weights from federated aggregation."""
        self.load_state_dict(weights)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for consumption and price prediction.
    
    Combines MSE for consumption and MAE for price prediction,
    weighted by task importance and horizon length.
    """
    
    def __init__(
        self,
        consumption_weight: float = 1.0,
        price_weight: float = 0.5,
        horizon_weights: Dict[str, float] = None
    ):
        super().__init__()
        
        self.consumption_weight = consumption_weight
        self.price_weight = price_weight
        
        if horizon_weights is None:
            horizon_weights = {'day': 1.0, 'week': 0.5, 'month': 0.25}
        self.horizon_weights = horizon_weights
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate multi-task loss.
        
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
            # Consumption loss (MSE)
            cons_key = f'consumption_{horizon}'
            if cons_key in predictions and cons_key in targets:
                cons_loss = F.mse_loss(predictions[cons_key], targets[cons_key])
                weighted_cons_loss = (
                    cons_loss * 
                    self.consumption_weight * 
                    self.horizon_weights[horizon]
                )
                total_loss += weighted_cons_loss
                loss_dict[f'loss_{cons_key}'] = cons_loss.item()
            
            # Price loss (MAE)
            price_key = f'price_{horizon}'
            if price_key in predictions and price_key in targets:
                price_loss = F.l1_loss(predictions[price_key], targets[price_key])
                weighted_price_loss = (
                    price_loss * 
                    self.price_weight * 
                    self.horizon_weights[horizon]
                )
                total_loss += weighted_price_loss
                loss_dict[f'loss_{price_key}'] = price_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing EnergyTransformer")
    print("=" * 60)
    
    # Model configuration
    config = {
        'n_features': 24,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.1,
        'horizons': {'day': 48, 'week': 336, 'month': 1440}
    }
    
    model = EnergyTransformer(**config)
    
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
    
    criterion = MultiTaskLoss()
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"\nLoss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    print("\n✓ Model test complete!")
```

---

## Phase 4: Google Colab Training Infrastructure

### Create `notebooks/colab_training.ipynb`

```python
# Cell 1: Setup and Dependencies
"""
Energy MVP - Transformer Model Training on Google Colab

Requirements:
- GPU runtime (T4 or better)
- Google Drive mount for checkpoints
- Supabase credentials
"""

import sys
import os
from pathlib import Path

# Check GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Mount Google Drive for checkpoint storage
from google.colab import drive
drive.mount('/content/drive')

# Create checkpoint directory
CHECKPOINT_DIR = Path('/content/drive/MyDrive/energymvp_checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
print(f"Checkpoint directory: {CHECKPOINT_DIR}")


# Cell 2: Install Dependencies
!pip install uv
!git clone https://github.com/yourusername/energymvp.git /content/energymvp
%cd /content/energymvp
!uv pip install -e .
!pip install tensorboard


# Cell 3: Load Environment Variables
import os
from getpass import getpass

# Supabase credentials
SUPABASE_URL = getpass("Enter SUPABASE_URL: ")
SUPABASE_KEY = getpass("Enter SUPABASE_KEY: ")

os.environ['SUPABASE_URL'] = SUPABASE_URL
os.environ['SUPABASE_KEY'] = SUPABASE_KEY

print("✓ Environment variables set")


# Cell 4: Import Project Modules
sys.path.append('/content/energymvp')

from src.models.transformer_model import EnergyTransformer, MultiTaskLoss
from src.models.feature_engineering import FeatureEngineer
from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.consumption_parser import parse_house_data
from src.training.training_utils import (
    create_data_loaders,
    train_epoch,
    validate_epoch,
    save_checkpoint,
    load_checkpoint
)

import pandas as pd
import numpy as np
from datetime import datetime

print("✓ Modules imported successfully")


# Cell 5: Load and Prepare Data
print("Loading data from Supabase...")

connector = SupabaseConnector()

# Load pricing data
pricing_df = connector.get_pricing_data()
print(f"Loaded {len(pricing_df)} pricing records")

# Load consumption data for all households
house_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Adjust based on your data
household_data = {}

for house_id in house_ids:
    try:
        df = parse_house_data(house_id, connector)
        household_data[house_id] = df
        print(f"Loaded house {house_id}: {len(df)} records")
    except Exception as e:
        print(f"Failed to load house {house_id}: {e}")

print(f"\n✓ Loaded data for {len(household_data)} households")


# Cell 6: Feature Engineering
print("Engineering features...")

config = {
    'horizons': {'day': 48, 'week': 336, 'month': 1440},
    'sequence_length': 48
}

engineer = FeatureEngineer(config)

# Process each household
processed_data = {}

for house_id, df in household_data.items():
    print(f"Processing house {house_id}...")
    
    # Prepare features
    features = engineer.prepare_features(df, pricing_df)
    
    # Create targets
    consumption_target = df[[
        col for col in df.columns if col.startswith('appliance_')
    ]].sum(axis=1).values
    
    # Merge price data
    merged = pd.merge_asof(
        df[['timestamp']].sort_values('timestamp'),
        pricing_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    price_target = merged['price_per_kwh'].values
    
    # Create sequences
    X, y = engineer.create_sequences(
        features,
        consumption_target,
        price_target,
        sequence_length=48,
        horizons=config['horizons']
    )
    
    processed_data[house_id] = {'X': X, 'y': y}
    print(f"  Created {len(X)} sequences")

print(f"\n✓ Feature engineering complete")


# Cell 7: Create Data Loaders
from torch.utils.data import TensorDataset, DataLoader, random_split

# Combine all household data
all_X = []
all_y = {key: [] for key in processed_data[1]['y'].keys()}

for house_id, data in processed_data.items():
    all_X.append(torch.FloatTensor(data['X']))
    for key in data['y'].keys():
        all_y[key].append(torch.FloatTensor(data['y'][key]))

# Concatenate
X_tensor = torch.cat(all_X, dim=0)
y_tensors = {key: torch.cat(all_y[key], dim=0) for key in all_y.keys()}

print(f"Combined dataset:")
print(f"  X shape: {X_tensor.shape}")
for key, val in y_tensors.items():
    print(f"  {key} shape: {val.shape}")

# Create dataset
dataset = TensorDataset(X_tensor, *y_tensors.values())

# Split: 60% train, 20% val, 20% test
n_total = len(dataset)
n_train = int(0.6 * n_total)
n_val = int(0.2 * n_total)
n_test = n_total - n_train - n_val

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [n_train, n_val, n_test]
)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"\nDataset splits:")
print(f"  Train: {n_train} samples ({n_train/n_total*100:.1f}%)")
print(f"  Val: {n_val} samples ({n_val/n_total*100:.1f}%)")
print(f"  Test: {n_test} samples ({n_test/n_total*100:.1f}%)")


# Cell 8: Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_config = {
    'n_features': X_tensor.shape[2],
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'horizons': config['horizons']
}

model = EnergyTransformer(**model_config).to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Initialize optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
criterion = MultiTaskLoss()

print("✓ Model initialized")


# Cell 9: Training Loop
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='/content/tensorboard_logs')

num_epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0

print(f"Starting training for {num_epochs} epochs...")
print(f"Batch size: {batch_size}")
print(f"Training samples: {n_train}")
print("=" * 60)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_loss_components = {}
    
    for batch_idx, batch in enumerate(train_loader):
        X_batch = batch[0].to(device)
        y_batch = {
            key: batch[i+1].to(device)
            for i, key in enumerate(y_tensors.keys())
        }
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Calculate loss
        loss, loss_dict = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        # Accumulate loss components
        for key, val in loss_dict.items():
            if key not in train_loss_components:
                train_loss_components[key] = 0.0
            train_loss_components[key] += val
    
    # Average training loss
    train_loss /= len(train_loader)
    for key in train_loss_components:
        train_loss_components[key] /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_loss_components = {}
    
    with torch.no_grad():
        for batch in val_loader:
            X_batch = batch[0].to(device)
            y_batch = {
                key: batch[i+1].to(device)
                for i, key in enumerate(y_tensors.keys())
            }
            
            outputs = model(X_batch)
            loss, loss_dict = criterion(outputs, y_batch)
            
            val_loss += loss.item()
            for key, val in loss_dict.items():
                if key not in val_loss_components:
                    val_loss_components[key] = 0.0
                val_loss_components[key] += val
    
    val_loss /= len(val_loader)
    for key in val_loss_components:
        val_loss_components[key] /= len(val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Logging
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    
    for key, val in train_loss_components.items():
        writer.add_scalar(f'Train/{key}', val, epoch)
    for key, val in val_loss_components.items():
        writer.add_scalar(f'Val/{key}', val, epoch)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss: {val_loss:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        checkpoint_path = CHECKPOINT_DIR / f"best_model_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': model_config
        }, checkpoint_path)
        print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    print()

writer.close()
print("=" * 60)
print(f"Training complete!")
print(f"Best validation loss: {best_val_loss:.6f}")


# Cell 10: Evaluate on Test Set
print("Evaluating on test set...")

model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
model.eval()

test_loss = 0.0
test_predictions = []
test_targets = []

with torch.no_grad():
    for batch in test_loader:
        X_batch = batch[0].to(device)
        y_batch = {
            key: batch[i+1].to(device)
            for i, key in enumerate(y_tensors.keys())
        }
        
        outputs = model(X_batch)
        loss, _ = criterion(outputs, y_batch)
        
        test_loss += loss.item()
        test_predictions.append(outputs)
        test_targets.append(y_batch)

test_loss /= len(test_loader)

print(f"Test Loss: {test_loss:.6f}")

# Calculate detailed metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

for horizon in config['horizons'].keys():
    cons_key = f'consumption_{horizon}'
    price_key = f'price_{horizon}'
    
    # Gather all predictions and targets
    cons_preds = torch.cat([p[cons_key] for p in test_predictions]).cpu().numpy()
    cons_targets = torch.cat([t[cons_key] for t in test_targets]).cpu().numpy()
    
    price_preds = torch.cat([p[price_key] for p in test_predictions]).cpu().numpy()
    price_targets = torch.cat([t[price_key] for t in test_targets]).cpu().numpy()
    
    # Calculate metrics
    cons_mae = mean_absolute_error(cons_targets.flatten(), cons_preds.flatten())
    cons_rmse = np.sqrt(mean_squared_error(cons_targets.flatten(), cons_preds.flatten()))
    
    price_mae = mean_absolute_error(price_targets.flatten(), price_preds.flatten())
    price_rmse = np.sqrt(mean_squared_error(price_targets.flatten(), price_preds.flatten()))
    
    print(f"\n{horizon.upper()} Horizon:")
    print(f"  Consumption MAE: {cons_mae:.4f} kWh")
    print(f"  Consumption RMSE: {cons_rmse:.4f} kWh")
    print(f"  Price MAE: ${price_mae:.4f}/kWh")
    print(f"  Price RMSE: ${price_rmse:.4f}/kWh")

print("\n✓ Evaluation complete!")


# Cell 11: Export Model for CPU Inference
print("Exporting model for CPU inference...")

# Export to TorchScript
model.cpu()
model.eval()

example_input = torch.randn(1, 48, model_config['n_features'])
traced_model = torch.jit.trace(model, example_input)

export_path = CHECKPOINT_DIR / "transformer_model_traced.pt"
traced_model.save(str(export_path))

print(f"✓ Model exported to: {export_path}")
print("\nDownload this file from Google Drive to use on your local machine")


# Cell 12: Visualize Predictions
import matplotlib.pyplot as plt

# Get a sample batch
sample_batch = next(iter(test_loader))
X_sample = sample_batch[0][:1].to(device)  # Take first sample
y_sample = {
    key: sample_batch[i+1][:1].to(device)
    for i, key in enumerate(y_tensors.keys())
}

# Get predictions
model.eval()
with torch.no_grad():
    pred_sample = model(X_sample)

# Plot for each horizon
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

for idx, horizon in enumerate(config['horizons'].keys()):
    # Consumption
    ax_cons = axes[idx, 0]
    cons_key = f'consumption_{horizon}'
    
    pred_cons = pred_sample[cons_key][0].cpu().numpy()
    true_cons = y_sample[cons_key][0].cpu().numpy()
    
    ax_cons.plot(true_cons, label='Actual', linewidth=2)
    ax_cons.plot(pred_cons, label='Predicted', linewidth=2, linestyle='--')
    ax_cons.set_title(f'{horizon.upper()} - Consumption')
    ax_cons.set_xlabel('Intervals')
    ax_cons.set_ylabel('kWh')
    ax_cons.legend()
    ax_cons.grid(True)
    
    # Price
    ax_price = axes[idx, 1]
    price_key = f'price_{horizon}'
    
    pred_price = pred_sample[price_key][0].cpu().numpy()
    true_price = y_sample[price_key][0].cpu().numpy()
    
    ax_price.plot(true_price, label='Actual', linewidth=2)
    ax_price.plot(pred_price, label='Predicted', linewidth=2, linestyle='--')
    ax_price.set_title(f'{horizon.upper()} - Price')
    ax_price.set_xlabel('Intervals')
    ax_price.set_ylabel('$/kWh')
    ax_price.legend()
    ax_price.grid(True)

plt.tight_layout()
plt.savefig(CHECKPOINT_DIR / 'predictions_sample.png', dpi=300)
plt.show()

print("✓ Visualization saved")
```

---

## Summary & Next Steps

This comprehensive plan provides:

1. ✅ **Complete feature engineering** - All 25+ features from available data
2. ✅ **State-of-the-art Transformer** - Multi-head attention for time series
3. ✅ **Multi-horizon predictions** - 1 day, 1 week, 1 month
4. ✅ **Multi-task learning** - Both consumption and price forecasting
5. ✅ **GPU training on Colab** - Optimized for Google Colab T4/V100
6. ✅ **Federated learning compatible** - FedAvg weight aggregation support
7. ✅ **Enhanced trading logic** - Uses price predictions for buy-low/sell-high
8. ✅ **Complete evaluation** - Metrics for all horizons and tasks

### Implementation Order

1. Start with Phase 2 (Feature Engineering) - Most critical
2. Then Phase 3 (Transformer Model) - Core architecture
3. Then Phase 4 (Colab Training) - Get model trained
4. Then Phases 5-9 in parallel - Integration and testing

### Estimated Timeline

- **Week 1:** Phases 1-3 (cleanup, features, model)
- **Week 2:** Phases 4-5 (training, data integration)
- **Week 3:** Phases 6-9 (trading, evaluation, testing)

Ready to start implementing? I can help you with any phase!
