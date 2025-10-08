# Dual-Transformer Implementation Plan

**Using Real Supabase Data**

## Overview

This document outlines the complete implementation plan for rebuilding the energy trading system with a dual-transformer architecture. We will remove the existing single-transformer system and build two specialized transformers from scratch.

---

## Available Data in Supabase

### 1. Consumption Data (Real Appliance Data)

- **Tables**: `august11homeconsumption` (1,000 records for Houses 1-2)
- **Date Range**: August 1-31, 2025 (31 days, hourly data)
- **Schema**:

  ```
  - House: integer (1-11)
  - Timestamp: timestamptz (hourly intervals)
  - Total kWh: float (⚠️ DAILY TOTAL - see warning below!)
  - Appliance_Breakdown_JSON: json with 9 appliances
  ```

- **⚠️ CRITICAL DATA STRUCTURE INSIGHT**:
  - `Total kWh` column shows **DAILY TOTALS** repeated for every hour!
  - **DO NOT** use `Total kWh` directly as hourly consumption
  - **MUST** calculate actual hourly consumption from appliance breakdown
  
  ```python
  # ❌ WRONG - This is daily total (31 kWh) for ALL 24 hours
  consumption = df['Total kWh']
  
  # ✅ CORRECT - Calculate from appliances
  appliance_cols = [col for col in df.columns if col.startswith('appliance_')]
  hourly_consumption = df[appliance_cols].sum(axis=1)  # 0.5-3.4 kWh per hour
  ```

- **Appliances** (from JSON):
  - A/C → `appliance_ac`
  - Washing/Drying → `appliance_washing_drying`
  - Refrig. → `appliance_fridge`
  - EV Charging → `appliance_ev_charging`
  - DishWasher → `appliance_dishwasher`
  - Computers → `appliance_computers`
  - Stovetop → `appliance_stove`
  - Water Heater → `appliance_water_heater`
  - Standby/ Misc. → `appliance_misc`

- **Actual Consumption Stats** (House 1):
  - Hourly: 0.5-3.4 kWh per hour (avg: 1.6 kWh/hour)
  - Daily: ~38 kWh total
  - Battery can power house for **20 hours** at average consumption

### 2. Pricing Data (Real California Market Pricing)

- **Table**: `cabuyingpricehistoryseptember2025` (1,000 LMP records)
- **Date Range**: October 1 - November 11, 2024
- **⚠️ CRITICAL**: Must filter for `LMP_TYPE = 'LMP'` only!
  
  ```python
  # Load pricing data with LMP filter
  query = client.table('cabuyingpricehistoryseptember2025') \
      .select('*') \
      .eq('LMP_TYPE', 'LMP') \
      .order('INTERVALSTARTTIME_GMT', desc=False)
  ```

- **Key Columns**:
  - INTERVALSTARTTIME_GMT (timestamp)
  - `Price MWH` - in $/MWh (MAIN PRICE FIELD)
  - `Price KWH` - convert from MWH: `Price MWH / 1000`
  - LMP_TYPE - MUST be 'LMP'
  
- **Price Distribution** (Real CA LMP Data):
  - Min: -$11/MWh (negative prices occur!)
  - 25th percentile: $32/MWh
  - Median: $45/MWh  
  - 75th percentile: $51/MWh
  - Max: $113/MWh
  - **11%** below $20/MWh (hard buy zone)
  - **25%** in $20-40/MWh (opportunistic zone)
  - **64%** above $40/MWh (hard sell zone)

### 3. Battery Trading Data (✅ Generated with Business Rules)

- **Tables**: `house1_battery`, `house2_battery` (744 and 256 records)
- **Generated using**: `src/data_integration/generate_battery_trading_data.py`
- **Contains intelligent trading labels** following business rules

- **Training Label Distribution**:
  - Buy: 13% (demand-driven + opportunistic)
  - Hold: 69%
  - Sell: 18%
  - Avg buy price: $46/MWh
  - Avg sell price: $47/MWh
  
- **Battery State Columns**:
  - timestamp
  - house_id
  - battery_soc_percent (20-60% range in current data)
  - battery_charge_kwh
  - battery_available_kwh
  - battery_capacity_remaining_kwh
  - battery_soh_percent
  - **action**: 'buy', 'hold', 'sell' (TRAINING LABELS)
  - **trade_amount_kwh**: Quantity to trade (TRAINING LABELS)
  - **price_per_kwh**: Price at this interval
  - **consumption_kwh**: Actual hourly consumption

- **Subatomic Battery Specifications**:
  - Houses 1-9: 1 battery (40 kWh)
  - Houses 10-11: 2 batteries (80 kWh total)
  - Max charge rate: 10 kW (10 kWh per hour)
  - Max discharge rate: 8 kW (8 kWh per hour)
  - SoC constraints: 20% min, 100% max
  - Efficiency: 95%

- **Energy Capacity Analysis**:
  - Max charging: **240 kWh/day** (10kW × 24h)
  - House 1 consumption: **38 kWh/day**
  - **Trading opportunity: 50-70 kWh/day** surplus for market!

---

## Architecture Design

### Transformer 1: Consumption Predictor

**Purpose**: Predict future energy consumption based on historical appliance usage and patterns

**Inputs** (16-17 features):

- 9 appliance consumption values (from Supabase, normalized)
- 4 temporal features (hour_sin, hour_cos, day_sin, day_cos)
- 3-4 historical pattern features:
  - Last week same time consumption
  - Weekday average consumption
  - 7-day rolling average consumption
  
**Outputs**:

- Day-ahead predictions: 48 values (next 24 hours in 30-min intervals)
- Week-ahead predictions: 336 values (next 7 days)

**Architecture**:

- d_model: 384
- n_heads: 6
- n_layers: 5
- Sequence length: 48 (24 hours of context)
- Loss: MSE weighted by horizon (day=1.0, week=0.5)

### Transformer 2: Trading Predictor

**Purpose**: Predict electricity prices and make optimal buy/sell/hold trading decisions

**Inputs** (30 features - ENHANCED):

**Core Features (20)**:
- 3 consumption prediction features (from T1: peak, total, average)
- 6 pricing features (current price, 4 lags, weekly average)
- 4 battery state features (SoC%, available kWh, remaining capacity, SoH%)
- 3 historical context features (actual consumption last 24h, T1 prediction error, recent profit/loss)
- 4 temporal features (hour_sin, hour_cos, day_sin, day_cos)

**Enhanced Features (10)** - ⭐ NEW! Encode business logic:
- **Price Trend Features (4)**:
  - price_percentile_recent: Where price ranks 0-100 in recent window
    * <20: "Very low price" → BUY signal
    * >80: "Very high price" → SELL signal
  - price_vs_min_ratio: current / min_recent
  - price_vs_max_ratio: current / max_recent
  - price_volatility: std dev of recent prices
  
- **Demand-Driven Features (3)**:
  - energy_deficit_next_hours: Shortage in next 8h (kWh)
    * >0: MUST BUY to cover household
  - energy_surplus: Excess beyond next 8h (kWh)
    * >0: SHOULD SELL to avoid waste
  - hours_of_coverage: How long battery can sustain
    * <4 hours: Critical need to buy
  
- **Trading Opportunity Features (3)**:
  - buy_signal_strength: 0-1 (1 = strong buy)
  - sell_signal_strength: 0-1 (1 = strong sell)
  - hold_score: 0-1 (1 = should hold)

✅ **These features teach the model WHEN to trade based on business rules!**

**Outputs**:

- Price predictions: 48 values (next 24 hours)
- Trading decisions: 48x3 (buy/sell/hold probabilities for each interval)
- Trade quantities: 48 values (kWh amounts to trade)

**Architecture**:

- d_model: 512
- n_heads: 8
- n_layers: 6
- Multi-task loss:
  - 30% price prediction accuracy (MAE)
  - 30% trading decision correctness (CrossEntropy)
  - 40% profitability reward

**Trading Strategy - Business Rules**:

⚠️ **CRITICAL**: Model must learn these rules from training data!

**Hard Constraints** (MUST follow):
1. **MUST BUY**: Price < $20/MWh AND SoC < 100%
2. **MUST SELL**: Price > $40/MWh AND SoC > 20%
3. **MUST HOLD**: SoC ≤ 20% (safety threshold)
4. **MUST HOLD**: Price < $2.7/MWh (10% of household rate)

**Demand-Driven Trading** (Priority Logic):
1. **MUST BUY**: Energy deficit exists (not enough for next 4-8 hours)
2. **SHOULD SELL**: Energy surplus exists (more than needed for next 2 hours)

**Opportunistic Trading** (Model should learn):
- Buy when price in bottom 20-30% of recent 48-96 hour window
- Sell when price in top 20-30% of recent window
- Use **price percentile** features to teach model relative price levels

**Priority Order**:
1. Match household power demands (must-have)
2. Sell (Max Capacity - Daily Usage) to market
3. Sell excess power not needed
4. Balance profit margin and minimize idle battery

---

## Implementation Plan

### Phase 1: Clean Up Old Architecture ✅

**Files to Remove**:

1. `src/models/transformer_model.py`
2. `src/models/feature_engineering.py`
3. `src/training/train_transformer_local.py`

**Action**: Delete these files to start fresh

---

### Phase 2: Build Data Adapter

**File**: `src/data_integration/data_adapter.py`

**Purpose**: Unified interface to load and merge data from multiple sources

**Functions**:

```python
def load_consumption_data(
    house_id: int = None,
    source: str = 'supabase'
) -> pd.DataFrame:
    """
    Load consumption data from Supabase.
    
    Args:
        house_id: Specific house (1-11) or None for all houses
        source: 'supabase' (from august11homeconsumption or houseX tables)
    
    Returns:
        DataFrame with columns:
        - timestamp
        - house_id
        - total_consumption_kwh
        - appliance_ac
        - appliance_washing_drying
        - appliance_fridge
        - appliance_ev_charging
        - appliance_dishwasher
        - appliance_computers
        - appliance_stove
        - appliance_water_heater
        - appliance_misc
    """

def load_pricing_data(
    start_date: datetime = None,
    end_date: datetime = None
) -> pd.DataFrame:
    """
    Load pricing data from Supabase.
    
    Returns:
        DataFrame with columns:
        - timestamp
        - price_per_kwh
    """

def generate_battery_data(
    timestamps: pd.DatetimeIndex,
    consumption_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate synthetic battery state data.
    
    Returns:
        DataFrame with columns:
        - timestamp
        - battery_soc_percent
        - battery_soh_percent
        - battery_charge_kwh
        - battery_available_kwh
    """

def merge_all_data(
    consumption_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    battery_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge consumption, pricing, and battery data on timestamp.
    
    Handles:
    - Date alignment (consumption is Aug 2025, pricing is Oct 2024)
    - Missing value interpolation
    - Timezone handling
    
    Returns:
        Complete DataFrame ready for feature engineering
    """
```

**Key Challenge**: Date mismatch

- Consumption data: August 2025
- Pricing data: October 2024
- **Solution**: Use pricing data patterns (time-of-day, day-of-week) applied to consumption data timestamps

---

### Phase 3: Build Consumption Transformer

#### 3.1 Feature Engineering

**File**: `src/models/feature_engineering_consumption.py`

```python
class ConsumptionFeatureEngineer:
    """
    Feature engineering for consumption prediction.
    """
    
    def __init__(self):
        self.appliance_scaler = StandardScaler()
        self._fitted = False
    
    def extract_appliance_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and normalize 9 appliance features from Supabase data.
        
        Returns: (n_samples, 9)
        """
    
    def extract_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract cyclical temporal features.
        
        Returns: (n_samples, 4)
        - hour_sin, hour_cos
        - day_sin, day_cos
        """
    
    def extract_historical_patterns(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate historical consumption patterns.
        
        Returns: (n_samples, 3-4)
        - last_week_same_time
        - weekday_average
        - rolling_7day_average
        """
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """
        Prepare complete feature matrix.
        
        Returns: (n_samples, 16-17 features)
        """
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets_consumption: np.ndarray,
        sequence_length: int = 48,
        horizons: Dict[str, int] = {'day': 48, 'week': 336}
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for training.
        
        Returns:
        - X: (n_sequences, 48, 16-17)
        - y: {'consumption_day': (n_sequences, 48),
              'consumption_week': (n_sequences, 336)}
        """
```

#### 3.2 Model Architecture

**File**: `src/models/consumption_transformer.py`

```python
class ConsumptionTransformer(nn.Module):
    """
    Transformer model for multi-horizon consumption prediction.
    """
    
    def __init__(
        self,
        n_features: int = 17,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 5,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        horizons: Dict[str, int] = {'day': 48, 'week': 336}
    ):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
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
        
        # Consumption prediction heads (one per horizon)
        self.consumption_heads = nn.ModuleDict()
        for horizon_name, horizon_len in horizons.items():
            self.consumption_heads[horizon_name] = nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, horizon_len)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, sequence_length=48, n_features=17)
        
        Returns:
            {
                'consumption_day': (batch, 48),
                'consumption_week': (batch, 336)
            }
        """


class ConsumptionLoss(nn.Module):
    """Loss function for consumption prediction."""
    
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
        Calculate weighted MSE loss.
        
        Returns:
            total_loss, loss_dict
        """
```

---

### Phase 4: Build Trading Transformer

#### 4.1 Trading Optimizer

**File**: `src/models/trading_optimizer.py`

```python
def calculate_optimal_trading_decisions(
    predicted_consumption: np.ndarray,
    actual_prices: np.ndarray,
    battery_state: Dict
) -> Dict:
    """
    Calculate optimal trading decisions in hindsight for supervised learning.
    
    Args:
        predicted_consumption: (48,) array from Consumption Transformer
        actual_prices: (48,) array from Supabase pricing data
        battery_state: Dict with current SoC, capacity, etc.
    
    Strategy:
        1. Calculate required energy with 10% buffer
        2. Buy when price is in bottom 25% percentile AND battery has capacity
        3. Sell when price is in top 25% percentile AND battery has excess
        4. Hold otherwise
        5. Respect battery constraints (20-90% SoC)
    
    Returns:
        {
            'optimal_decisions': (48,) array [0=Buy, 1=Hold, 2=Sell],
            'optimal_quantities': (48,) array of kWh amounts,
            'expected_profit': float
        }
    """
```

#### 4.2 Feature Engineering

**File**: `src/models/feature_engineering_trading.py`

```python
class TradingFeatureEngineer:
    """
    Feature engineering for trading decisions.
    """
    
    def extract_consumption_predictions(
        self,
        consumption_transformer: nn.Module,
        recent_data: np.ndarray
    ) -> np.ndarray:
        """
        Get predictions from Consumption Transformer.
        
        Returns: (n_samples, 3)
        - predicted_peak
        - predicted_total
        - predicted_average
        """
    
    def extract_pricing_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract pricing features from Supabase data.
        
        Returns: (n_samples, 6)
        - current_price
        - price_lag_1, price_lag_2, price_lag_3, price_lag_4
        - weekly_avg_price
        """
    
    def extract_battery_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract battery state features (synthetic).
        
        Returns: (n_samples, 4)
        - battery_soc_percent
        - battery_available_kwh
        - battery_remaining_capacity_kwh
        - battery_soh_percent
        """
    
    def extract_historical_context(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract recent performance context.
        
        Returns: (n_samples, 3)
        - actual_consumption_last_24h
        - consumption_prediction_error
        - recent_profit_loss
        """
    
    def prepare_features(
        self,
        consumption_predictions: np.ndarray,
        df: pd.DataFrame
    ) -> np.ndarray:
        """
        Prepare complete feature matrix for trading.
        
        Returns: (n_samples, 20 features)
        """
```

#### 4.3 Model Architecture

**File**: `src/models/trading_transformer.py`

```python
class TradingTransformer(nn.Module):
    """
    Multi-task transformer for price prediction and trading decisions.
    """
    
    def __init__(
        self,
        n_features: int = 20,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
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
        
        # Multi-task heads
        self.price_head = nn.Linear(d_model, 48)
        self.trading_decision_head = nn.Linear(d_model, 48 * 3)
        self.trade_quantity_head = nn.Linear(d_model, 48)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, sequence_length=48, n_features=20)
        
        Returns:
            {
                'predicted_price': (batch, 48),
                'trading_decisions': (batch, 48, 3),  # buy/hold/sell logits
                'trade_quantities': (batch, 48)
            }
        """


class TradingLoss(nn.Module):
    """Multi-task loss for trading transformer."""
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate multi-task loss.
        
        Components:
        - 30% price prediction (MAE)
        - 30% trading decision (CrossEntropy)
        - 40% profitability (negative profit = higher loss)
        
        Returns:
            total_loss, loss_dict
        """
```

---

### Phase 5: Training Utilities

**File**: `src/training/training_utils.py`

```python
def create_data_loaders(
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    batch_size: int = 32,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train and validation data loaders."""

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    y_keys: List[str]
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    y_keys: List[str]
) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """Save model checkpoint."""

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """Load model checkpoint."""
```

---

### Phase 6: Google Colab Training Notebook

**File**: `notebooks/train_dual_transformers.ipynb`

#### Section 1: Setup & Environment

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository or navigate to project
%cd /content/drive/MyDrive/energymvp

# Install dependencies using uv
!pip install uv
!uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!uv pip install pandas numpy scikit-learn matplotlib seaborn supabase python-dotenv

# Set random seeds
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
```

#### Section 2: Data Loading from Supabase

```python
# Import project modules
import sys
sys.path.append('/content/drive/MyDrive/energymvp')

from src.data_integration.data_adapter import (
    load_consumption_data,
    load_pricing_data,
    generate_battery_data,
    merge_all_data
)

print("="*70)
print("LOADING DATA FROM SUPABASE")
print("="*70)

# Load consumption data (August 2025, 11 houses)
print("\n1. Loading consumption data...")
consumption_df = load_consumption_data(source='supabase')
print(f"   ✓ Loaded {len(consumption_df):,} consumption records")
print(f"   ✓ Date range: {consumption_df['timestamp'].min()} to {consumption_df['timestamp'].max()}")
print(f"   ✓ Houses: {consumption_df['house_id'].unique()}")

# Load pricing data (October 2024, CA market)
print("\n2. Loading pricing data...")
pricing_df = load_pricing_data()
print(f"   ✓ Loaded {len(pricing_df):,} pricing records")
print(f"   ✓ Price range: ${pricing_df['price_per_kwh'].min():.4f} to ${pricing_df['price_per_kwh'].max():.4f}")

# Generate battery data
print("\n3. Generating synthetic battery data...")
battery_df = generate_battery_data(
    timestamps=consumption_df['timestamp'],
    consumption_data=consumption_df
)
print(f"   ✓ Generated {len(battery_df):,} battery state records")

# Merge all data
print("\n4. Merging all data sources...")
df_complete = merge_all_data(consumption_df, pricing_df, battery_df)
print(f"   ✓ Complete dataset: {len(df_complete):,} records")
print(f"   ✓ Columns: {df_complete.columns.tolist()}")

print("\n" + "="*70)
print("DATA LOADING COMPLETE")
print("="*70)
```

#### Section 3: Preflight Validation (CPU)

```python
print("="*70)
print("PREFLIGHT VALIDATION - CPU ONLY")
print("="*70)

# Import models
from src.models.consumption_transformer import ConsumptionTransformer, ConsumptionLoss
from src.models.trading_transformer import TradingTransformer, TradingLoss
from src.models.feature_engineering_consumption import ConsumptionFeatureEngineer
from src.models.feature_engineering_trading import TradingFeatureEngineer
from src.training.training_utils import create_data_loaders, train_epoch, validate

# Test 1: Consumption Transformer forward pass
print("\n1. Testing Consumption Transformer...")
engineer_consumption = ConsumptionFeatureEngineer()
features_consumption = engineer_consumption.prepare_features(df_complete, fit=True)
X_cons, y_cons = engineer_consumption.create_sequences(
    features_consumption,
    df_complete['total_consumption_kwh'].values,
    sequence_length=48,
    horizons={'day': 48, 'week': 336}
)
print(f"   ✓ Features: {features_consumption.shape}")
print(f"   ✓ Sequences: X={X_cons.shape}")

model_consumption = ConsumptionTransformer(n_features=features_consumption.shape[1])
x_test = torch.FloatTensor(X_cons[:2])
output_cons = model_consumption(x_test)
print(f"   ✓ Forward pass successful")
print(f"   ✓ Output shapes: {[(k, v.shape) for k, v in output_cons.items()]}")

# Test 2: Trading Transformer forward pass
print("\n2. Testing Trading Transformer...")
engineer_trading = TradingFeatureEngineer()
features_trading = engineer_trading.prepare_features(
    output_cons['consumption_day'].detach().numpy(),
    df_complete.iloc[:len(output_cons['consumption_day'])]
)
print(f"   ✓ Trading features: {features_trading.shape}")

model_trading = TradingTransformer(n_features=features_trading.shape[1])
x_test_trading = torch.FloatTensor(features_trading[:2].reshape(2, 48, -1))
output_trading = model_trading(x_test_trading)
print(f"   ✓ Forward pass successful")
print(f"   ✓ Output shapes: {[(k, v.shape) for k, v in output_trading.items()]}")

# Test 3: Data pipeline validation
print("\n3. Validating data pipeline...")
assert not np.isnan(features_consumption).any(), "NaN values in consumption features!"
assert not np.isinf(features_consumption).any(), "Inf values in consumption features!"
assert not np.isnan(features_trading).any(), "NaN values in trading features!"
assert not np.isinf(features_trading).any(), "Inf values in trading features!"
print("   ✓ No NaN/Inf values detected")

# Test 4: Quick training test (2 epochs on CPU)
print("\n4. Quick training test (2 epochs on CPU)...")
train_loader, val_loader, y_keys = create_data_loaders(
    X_cons[:100],
    {k: v[:100] for k, v in y_cons.items()},
    batch_size=8,
    train_split=0.8
)

criterion = ConsumptionLoss()
optimizer = torch.optim.AdamW(model_consumption.parameters(), lr=1e-4)

for epoch in range(2):
    train_loss, _ = train_epoch(
        model_consumption, train_loader, criterion, optimizer,
        torch.device('cpu'), y_keys
    )
    val_loss, _ = validate(
        model_consumption, val_loader, criterion,
        torch.device('cpu'), y_keys
    )
    print(f"   Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

print("\n" + "="*70)
print("✅ PREFLIGHT VALIDATION PASSED - READY FOR GPU TRAINING")
print("="*70)
```

#### Section 4: Train Consumption Transformer (GPU)

```python
print("="*70)
print("TRAINING CONSUMPTION TRANSFORMER")
print("="*70)

# Move to GPU
device = torch.device('cuda')
model_consumption = ConsumptionTransformer(n_features=features_consumption.shape[1]).to(device)

# Create full data loaders
train_loader, val_loader, y_keys = create_data_loaders(
    X_cons, y_cons,
    batch_size=32,
    train_split=0.8
)

# Training setup
criterion = ConsumptionLoss()
optimizer = torch.optim.AdamW(model_consumption.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Training loop
num_epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0

history = {'train_loss': [], 'val_loss': []}

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    
    # Train
    train_loss, train_metrics = train_epoch(
        model_consumption, train_loader, criterion,
        optimizer, device, y_keys
    )
    
    # Validate
    val_loss, val_metrics = validate(
        model_consumption, val_loader, criterion,
        device, y_keys
    )
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Track history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    
    # Save checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        save_checkpoint(
            model_consumption, optimizer, epoch,
            {'train_loss': train_loss, 'val_loss': val_loss},
            '/content/drive/MyDrive/energymvp/checkpoints/consumption_transformer_best.pt'
        )
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n  Early stopping after {epoch} epochs")
            break

print("\n" + "="*70)
print("✅ CONSUMPTION TRANSFORMER TRAINING COMPLETE")
print("="*70)
```

#### Section 5: Train Trading Transformer (GPU)

```python
print("="*70)
print("TRAINING TRADING TRANSFORMER")
print("="*70)

# Load best consumption model
checkpoint = load_checkpoint(
    '/content/drive/MyDrive/energymvp/checkpoints/consumption_transformer_best.pt',
    model_consumption
)
model_consumption.eval()

# Generate consumption predictions for all data
print("\n1. Generating consumption predictions...")
all_predictions = []
with torch.no_grad():
    for i in range(0, len(X_cons), 32):
        batch = torch.FloatTensor(X_cons[i:i+32]).to(device)
        pred = model_consumption(batch)
        all_predictions.append(pred['consumption_day'].cpu().numpy())

consumption_predictions = np.vstack(all_predictions)
print(f"   ✓ Generated {len(consumption_predictions):,} predictions")

# Calculate optimal trading labels
print("\n2. Calculating optimal trading labels...")
from src.models.trading_optimizer import calculate_optimal_trading_decisions

optimal_labels = []
for i in range(len(consumption_predictions)):
    labels = calculate_optimal_trading_decisions(
        predicted_consumption=consumption_predictions[i],
        actual_prices=pricing_df['price_per_kwh'].values[i:i+48],
        battery_state={
            'current_charge_kwh': battery_df['battery_charge_kwh'].iloc[i],
            'capacity_kwh': 13.5,
            'min_soc': 0.20,
            'max_soc': 0.90
        }
    )
    optimal_labels.append(labels)

print(f"   ✓ Calculated {len(optimal_labels):,} optimal trading labels")

# Prepare trading features
print("\n3. Preparing trading features...")
engineer_trading = TradingFeatureEngineer()
features_trading = engineer_trading.prepare_features(
    consumption_predictions,
    df_complete.iloc[:len(consumption_predictions)]
)

# Create sequences
X_trading, y_trading = engineer_trading.create_sequences(
    features_trading,
    optimal_labels,
    sequence_length=48
)

print(f"   ✓ Trading sequences: X={X_trading.shape}")

# Initialize trading model
model_trading = TradingTransformer(n_features=features_trading.shape[1]).to(device)

# Create data loaders
train_loader_trading, val_loader_trading, y_keys_trading = create_data_loaders(
    X_trading, y_trading,
    batch_size=32,
    train_split=0.8
)

# Training setup
criterion_trading = TradingLoss()
optimizer_trading = torch.optim.AdamW(
    model_trading.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# Training loop (similar to consumption transformer)
num_epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0

history_trading = {'train_loss': [], 'val_loss': []}

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    
    # Train
    train_loss, train_metrics = train_epoch(
        model_trading, train_loader_trading, criterion_trading,
        optimizer_trading, device, y_keys_trading
    )
    
    # Validate
    val_loss, val_metrics = validate(
        model_trading, val_loader_trading, criterion_trading,
        device, y_keys_trading
    )
    
    # Track history
    history_trading['train_loss'].append(train_loss)
    history_trading['val_loss'].append(val_loss)
    
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Price Loss: {train_metrics['price_loss']:.4f}")
    print(f"  Trading Loss: {train_metrics['trading_loss']:.4f}")
    print(f"  Profit: ${-train_metrics['profit_loss']:.2f}")
    
    # Save checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        save_checkpoint(
            model_trading, optimizer_trading, epoch,
            train_metrics,
            '/content/drive/MyDrive/energymvp/checkpoints/trading_transformer_best.pt'
        )
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n  Early stopping after {epoch} epochs")
            break

print("\n" + "="*70)
print("✅ TRADING TRANSFORMER TRAINING COMPLETE")
print("="*70)
```

#### Section 6: End-to-End Testing

```python
print("="*70)
print("END-TO-END TESTING")
print("="*70)

# Load both best models
load_checkpoint(
    '/content/drive/MyDrive/energymvp/checkpoints/consumption_transformer_best.pt',
    model_consumption
)
load_checkpoint(
    '/content/drive/MyDrive/energymvp/checkpoints/trading_transformer_best.pt',
    model_trading
)

model_consumption.eval()
model_trading.eval()

# Run inference on validation set
print("\n1. Running inference...")
with torch.no_grad():
    # Get sample batch
    sample_X_cons = torch.FloatTensor(X_cons[-10:]).to(device)
    
    # Predict consumption
    consumption_pred = model_consumption(sample_X_cons)
    
    # Prepare trading features
    trading_features = engineer_trading.prepare_features(
        consumption_pred['consumption_day'].cpu().numpy(),
        df_complete.iloc[-10:]
    )
    sample_X_trading = torch.FloatTensor(trading_features.reshape(-1, 48, 20)).to(device)
    
    # Predict trading
    trading_pred = model_trading(sample_X_trading)

print("   ✓ Inference successful")

# Calculate metrics
print("\n2. Calculating metrics...")
consumption_mape = np.mean(
    np.abs(
        (consumption_pred['consumption_day'].cpu().numpy() - y_cons['consumption_day'][-10:]) /
        y_cons['consumption_day'][-10:]
    )
) * 100

print(f"   Consumption MAPE: {consumption_mape:.2f}%")

# Visualize results
print("\n3. Generating visualizations...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Consumption predictions
axes[0, 0].plot(y_cons['consumption_day'][-1], label='Actual')
axes[0, 0].plot(consumption_pred['consumption_day'][-1].cpu().numpy(), label='Predicted')
axes[0, 0].set_title('Consumption Prediction (Next 24h)')
axes[0, 0].legend()

# Plot 2: Price predictions
axes[0, 1].plot(trading_pred['predicted_price'][-1].cpu().numpy())
axes[0, 1].set_title('Price Prediction (Next 24h)')

# Plot 3: Trading decisions
decisions = torch.argmax(trading_pred['trading_decisions'][-1], dim=-1).cpu().numpy()
axes[1, 0].plot(decisions)
axes[1, 0].set_title('Trading Decisions (0=Buy, 1=Hold, 2=Sell)')

# Plot 4: Training history
axes[1, 1].plot(history['train_loss'], label='Consumption Train')
axes[1, 1].plot(history['val_loss'], label='Consumption Val')
axes[1, 1].plot(history_trading['train_loss'], label='Trading Train')
axes[1, 1].plot(history_trading['val_loss'], label='Trading Val')
axes[1, 1].set_title('Training History')
axes[1, 1].legend()
axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/energymvp/checkpoints/training_results.png')
print("   ✓ Saved visualizations")

print("\n" + "="*70)
print("✅ END-TO-END TESTING COMPLETE")
print("="*70)
```

#### Section 7: Model Export

```python
print("="*70)
print("MODEL EXPORT")
print("="*70)

# Save model configs
import json

config_consumption = {
    'n_features': features_consumption.shape[1],
    'd_model': 384,
    'n_heads': 6,
    'n_layers': 5,
    'horizons': {'day': 48, 'week': 336}
}

config_trading = {
    'n_features': features_trading.shape[1],
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6
}

with open('/content/drive/MyDrive/energymvp/checkpoints/consumption_config.json', 'w') as f:
    json.dump(config_consumption, f, indent=2)

with open('/content/drive/MyDrive/energymvp/checkpoints/trading_config.json', 'w') as f:
    json.dump(config_trading, f, indent=2)

# Save training logs
training_logs = {
    'consumption': {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'best_val_loss': best_val_loss,
        'final_mape': float(consumption_mape)
    },
    'trading': {
        'train_loss': history_trading['train_loss'],
        'val_loss': history_trading['val_loss']
    }
}

with open('/content/drive/MyDrive/energymvp/checkpoints/training_logs.json', 'w') as f:
    json.dump(training_logs, f, indent=2)

print("✓ Saved model configs")
print("✓ Saved training logs")
print("\n✅ All files ready for download from Google Drive")
print("="*70)
```

---

## Success Criteria

### Consumption Transformer

- ✅ MAPE < 15% on day-ahead consumption predictions
- ✅ Model converges within 50-100 epochs
- ✅ No overfitting (train/val loss gap < 20%)

### Trading Transformer

- ✅ Price MAE < $0.05/kWh
- ✅ Trading decisions achieve 20-40% cost savings vs baseline
- ✅ 10% energy buffer maintained
- ✅ Battery constraints respected (20-90% SoC)

### System Integration

- ✅ Consumption → Trading pipeline works smoothly
- ✅ Preflight validation catches errors before GPU training
- ✅ Models train successfully on Colab GPU
- ✅ Checkpoints save correctly to Google Drive

---

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1 | Remove old architecture | 15 min |
| Phase 2 | Build data adapter | 3-4 hours |
| Phase 3 | Build Consumption Transformer | 6-8 hours |
| Phase 4 | Build Trading Transformer | 8-10 hours |
| Phase 5 | Training utilities | 2-3 hours |
| Phase 6 | Colab notebook | 4-5 hours |
| **Total** | | **24-30 hours** |

**GPU Training Time** (on Colab):

- Consumption Transformer: 2-3 hours
- Trading Transformer: 2-3 hours
- Total GPU time: **4-6 hours**

---

## Implementation Status

### ✅ Completed (as of 2025-10-08 - UPDATED)

1. **Phase 1: Clean Up** ✅
   - Removed old transformer architecture
   - Created directory structure

2. **Phase 2: Data Infrastructure** ✅
   - `src/data_integration/data_adapter.py` - Loads real Supabase data
   - `src/data_integration/supabase_connector.py` - ✅ Updated with LMP filter
   - `src/data_integration/generate_battery_trading_data.py` - ✅ Generates training labels
   - `src/models/trading_optimizer.py` - ✅ Business rules for label generation
   - ✅ Fixed consumption data loading (use appliances, not daily total)
   - ✅ Fixed pricing data loading (filter for LMP only)
   - ✅ Generated 1,000 battery trading labels (744 House 1, 256 House 2)
   - ✅ Uploaded training data to Supabase (house1_battery, house2_battery tables)

3. **Phase 3.1: Consumption Feature Engineering** ✅
   - `src/models/feature_engineering_consumption.py` - 17 features
   - Tested and working

4. **Phase 4.1: Trading Feature Engineering** ✅ **NEW!**
   - `src/models/feature_engineering_trading.py` - ✅ **Enhanced with 30 features!**
   - Core features (20): consumption, pricing, battery, historical, temporal
   - **Enhanced features (10)**: price trends, demand-driven, trading opportunities
   - ✅ These features encode all business rules for model learning
   - ✅ Handles hourly consumption calculation correctly
   - ✅ Tested and working

### 🔨 In Progress

4. **Phase 3.2: Consumption Transformer Model**
   - Need to create: `src/models/consumption_transformer.py`
   - Architecture designed (384d, 6 heads, 5 layers)

5. **Phase 4: Trading Transformer (3 files)**
   - Need to create: `src/models/feature_engineering_trading.py`
   - Need to create: `src/models/trading_optimizer.py`
   - Need to create: `src/models/trading_transformer.py`

6. **Phase 5: Training Utilities**
   - Need to create: `src/training/training_utils.py`

7. **Phase 6: Colab Notebook**
   - Need to create: `notebooks/train_dual_transformers.ipynb`

### 📊 Progress: 60% Complete ⬆️ UPDATED!

**What Works:**

- ✅ Data loading from Supabase (consumption + pricing) - **FIXED data structure issues**
- ✅ Battery trading label generation with comprehensive business rules
- ✅ Feature engineering for consumption prediction (17 features)
- ✅ **Feature engineering for trading with enhanced features (30 features)**
- ✅ **Business rules implementation (hard constraints + opportunistic)**
- ✅ **Training data in Supabase (1,000 labeled examples)**

**What's Next:**

- ⚠️ Build consumption transformer model architecture
- ⚠️ Build trading transformer model architecture
- ⚠️ Implement custom loss function with business rule penalties
- ⚠️ Create PyTorch dataset loaders
- ⚠️ Create training utilities
- ⚠️ Create Colab training notebook

**Estimated Time Remaining:** 10-15 hours development + 4-6 hours GPU training

**Key Insights from Today:**
1. ⚠️ Consumption `Total kWh` is DAILY total - must calculate from appliances!
2. ⚠️ Pricing data must filter for `LMP_TYPE='LMP'` only
3. ✅ Enhanced features teach model business logic (price percentiles, energy deficit/surplus)
4. ✅ Training labels generated with demand-driven + opportunistic trading rules
5. ✅ Battery capacity (240 kWh/day) far exceeds consumption (38 kWh/day) - plenty of trading opportunity!

### 🚀 Ready for Handoff

See `HANDOFF.md` for next steps and what to build.
