# System Architecture

**Energy Trading System with Transformer-Based Forecasting**

---

## Overview

The system uses a multi-task Transformer neural network to predict energy consumption and prices across multiple time horizons, enabling autonomous peer-to-peer energy trading among households with battery storage.

---

## Core Components

### 1. Data Pipeline

**Synthetic Data Generator** (`src/data_integration/synthetic_data_generator.py`)

- Generates realistic household energy consumption patterns
- Simulates 9 appliances with time-dependent behavior
- Battery sensor data (SoC, SoH, charge, cycles)
- Weather data (temperature, solar irradiance)
- Time-of-use pricing with daily cycles

**Feature Engineering** (`src/models/feature_engineering.py`)

- Extracts 24 features from raw data:
  - 9 appliance consumption features (scaled)
  - 4 battery state features (scaled)
  - 2 weather features (scaled)
  - 4 temporal features (cyclical sin/cos encoding)
  - 5 pricing features (current + 4 lags, scaled)
- Creates sequences: (batch, 48 timesteps, 24 features)
- Train/validation/test splitting

---

### 2. Transformer Model

**EnergyTransformer** (`src/models/transformer_model.py`)

```
Architecture:
- Input: (batch, 48, 24) - 24-hour lookback
- Projection: Linear(24 → 512)
- Positional Encoding: Sinusoidal
- Transformer Encoder: 6 layers, 8 heads, 2048 dim FFN
- Output: 6 prediction streams

Parameters: 25.8M (~103 MB FP32)
```

**Multi-Task Heads:**

- Consumption prediction (MSE loss, weight=0.6)
- Price prediction (MAE loss, weight=0.4)

**Multi-Horizon Outputs:**

- Day: 48 intervals (24 hours)
- Week: 336 intervals (7 days)
- Month: 1440 intervals (30 days)

**Loss Function:**

```python
total_loss = Σ (horizon_weight × task_weight × task_loss)

Horizon weights: {day: 1.0, week: 0.5, month: 0.25}
Task weights: {consumption: 0.6, price: 0.4}
```

---

### 3. Training Pipeline

**Local Training** (`src/training/train_transformer_local.py`)

- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: ReduceLROnPlateau (patience=3)
- Early stopping: patience=5 epochs
- Gradient clipping: max_norm=1.0
- Checkpoint: Best model by validation loss

**Training Configuration:**

```yaml
batch_size: 32
learning_rate: 0.0001
epochs: 50
sequence_length: 48
patience: 5
```

---

### 4. Supporting Systems

**Battery Management** (`src/models/battery_manager.py`)

- Capacity: 13.5 kWh (Tesla Powerwall equivalent)
- Efficiency: 90% round-trip
- Charge/discharge limits: 5 kW
- Reserve: 10% minimum, 80% maximum

**Grid Constraints** (`src/models/grid_constraints.py`)

- Import limit: 10 kWh per 30-min interval
- Export limit: 4 kWh per 30-min interval
- Validates all transactions

**Profitability Calculator** (`src/models/profitability.py`)

- Scores trading opportunities (0-100)
- Power need signals (GREEN/RED)
- Considers battery state, predictions, market conditions

**Trading System** (`src/trading/`)

- Autonomous buy/sell decisions
- Instant P2P matching
- Profit margin enforcement (5% minimum)
- Transmission loss modeling (5%)

**Federated Learning** (`src/federated/federated_aggregator.py`)

- FedAvg algorithm
- Privacy-preserving model updates
- Aggregates weights from multiple nodes

---

## Data Flow

```
1. Raw Data (Supabase/Synthetic)
   ↓
2. Feature Engineering
   - Extract 24 features
   - Create sequences (batch, 48, 24)
   ↓
3. Transformer Model
   - Encode with attention
   - Multi-task prediction
   ↓
4. Predictions
   - Consumption: day/week/month
   - Price: day/week/month
   ↓
5. Trading Logic
   - Buy/sell/hold decisions
   - Battery optimization
   ↓
6. Market Mechanism
   - P2P matching
   - Transaction execution
```

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Central Coordinator                    │
│  - Price Signals                                         │
│  - FedAvg Aggregation                                    │
│  - Market Rules                                          │
└─────────────────┬───────────────────────────────────────┘
                  │
         ┌────────┴────────┐
         │  Market Matching │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐   ┌───▼────┐   ┌───▼────┐
│ Node 1 │   │ Node 2 │...│ Node N │
│        │   │        │   │        │
│ Trans- │   │ Trans- │   │ Trans- │
│ former │   │ former │   │ former │
│        │   │        │   │        │
│ Battery│   │ Battery│   │ Battery│
│ (13.5  │   │ (13.5  │   │ (13.5  │
│  kWh)  │   │  kWh)  │   │  kWh)  │
└────────┘   └────────┘   └────────┘
```

---

## Model Inputs & Outputs

### Inputs (per household)

```
Sequence: 48 intervals (24 hours lookback)

Features per timestep (24 total):
[
  fridge, washing_machine, dishwasher, ev_charging,
  ac, stove, water_heater, computers, misc,           # 9 appliances
  
  battery_soc, battery_soh, battery_charge,
  battery_cycles,                                      # 4 battery
  
  temperature, solar_irradiance,                       # 2 weather
  
  hour_sin, hour_cos, day_sin, day_cos,              # 4 temporal
  
  price_t0, price_t-1, price_t-2,
  price_t-3, price_t-4                                 # 5 pricing
]
```

### Outputs (per household)

```
6 prediction streams:

Consumption:
- consumption_day:   48 values   (24 hours)
- consumption_week:  336 values  (7 days)
- consumption_month: 1440 values (30 days)

Price:
- price_day:   48 values   (24 hours)
- price_week:  336 values  (7 days)
- price_month: 1440 values (30 days)
```

---

## Trading Logic

### Decision Flow

```python
# Every 30-minute interval:

1. Get predictions from Transformer
   consumption_pred = model(features)['consumption_day']
   price_pred = model(features)['price_day']

2. Calculate profitability
   score = profitability_calculator.score(
       battery_state,
       consumption_pred,
       price_pred,
       current_price
   )

3. Make decision
   if score > 75 and price_pred < current_price:
       decision = BUY
   elif score > 75 and price_pred > current_price:
       decision = SELL
   else:
       decision = HOLD

4. Execute trades
   market.match_trades(all_node_decisions)

5. Update battery
   battery.charge() or battery.discharge()
```

---

## Federated Learning

Every 3 hours (6 intervals):

1. Each node trains locally on its data
2. Nodes send weight updates to central server
3. Central server aggregates using FedAvg
4. Updated global model distributed back to nodes

**Privacy:** Raw data never leaves the household.

---

## Configuration

All parameters in `config/config.yaml`:

- Transformer architecture
- Training hyperparameters
- Battery specifications
- Trading rules
- Grid constraints
- Federated learning settings

---

## Scalability

**Current:** 10 households  
**Designed for:** 100+ households with minimal changes

**Key scaling factors:**

- Transformer: Fixed size (25.8M params)
- Market matching: O(N²) worst case
- Federated aggregation: O(N) linear

**Bottleneck:** Market matching for large N
**Solution:** Hierarchical matching or auction mechanism

---

## Technology Stack

- **ML Framework:** PyTorch 2.x
- **Data Processing:** Pandas, NumPy
- **Feature Scaling:** scikit-learn StandardScaler
- **Configuration:** YAML
- **Database:** Supabase (PostgreSQL)
- **Training:** Google Colab GPU (for production)
- **Language:** Python 3.12

---

## File Structure

```
src/
├── data_integration/
│   ├── synthetic_data_generator.py  # Synthetic data for testing
│   ├── supabase_connector.py        # Database interface
│   └── data_adapter.py               # Schema transformations
│
├── models/
│   ├── transformer_model.py         # EnergyTransformer & loss
│   ├── feature_engineering.py       # Feature extraction
│   ├── battery_manager.py           # Battery simulation
│   ├── grid_constraints.py          # Grid limits
│   ├── profitability.py             # Trade scoring
│   └── central_model.py             # Coordinator
│
├── training/
│   └── train_transformer_local.py   # Training pipeline
│
├── trading/
│   ├── trading_logic.py             # Buy/sell decisions
│   └── market_mechanism.py          # P2P matching
│
├── federated/
│   └── federated_aggregator.py      # FedAvg implementation
│
└── simulation/
    ├── household_node.py            # Node integration
    └── run_demo.py                  # Full simulation
```

---

## Deployment Strategy

### Development (Current)

- Synthetic data on local machine
- CPU training for verification
- Single household testing

### Staging

- Real Supabase data
- Google Colab GPU training
- Multi-household simulation

### Production

- Live Supabase connection
- Trained model deployment
- Real-time trading execution
- Federated learning updates

---

## Performance Characteristics

**Model:**

- Inference time: <100ms per household (GPU)
- Training time: 2-4 hours on Colab T4 (50 epochs)
- Memory: ~500MB during inference

**System:**

- Interval processing: <1 second for 10 households
- Market matching: <100ms
- Database queries: <200ms

---

## Next Integration Points

When real data arrives:

1. Swap `generate_synthetic_household_data()` with `load_real_data_from_supabase()`
2. Verify 24 features match expected schema
3. Retrain on GPU with real data
4. Deploy trained model to nodes
5. Enable live trading

**Estimated integration time:** 4-6 hours
