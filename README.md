# Federated Energy Trading System

> Dual-transformer AI system for household energy management: consumption forecasting + intelligent price-aware trading to maximize savings against PG&E rates.

## Overview

This system uses a state-of-the-art Transformer model to predict energy consumption and prices across multiple time horizons (day, week, month), enabling autonomous peer-to-peer energy trading among households with battery storage. The goal is to achieve 20-40% cost savings compared to standard utility pricing through intelligent trading and battery optimization.

### Key Features

- **Dual-Transformer Architecture**: Separate models for consumption prediction and price-aware trading
- **Transformer 1 (Consumption)**: Predicts household energy usage using appliances, weather, and behavioral patterns
- **Transformer 2 (Trading)**: Predicts prices and makes optimal buy/sell decisions based on battery state
- **Weather Integration**: Live API data for accurate consumption forecasting (AC, heating)
- **Intelligent Trading**: ML-learned trading strategies that maximize savings vs PG&E baseline
- **Battery Optimization**: Ensures 10% energy buffer while minimizing costs
- **Multi-Horizon Predictions**: Day (24h), week (7d) forecasts in 30-min intervals
- **Federated Learning**: Privacy-preserving model updates with FedAvg
- **No Circular Dependencies**: Clean T1 → T2 pipeline with correct causality

## Quick Start

### Prerequisites

- Python 3.12
- `uv` package manager
- PyTorch 2.x
- (Optional) Supabase account for real data

### Installation

```bash
cd /path/to/energymvp
source .venv/bin/activate
uv pip install -e .
```

### Test Components

```bash
# Test synthetic data generation
python src/data_integration/synthetic_data_generator.py

# Test feature engineering
python src/models/feature_engineering.py

# Test transformer model
python src/models/transformer_model.py

# Run local training (CPU)
python src/training/train_transformer_local.py
```

### Configuration

All parameters in `config/config.yaml`:  

- Transformer architecture  
- Training hyperparameters  
- Battery specifications  
- Trading rules  
- Grid constraints

## Project Structure

```
energymvp/
├── README.md                        # This file
├── docs/
│   ├── ARCHITECTURE.md              # System architecture
│   ├── EXECUTIVE_SUMMARY.md         # High-level overview
│   ├── TECHNICAL_REFERENCE.md       # Algorithms & formulas
│   ├── PRE_FLIGHT_RESULTS.md        # Verification results
│   └── archive/                     # Old documentation
├── config/
│   └── config.yaml                  # Configuration
├── src/
│   ├── data_integration/
│   │   ├── synthetic_data_generator.py
│   │   ├── supabase_connector.py
│   │   └── data_adapter.py
│   ├── models/
│   │   ├── transformer_model.py     # 25.8M param model
│   │   ├── feature_engineering.py   # 24 features
│   │   ├── battery_manager.py
│   │   ├── grid_constraints.py
│   │   ├── profitability.py
│   │   └── central_model.py
│   ├── training/
│   │   └── train_transformer_local.py
│   ├── trading/
│   │   ├── trading_logic.py
│   │   └── market_mechanism.py
│   ├── federated/
│   │   └── federated_aggregator.py
│   └── simulation/
│       ├── household_node.py
│       └── run_demo.py
├── checkpoints/                     # Model checkpoints
└── tests/                           # Test suite
```

## System Architecture

```
Raw Data Sources
    ├─ Household: 9 Appliances (kWh per 30-min)
    ├─ Weather API: Temperature, humidity, solar irradiance
    ├─ Historical: Daily usage patterns by day of week
    └─ Market: PG&E pricing (TOU rates)
    ↓
┌─────────────────────────────────────────────────────────┐
│ TRANSFORMER 1: Consumption Predictor                    │
│ Purpose: Pure energy usage forecasting                  │
├─────────────────────────────────────────────────────────┤
│ Inputs (18-20 features):                                │
│   ├─ 9 Appliance consumption (scaled)                  │
│   ├─ 3-4 Weather features (temp, humidity, solar)      │
│   ├─ 4 Temporal patterns (hour/day cyclical)           │
│   └─ 3-4 Historical patterns (daily avg, seasonal)     │
│                                                         │
│ Architecture: 4-6 layers, 6 heads, ~5-8M params        │
│                                                         │
│ Output: Predicted consumption for day/week (30-min)    │
└─────────────────────────────────────────────────────────┘
    ↓
Consumption Predictions (from T1)
    ├─ predicted_consumption_day: 48 values (24h)
    └─ predicted_consumption_week: 336 values (7d)
    ↓
┌─────────────────────────────────────────────────────────┐
│ TRANSFORMER 2: Price Predictor + Trading Engine         │
│ Purpose: Predict prices & optimize buy/sell decisions   │
├─────────────────────────────────────────────────────────┤
│ Inputs (15-18 features):                                │
│   ├─ Consumption predictions from T1                   │
│   ├─ Historical pricing (PG&E rates, lags)             │
│   ├─ Battery state (SoC, available capacity)           │
│   ├─ Historical usage context                          │
│   └─ Temporal patterns                                  │
│                                                         │
│ Architecture: 4-6 layers, 6-8 heads, ~8-12M params     │
│                                                         │
│ Outputs:                                                │
│   ├─ Predicted prices (day/week)                       │
│   ├─ Trading decisions (Buy/Sell/Hold)                 │
│   ├─ Trade quantities (kWh)                            │
│   └─ Battery charging schedule                         │
└─────────────────────────────────────────────────────────┘
    ↓
Trading Execution
    ├─ Buy when: Battery low + Price favorable
    ├─ Sell when: Battery high + Price advantageous
    ├─ Maintain: 10% energy buffer at all times
    └─ Optimize: Maximize savings vs PG&E baseline
    ↓
Cost Savings: 20-40% vs standard PG&E rates
```

## Core Components

### 1. Transformer 1: Consumption Predictor (`consumption_transformer.py`)

- **Architecture**: 4-6 layers, 6 attention heads, 1024 FFN dim
- **Parameters**: 5-8M (~20-32 MB)
- **Input**: 48 timesteps × 18-20 features (appliances + weather + temporal + patterns)
- **Output**: Consumption predictions for day/week (48 and 336 intervals)
- **Loss**: MSE on consumption accuracy
- **Training**: Independent, can validate against actual consumption

### 2. Transformer 2: Price Predictor + Trading (`trading_transformer.py`)

- **Architecture**: 4-6 layers, 6-8 attention heads, 1536 FFN dim
- **Parameters**: 8-12M (~32-48 MB)
- **Input**: Consumption predictions + pricing + battery state + historical context
- **Output**: Price predictions + trading decisions + quantities + battery schedule
- **Loss**: Multi-task (price prediction MAE + trading profit optimization)
- **Training**: Uses T1's predictions as input, learns optimal trading strategy

### 3. Feature Engineering

**For Transformer 1 (18-20 features):**
- 9 appliances (scaled)
- 3-4 weather features (scaled)
- 4 temporal (cyclical)
- 3-4 historical patterns

**For Transformer 2 (15-18 features):**
- Consumption predictions from T1
- Historical pricing (5-6 features)
- Battery state (4 features)
- Historical usage context (3 features)
- Temporal (4 features)

### 3. Training Pipeline (`train_transformer_local.py`)

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: patience=5 epochs
- **Checkpointing**: Best model by validation loss

### 4. Battery Management (`battery_manager.py`)

- **Capacity**: 13.5 kWh (Tesla Powerwall)
- **Efficiency**: 90% round-trip
- **Limits**: 5 kW charge/discharge, 10-80% SoC range

### 5. Battery Management (`battery_manager.py`)

- **Capacity**: 13.5 kWh (Tesla Powerwall)
- **State Tracking**: SoC, SoH, available energy, available capacity
- **Trading Logic**: Integrated with Transformer 2
  - Battery full (>80%): Avoid buying, consider selling
  - Battery low (<20%): Need to buy at optimal prices
  - Always maintain 10% buffer above predicted consumption needs
- **Optimization**: T2 learns when to charge based on price predictions

### 6. Federated Learning (`federated_aggregator.py`)

- **Algorithm**: FedAvg
- **Frequency**: Every 6 intervals (3 hours)
- **Privacy**: Data stays local, only weights shared

## Model Specifications

### Transformer 1 Input Features (18-20 total)

1. **Appliances (9)**: fridge, washing_machine, dishwasher, ev_charging, ac, stove, water_heater, computers, misc
2. **Weather (3-4)**: temperature, humidity, solar_irradiance, (optional: precipitation)
3. **Temporal (4)**: hour_sin, hour_cos, day_sin, day_cos (cyclical encoding)
4. **Historical Patterns (3-4)**:
   - last_week_same_time_consumption
   - avg_consumption_this_weekday  
   - avg_daily_consumption_last_7_days
   - seasonal_factor (optional)

### Transformer 1 Outputs (Consumption Predictions)

- **consumption_day**: 48 intervals (next 24 hours in 30-min chunks)
- **consumption_week**: 336 intervals (next 7 days in 30-min chunks)

### Transformer 2 Input Features (15-18 total)

1. **From T1 (1-3)**: predicted_consumption_next_24h, predicted_peak, predicted_daily_total
2. **Pricing (5-6)**: current_price, price lags (t-1 to t-4), avg_price_same_hour_last_week
3. **Battery State (4)**: soc_percent, available_kwh, available_capacity_kwh, soh_percent
4. **Historical Context (3)**: actual_consumption_last_24h, prediction_error, recent_trading_pnl
5. **Temporal (4)**: hour_sin, hour_cos, day_sin, day_cos

### Transformer 2 Outputs (Trading Decisions)

- **predicted_price_day**: 48 price predictions ($/kWh for next 24h)
- **predicted_price_week**: 336 price predictions ($/kWh for next 7 days)
- **trading_decisions**: Buy/Sell/Hold for each interval
- **trade_quantities**: kWh amounts for each trade
- **battery_schedule**: Charge/discharge plan optimized for cost savings

## Technology Stack

- **Language**: Python 3.12
- **ML Framework**: PyTorch 2.x
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Database**: Supabase (PostgreSQL)
- **Training**: Google Colab GPU
- **Package Manager**: uv
- **Configuration**: YAML

## Documentation

- **ARCHITECTURE.md**: Detailed system architecture and data flow
- **EXECUTIVE_SUMMARY.md**: High-level project overview
- **TECHNICAL_REFERENCE.md**: Algorithms, formulas, and implementation details
