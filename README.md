# Federated Energy Trading System

> AI-powered peer-to-peer energy trading using Transformer neural networks for multi-horizon consumption and price forecasting.

## Overview

This system uses a state-of-the-art Transformer model to predict energy consumption and prices across multiple time horizons (day, week, month), enabling autonomous peer-to-peer energy trading among households with battery storage. The goal is to achieve 20-40% cost savings compared to standard utility pricing through intelligent trading and battery optimization.

### Key Features

- **Transformer Forecasting**: Multi-task, multi-horizon predictions (25.8M parameter model)
- **24 Input Features**: Appliances, battery state, weather, temporal, and pricing data
- **Multi-Horizon Predictions**: Day (24h), week (7d), month (30d) forecasts
- **Federated Learning**: Privacy-preserving model updates with FedAvg
- **Autonomous Trading**: Buy/sell/hold decisions based on forecasts
- **Battery Optimization**: Smart charging/discharging with grid constraints
- **Instant P2P Matching**: Real-time trade execution

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
Raw Data (480 samples, 10 days)
    │
    ├─ 9 Appliances (fridge, EV, AC, etc.)
    ├─ 4 Battery Sensors (SoC, SoH, charge, cycles)
    ├─ 2 Weather (temp, solar)
    └─ Pricing (TOU)
    ↓
Feature Engineering (24 features)
    ↓
Transformer Model (25.8M params)
    ├─ 6 layers, 8 heads
    ├─ Multi-task: consumption + price
    └─ Multi-horizon: day/week/month
    ↓
Predictions
    ├─ Consumption: 48/336/1440 intervals
    └─ Price: 48/336/1440 intervals
    ↓
Trading Logic
    ├─ Buy/Sell/Hold decisions
    └─ Battery optimization
    ↓
Market Mechanism
    └─ P2P matching & execution
```

## Core Components

### 1. Transformer Model (`transformer_model.py`)

- **Architecture**: 6 layers, 8 attention heads, 2048 FFN dim
- **Parameters**: 25.8M (~103 MB)
- **Input**: 48 timesteps × 24 features
- **Output**: 6 prediction streams (consumption + price × 3 horizons)
- **Loss**: Multi-task weighted (MSE + MAE)

### 2. Feature Engineering (`feature_engineering.py`)

- **24 Features**: 9 appliances + 4 battery + 2 weather + 4 temporal + 5 pricing
- **Scaling**: StandardScaler per feature group
- **Sequences**: 48-timestep windows for training

### 3. Training Pipeline (`train_transformer_local.py`)

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: patience=5 epochs
- **Checkpointing**: Best model by validation loss

### 4. Battery Management (`battery_manager.py`)

- **Capacity**: 13.5 kWh (Tesla Powerwall)
- **Efficiency**: 90% round-trip
- **Limits**: 5 kW charge/discharge, 10-80% SoC range

### 5. Trading System (`trading/`)

- **Decisions**: Buy/sell/hold based on predictions
- **Constraints**: 10 kWh import, 4 kWh export per interval
- **Matching**: Instant P2P with 5% transmission loss

### 6. Federated Learning (`federated_aggregator.py`)

- **Algorithm**: FedAvg
- **Frequency**: Every 6 intervals (3 hours)
- **Privacy**: Data stays local, only weights shared

## Model Specifications

### Input Features (24 total)

1. **Appliances (9)**: fridge, washing_machine, dishwasher, ev_charging, ac, stove, water_heater, computers, misc
2. **Battery (4)**: SoC %, SoH %, charge kWh, cycle count
3. **Weather (2)**: temperature, solar irradiance
4. **Temporal (4)**: hour_sin, hour_cos, day_sin, day_cos
5. **Pricing (5)**: current price + 4 lagged prices

### Output Predictions (6 streams)

- **Consumption**: Day (48), Week (336), Month (1440) intervals
- **Price**: Day (48), Week (336), Month (1440) intervals

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
