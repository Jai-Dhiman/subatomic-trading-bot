# Energy Trading Transformers (Training Repo)

This repository contains the training code and documentation for a dual-transformer system that forecasts household energy consumption and generates market-aware trading signals. The repo’s purpose is model training (not the sample API).

- Transformer 1 (Consumption): Predicts consumption for day/week using 48-step sequences of 17 features
- Transformer 2 (Trading): Uses consumption predictions + market/battery/temporal context (30 features) to predict prices and buy/hold/sell actions and quantities for the next 24 hours

All documentation uses cents/kWh units. Conversion: 1 $/MWh = 0.1 cents/kWh (e.g., 2.0 cents/kWh ≈ $20/MWh; 4.0 cents/kWh ≈ $40/MWh; 27.0 cents/kWh ≈ $270/MWh).

## Quick start

### Prerequisites
- Python 3.12+
- uv (preferred Python package manager)
- PyTorch 2.x
- (Optional) Supabase account/credentials for real data
- (Optional) Google Colab GPU (preferred for training)

### Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Training
- Train using Supabase battery labels (source of truth):
  - scripts/train_trading_transformer_from_supabase.py
  - Trains TradingTransformer V2 on labeled battery actions in Supabase tables

- Local CPU test training (optional):
  - scripts/train_trading_transformer_local.py
  - Uses the optimizer to generate labels for a quick CPU-only check

- Colab GPU training (recommended):
  - See notebooks/README.md for step-by-step Colab instructions (aligned with this repo). You can keep your .env in Drive and install with uv inside Colab.

### Configuration
Edit config/config.yaml for high-level parameters (model shapes, training settings, and limits). Supabase credentials should be provided via .env (never commit secrets).

## Project structure
```
energymvp/
├── README.md
├── config/
│   └── config.yaml
├── data/
│   └── demo/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BUSINESS_RULES.md
│   └── TECHNICAL_REFERENCE.md
├── notebooks/
│   ├── README.md
│   └── train_dual_transformers.ipynb
├── scripts/
│   ├── train_trading_transformer_from_supabase.py
│   ├── train_trading_transformer_local.py
│   ├── regenerate_battery_data.py
│   ├── evaluate_trading_models.py
│   └── ...
├── src/
│   ├── data_integration/
│   ├── models/
│   │   ├── consumption_transformer.py
│   │   ├── trading_transformer_v2.py
│   │   ├── feature_engineering_consumption.py
│   │   └── feature_engineering_trading.py
│   ├── training/
│   └── api/           # sample interface (not the main focus here)
└── tests/
```

## What’s included
- Model code: src/models/consumption_transformer.py and src/models/trading_transformer_v2.py
- Feature engineering: src/models/feature_engineering_consumption.py, src/models/feature_engineering_trading.py
- Training utilities: src/training/training_utils.py
- Data integration + label generation scripts: src/data_integration/*, scripts/*
- Demo data for examples/backtests: data/demo/

## Units
- All docs use cents/kWh
- Conversion reference: 1 $/MWh = 0.1 cents/kWh

## Documentation
- docs/ARCHITECTURE.md — System architecture and T1→T2 data flow
- docs/BUSINESS_RULES.md — Trading rules and constraints (in cents/kWh)
- docs/TECHNICAL_REFERENCE.md — Model specs, features, losses, and metrics
