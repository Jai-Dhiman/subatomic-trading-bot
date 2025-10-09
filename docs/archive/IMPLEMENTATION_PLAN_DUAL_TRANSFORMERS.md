# Dual-Transformer Energy Trading (Training Repo)

This repository contains the training code and documentation for a dual-transformer system that forecasts household energy consumption and generates market-aware trading signals. The primary focus is model training and data preparation; an example API exists in src/api/ but is not the focus of this repo.

- Transformer 1 (Consumption): Predicts consumption for day/week from 48-step sequences of 17 features.
- Transformer 2 (Trading): Uses consumption predictions + market/battery context (30 features) to predict prices and buy/hold/sell actions and quantities for the next 24 hours.

All documentation uses cents/kWh units. Conversion: 1 $/MWh = 0.1 cents/kWh.


## What’s included
- Model code: src/models/consumption_transformer.py and src/models/trading_transformer_v2.py
- Feature engineering: src/models/feature_engineering_consumption.py, src/models/feature_engineering_trading.py
- Training utils: src/training/training_utils.py
- Data integration helpers and label generation scripts: src/data_integration/*, scripts/*
- Demo data for backtesting/examples: data/demo/
- Configuration: config/config.yaml


## Prerequisites
- Python 3.12+
- uv (preferred Python package manager)
- (Optional) Supabase credentials for real data (.env)
- (Optional) Google Colab GPU for training (preferred by default)


## Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Environment variables (for Supabase) should be stored in an .env file (never commit secrets):
```bash
SUPABASE_URL=...  # e.g., https://xyzcompany.supabase.co
SUPABASE_KEY=...  # service role or appropriate key
```


## Training flows
- Train using Supabase battery labels (source of truth):
  - scripts/train_trading_transformer_from_supabase.py
  - Uses labeled battery actions in Supabase tables; trains TradingTransformer V2 directly on real labels.

- Local CPU test training (optional):
  - scripts/train_trading_transformer_local.py
  - Generates labels from the optimizer for quick iteration on CPU.

- Colab GPU training (recommended):
  - See notebooks/README.md for step-by-step guidance. Follow the same modules and parameters as above.


## Data notes
- Consumption data: real appliance breakdown per household; hourly or 30-minute alignment handled in data adapter/feature engineering.
- Pricing data: real CA market pricing (ensure correct LMP filter upstream).
- Battery label generation (optional): scripts/regenerate_battery_data.py uses src/models/trading_optimizer.py to create training labels consistent with current business rules.


## Units
- Documentation and examples use cents/kWh.
- Conversion reference: 1 $/MWh = 0.1 cents/kWh.
  - 2.0 cents/kWh ≈ $20/MWh
  - 4.0 cents/kWh ≈ $40/MWh
  - 27.0 cents/kWh ≈ $270/MWh


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
│   ├── TECHNICAL_REFERENCE.md
│   └── archive/
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


## Documentation
- docs/ARCHITECTURE.md — System architecture and model/data flow
- docs/BUSINESS_RULES.md — Trading rules and constraints (in cents/kWh)
- docs/TECHNICAL_REFERENCE.md — Detailed model specs, features, losses, and metrics


## Notes
- Package management uses uv (preferred).
- No emojis in code, comments, or docs.
