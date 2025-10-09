# System Architecture

This repo trains a dual-transformer system for energy trading. The system runs in two sequential stages:

1) Consumption Transformer (T1)
- Input shape: (batch, 48, 17) — 48 recent 30-min intervals, 17 features per step
- Output: day-ahead (48) and week-ahead (336) consumption predictions

2) Trading Transformer (T2)
- Input shape: (batch, 48, 30)
- Outputs: predicted_price (48), trading_decisions (48×3 logits), trade_quantities (48)
- Consumes T1’s day-ahead predictions alongside pricing, battery, temporal, and contextual features

All documentation uses cents/kWh units. Conversion: 1 $/MWh = 0.1 cents/kWh.

---

## Data flow (training)

```
1) Load real data (Supabase) and/or demo CSVs
2) Feature engineering for T1 (17 features)
3) Train T1 to predict consumption (48 and 336 horizons)
4) Prepare features for T2 (30 features), including T1 outputs
5) Train T2 on:
   - Real labels from Supabase battery tables (preferred), or
   - Optimizer-generated labels (for local CPU tests)
6) Evaluate and export checkpoints
```

---

## Data flow (inference)

```
1) Prepare a single (1, 48, 17) input sequence for T1
2) Run T1 → get consumption_day (1, 48)
3) Build T2 features (1, 48, 30) using T1 outputs + context
4) Run T2 → get:
   - predicted_price (1, 48)
   - trading_decisions (1, 48, 3)
   - trade_quantities (1, 48)
```

---

## Features

- T1 (Consumption) — 17 features per timestep
  - Appliances (9): normalized consumption features
  - Temporal (4): hour_sin, hour_cos, day_sin, day_cos
  - Historical patterns (4): last-week same time, weekday-hour average, rolling averages, seasonal factor

- T2 (Trading) — 30 features per timestep
  - Consumption (from T1): and basic aggregations
  - Pricing (lags, moving stats)
  - Battery state: SoC, available energy/capacity, health
  - Temporal: hour-of-day and day-of-week encodings
  - Additional contextual features (volatility, moving averages, indicators)

---

## Training artifacts

- src/models/consumption_transformer.py
  - Input: (batch, 48, 17)
  - Outputs: {'consumption_day': (batch, 48), 'consumption_week': (batch, 336)}

- src/models/trading_transformer_v2.py
  - Input: (batch, 48, 30)
  - Outputs: {
      'predicted_price': (batch, 48),
      'trading_decisions': (batch, 48, 3),
      'trade_quantities': (batch, 48)
    }

- src/training/training_utils.py — common training routines (data loaders, train/val, checkpoints)

- scripts/train_trading_transformer_from_supabase.py — primary training flow using Supabase battery labels
- scripts/train_trading_transformer_local.py — optional CPU flow that generates labels via optimizer

---

## Units and constraints

- Units: cents/kWh throughout
- Example thresholds: 2.0 (buy), 4.0 (sell), 27.0 (grid price reference)
- Physical constraints enforced in data generation/optimizer and loss design:
  - SoC bounds (e.g., 20%–90% typical)
  - Charge/discharge energy per 30-min interval limited by kW ratings

---

## Notes

- This repo focuses on model training. A sample API exists under src/api for local testing/demo but is not the focus.
- Prefer Google Colab with GPU for full training runs. Use uv for Python package management.
