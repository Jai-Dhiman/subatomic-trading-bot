# Technical Reference

This technical reference describes the models, features, training losses, metrics, and configurations currently used in this repo. All units are cents/kWh unless otherwise noted.

Conversion: 1 $/MWh = 0.1 cents/kWh.

---

## Models

### ConsumptionTransformer (T1)
- File: src/models/consumption_transformer.py
- Input: (batch, 48, 17)
- Outputs: {
  'consumption_day': (batch, 48),
  'consumption_week': (batch, 336)
}
- Loss: weighted MSE across horizons (e.g., day=1.0, week=0.5)

Features (17 per timestep):
- Appliances (9)
- Temporal (4): hour_sin, hour_cos, day_sin, day_cos
- Historical (4): last week same time, weekday-hour average, rolling averages, seasonal factor

### TradingTransformer V2 (T2)
- File: src/models/trading_transformer_v2.py
- Input: (batch, 48, 30)
- Outputs: {
  'predicted_price': (batch, 48),
  'trading_decisions': (batch, 48, 3),
  'trade_quantities': (batch, 48)
}

Features (30 per timestep):
- Consumption from T1 and aggregates
- Pricing (current + lags + moving stats)
- Battery state (SoC, available kWh/capacity, health)
- Temporal encodings (hour/day)
- Additional contextual indicators (volatility, indicators)

---

## Training losses

### T1 loss
- Weighted MSE over horizons

### T2 loss (TradingLossV2)
Used by scripts/train_trading_transformer_from_supabase.py with:
- 20% price prediction (MAE) — next-interval target
- 60% trading decision (CrossEntropy) — class-balanced
- 20% market profit — profit = sell_revenue − buy_costs using predicted decisions/quantities

Household revenue is constant and excluded from the loss.

---

## Metrics

- Price MAE (cents/kWh)
- Decision accuracy (%), including class-wise precision/recall if desired
- Profit (USD) or in consistent cents/kWh units over energy volumes
- Consumption accuracy for T1: MAPE/RMSE on the 48-step horizon

---

## Configuration examples

config/config.yaml (selected):
```yaml
simulation:
  num_intervals: 48
  interval_duration_minutes: 30

battery:
  capacity_kwh: 13.5
  efficiency: 0.90
  max_charge_rate_kw: 5.0
  max_discharge_rate_kw: 5.0
  min_reserve_percent: 0.10

model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  dropout: 0.1
  batch_size: 32
  sequence_length: 48

trading:
  max_single_trade_kwh: 2.0
  transmission_efficiency: 0.95
```

---

## Data & labels

- Training with real labels (preferred): Supabase battery tables contain decision (buy/hold/sell) and quantities aligned to pricing and consumption
- Optional label generation (local CPU): scripts/regenerate_battery_data.py uses src/models/trading_optimizer.py to generate labels consistent with business rules and physical limits

---

## Notes

- Use uv for package management
- Prefer Google Colab GPU for full training runs
- Keep documentation in cents/kWh to avoid confusion; include conversions only when necessary
