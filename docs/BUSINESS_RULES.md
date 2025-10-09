# Business Rules & Trading Strategy

This document describes the trading rules and constraints used to generate labels and train the Trading Transformer. All units are cents/kWh.

Conversion reference: 1 $/MWh = 0.1 cents/kWh.

- 2.0 cents/kWh ≈ $20/MWh
- 4.0 cents/kWh ≈ $40/MWh
- 27.0 cents/kWh ≈ $270/MWh

---

## Hard rules (non-negotiable)

1) Battery safety (SoC bounds)

- Never discharge below 20% SoC
- Do not exceed max SoC (typically 90% for safety)

2) Buy low, sell high (thresholds)

- Buy when price < 2.0 cents/kWh and capacity is available
- Sell when price > 4.0 cents/kWh and SoC > safety minimum

3) Household demand priority

- Always keep enough energy to satisfy near-term household consumption with a small buffer; never jeopardize essential load

---

## Priority order

1) Meet household power demands (must-have)
2) Sell excess energy beyond household needs
3) Opportunistically buy/sell around thresholds to improve profit while staying within physical limits

---

## Optimizer behavior (for label generation)

When labels are generated locally (e.g., scripts/regenerate_battery_data.py), the optimizer:

- Respects SoC bounds and per-interval energy limits set by charge/discharge ratings
- Buys aggressively below 2.0 cents/kWh when capacity allows
- Sells aggressively above 4.0 cents/kWh when energy is available
- Defaults to HOLD when price is between thresholds or capacity/SoC constraints are hit

The optimizer simulates battery energy with efficiency and subtracts consumption after each interval to maintain realistic state transitions.

---

## Training objective (used by source-of-truth script)

scripts/train_trading_transformer_from_supabase.py uses real labels from Supabase and trains TradingLossV2 with the following weights:

- 20% price prediction (MAE)
- 60% trading decision (CrossEntropy; class-balanced)
- 20% market profit (sell_revenue − buy_costs)

Household revenue is constant and not included in the loss. Profit is computed using predicted decisions/quantities and the target next-interval price.

---

## Example decision policy (illustrative)

```
if SoC <= 20%:
    HOLD  # safety
elif price < 2.0 and SoC < 90%:
    BUY as capacity allows
elif price > 4.0 and SoC > 20%:
    SELL excess energy
else:
    HOLD
```

Notes:

- “BUY/SELL as capacity allows” is limited by per-interval kW ratings and battery efficiency.
- “Excess energy” means energy beyond the safety minimum plus near-term consumption buffer.

---

## Success criteria (operational)

- 100% household demand met
- SoC within bounds across the day
- Positive market profit over evaluation windows
- Rule compliance (buy/sell decisions align with thresholds and safety constraints)
