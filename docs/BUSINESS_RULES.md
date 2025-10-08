# Business Rules & Trading Strategy

**Updated**: October 8, 2025  
**Focus**: Profit Maximization through Strategic Energy Trading

---

## 💰 **Business Model**

### Revenue Streams
1. **Household Service**: Charge households **$270/MWh ($0.27/kWh)** for electricity
2. **Market Trading**: Buy low from market, sell high to market
3. **Battery Optimization**: Minimize holding costs, maximize utilization

### Goal
**Maximize total profit** = Household Revenue + Market Profit - Buy Costs

---

## 🎯 **Hard Rules** (Non-Negotiable)

### Rule 1: Always Buy Below $20/MWh
```
IF market_price < $20/MWh AND battery_soc < 100%:
    BUY (as much as possible)
```
- **Rationale**: Prices below $20/MWh are exceptionally cheap
- **Action**: Fill battery to maximize future profit potential
- **Constraint**: Only if battery has capacity (SoC < 100%)

### Rule 2: Always Sell Above $40/MWh
```
IF market_price > $40/MWh AND battery_soc > 20%:
    SELL (excess power)
```
- **Rationale**: Prices above $40/MWh are profitable
- **Action**: Sell un-needed power to market
- **Constraint**: Keep enough for household demand + buffer

### Rule 3: Hold When SoC Below 20%
```
IF battery_soc <= 20%:
    HOLD (safety threshold)
```
- **Rationale**: Protect battery health and ensure household power
- **Action**: No discharge below 20%
- **Constraint**: Absolute minimum

### Rule 4: Hold When Price Below 10% of Grid
```
IF market_price < $27/MWh (10% of $270/MWh):
    HOLD (price too low)
```
- **Rationale**: Not worth selling at such low prices
- **Action**: Wait for better market conditions
- **Constraint**: Minimum acceptable price floor

---

## 📋 **Priority List** (In Order)

### Priority 1: Match Household Power Demands
```
MUST: Ensure enough energy for household consumption
```
- **Target**: Meet 100% of predicted daily consumption
- **Buffer**: Keep 5% safety margin
- **Critical**: This is non-negotiable - households must have power

### Priority 2: Sell (Max Capacity - Daily Usage)
```
Sellable Capacity = Battery Max Capacity - Daily Predicted Usage
```
- **Calculation**: 
  - If battery is 40 kWh and daily usage is 30 kWh
  - Sellable capacity = 10 kWh
- **Action**: Sell this excess when market price > $40/MWh

### Priority 3: Sell Remaining Un-needed Power
```
IF battery_soc > target_soc AND price > threshold:
    Sell excess to market
```
- **Target SoC**: 60% (comfortable operating level)
- **Action**: Sell power above this level when profitable
- **Benefit**: Reduce holding costs, increase profit

### Priority 4: Minimize Excess Battery Storage
```
Penalty for holding power when:
- Battery SoC > 80% AND price > $20/MWh
- Good selling opportunity exists but holding
```
- **Rationale**: Holding power has opportunity cost
- **Action**: Bias towards selling excess
- **Benefit**: Maximize capital efficiency

---

## 🏆 **Reward Mechanism**

### For Transformer Training

**Primary Objective**: Maximize Profit

```python
total_profit = household_revenue + market_profit - buy_costs

Where:
- household_revenue = consumption * $0.27/kWh
- market_profit = sell_revenue - buy_costs
- sell_revenue = sell_quantity * market_price
- buy_costs = buy_quantity * market_price
```

### Loss Function Weights
- **20%**: Price prediction accuracy (guidance)
- **20%**: Trading decision correctness (rules compliance)
- **60%**: Profitability reward (PRIMARY GOAL)

### Reward Calculation
```
higher_profit → lower_loss → better_model
```

### Penalties
1. **Excess Storage**: Holding power when could profitably sell
2. **Missed Opportunities**: Not buying at < $20/MWh
3. **Poor Timing**: Selling at < $40/MWh when could wait

---

## 📊 **Example Scenarios**

### Scenario 1: Cheap Market Price
```
Market Price: $15/MWh ($0.015/kWh)
Battery SoC: 50%
Action: BUY maximum amount
Rationale: Price < $20/MWh, excellent buying opportunity
```

### Scenario 2: Expensive Market Price
```
Market Price: $60/MWh ($0.060/kWh)
Battery SoC: 70%
Household Demand: 2 kWh (next 4 hours)
Action: SELL 8 kWh (keep 2 kWh + buffer)
Rationale: Price > $40/MWh, sell excess while meeting demand
```

### Scenario 3: Mid-Range Price
```
Market Price: $30/MWh ($0.030/kWh)
Battery SoC: 40%
Action: HOLD
Rationale: Price between thresholds, wait for better opportunity
```

### Scenario 4: Low Battery
```
Market Price: $50/MWh ($0.050/kWh)
Battery SoC: 22%
Action: HOLD (mostly)
Rationale: Near safety threshold, prioritize household demand
```

---

## 💡 **Business Logic Summary**

### Decision Tree
```
1. Is SoC <= 20%?
   → YES: HOLD (safety)
   → NO: Continue

2. Is price < $27/MWh (10% of household rate)?
   → YES: HOLD (too cheap to sell)
   → NO: Continue

3. Is price < $20/MWh AND SoC < 100%?
   → YES: BUY (excellent price)
   → NO: Continue

4. Is price > $40/MWh AND SoC > 20%?
   → YES: SELL excess (profitable)
   → NO: Continue

5. Is SoC > 80% AND price > $20/MWh?
   → YES: SELL excess (reduce storage)
   → NO: HOLD
```

---

## 📈 **Expected Profit Margins**

### Conservative Scenario
- Household revenue: $0.27/kWh × consumption
- Market profit: 10-15% additional
- **Total margin**: 110-115%

### Optimal Scenario
- Household revenue: $0.27/kWh × consumption
- Market profit: 25-40% additional
- **Total margin**: 125-140%

### Aggressive Scenario
- Household revenue: $0.27/kWh × consumption
- Market profit: 40-60% additional
- **Total margin**: 140-160%

---

## ⚙️ **Implementation Details**

### Optimizer Configuration
```python
calculate_optimal_trading_decisions(
    predicted_consumption,
    actual_prices,
    battery_state,
    household_price_kwh=0.27,      # $270/MWh
    buy_threshold_mwh=20.0,        # Buy below $20/MWh
    sell_threshold_mwh=40.0,       # Sell above $40/MWh
    min_grid_price_pct=0.10        # 10% of household price
)
```

### Loss Function Configuration
```python
TradingLoss(
    price_weight=0.20,             # 20% price accuracy
    decision_weight=0.20,          # 20% decision correctness
    profit_weight=0.60,            # 60% profitability (PRIMARY)
    household_price_kwh=0.27,
    excess_storage_penalty=0.10
)
```

---

## ✅ **Success Metrics**

### Must Achieve
- ✅ 100% household demand met
- ✅ Battery SoC maintained 20-100%
- ✅ Total profit > 0

### Target Performance
- 🎯 Profit margin: 125-140%
- 🎯 Market profit: 20-40% of household revenue
- 🎯 Average SoC: 40-70% (optimal range)
- 🎯 Trading decisions: >80% rule compliance

### Stretch Goals
- 🚀 Profit margin: >140%
- 🚀 Market profit: >40% of household revenue
- 🚀 Zero missed opportunities (buy at < $20, sell at > $40)
- 🚀 Minimal excess storage (<10% of time at >80% SoC)

---

## 🔄 **Continuous Improvement**

### Model Training
- Reward high-profit outcomes
- Penalize missed opportunities
- Balance short-term vs long-term profit

### Strategy Refinement
- Adjust thresholds based on market patterns
- Optimize buffer sizes
- Fine-tune SoC targets

---

**Bottom Line**: The transformer should learn to maximize profit by intelligently trading in the market while always ensuring household power needs are met. Higher profit = better model performance.
