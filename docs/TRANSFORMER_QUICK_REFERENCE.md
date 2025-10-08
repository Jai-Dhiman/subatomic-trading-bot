# Transformer Quick Reference

**Last Updated:** 2025-10-08  
**Status:** Implementation Plan Ready

---

## Architecture Overview

### Current System (LSTM)

- **Input:** 5 features (consumption, temp, solar, hour, day)
- **Sequence:** 48 intervals (24h lookback)
- **Output:** 6 intervals (3h ahead)
- **Parameters:** ~50K
- **Training:** Local CPU

### New System (Transformer)

- **Input:** 24 features (appliances, battery, pricing, weather, temporal)
- **Sequence:** 48 intervals (24h lookback)
- **Output:** 3 horizons × 2 tasks = 6 outputs
  - Day (48), Week (336), Month (1440)
  - Consumption + Price predictions
- **Parameters:** ~12M
- **Training:** Google Colab GPU

---

## Feature Specification

### Input Features (24 total)

| Category | Features | Count | Description |
|----------|----------|-------|-------------|
| **Appliances** | Washing machine, Dishwasher, EV charging, Fridge, AC, Stove, Water heater, Computers, Misc | 9 | Per-appliance consumption (kWh) |
| **Battery** | SoC %, SoH %, Charge (kWh), Cycle count | 4 | Battery state sensors |
| **Weather** | Temperature (F), Solar irradiance (W/m²) | 2 | Local weather conditions |
| **Temporal** | Hour sin/cos, Day sin/cos | 4 | Cyclical time encoding |
| **Pricing** | Current + 4 lags | 5 | Historical price data |

### Output Predictions (6 total)

| Horizon | Intervals | Hours | Consumption | Price |
|---------|-----------|-------|-------------|-------|
| **Day** | 48 | 24h | ✓ | ✓ |
| **Week** | 336 | 7d | ✓ | ✓ |
| **Month** | 1440 | 30d | ✓ | ✓ |

---

## Model Configuration

```yaml
# config/config.yaml
model:
  type: transformer
  d_model: 512
  n_heads: 8
  n_layers: 6
  dim_feedforward: 2048
  dropout: 0.1
  batch_size: 32
  input_sequence_length: 48
  
  horizons:
    day: 48
    week: 336
    month: 1440
  
  features:
    appliances: 9
    battery: 4
    weather: 2
    temporal: 4
    pricing: 5
    total: 24

training:
  learning_rate: 1e-4
  epochs: 50
  patience: 5
  gradient_clip: 1.0
  
  loss_weights:
    consumption: 1.0
    price: 0.5
  
  horizon_weights:
    day: 1.0
    week: 0.5
    month: 0.25
```

---

## Implementation Phases

### Phase 1: Cleanup (1 day)

- [ ] Backup LSTM code (`node_model.py.backup`)
- [ ] Update `config/config.yaml`
- [ ] Update imports in `household_node.py`

### Phase 2: Feature Engineering (2-3 days)

- [ ] Create `src/models/feature_engineering.py`
- [ ] Implement `FeatureEngineer` class
- [ ] Test with sample data
- [ ] Validate feature shapes

### Phase 3: Transformer Model (2-3 days)

- [ ] Create `src/models/transformer_model.py`
- [ ] Implement `EnergyTransformer` class
- [ ] Implement `MultiTaskLoss` class
- [ ] Test model forward pass
- [ ] Verify output shapes

### Phase 4: Colab Training (2-3 days)

- [ ] Create `notebooks/colab_training.ipynb`
- [ ] Setup GPU environment
- [ ] Load and process data
- [ ] Train model
- [ ] Export trained model

### Phase 5: Data Integration (1-2 days)

- [ ] Update `consumption_parser.py`
- [ ] Update `data_adapter.py`
- [ ] Test data pipeline

### Phase 6: Enhanced Trading (2-3 days)

- [ ] Create `EnhancedTradingStrategy` class
- [ ] Implement price-based buy/sell logic
- [ ] Test trading decisions

### Phase 7: Model Integration (1-2 days)

- [ ] Update `household_node.py`
- [ ] Test end-to-end prediction
- [ ] Verify federated learning compatibility

### Phase 8: Evaluation (2 days)

- [ ] Create `src/evaluation/metrics.py`
- [ ] Implement MAPE, RMSE, MAE
- [ ] Calculate trading metrics

### Phase 9: Testing (2-3 days)

- [ ] Unit tests for transformer
- [ ] Unit tests for features
- [ ] Integration tests
- [ ] End-to-end validation

**Total Estimated Time:** 15-21 days (2-3 weeks)

---

## Key Decisions

### Why Transformer over LSTM?

1. **Better long-range dependencies** - Self-attention captures patterns across entire sequence
2. **Parallel training** - Faster on GPU than sequential LSTM
3. **Multi-horizon prediction** - Natural fit for predicting multiple time scales
4. **State-of-the-art** - Superior performance on time series benchmarks

### Why Multi-Task Learning?

1. **Shared representations** - Consumption and price patterns are related
2. **Better generalization** - Prevents overfitting to single task
3. **Efficient training** - Single model for both predictions

### Why 3 Horizons?

1. **Short-term (1 day)** - Immediate trading decisions
2. **Medium-term (1 week)** - Strategic battery management
3. **Long-term (1 month)** - Capacity planning and trend analysis

---

## Training Strategy

### Data Split

- **Train:** 60% (for learning patterns)
- **Validation:** 20% (for hyperparameter tuning)
- **Test:** 20% (for final evaluation)

### Hardware Requirements

- **GPU:** T4 or better (15GB+ VRAM)
- **RAM:** 16GB+ system memory
- **Storage:** 5GB+ for checkpoints

### Training Time Estimates

- **Per epoch:** ~5-10 minutes (T4 GPU)
- **Total training:** ~4-8 hours (50 epochs with early stopping)

### Checkpoint Strategy

- Save best model based on validation loss
- Export to TorchScript for CPU inference
- Store in Google Drive for persistence

---

## Enhanced Trading Logic

### Price Prediction Strategy

```python
# Buy Logic
if (predicted_price_1h < current_price * 0.95 and
    predicted_price_24h > current_price * 1.05 and
    battery_soc < 80%):
    → BUY (price expected to rise)

# Sell Logic  
if (predicted_price_1h > current_price * 1.05 and
    predicted_price_24h < current_price * 0.95 and
    battery_soc > 20%):
    → SELL (price expected to fall)

# Hold Logic
else:
    → HOLD (unclear price direction or battery constraints)
```

### Confidence Thresholds

- Only trade when prediction confidence > 70%
- Adjust trade size based on prediction uncertainty
- Maintain existing constraints (5% profit margin, daily limits)

---

## Evaluation Metrics

### Consumption Prediction

- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- Per horizon: day, week, month

### Price Prediction

- **Direction Accuracy** (% correct up/down)
- **RMSE** (price forecast error)
- **MAE** (absolute price error)
- Per horizon: day, week, month

### Trading Performance

- **Total Profit/Loss** ($)
- **Success Rate** (% profitable trades)
- **Risk-Adjusted Returns** (Sharpe ratio)
- **Battery Utilization** (% of capacity used)

### Training Metrics

- **GPU Utilization** (%)
- **Training Time** (min/epoch)
- **Memory Usage** (GB)
- **Convergence Rate** (epochs to best)

---

## Common Issues & Solutions

### Issue: Out of Memory (OOM)

**Solutions:**

- Reduce batch size (32 → 16 → 8)
- Reduce model size (d_model: 512 → 256)
- Use gradient checkpointing
- Enable mixed precision training

### Issue: Poor Convergence

**Solutions:**

- Reduce learning rate (1e-4 → 1e-5)
- Increase warmup steps
- Try different optimizer (AdamW)
- Check for data leakage

### Issue: Overfitting

**Solutions:**

- Increase dropout (0.1 → 0.2)
- Add L2 regularization
- Use more training data
- Reduce model complexity

### Issue: Slow Training

**Solutions:**

- Use mixed precision (FP16)
- Increase batch size (if memory allows)
- Use gradient accumulation
- Optimize data loading pipeline

---

## File Structure

```
energymvp/
├── src/
│   ├── models/
│   │   ├── transformer_model.py          # NEW: Transformer architecture
│   │   ├── feature_engineering.py        # NEW: Feature extraction
│   │   ├── node_model.py.backup          # Backup of LSTM
│   │   ├── battery_manager.py            # Unchanged
│   │   ├── grid_constraints.py           # Unchanged
│   │   ├── profitability.py              # Unchanged
│   │   └── central_model.py              # Unchanged
│   ├── trading/
│   │   ├── trading_logic.py              # MODIFY: Add EnhancedTradingStrategy
│   │   └── market_mechanism.py           # Unchanged
│   ├── simulation/
│   │   ├── household_node.py             # MODIFY: Use Transformer
│   │   └── run_demo.py                   # MODIFY: Update for new model
│   ├── data_integration/
│   │   ├── consumption_parser.py         # MODIFY: Parse all features
│   │   ├── data_adapter.py               # MODIFY: Add pricing merge
│   │   └── supabase_connector.py         # Unchanged
│   ├── training/
│   │   └── training_utils.py             # NEW: GPU training utilities
│   └── evaluation/
│       └── metrics.py                    # NEW: Evaluation framework
├── notebooks/
│   └── colab_training.ipynb              # NEW: Google Colab notebook
├── config/
│   └── config.yaml                       # MODIFY: Transformer config
├── docs/
│   ├── TRANSFORMER_IMPLEMENTATION_PLAN.md  # NEW: Full plan
│   └── TRANSFORMER_QUICK_REFERENCE.md      # This file
└── tests/
    ├── test_transformer_model.py         # NEW: Model tests
    ├── test_feature_engineering.py       # NEW: Feature tests
    └── test_trading_logic.py             # NEW: Trading tests
```

---

## Next Steps

1. **Read the full implementation plan:** `docs/TRANSFORMER_IMPLEMENTATION_PLAN.md`
2. **Start with Phase 2:** Feature engineering is the foundation
3. **Test incrementally:** Don't wait until everything is complete
4. **Use the todo list:** Track progress through all 9 phases
5. **Ask for help:** Each phase has detailed code examples

---

## Quick Commands

```bash
# Backup current LSTM
cp src/models/node_model.py src/models/node_model.py.backup

# Test feature engineering
python src/models/feature_engineering.py

# Test transformer model
python src/models/transformer_model.py

# Upload to Colab
# (manually upload notebook to Google Colab)

# After training, test locally
python -c "import torch; model = torch.jit.load('transformer_model_traced.pt'); print('Model loaded!')"
```

---

## Success Criteria

### Technical

- [ ] Model trains without OOM errors
- [ ] Validation loss converges (< 10 epochs plateau)
- [ ] Test MAPE < 15% for 1-day predictions
- [ ] Test MAPE < 25% for 1-week predictions
- [ ] Price direction accuracy > 60%

### Functional

- [ ] Federated learning weight aggregation works
- [ ] Trading logic uses price predictions
- [ ] Model inference runs on local CPU
- [ ] End-to-end simulation completes
- [ ] Achieves 20-40% cost savings target

### Performance

- [ ] Training time < 10 hours
- [ ] Inference time < 100ms per prediction
- [ ] GPU utilization > 80%
- [ ] Model size < 50MB (TorchScript)

---

**Ready to implement? Start with Phase 2 (Feature Engineering)!**
