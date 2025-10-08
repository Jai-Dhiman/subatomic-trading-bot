# Transformer Pre-Flight Plan

**Created:** 2025-10-08  
**Status:** Ready to Execute  
**Goal:** Train full multi-task Transformer on synthetic data as pre-flight check  
**Timeline:** 1-2 days

---

## Executive Summary

Since real consumption, appliance, battery, and weather data will be available soon, we're implementing the **FULL** multi-task Transformer architecture now (not the simplified pricing-only version). We'll train it on minimal synthetic data to verify the entire pipeline works, so when real data arrives, we can plug-and-play with minimal issues.

**Key Decision:** Skip pricing-only Phase 1A → Go straight to full architecture with synthetic data pre-flight.

---

## Architecture Overview

### Full Multi-Task Transformer (Production Target)

**Input Features:** 24 total
- 9 appliance consumption features
- 4 battery state features  
- 2 weather features
- 4 temporal features (cyclical encoding)
- 5 pricing features (current + 4 lags)

**Model Architecture:**
- d_model: 512
- n_heads: 8
- n_layers: 6
- dim_feedforward: 2048
- dropout: 0.1
- **Parameters:** ~12M

**Output Predictions:** 6 total
- Consumption: day (48), week (336), month (1440)
- Price: day (48), week (336), month (1440)

---

## Why Synthetic Data First?

### Strategy: Pre-Flight Before Real Data

1. **Verify architecture** - Ensure model can handle full feature set
2. **Test training pipeline** - Catch errors early on CPU/local machine
3. **Validate shapes** - Confirm all tensor dimensions match
4. **Quick iteration** - Don't wait for data to start building
5. **Smooth transition** - When real data arrives, just swap data source

### What Pre-Flight Achieves

- ✅ Model instantiates without errors
- ✅ Feature engineering handles all 24 features
- ✅ Training loop runs without crashes
- ✅ Loss decreases (proves gradient flow works)
- ✅ Checkpoint save/load works
- ✅ Multi-task loss balancing is correct

---

## Implementation Tasks

### Task 1: Create Synthetic Data Generator

**File:** `src/data_integration/synthetic_data_generator.py`

**Purpose:** Generate minimal realistic data (7-10 days) with patterns mimicking real household consumption.

**Data to Generate:**

```python
# Time range
7-10 days × 48 intervals/day = 336-480 samples

# Pricing data (1 pattern)
- Sinusoidal daily cycle
- Peak: 8-10am ($0.45-0.55/kWh), 6-9pm ($0.48-0.58/kWh)
- Low: 2-5am ($0.25-0.30/kWh)

# Appliances (9 patterns)
1. fridge: constant ~0.15 kW + noise
2. washing_machine: 2-3 cycles/week, 1.5 kW, 90 min
3. dishwasher: daily 7pm, 1.8 kW, 120 min
4. ev_charging: nightly 11pm-6am, 7.4 kW
5. ac: peaks 12pm-6pm, 2-3.5 kW (temperature dependent)
6. stove: peaks at meal times (7-8am, 12-1pm, 6-7pm), 2.5 kW
7. water_heater: morning (6-8am) and evening (6-9pm), 3 kW
8. computers: business hours 9am-6pm, 0.3 kW
9. misc: baseline 0.1-0.2 kW + noise

# Battery sensors (4 features)
- soc_percent: cycles between 20-80%
- soh_percent: starts at 100%, degrades to 99.5%
- charge_kwh: calculated from SoC × capacity (13.5 kWh)
- cycle_count: increments on full charge/discharge

# Weather (2 features)
- temperature: daily cycle 15-30°C, peaks at 2pm
- solar_irradiance: 0 at night, peaks at noon (800 W/m²)
```

**Validation:**
- All 24 feature columns present
- Timestamps are continuous 30-min intervals
- Value ranges are realistic
- Patterns are visible (e.g., EV charges at night, AC peaks afternoon)

---

### Task 2: Full Feature Engineering Pipeline

**File:** `src/models/feature_engineering.py`

**Purpose:** Extract and scale all 24 features for Transformer input.

**Key Methods:**

```python
class FeatureEngineer:
    def __init__(self, config):
        # Initialize scalers for each feature group
        self.appliance_scaler = StandardScaler()
        self.battery_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        self.price_scaler = StandardScaler()
    
    def prepare_features(self, df, pricing_df):
        # Extract 9 appliance features
        # Extract 4 battery features
        # Extract 2 weather features
        # Extract 4 temporal features (sin/cos encoding)
        # Extract 5 pricing features (current + 4 lags)
        # → Returns (n_samples, 24)
    
    def create_sequences(self, features, targets_consumption, targets_price):
        # Generate sequences: (n_samples, seq_len=48, n_features=24)
        # Generate targets for 3 horizons: day(48), week(336), month(1440)
        # → Returns X and y dict
```

**Expected Output:**
- X shape: (n_sequences, 48, 24)
- y['consumption_day'] shape: (n_sequences, 48)
- y['consumption_week'] shape: (n_sequences, 336)
- y['consumption_month'] shape: (n_sequences, 1440)
- y['price_day'] shape: (n_sequences, 48)
- y['price_week'] shape: (n_sequences, 336)
- y['price_month'] shape: (n_sequences, 1440)

---

### Task 3: Full EnergyTransformer Model

**File:** `src/models/transformer_model.py`

**Purpose:** Implement production Transformer with multi-task, multi-horizon predictions.

**Components:**

```python
class PositionalEncoding(nn.Module):
    # Sine/cosine positional encoding for sequences

class EnergyTransformer(nn.Module):
    def __init__(
        self,
        n_features=24,
        d_model=512,
        n_heads=8,
        n_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        horizons={'day': 48, 'week': 336, 'month': 1440}
    ):
        # Input projection: Linear(24 → 512)
        # Positional encoding
        # Transformer encoder (6 layers, 8 heads)
        # Multi-task heads:
        #   - consumption_heads (3 horizons)
        #   - price_heads (3 horizons)
    
    def forward(self, x):
        # x: (batch, seq_len=48, features=24)
        # → Returns dict with 6 predictions

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        consumption_weight=0.6,
        price_weight=0.4,
        horizon_weights={'day': 1.0, 'week': 0.5, 'month': 0.25}
    ):
        # MSE for consumption, MAE for price
        # Weighted by task and horizon importance
```

**Model Size:**
- Expected parameters: ~12M
- Model size: ~48 MB (FP32)
- GPU memory: ~2-4 GB during training

---

### Task 4: Local Training Script

**File:** `src/training/train_transformer_local.py`

**Purpose:** Verify training works end-to-end on CPU/local GPU.

**Training Configuration:**
- Epochs: 5-10 (just for verification)
- Batch size: 32 (adjust based on memory)
- Learning rate: 1e-4
- Optimizer: AdamW
- Data split: 80% train, 20% validation

**Expected Behavior:**
```
Epoch 1/10
  Batch 1/20 - Loss: 0.8453 (consumption: 0.6234, price: 0.2219)
  Batch 10/20 - Loss: 0.7821 (consumption: 0.5897, price: 0.1924)
  Val Loss: 0.7645 (consumption: 0.5734, price: 0.1911)

Epoch 2/10
  Batch 1/20 - Loss: 0.7234 (consumption: 0.5456, price: 0.1778)
  ...
  Val Loss: 0.6987 (consumption: 0.5123, price: 0.1864)
  ✓ Best model saved!

...

Training complete!
Final Val Loss: 0.5234
✓ Checkpoint saved to: checkpoints/transformer_preflight.pt
```

**Verification Checks:**
- [ ] Training loop runs without errors
- [ ] Loss decreases monotonically (or mostly)
- [ ] Both consumption and price losses decrease
- [ ] Validation loss tracks training loss
- [ ] Checkpoint save/load works
- [ ] Can make predictions on val data

---

### Task 5: Automated Testing

**File:** `tests/test_transformer_pipeline.py`

**Test Coverage:**

```python
def test_synthetic_data_generation():
    # Verify 24 columns present
    # Check realistic value ranges
    # Confirm temporal patterns

def test_feature_engineering():
    # 24 features extracted correctly
    # Sequences shape: (n, 48, 24)
    # Scaling/inverse scaling works

def test_transformer_architecture():
    # ~12M parameters
    # Forward pass produces correct shapes
    # Multi-task loss computes correctly

def test_training_pipeline():
    # Mini training (2 epochs, 10 samples)
    # Loss decreases
    # Checkpoint save/load

def test_end_to_end():
    # Generate data → Features → Train → Predict
    # No errors in full pipeline
```

**Run Tests:**
```bash
pytest tests/test_transformer_pipeline.py -v
```

---

### Task 6: Update Configuration

**File:** `config/config.yaml`

**Changes:**

```yaml
# REMOVE old LSTM config
# model:
#   lstm_hidden_size: 64
#   lstm_num_layers: 2
#   ...

# ADD Transformer config
transformer:
  architecture:
    d_model: 512
    n_heads: 8
    n_layers: 6
    dim_feedforward: 2048
    dropout: 0.1
  
  features:
    appliances: 9
    battery: 4
    weather: 2
    temporal: 4
    pricing: 5
    total: 24
  
  horizons:
    day: 48
    week: 336
    month: 1440
  
  training:
    batch_size: 32
    learning_rate: 0.0001
    epochs: 10  # Just for pre-flight
    sequence_length: 48

# KEEP existing sections unchanged
battery: ...
trading: ...
grid_constraints: ...
```

---

### Task 7: Pre-Flight Execution & Documentation

**Run Complete Pipeline:**

```bash
# 1. Generate synthetic data
python src/data_integration/synthetic_data_generator.py

# 2. Test feature engineering
python src/models/feature_engineering.py

# 3. Test transformer model
python src/models/transformer_model.py

# 4. Run local training
python src/training/train_transformer_local.py

# 5. Run all tests
pytest tests/test_transformer_pipeline.py -v
```

**Document Results:**

Create `docs/PRE_FLIGHT_RESULTS.md`:

```markdown
# Pre-Flight Results - Transformer Architecture

**Date:** 2025-10-08  
**Goal:** Verify full Transformer works before real data arrives

## Summary
- ✅ Synthetic data generation: PASS
- ✅ Feature engineering: PASS
- ✅ Transformer model: PASS (~12.3M parameters)
- ✅ Training pipeline: PASS (loss decreased from 0.84 → 0.52)
- ✅ All tests: PASS (5/5)

## Training Results
- Initial loss: 0.8453
- Final loss: 0.5234
- Training time: ~2 min/epoch (CPU)
- Model converged after 7 epochs

## Issues Found
- None major
- Note: CPU training is slow (~2 min/epoch)
  → Use Colab GPU for real training

## Next Steps
- ✅ Architecture verified
- ⏳ Waiting for real data from Supabase
- Ready to plug real data into pipeline
```

---

## Timeline

### Day 1: Implementation (4-6 hours)
- [ ] Task 1: Synthetic data generator (1 hour)
- [ ] Task 2: Feature engineering (1.5 hours)
- [ ] Task 3: Transformer model (2 hours)
- [ ] Task 4: Training script (1 hour)
- [ ] Task 5: Tests (30 min)
- [ ] Task 6: Config update (15 min)

### Day 2: Verification (1-2 hours)
- [ ] Task 7: Run full pipeline
- [ ] Fix any issues
- [ ] Document results
- [ ] Commit to repo

---

## Success Criteria

### Must Have ✅
- [x] All 7 tasks completed
- [ ] Synthetic data generates correctly
- [ ] Model has ~12M parameters
- [ ] Training runs without errors
- [ ] Loss decreases during training
- [ ] All tests pass

### Nice to Have 🎯
- [ ] Training converges in <10 epochs
- [ ] Consumption and price losses balanced
- [ ] Predictions look reasonable visually
- [ ] Training time documented

---

## After Pre-Flight: Real Data Integration

Once this pre-flight succeeds, integrating real data is straightforward:

1. **Keep synthetic generator** - Useful for testing
2. **Create real data loader** - `src/data_integration/real_data_loader.py`
3. **Swap data source** - In training script, change:
   ```python
   # Before
   data = generate_synthetic_data(days=10)
   
   # After  
   data = load_real_data_from_supabase()
   ```
4. **Verify feature compatibility** - Same 24 features
5. **Retrain on Colab GPU** - Much faster than CPU
6. **Compare results** - Synthetic vs real data performance

---

## Common Issues & Solutions

### Issue: Out of Memory
**Solution:** Reduce batch size (32 → 16 → 8)

### Issue: Training too slow
**Solution:** This is expected on CPU. Move to Colab GPU for real training.

### Issue: Loss not decreasing
**Solution:** 
- Check learning rate (try 1e-5 if 1e-4 too high)
- Verify gradient flow: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Check data quality: no NaNs or extreme values

### Issue: Test failures
**Solution:**
- Check tensor shapes match expectations
- Verify feature counts (should be 24)
- Print intermediate values to debug

---

## File Structure After Pre-Flight

```
energymvp/
├── src/
│   ├── models/
│   │   ├── transformer_model.py        # NEW: Full transformer
│   │   ├── feature_engineering.py      # NEW: Full features
│   │   ├── node_model.py               # OLD: Keep for reference
│   │   └── ...
│   ├── training/
│   │   └── train_transformer_local.py  # NEW: Local training
│   ├── data_integration/
│   │   ├── synthetic_data_generator.py # NEW: Synthetic data
│   │   └── ...
├── tests/
│   └── test_transformer_pipeline.py    # NEW: Full tests
├── config/
│   └── config.yaml                     # MODIFIED: Transformer config
├── docs/
│   ├── TRANSFORMER_PRE_FLIGHT_PLAN.md  # This file
│   └── PRE_FLIGHT_RESULTS.md           # To be created
└── checkpoints/                         # NEW: Model checkpoints
    └── transformer_preflight.pt
```

---

## Next Actions

1. **Review this plan** - Make sure approach makes sense
2. **Start with Task 1** - Synthetic data generator
3. **Test incrementally** - Don't wait until all tasks done
4. **Document issues** - Note any problems for troubleshooting
5. **Commit frequently** - Save progress after each task

---

**Ready to start? Begin with Task 1: Create Synthetic Data Generator**

**Questions?** Refer to:
- Full architecture details: `docs/TRANSFORMER_IMPLEMENTATION_PLAN.md`
- Quick reference: `docs/TRANSFORMER_QUICK_REFERENCE.md`
- Phase 2-3 code examples in implementation plan

**Good luck! 🚀**
