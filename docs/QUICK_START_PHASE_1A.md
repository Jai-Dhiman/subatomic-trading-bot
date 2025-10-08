# Quick Start: Phase 1A - Pricing-Only Transformer

**Created:** 2025-10-08  
**Your Data:** Historical pricing from Supabase  
**Your Strategy:** Train on all data except last week, test on last week  
**Environment:** Google Colab Pro with GPU

---

## ✅ What's Ready NOW

### 1. Feature Engineering (COMPLETE)

**File:** `src/models/feature_engineering_v1.py`

- ✅ Tested and working
- ✅ Creates 9 features (5 price + 4 temporal)
- ✅ Walk-forward validation split (last 7 days for testing)
- ✅ No deprecation warnings

**Test it:**

```bash
python src/models/feature_engineering_v1.py
```

### 2. Documentation (COMPLETE)

- `docs/TRANSFORMER_IMPLEMENTATION_PLAN.md` - Full plan with all phases
- `docs/TRANSFORMER_QUICK_REFERENCE.md` - Quick reference guide
- `docs/IMPLEMENTATION_PHASE_1_PRICING_ONLY.md` - Phase 1A details
- `docs/QUICK_START_PHASE_1A.md` - This file

---

## 🎯 Your Next Steps (This Week)

### Step 1: Test with Real Supabase Data (30 minutes)

Create `src/models/test_real_data.py`:

```python
"""Test feature engineering with real Supabase pricing data."""

import sys
sys.path.append('/Users/jdhiman/Documents/energymvp')

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import DataAdapter
from src.models.feature_engineering_v1 import FeatureEngineerV1

print("Loading pricing data from Supabase...")
connector = SupabaseConnector()
raw_pricing = connector.get_pricing_data()

print(f"Loaded {len(raw_pricing)} records")
print(f"Date range: {raw_pricing['INTERVALSTARTTIME_GMT'].min()} to {raw_pricing['INTERVALSTARTTIME_GMT'].max()}")

# Adapt pricing data
print("\nAdapting pricing data...")
pricing_df = DataAdapter.adapt_pricing_data(raw_pricing)

print(f"Adapted: {len(pricing_df)} records")
print(f"Columns: {pricing_df.columns.tolist()}")
print(f"Price range: ${pricing_df['price_per_kwh'].min():.3f} - ${pricing_df['price_per_kwh'].max():.3f}/kWh")

# Feature engineering
print("\nEngineering features...")
engineer = FeatureEngineerV1({})
features, df_enhanced = engineer.prepare_features(pricing_df)

print(f"Features shape: {features.shape}")

# Create sequences
print("\nCreating sequences...")
X, y = engineer.create_sequences(
    features,
    pricing_df['price_per_kwh'].values,
    sequence_length=48,
    horizons={'day': 48, 'week': 336}
)

print(f"X shape: {X.shape}")
print(f"y['price_day'] shape: {y['price_day'].shape}")
print(f"y['price_week'] shape: {y['price_week'].shape}")

# Create walk-forward split
print("\nCreating walk-forward split...")
train_data, val_data, test_data = engineer.create_walk_forward_split(
    X, y, test_days=7, val_days=3
)

print("\n✓ Real data test complete!")
print(f"\nReady for Colab training with:")
print(f"  - Train: {len(train_data['X'])} sequences")
print(f"  - Val: {len(val_data['X'])} sequences")
print(f"  - Test: {len(test_data['X'])} sequences (last week!)")
```

**Run it:**

```bash
python src/models/test_real_data.py
```

---

### Step 2: Create Simplified Transformer (Day 1-2)

**File to create:** `src/models/transformer_model_v1.py`

The full code is in `docs/IMPLEMENTATION_PHASE_1_PRICING_ONLY.md` starting at line 370.

**Key specs:**

- d_model: 256 (smaller than full version)
- n_heads: 4
- n_layers: 4
- ~2M parameters (fast training)
- Only price predictions (day + week horizons)

**After creating it, test:**

```bash
python src/models/transformer_model_v1.py
```

---

### Step 3: Create Simplified Colab Notebook (Day 3-5)

**File:** `notebooks/colab_phase1a_training.ipynb`

**Key cells:**

1. **Setup GPU** - Check CUDA, mount Google Drive
2. **Install packages** - uv, PyTorch, dependencies
3. **Load data** - Connect to Supabase, load pricing
4. **Feature engineering** - Run FeatureEngineerV1
5. **Create model** - Initialize PriceTransformer
6. **Training loop** - Train with early stopping
7. **Evaluate** - Test on last week
8. **Export** - Save model for CPU inference

---

## 📊 Expected Results

### Minimum Success (Week 1)

- [ ] Model trains without OOM errors on Colab
- [ ] Training converges (val loss stops improving)
- [ ] Can predict last week's prices
- [ ] MAE < $0.10/kWh for 1-day ahead
- [ ] Beats naive baseline (repeat last value)

### Good Success

- [ ] MAE < $0.05/kWh for 1-day predictions
- [ ] MAE < $0.08/kWh for 1-week predictions
- [ ] Direction accuracy > 60% (up/down correct)
- [ ] Captures daily price patterns visually

### Excellent Success

- [ ] MAE < $0.03/kWh for 1-day predictions
- [ ] Direction accuracy > 70%
- [ ] Identifies peak pricing times correctly
- [ ] Model ready for trading integration

---

## 🔄 Incremental Expansion Plan

Once Phase 1A works, you can expand incrementally:

### Phase 1B: Add Synthetic Consumption (testing only)

- Generate fake appliance data for architecture testing
- Add consumption prediction heads to transformer
- Verify multi-task learning works

### Phase 1C: Add Real Appliance Data (when available)

- Replace synthetic with real 9-appliance data
- Retrain with both consumption and price predictions
- Compare accuracy vs pricing-only baseline

### Phase 1D: Add Battery Sensors (when available)

- Add 4 battery features (SoC, SoH, charge_kwh, cycles)
- May improve consumption predictions
- Enables battery-aware trading strategies

### Phase 1E: Add Weather Data (when available)

- Add temperature and solar_irradiance
- Should improve long-term predictions (week/month)
- Enables weather-aware strategies

---

## 🎓 Key Learning Points

### Why Pricing-Only First?

1. **You have the data NOW** - Don't wait
2. **Simpler to debug** - Fewer moving parts
3. **Validates architecture** - Proves transformer works
4. **Immediate value** - Price forecasting enables trading
5. **Incremental is safer** - Add complexity gradually

### Why Walk-Forward Validation?

1. **Realistic** - Models never see the future
2. **Production-like** - Train on past, predict future
3. **Clear metrics** - Last week = your real test
4. **No data leakage** - Test set is truly held out

### Why Smaller Model for V1?

1. **Faster iteration** - ~5 min/epoch vs ~10 min/epoch
2. **Less memory** - Fits comfortably on Colab T4
3. **Good baseline** - Can always scale up later
4. **Easier to debug** - Simpler architecture

---

## 🆘 Common Issues & Solutions

### Issue: "Insufficient data"

**Solution:** Your pricing data might be shorter than expected. Adjust:

- `test_days=3` instead of 7
- `val_days=1` instead of 3
- Or get more historical data

### Issue: Colab disconnects during training

**Solution:**

- Checkpoint every epoch (already in plan)
- Use Colab Pro (you have it!)
- Keep browser tab active

### Issue: Model doesn't converge

**Solution:**

- Lower learning rate: 1e-4 → 1e-5
- Increase patience: 5 → 10
- Check data quality (NaNs, outliers)

### Issue: Predictions are constant

**Solution:**

- Model might be stuck in local minimum
- Try restarting with different random seed
- Check that features are normalized

---

## 📝 Checklist for Week 1

### Day 1

- [x] Feature engineering created and tested
- [ ] Test with real Supabase data
- [ ] Verify data has enough history for last-week test

### Day 2

- [ ] Create transformer_model_v1.py
- [ ] Test model forward pass
- [ ] Verify output shapes

### Day 3

- [ ] Create Colab notebook
- [ ] Setup GPU environment
- [ ] Load and prepare data

### Day 4-5

- [ ] Train model on Colab
- [ ] Monitor training (TensorBoard)
- [ ] Checkpoint best model

### Day 6

- [ ] Evaluate on last week (test set)
- [ ] Calculate MAE, RMSE, direction accuracy
- [ ] Visualize predictions vs actual

### Day 7

- [ ] Export model for CPU inference
- [ ] Document results
- [ ] Plan Phase 1B (if needed) or integrate into trading

---

## 🎉 Success Looks Like

After Week 1, you should have:

1. **Working feature engineering** - Transforms Supabase data → model input
2. **Trained transformer** - Stored in Google Drive
3. **Test metrics** - MAE, RMSE, plots for last week
4. **Exported model** - TorchScript file for local inference
5. **Clear next steps** - Whether to add features or integrate

This becomes your **baseline** for all future improvements!

---

## 💬 Questions?

If you get stuck:

1. Check the detailed implementation plan: `docs/IMPLEMENTATION_PHASE_1_PRICING_ONLY.md`
2. Review the full plan: `docs/TRANSFORMER_IMPLEMENTATION_PLAN.md`
3. Look at code examples in each document
4. Ask for help with specific errors

---

**Ready to start? Run this now:**

```bash
# Test feature engineering works
python src/models/feature_engineering_v1.py

# Test with real Supabase data
python src/models/test_real_data.py

# Create transformer model next
# (copy code from IMPLEMENTATION_PHASE_1_PRICING_ONLY.md)
```

**Good luck! 🚀**
