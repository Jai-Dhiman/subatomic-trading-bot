# Training Notebook Guide

## Overview

The `train_dual_transformers.ipynb` notebook trains both transformers on Google Colab with GPU acceleration. This is the final step to reach MVP!

## What This Notebook Does

1. **Setup & Environment** - Installs dependencies, mounts Google Drive, configures GPU
2. **Data Loading** - Loads real consumption, pricing, and battery data from Supabase
3. **Preflight Validation** - Tests everything on CPU before expensive GPU training
4. **Consumption Training** - Trains the consumption prediction model (target: MAPE < 15%)
5. **Trading Training** - Trains the trading decision model (target: 20-40% cost savings)
6. **End-to-End Testing** - Validates the complete system with comprehensive metrics
7. **Model Export** - Saves models, configs, logs, and visualizations to Google Drive

## Prerequisites

Before running the notebook, you need:

1. **Google Drive** with the project folder uploaded
2. **Supabase credentials** in a `.env` file
3. **Google Colab** account (free tier with GPU is sufficient)

## Step-by-Step Instructions

### 1. Upload Project to Google Drive

```bash
# On your local machine, compress the project
cd ~/Documents
zip -r energymvp.zip energymvp/ -x "energymvp/.git/*" -x "energymvp/.venv/*" -x "energymvp/__pycache__/*"

# Upload energymvp.zip to Google Drive at:
# My Drive/energymvp/
```

### 2. Prepare .env File

Create a `.env` file in your Google Drive at `/MyDrive/energymvp/.env` with:

```bash
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
```

### 3. Open in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Open notebook → Upload
3. Upload `notebooks/train_dual_transformers.ipynb`
4. **IMPORTANT**: Enable GPU
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4 recommended)

### 4. Run the Notebook

Execute cells in order:

**Section 1: Setup (5-10 minutes)**
- Mounts Google Drive
- Installs dependencies with `uv`
- Verifies GPU is available
- Imports all project modules

**Section 2: Data Loading (2-5 minutes)**
- Loads consumption data from Supabase
- Loads pricing data with LMP filter
- Generates battery state data
- Validates data quality

**Section 3: Preflight Validation (5-10 minutes on CPU)**
- Tests both transformers
- Verifies feature engineering
- Runs quick training test
- **CRITICAL**: Catches errors before GPU training

**Section 4: Consumption Training (1-3 hours on GPU)**
- Trains ConsumptionTransformer
- Monitors MAPE (target < 15%)
- Saves checkpoints every 5 epochs
- Early stopping with patience=10

**Section 5: Trading Training (1-3 hours on GPU)**
- Uses trained consumption model
- Generates optimal trading labels
- Trains TradingTransformer
- Monitors cost savings (target 20-40%)

**Section 6: End-to-End Testing (10-15 minutes)**
- Loads best models
- Runs complete inference
- Calculates all metrics
- Generates visualizations

**Section 7: Model Export (5 minutes)**
- Saves model configs
- Exports training logs
- Creates summary report
- Everything saved to Google Drive

## Expected Results

### Success Criteria

- ✅ Consumption MAPE < 15%
- ✅ Price MAE < $0.05/kWh
- ✅ Cost savings: 20-40% vs baseline
- ✅ Models converge within 50-100 epochs
- ✅ All checkpoints saved to Google Drive

### Output Files

All files saved to `/content/drive/MyDrive/energymvp/checkpoints/`:

```
checkpoints/
├── consumption_transformer_best.pt       # Best consumption model
├── trading_transformer_best.pt           # Best trading model
├── consumption_config.json               # Model architecture config
├── trading_config.json                   # Model architecture config
├── training_logs.json                    # Complete training history
├── TRAINING_SUMMARY.txt                  # Human-readable summary
├── consumption_training_history.png      # Loss/MAPE plots
├── trading_training_history.png          # Trading loss plots
├── confusion_matrix.png                  # Trading decision matrix
└── end_to_end_results.png               # Comprehensive results
```

## Total Time Estimate

- Setup: 10-15 minutes
- Data loading: 5 minutes
- Preflight: 10 minutes
- Consumption training: 1-3 hours (GPU)
- Trading training: 1-3 hours (GPU)
- Testing & export: 20 minutes

**Total: 3-7 hours** (mostly unattended GPU training)

## Troubleshooting

### GPU Not Available
```python
# In Colab: Runtime → Change runtime type → Hardware accelerator → GPU
# Verify with: torch.cuda.is_available()
```

### Out of Memory
```python
# Reduce batch size in Section 4 & 5:
batch_size=16  # Instead of 32
```

### Import Errors
```python
# Make sure project is at /content/drive/MyDrive/energymvp/
# Check path with: !ls /content/drive/MyDrive/energymvp/src
```

### Supabase Connection Failed
```python
# Verify .env file exists and has correct credentials
# Test with: !cat /content/drive/MyDrive/energymvp/.env
```

## After Training Completes

1. **Download models** from Google Drive → checkpoints folder
2. **Review training logs** in `training_logs.json`
3. **Check success criteria** in `TRAINING_SUMMARY.txt`
4. **Proceed to deployment** if all metrics are met

## Next Steps After MVP

Once training succeeds:

1. **Test locally** - Load checkpoints and run inference
2. **Build demo** - Create inference pipeline for real-time predictions
3. **Deploy API** - Expose models via FastAPI or similar
4. **Monitor performance** - Track metrics in production
5. **Retrain periodically** - Update models with new data

## Support

If you encounter issues:

1. Check the **Preflight Validation** section output
2. Review the **error messages** carefully (explicit handling, no fallbacks per your preference)
3. Verify **data loading** succeeded (Section 2 outputs)
4. Check **GPU availability** (Section 1 outputs)

---

**Ready to train!** Upload the notebook to Colab and start with Section 1.
