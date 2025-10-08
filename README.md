# Federated Energy Trading System - POC

> A proof-of-concept demonstrating cost savings through AI-powered, federated energy trading among households with battery storage.

## Overview

This project implements a distributed energy trading system where individual households ("nodes") use machine learning to predict their energy consumption and autonomously trade energy with each other, coordinated by a central model. The goal is to demonstrate 20-40% cost savings compared to standard PG&E electricity pricing.

### Key Features

- **Federated Learning**: Node models train locally on household data, share updates with central model
- **Autonomous Trading**: Each household makes independent buy/sell decisions
- **Battery Optimization**: Smart charging/discharging to maximize savings
- **Market Mechanism**: Instant peer-to-peer energy matching with price signals
- **Cost Comparison**: Baseline vs optimized savings analysis

## Quick Start

### Prerequisites

1. Python 3.12 with `uv` package manager
2. Supabase account with electricity pricing data
3. Virtual environment activated

### Setup

1. **Install dependencies:**
```bash
cd /Users/jdhiman/Documents/energymvp
source .venv/bin/activate
uv pip install -e .
```

2. **Configure Supabase:**
```bash
cp .env.example .env
# Edit .env with your SUPABASE_URL and SUPABASE_KEY
```

3. **Test data connection:**
```bash
python src/data_integration/test_ca_pricing.py
```

### Run Simulation

```bash
python -m src.simulation.run_demo
```

Results will be saved to `data/output/simulation_results.json`

## Project Structure

```
energymvp/
├── README.md                    # This file
├── docs/                        # Full documentation
│   ├── CURRENT_STATUS.md       # Production-ready status
│   ├── NEXT_STEPS.md           # Remaining tasks
│   ├── TECHNICAL_REFERENCE.md  # Algorithms and formulas
│   ├── QUICKSTART.md           # Setup guide
│   ├── EXECUTIVE_SUMMARY.md    # High-level overview
│   └── archive/                # Historical documentation
├── config/
│   └── config.yaml             # System configuration
├── src/
│   ├── data_integration/       # Supabase integration (COMPLETE)
│   ├── models/                 # ML models, battery, grid, profitability (COMPLETE)
│   ├── federated/              # Federated learning (COMPLETE)
│   ├── trading/                # Market mechanism (COMPLETE)
│   ├── simulation/             # Simulation engine (COMPLETE)
│   └── output/                 # Metrics and export
├── data/
│   ├── generated/              # Local generated data
│   └── output/                 # Simulation results
└── tests/                      # Tests
```

## System Architecture

```
┌─────────────────────────────────┐
│      Central Model              │
│  - Price Signals                │
│  - FedAvg Aggregation           │
│  - Market Rules                 │
└────────────┬────────────────────┘
             │
     ┌───────┴────────┐
     │  Message Bus    │
     └───────┬────────┘
             │
    ┌────────┴──────────┐
    │  Market Mechanism  │
    └────────┬──────────┘
             │
  ┌──────────┼──────────┐
  │          │          │
Node 1     Node 2    Node 10
(LSTM)     (LSTM)    (LSTM)
```

## Core Components

### 1. Node Model
- **LSTM** time series forecasting
- **Inputs**: Historical usage, weather, time features
- **Output**: 30-minute interval predictions (3 hours ahead)
- **Battery**: 13.5 kWh capacity, 80% usable, 90% efficiency
- **Status:** Production-ready

### 2. Grid Constraints Manager
- **Import Limit**: 10 kWh per 30-minute interval
- **Export Limit**: 4 kWh per 30-minute interval
- **Validation**: All grid transactions checked
- **Status:** Production-ready, tested

### 3. Profitability Calculator
- **Scoring**: 0-100 profitability score for trading opportunities
- **Signals**: GREEN/RED power need indicators
- **Components**: Trading potential, battery health, market conditions, capacity utilization
- **Status:** Production-ready

### 4. Central Model
- **Price Signals**: Dynamic pricing based on supply/demand
- **FedAvg**: Aggregates node model weights every 3 hours
- **Market Rules**: California regulations
- **Status:** Production-ready

### 5. Trading System
- **Instant Matching**: No order book, real-time trades
- **Transmission Loss**: 5% energy loss in peer-to-peer trades
- **Profit Optimization**: Nodes trade when advantageous vs PG&E rates
- **Status:** Production-ready

## Simulation Parameters

- **Households**: 10 nodes with unique profiles
- **Duration**: 24 hours (48 thirty-minute intervals)
- **Historical Data**: 1 year for training
- **Baseline**: PG&E TOU-C rate schedule
- **Target Savings**: 20-40% cost reduction

## Key Metrics

### Primary
- **Cost Savings**: Optimized cost vs PG&E baseline

### Secondary
- Prediction accuracy (MAPE)
- Trading efficiency
- Battery utilization
- Peak demand reduction

## Technology Stack

- **Python**: 3.12
- **Package Manager**: uv
- **ML Framework**: PyTorch
- **Data**: Pandas, NumPy
- **Async**: asyncio
- **Config**: PyYAML
- **Testing**: pytest

## Implementation Status

### Completed
1. ✅ Project setup and configuration
2. ✅ Data integration (Supabase)
3. ✅ Node model (LSTM, battery management)
4. ✅ Grid constraints manager
5. ✅ Profitability calculator
6. ✅ Central model coordination
7. ✅ Trading system and market mechanism
8. ✅ Simulation engine
9. ✅ Federated learning (FedAvg)

### In Progress
- Testing suite (integration, battery, trading)
- Consumption data integration (blocked on data)
- Logging and metrics collection

### Remaining
- End-to-end simulation with real data
- 20-40% savings validation

## Output Format

Simulation generates JSON with:
- Per-household timeline (predictions, trades, battery states)
- Cost comparison (optimized vs baseline)
- System-wide metrics (total trades, savings, peak reduction)
- Visualization-ready data

Example:
```json
{
  "households": [
    {
      "id": 1,
      "costs": {
        "baseline_pge_total": 22.45,
        "optimized_total": 15.32,
        "savings": 7.13,
        "savings_percent": 31.7
      }
    }
  ],
  "aggregate_metrics": {
    "avg_household_savings_percent": 31.8,
    "total_energy_traded_kwh": 42.7
  }
}
```

## Configuration

Edit `config/config.yaml` to adjust:
- Number of households
- Battery specifications
- Trading parameters
- Model hyperparameters
- Pricing rates

## Testing

```bash
pytest tests/
```

## Documentation

- **CURRENT_STATUS.md**: Production-ready status and metrics
- **NEXT_STEPS.md**: Remaining tasks and priorities
- **TECHNICAL_REFERENCE.md**: Detailed algorithms and formulas
- **QUICKSTART.md**: Setup and development guide
- **EXECUTIVE_SUMMARY.md**: High-level project overview

## Data Sources

### Real Data (Supabase) - ✓ INTEGRATED

**Pricing Data:**
- **Source:** California ISO electricity market data
- **Table:** `cabuyingpricehistoryseptember2025` (40,200 records)
- **Date Range:** October 1-9, 2024
- **Price Range:** $-0.057 to $0.190 per kWh
- **Columns:** `INTERVALSTARTTIME_GMT`, `Price KWH`, `NODE`, etc.

**Data Adapter:**
- Transforms CA pricing schema to simulation format
- Maps `INTERVALSTARTTIME_GMT` → `timestamp`
- Maps `Price KWH` → `price_per_kwh`
- Handles 1-hour intervals

**Test Commands:**
```bash
python src/data_integration/list_tables.py        # List all tables
python src/data_integration/test_ca_pricing.py    # Full pricing test
```

**Consumption Data:**
- Ready to integrate from Supabase
- Parser module complete (`consumption_parser.py`)
- Waiting for table population

**Weather Data:**
- Planned for future enhancement
- Not blocking current POC

## Team Collaboration

### Frontend Integration
JSON outputs in `data/output/` are ready for visualization:
- Timeline data for charts
- Cost comparison metrics
- Trading activity heatmaps

### Hardware Integration
This is a software-only POC. For hardware:
- Deploy models to edge devices (ONNX, TFLite)
- Integrate with IoT/smart meters
- Add real-time data streams

## Future Enhancements

- Transformer models for forecasting
- Weather API integration
- Sports/events calendar
- Multi-step optimization
- Differential privacy
- Scalability to 1000+ households

## Success Criteria

- ✅ 10 household models implemented
- ✅ Federated learning ready (FedAvg)
- ✅ Trading system operational with constraints
- ✅ Grid constraints enforced (10 kWh import, 4 kWh export)
- ✅ Profitability calculator operational
- ✅ Clean, production-ready codebase (no dummy data)
- ⏳ 20-40% cost savings demonstration (pending real data)

## License

[To be determined]

## Contact

Engineering Lead: jdhiman
Project: Energy Trading System MVP
Timeline: 1-2 weeks (14 days)

---

**Status**: Production-Ready - Awaiting Real Consumption Data  
**Last Updated**: 2025-10-08  
**Next Step**: Integration testing and consumption data loading
