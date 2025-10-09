"""
Model Evaluation Script - Matching Notebook Section 6.

Runs the same evaluations as the train_dual_transformers.ipynb notebook:
1. Consumption MAPE
2. Price MAE  
3. Trading decision accuracy
4. Cost savings vs baseline
5. Confusion matrix
6. Visualizations

Models evaluated:
- consumption_transformer_best.pt
- trading_transformer_supabase_best.pt
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import load_consumption_data, load_pricing_data
from src.models.consumption_transformer import ConsumptionTransformer
from src.models.trading_transformer_v2 import TradingTransformer
from src.models.feature_engineering_consumption import ConsumptionFeatureEngineer
from src.models.feature_engineering_trading import TradingFeatureEngineer
from src.training.training_utils import load_checkpoint


def calculate_mape(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = targets > 0  # Avoid division by zero
    return np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100


def load_battery_data_from_supabase(connector: SupabaseConnector, house_ids: list) -> pd.DataFrame:
    """Load battery data from Supabase tables."""
    print("Loading battery data from Supabase...")
    
    all_battery_data = []
    
    for house_id in house_ids:
        table_name = f"house{house_id}_battery"
        print(f"  Loading {table_name}...")
        
        try:
            response = connector.client.table(table_name).select("*").execute()
            if response.data:
                df = pd.DataFrame(response.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                all_battery_data.append(df)
                print(f"    ✓ Loaded {len(df)} records")
        except Exception as e:
            print(f"    ⚠ Error loading {table_name}: {e}")
    
    battery_df = pd.concat(all_battery_data, ignore_index=True)
    battery_df = battery_df.sort_values(['house_id', 'timestamp']).reset_index(drop=True)
    
    return battery_df


def prepare_data_for_evaluation(consumption_df, pricing_df, battery_df):
    """Prepare data for evaluation (matching notebook preprocessing)."""
    print("\nPreparing data for evaluation...")
    
    # Merge data
    from src.data_integration.data_adapter import merge_all_data, generate_battery_data
    
    # Generate synthetic battery for merge (we'll override with real battery data)
    battery_synthetic = generate_battery_data(
        consumption_df['timestamp'],
        consumption_df
    )
    
    df_complete = merge_all_data(consumption_df, pricing_df, battery_synthetic)
    
    # Feature engineering for consumption
    engineer_consumption = ConsumptionFeatureEngineer()
    features_consumption = engineer_consumption.prepare_features(df_complete, fit=True)
    
    # Calculate hourly consumption from appliances
    appliance_cols = [col for col in df_complete.columns if col.startswith('appliance_')]
    if appliance_cols:
        df_complete['hourly_consumption_kwh'] = df_complete[appliance_cols].sum(axis=1)
    else:
        df_complete['hourly_consumption_kwh'] = df_complete['total_consumption_kwh'] / 24.0
    
    # Create sequences for consumption transformer
    sequence_length = 48
    X_cons, y_cons = engineer_consumption.create_sequences(
        features_consumption,
        df_complete['hourly_consumption_kwh'].values,
        sequence_length=sequence_length,
        horizons={'day': 48, 'week': 336}
    )
    
    print(f"  ✓ Consumption sequences: {X_cons.shape}")
    print(f"  ✓ y_cons shapes: {', '.join(f'{k}={v.shape}' for k, v in y_cons.items())}")
    
    # Prepare trading data from battery records
    action_map = {'buy': 0, 'hold': 1, 'sell': 2}
    battery_df['decision'] = battery_df['action'].map(action_map)
    
    # Create trading sequences
    sequences = []
    for house_id in battery_df['house_id'].unique():
        house_data = battery_df[battery_df['house_id'] == house_id].copy()
        house_data = house_data.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(house_data) - sequence_length):
            sequence = house_data.iloc[i:i + sequence_length]
            sequences.append({
                'consumption': sequence['consumption_kwh'].values,
                'prices': sequence['price_per_kwh'].values,
                'soc': sequence['battery_soc_percent'].values,
                'target_price': sequence.iloc[0]['price_per_kwh'],
                'target_decision': sequence.iloc[0]['decision'],
                'target_quantity': sequence.iloc[0]['trade_amount_kwh'],
                'target_consumption': sequence.iloc[0]['consumption_kwh']
            })
    
    # Build trading feature matrix
    X_trading = []
    y_trading = {
        'price': [],
        'decisions': [],
        'quantities': [],
        'consumption': []
    }
    
    for seq in sequences:
        features_timesteps = []
        for t in range(sequence_length):
            features_t = [
                seq['consumption'][t],
                seq['prices'][t],
                seq['soc'][t],
            ]
            features_t += [0.0] * (30 - len(features_t))
            features_timesteps.append(features_t)
        
        X_trading.append(features_timesteps)
        y_trading['price'].append(seq['target_price'])
        y_trading['decisions'].append(seq['target_decision'])
        y_trading['quantities'].append(seq['target_quantity'])
        y_trading['consumption'].append(seq['target_consumption'])
    
    X_trading = np.array(X_trading, dtype=np.float32)
    for key in y_trading:
        y_trading[key] = np.array(y_trading[key], dtype=np.float32)
    
    print(f"  ✓ Trading sequences: {X_trading.shape}")
    print(f"  ✓ y_trading shapes: {', '.join(f'{k}={v.shape}' for k, v in y_trading.items())}")
    
    return X_cons, y_cons, X_trading, y_trading


def run_evaluation(model_consumption, model_trading, X_cons, y_cons, X_trading, y_trading, device):
    """Run comprehensive evaluation (matching notebook Section 6)."""
    
    print("\n" + "="*70)
    print("SECTION 6: END-TO-END TESTING")
    print("="*70)
    print("\nRunning complete inference pipeline on test data...\n")
    
    # Set models to eval mode
    model_consumption.eval()
    model_trading.eval()
    
    # Use last 100 samples as test set (matching notebook)
    test_size = min(100, len(X_cons), len(X_trading))
    print(f"Using last {test_size} samples for testing")
    
    test_X_cons = torch.FloatTensor(X_cons[-test_size:]).to(device)
    test_X_trading = torch.FloatTensor(X_trading[-test_size:]).to(device)
    
    # Run inference
    print("\n2. Running inference on test samples...")
    with torch.no_grad():
        # Consumption predictions
        consumption_pred = model_consumption(test_X_cons)
        
        # Trading predictions
        trading_pred = model_trading(test_X_trading)
    
    print("   ✓ Inference complete")
    
    # Calculate comprehensive metrics
    print("\n3. Calculating metrics...")
    
    # Consumption MAPE
    cons_pred_day = consumption_pred['consumption_day'].cpu().numpy()
    cons_target_day = y_cons['consumption_day'][-test_size:]
    consumption_mape = calculate_mape(cons_pred_day, cons_target_day)
    
    # Price MAE
    price_pred = trading_pred['predicted_price'][:, 0].cpu().numpy()
    price_target = y_trading['price'][-test_size:]
    price_mae = np.mean(np.abs(price_pred - price_target))
    
    # Trading decision accuracy
    decision_pred = torch.argmax(trading_pred['trading_decisions'][:, 0, :], dim=1).cpu().numpy()
    decision_target = y_trading['decisions'][-test_size:].astype(int)
    decision_accuracy = (decision_pred == decision_target).mean() * 100
    
    print(f"\n   📊 PERFORMANCE METRICS:")
    print(f"   {'='*50}")
    print(f"   Consumption MAPE:      {consumption_mape:.2f}% {'✓' if consumption_mape < 15 else '❌'} (target < 15%)")
    print(f"   Price MAE:             ${price_mae:.4f} {'✓' if price_mae < 0.05 else '❌'} (target < $0.05)")
    print(f"   Trading Accuracy:      {decision_accuracy:.2f}%")
    print(f"   {'='*50}")
    
    # Calculate cost savings vs baseline
    print("\n4. Calculating cost savings vs baseline...")
    
    # Baseline: buy all energy from grid at market price
    baseline_cost = (cons_target_day * price_target[:, np.newaxis]).sum()
    
    # Optimized: use trading decisions
    quantities_pred = trading_pred['trade_quantities'][:, 0].cpu().numpy()
    buy_mask = (decision_pred == 0)
    sell_mask = (decision_pred == 2)
    
    trading_cost = baseline_cost  # Start with baseline
    trading_cost -= (quantities_pred[sell_mask] * price_target[sell_mask]).sum()  # Revenue from selling
    trading_cost += (quantities_pred[buy_mask] * price_target[buy_mask]).sum()  # Cost of buying
    
    savings = baseline_cost - trading_cost
    savings_percent = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
    
    print(f"\n   💰 COST ANALYSIS:")
    print(f"   {'='*50}")
    print(f"   Baseline Cost:         ${baseline_cost:.2f}")
    print(f"   Optimized Cost:        ${trading_cost:.2f}")
    print(f"   Savings:               ${savings:.2f}")
    print(f"   Savings Percentage:    {savings_percent:.2f}% {'✓' if 20 <= savings_percent <= 40 else '⚠️'} (target 20-40%)")
    print(f"   {'='*50}")
    
    # Trading decision confusion matrix
    print("\n5. Trading decision distribution...")
    
    cm = confusion_matrix(decision_target, decision_pred)
    labels = ['Buy', 'Hold', 'Sell']
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Trading Decision Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    print("   ✓ Confusion matrix saved to results/confusion_matrix.png")
    
    # Visualize sample predictions
    print("\n6. Visualizing sample predictions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Consumption prediction
    sample_idx = 0
    axes[0, 0].plot(cons_target_day[sample_idx], label='Actual', marker='o', markersize=3)
    axes[0, 0].plot(cons_pred_day[sample_idx], label='Predicted', marker='x', markersize=3)
    axes[0, 0].set_xlabel('Time Interval')
    axes[0, 0].set_ylabel('Consumption (kWh)')
    axes[0, 0].set_title('Consumption Prediction (Next 24 Hours)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Price prediction
    axes[0, 1].scatter(price_target, price_pred, alpha=0.5)
    axes[0, 1].plot([price_target.min(), price_target.max()], 
                    [price_target.min(), price_target.max()], 
                    'r--', label='Perfect Prediction')
    axes[0, 1].set_xlabel('Actual Price ($/kWh)')
    axes[0, 1].set_ylabel('Predicted Price ($/kWh)')
    axes[0, 1].set_title('Price Prediction Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Trading decisions over time
    plot_length = min(50, len(decision_pred))
    axes[1, 0].plot(decision_pred[:plot_length], marker='o', label='Predicted')
    axes[1, 0].plot(decision_target[:plot_length], marker='x', label='Optimal', alpha=0.7)
    axes[1, 0].set_xlabel('Time Interval')
    axes[1, 0].set_ylabel('Decision (0=Buy, 1=Hold, 2=Sell)')
    axes[1, 0].set_title(f'Trading Decisions (First {plot_length} Intervals)')
    axes[1, 0].set_yticks([0, 1, 2])
    axes[1, 0].set_yticklabels(['Buy', 'Hold', 'Sell'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Decision distribution comparison
    unique_pred, counts_pred = np.unique(decision_pred, return_counts=True)
    unique_target, counts_target = np.unique(decision_target, return_counts=True)
    
    x = np.arange(3)
    width = 0.35
    axes[1, 1].bar(x - width/2, counts_target, width, label='Target', alpha=0.8)
    axes[1, 1].bar(x + width/2, counts_pred, width, label='Predicted', alpha=0.8)
    axes[1, 1].set_xlabel('Decision')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Decision Distribution Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Buy', 'Hold', 'Sell'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'end_to_end_results.png', dpi=150)
    plt.close()
    
    print("   ✓ Visualizations saved to results/end_to_end_results.png")
    
    print("\n" + "="*70)
    print("✅ END-TO-END TESTING COMPLETE")
    print("="*70)
    
    return {
        'consumption_mape': consumption_mape,
        'price_mae': price_mae,
        'trading_accuracy': decision_accuracy,
        'baseline_cost': baseline_cost,
        'optimized_cost': trading_cost,
        'savings': savings,
        'savings_percent': savings_percent
    }


def generate_summary_report(metrics: dict):
    """Generate summary report (matching notebook Section 7)."""
    
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    summary = f"""
{'='*70}
DUAL-TRANSFORMER ENERGY TRADING SYSTEM
Evaluation Summary Report
{'='*70}

Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
EVALUATION METRICS
{'='*70}
Consumption MAPE:    {metrics['consumption_mape']:.2f}% {'✓' if metrics['consumption_mape'] < 15 else '❌'} (target < 15%)
Price MAE:           ${metrics['price_mae']:.4f} {'✓' if metrics['price_mae'] < 0.05 else '❌'} (target < $0.05)
Trading Accuracy:    {metrics['trading_accuracy']:.2f}%

Cost Analysis:
  Baseline Cost:     ${metrics['baseline_cost']:.2f}
  Optimized Cost:    ${metrics['optimized_cost']:.2f}
  Savings:           ${metrics['savings']:.2f}
  Savings %:         {metrics['savings_percent']:.2f}% {'✓' if 20 <= metrics['savings_percent'] <= 40 else '⚠️'} (target 20-40%)

{'='*70}
SUCCESS CRITERIA
{'='*70}
✓ Consumption MAPE < 15%:     {'✓ PASSED' if metrics['consumption_mape'] < 15 else '❌ NOT MET'}
✓ Price MAE < $0.05/kWh:      {'✓ PASSED' if metrics['price_mae'] < 0.05 else '❌ NOT MET'}
✓ Cost savings 20-40%:        {'✓ PASSED' if 20 <= metrics['savings_percent'] <= 40 else '⚠️ NOT MET'}

{'='*70}
FILES GENERATED
{'='*70}
Results:
  - results/confusion_matrix.png
  - results/end_to_end_results.png
  - results/EVALUATION_SUMMARY.txt

{'='*70}
EVALUATION COMPLETE
{'='*70}
"""
    
    print(summary)
    
    # Save summary to file
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'EVALUATION_SUMMARY.txt', 'w') as f:
        f.write(summary)
    
    print("\n   ✓ Summary report saved to results/EVALUATION_SUMMARY.txt")


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("MODEL EVALUATION - NOTEBOOK STYLE")
    print("="*70)
    
    device = torch.device('cpu')
    checkpoint_dir = Path('checkpoints')
    
    # Load data
    print("\n1. Loading data from Supabase...")
    connector = SupabaseConnector()
    consumption_df = load_consumption_data(source='supabase')
    pricing_df = load_pricing_data()
    house_ids = sorted(consumption_df['house_id'].unique())
    battery_df = load_battery_data_from_supabase(connector, house_ids)
    
    # Prepare data
    X_cons, y_cons, X_trading, y_trading = prepare_data_for_evaluation(
        consumption_df, pricing_df, battery_df
    )
    
    # Load models
    print("\n2. Loading best models...")
    
    model_consumption = ConsumptionTransformer(
        n_features=17,
        d_model=384,
        n_heads=6,
        n_layers=5,
        dim_feedforward=1536,
        dropout=0.1,
        horizons={'day': 48, 'week': 336}
    ).to(device)
    
    model_trading = TradingTransformer(
        n_features=30,
        d_model=512,
        n_heads=8,
        n_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        prediction_horizon=48
    ).to(device)
    
    load_checkpoint(
        str(checkpoint_dir / 'consumption_transformer_best.pt'),
        model_consumption
    )
    load_checkpoint(
        str(checkpoint_dir / 'trading_transformer_supabase_best.pt'),
        model_trading
    )
    
    print("   ✓ Both models loaded and set to eval mode")
    
    # Run evaluation
    metrics = run_evaluation(
        model_consumption, model_trading,
        X_cons, y_cons, X_trading, y_trading,
        device
    )
    
    # Generate summary report
    generate_summary_report(metrics)
    
    print("\n" + "="*70)
    print("✅ ALL EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
