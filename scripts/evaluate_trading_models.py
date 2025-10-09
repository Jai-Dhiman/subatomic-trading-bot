"""
Evaluation Script for Trading Transformers.

Compares two models:
1. Old model: trained_transformer_best_local.pt (with wrong optimizer labels)
2. New model: trading_transformer_supabase_best.pt (with corrected Supabase data)

Evaluates on the corrected Supabase battery data.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import load_consumption_data, load_pricing_data
from src.models.trading_transformer_v2 import TradingTransformer
from src.training.training_utils import load_checkpoint


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
    
    print(f"  ✓ Total battery records: {len(battery_df):,}")
    
    return battery_df


def prepare_test_data(battery_df: pd.DataFrame, sequence_length: int = 48):
    """Prepare test sequences from battery data."""
    print("\nPreparing test data...")
    
    # Map decisions to integers
    action_map = {'buy': 0, 'hold': 1, 'sell': 2}
    battery_df['decision'] = battery_df['action'].map(action_map)
    
    # Create sequences
    sequences = []
    
    for house_id in battery_df['house_id'].unique():
        house_data = battery_df[battery_df['house_id'] == house_id].copy()
        house_data = house_data.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(house_data) - sequence_length):
            sequence = house_data.iloc[i:i + sequence_length]
            
            sequences.append({
                'house_id': house_id,
                'timestamp': sequence.iloc[-1]['timestamp'],
                'consumption': sequence['consumption_kwh'].values,
                'prices': sequence['price_per_kwh'].values,
                'soc': sequence['battery_soc_percent'].values,
                'decisions': sequence['decision'].values,
                'quantities': sequence['trade_amount_kwh'].values,
                'target_price': sequence.iloc[0]['price_per_kwh'],
                'target_decision': sequence.iloc[0]['decision'],
                'target_quantity': sequence.iloc[0]['trade_amount_kwh'],
                'target_consumption': sequence.iloc[0]['consumption_kwh']
            })
    
    print(f"  ✓ Created {len(sequences)} test sequences")
    
    # Build feature matrix
    X = []
    y = {
        'price': [],
        'decisions': [],
        'quantities': [],
        'consumption': []
    }
    
    for seq in tqdm(sequences, desc="Engineering features"):
        features_timesteps = []
        for t in range(sequence_length):
            features_t = [
                seq['consumption'][t],
                seq['prices'][t],
                seq['soc'][t],
            ]
            features_t += [0.0] * (30 - len(features_t))
            features_timesteps.append(features_t)
        
        X.append(features_timesteps)
        y['price'].append(seq['target_price'])
        y['decisions'].append(seq['target_decision'])
        y['quantities'].append(seq['target_quantity'])
        y['consumption'].append(seq['target_consumption'])
    
    X = np.array(X, dtype=np.float32)
    for key in y:
        y[key] = np.array(y[key], dtype=np.float32)
    
    print(f"  ✓ X shape: {X.shape}")
    print(f"  ✓ y shapes: {', '.join(f'{k}={v.shape}' for k, v in y.items())}")
    
    return X, y


def evaluate_model(model: torch.nn.Module, X: np.ndarray, y: dict, device: torch.device, model_name: str):
    """Evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    model.eval()
    
    # Batch evaluation
    batch_size = 32
    all_pred_decisions = []
    all_pred_quantities = []
    all_pred_prices = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="Evaluating"):
            batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
            predictions = model(batch_X)
            
            pred_decisions = torch.argmax(predictions['trading_decisions'][:, 0, :], dim=1)
            pred_quantities = predictions['trade_quantities'][:, 0]
            pred_prices = predictions['predicted_price'][:, 0]
            
            all_pred_decisions.append(pred_decisions.cpu().numpy())
            all_pred_quantities.append(pred_quantities.cpu().numpy())
            all_pred_prices.append(pred_prices.cpu().numpy())
    
    all_pred_decisions = np.concatenate(all_pred_decisions)
    all_pred_quantities = np.concatenate(all_pred_quantities)
    all_pred_prices = np.concatenate(all_pred_prices)
    
    # Calculate metrics
    target_decisions = y['decisions'].astype(int)
    target_quantities = y['quantities']
    target_prices = y['price']
    
    # Decision accuracy
    decision_accuracy = (all_pred_decisions == target_decisions).mean() * 100
    
    # Per-class accuracy
    class_names = ['Buy', 'Hold', 'Sell']
    class_accuracies = []
    for cls in range(3):
        mask = target_decisions == cls
        if mask.sum() > 0:
            cls_acc = (all_pred_decisions[mask] == target_decisions[mask]).mean() * 100
            class_accuracies.append(cls_acc)
            print(f"\n{class_names[cls]} Accuracy: {cls_acc:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(target_decisions, all_pred_decisions)
    
    # Price prediction error
    price_mae = np.abs(all_pred_prices - target_prices).mean()
    price_rmse = np.sqrt(((all_pred_prices - target_prices) ** 2).mean())
    
    # Quantity prediction error
    quantity_mae = np.abs(all_pred_quantities - target_quantities).mean()
    
    # Calculate profit metrics
    target_profit = calculate_profit(target_decisions, target_quantities, target_prices)
    predicted_profit = calculate_profit(all_pred_decisions, all_pred_quantities, target_prices)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"\nDecision Metrics:")
    print(f"  Overall Accuracy: {decision_accuracy:.2f}%")
    print(f"  Buy Accuracy: {class_accuracies[0]:.2f}%")
    print(f"  Hold Accuracy: {class_accuracies[1]:.2f}%")
    print(f"  Sell Accuracy: {class_accuracies[2]:.2f}%")
    
    print(f"\nPrice Prediction:")
    print(f"  MAE: ${price_mae:.4f}/kWh (${price_mae*1000:.2f}/MWh)")
    print(f"  RMSE: ${price_rmse:.4f}/kWh (${price_rmse*1000:.2f}/MWh)")
    
    print(f"\nQuantity Prediction:")
    print(f"  MAE: {quantity_mae:.3f} kWh")
    
    print(f"\nProfit Metrics:")
    print(f"  Target Total Profit: ${target_profit:.2f}")
    print(f"  Predicted Total Profit: ${predicted_profit:.2f}")
    print(f"  Profit Ratio: {predicted_profit/target_profit:.2%}")
    print(f"  Profit Difference: ${predicted_profit - target_profit:.2f}")
    
    # Decision distribution
    print(f"\nDecision Distribution:")
    for cls in range(3):
        target_count = (target_decisions == cls).sum()
        pred_count = (all_pred_decisions == cls).sum()
        print(f"  {class_names[cls]}: Target={target_count} ({target_count/len(target_decisions)*100:.1f}%), " +
              f"Predicted={pred_count} ({pred_count/len(all_pred_decisions)*100:.1f}%)")
    
    return {
        'name': model_name,
        'decision_accuracy': decision_accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm,
        'price_mae': price_mae,
        'quantity_mae': quantity_mae,
        'target_profit': target_profit,
        'predicted_profit': predicted_profit,
        'pred_decisions': all_pred_decisions,
        'target_decisions': target_decisions
    }


def calculate_profit(decisions: np.ndarray, quantities: np.ndarray, prices: np.ndarray) -> float:
    """Calculate total market profit."""
    profit = 0.0
    
    # Buy decisions (cost = negative profit)
    buy_mask = decisions == 0
    if buy_mask.sum() > 0:
        profit -= (quantities[buy_mask] * prices[buy_mask]).sum()
    
    # Sell decisions (revenue = positive profit)
    sell_mask = decisions == 2
    if sell_mask.sum() > 0:
        profit += (quantities[sell_mask] * prices[sell_mask]).sum()
    
    return profit


def plot_comparison(results_old: dict, results_new: dict, save_path: Path):
    """Create comparison visualizations."""
    print(f"\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Decision Accuracy Comparison
    ax = axes[0, 0]
    models = ['Old Model', 'New Model']
    accuracies = [results_old['decision_accuracy'], results_new['decision_accuracy']]
    bars = ax.bar(models, accuracies, color=['#ff6b6b', '#51cf66'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Decision Accuracy')
    ax.set_ylim([0, 100])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Per-Class Accuracy
    ax = axes[0, 1]
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, results_old['class_accuracies'], width, label='Old Model', color='#ff6b6b')
    ax.bar(x + width/2, results_new['class_accuracies'], width, label='New Model', color='#51cf66')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(['Buy', 'Hold', 'Sell'])
    ax.legend()
    ax.set_ylim([0, 100])
    
    # 3. Profit Comparison
    ax = axes[1, 0]
    x = np.arange(2)
    old_profits = [results_old['target_profit'], results_old['predicted_profit']]
    new_profits = [results_new['target_profit'], results_new['predicted_profit']]
    
    ax.bar(x - width/2, old_profits, width, label='Old Model', color='#ff6b6b')
    ax.bar(x + width/2, new_profits, width, label='New Model', color='#51cf66')
    ax.set_ylabel('Total Profit ($)')
    ax.set_title('Profit Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Target', 'Predicted'])
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 4. Confusion Matrix Comparison (New Model)
    ax = axes[1, 1]
    cm = results_new['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax,
                xticklabels=['Buy', 'Hold', 'Sell'],
                yticklabels=['Buy', 'Hold', 'Sell'])
    ax.set_title('New Model Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comparison plot to {save_path}")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("TRADING TRANSFORMER MODEL COMPARISON")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Model configurations
    model_params = {
        'n_features': 30,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'prediction_horizon': 48
    }
    
    checkpoint_dir = Path('checkpoints')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading test data...")
    connector = SupabaseConnector()
    consumption_df = load_consumption_data(source='supabase')
    pricing_df = load_pricing_data()
    house_ids = sorted(consumption_df['house_id'].unique())
    
    battery_df = load_battery_data_from_supabase(connector, house_ids)
    X, y = prepare_test_data(battery_df)
    
    # Evaluate Old Model
    print("\n2. Evaluating OLD model (wrong optimizer)...")
    model_old = TradingTransformer(**model_params).to(device)
    old_checkpoint = checkpoint_dir / 'trading_transformer_best_local.pt'
    
    if old_checkpoint.exists():
        load_checkpoint(str(old_checkpoint), model_old)
        results_old = evaluate_model(model_old, X, y, device, "Old Model (Wrong Optimizer)")
    else:
        print(f"  ⚠ Old model checkpoint not found: {old_checkpoint}")
        results_old = None
    
    # Evaluate New Model
    print("\n3. Evaluating NEW model (Supabase data)...")
    model_new = TradingTransformer(**model_params).to(device)
    new_checkpoint = checkpoint_dir / 'trading_transformer_supabase_best.pt'
    
    if new_checkpoint.exists():
        load_checkpoint(str(new_checkpoint), model_new)
        results_new = evaluate_model(model_new, X, y, device, "New Model (Supabase Data)")
    else:
        print(f"  ⚠ New model checkpoint not found: {new_checkpoint}")
        results_new = None
    
    # Comparison
    if results_old and results_new:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        print(f"\nDecision Accuracy:")
        print(f"  Old Model: {results_old['decision_accuracy']:.2f}%")
        print(f"  New Model: {results_new['decision_accuracy']:.2f}%")
        print(f"  Improvement: {results_new['decision_accuracy'] - results_old['decision_accuracy']:+.2f}%")
        
        print(f"\nProfit:")
        print(f"  Old Model: ${results_old['predicted_profit']:.2f}")
        print(f"  New Model: ${results_new['predicted_profit']:.2f}")
        print(f"  Improvement: ${results_new['predicted_profit'] - results_old['predicted_profit']:+.2f}")
        
        print(f"\nTarget Profit: ${results_new['target_profit']:.2f}")
        print(f"Old Model vs Target: {results_old['predicted_profit']/results_new['target_profit']:.1%}")
        print(f"New Model vs Target: {results_new['predicted_profit']/results_new['target_profit']:.1%}")
        
        # Generate plots
        plot_path = results_dir / 'model_comparison.png'
        plot_comparison(results_old, results_new, plot_path)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
