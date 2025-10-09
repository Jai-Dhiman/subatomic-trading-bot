"""
Local CPU Training Script for Trading Transformer.

This script trains the trading transformer locally on CPU using:
1. Pre-trained consumption transformer from checkpoints/consumption_transformer_best.pt
2. Real data from Supabase (consumption, pricing, battery)
3. Same architecture and hyperparameters as the Colab notebook

Usage:
    python scripts/train_trading_transformer_local.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging

from src.data_integration.data_adapter import (
    load_consumption_data,
    load_pricing_data,
    generate_battery_data,
    merge_all_data
)
from src.models.consumption_transformer import ConsumptionTransformer
from src.models.trading_transformer_v2 import (
    TradingTransformer,
    TradingLossV2,
    calculate_class_weights
)
from src.models.feature_engineering_consumption import ConsumptionFeatureEngineer
from src.models.feature_engineering_trading import TradingFeatureEngineer
from src.models.trading_optimizer import calculate_optimal_trading_decisions
from src.training.training_utils import (
    create_data_loaders,
    train_epoch,
    validate,
    save_checkpoint,
    load_checkpoint
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for local CPU training."""
    
    device = torch.device('cpu')
    
    consumption_model_checkpoint = 'checkpoints/consumption_transformer_best.pt'
    
    consumption_model_params = {
        'n_features': 17,
        'd_model': 384,
        'n_heads': 6,
        'n_layers': 5,
        'dim_feedforward': 1536,
        'dropout': 0.1,
        'horizons': {'day': 48, 'week': 336}
    }
    
    trading_model_params = {
        'n_features': 30,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'prediction_horizon': 48
    }
    
    trading_loss_params = {
        'price_weight': 0.20,
        'decision_weight': 0.60,  # Primary focus: learn correct decisions
        'profit_weight': 0.20,  # Secondary: profit guidance
        'household_price_kwh': 0.027,  # $27/MWh threshold for conditional trades
        'profit_scale': 10.0  # Very conservative scale to avoid instability
    }
    
    training_params = {
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'train_split': 0.8,
        'patience': 10
    }
    
    optimizer_params = {
        'buy_threshold_mwh': 20.0,   # Buy below $20/MWh (aggressive, based on real CA pricing)
        'sell_threshold_mwh': 40.0,  # Sell above $40/MWh (median is ~$39/MWh)
        'buy_percentile': 25.0,      # Fallback if dynamic thresholds needed
        'sell_percentile': 75.0,     # Fallback if dynamic thresholds needed
        'min_soc_for_sell': 0.20,    # Allow selling down to minimum SoC
        'target_soc_on_buy': 0.90    # Fill to 90% when buying
    }
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    sequence_length = 48
    prediction_horizon = 48


def load_and_prepare_data(config: TrainingConfig):
    """Load and prepare all data for training."""
    logger.info("="*70)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("="*70)
    
    logger.info("\n1. Loading consumption data from Supabase...")
    consumption_df = load_consumption_data(source='supabase')
    
    logger.info("\n2. Loading pricing data from Supabase...")
    pricing_df = load_pricing_data()
    
    logger.info("\n3. Generating battery data...")
    battery_df = generate_battery_data(
        timestamps=consumption_df['timestamp'],
        consumption_data=consumption_df
    )
    
    logger.info("\n4. Merging all data sources...")
    df_complete = merge_all_data(consumption_df, pricing_df, battery_df)
    
    logger.info(f"Complete dataset: {len(df_complete):,} records")
    logger.info(f"Date range: {df_complete['timestamp'].min()} to {df_complete['timestamp'].max()}")
    
    return df_complete, consumption_df, pricing_df


def prepare_consumption_features(df_complete: pd.DataFrame, config: TrainingConfig):
    """Prepare features for consumption transformer."""
    logger.info("\n5. Preparing consumption features...")
    
    engineer = ConsumptionFeatureEngineer()
    features = engineer.prepare_features(df_complete, fit=True)
    
    logger.info(f"Features extracted: {features.shape}")
    
    appliance_cols = [col for col in df_complete.columns if col.startswith('appliance_')]
    if appliance_cols:
        df_complete['hourly_consumption_kwh'] = df_complete[appliance_cols].sum(axis=1)
        logger.info(f"Calculated hourly consumption from {len(appliance_cols)} appliances")
    else:
        df_complete['hourly_consumption_kwh'] = df_complete['total_consumption_kwh'] / 24.0
        logger.info("Using total_consumption_kwh / 24 as hourly consumption")
    
    X_cons, y_cons = engineer.create_sequences(
        features,
        df_complete['hourly_consumption_kwh'].values,
        sequence_length=config.sequence_length,
        horizons=config.consumption_model_params['horizons']
    )
    
    logger.info(f"X_cons shape: {X_cons.shape}")
    for key, value in y_cons.items():
        logger.info(f"y_cons['{key}'] shape: {value.shape}")
    
    return X_cons, y_cons, engineer


def generate_consumption_predictions(
    model: nn.Module,
    X_cons: np.ndarray,
    config: TrainingConfig
):
    """Generate consumption predictions for all sequences."""
    logger.info("\n6. Generating consumption predictions...")
    
    model.eval()
    all_predictions = []
    batch_size = config.training_params['batch_size']
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_cons), batch_size), desc="Predicting consumption"):
            batch = torch.FloatTensor(X_cons[i:i+batch_size]).to(config.device)
            pred = model(batch)
            all_predictions.append(pred['consumption_day'].cpu().numpy())
    
    consumption_predictions = np.vstack(all_predictions)
    
    # IMPORTANT: Consumption data is HOURLY, but we need 30-min intervals
    # Scale down by 2 to get consumption per 30-min period
    consumption_predictions = consumption_predictions / 2.0
    
    logger.info(f"Generated {len(consumption_predictions):,} predictions")
    logger.info(f"Shape: {consumption_predictions.shape}")
    logger.info(f"Scaled from hourly to 30-min intervals (divided by 2)")
    logger.info(f"Mean consumption per 30-min: {consumption_predictions.mean():.3f} kWh")
    
    return consumption_predictions


def calculate_optimal_labels(
    consumption_predictions: np.ndarray,
    pricing_df: pd.DataFrame,
    config: TrainingConfig
):
    """Calculate optimal trading labels using business logic with battery state continuity."""
    logger.info("\n7. Calculating optimal trading labels...")
    
    optimal_decisions = []
    optimal_quantities = []
    optimal_prices = []
    
    # Initialize battery at 80% SoC to allow selling opportunities
    # Most prices < $0.27 means lots of buying, so start high
    # This gives battery energy to sell when prices are good
    initial_soc = 0.80
    current_battery_charge = 40.0 * initial_soc
    
    # Prepare pricing data by tiling to match consumption sequences
    all_prices = pricing_df['price_per_kwh'].values
    num_sequences = len(consumption_predictions)
    
    # Tile prices to cover all sequences (repeat pricing pattern)
    # Each sequence needs 48 intervals
    num_price_intervals_needed = num_sequences
    if len(all_prices) < num_price_intervals_needed + 48:
        # Tile the prices to have enough data
        repeats_needed = (num_price_intervals_needed + 48) // len(all_prices) + 1
        tiled_prices = np.tile(all_prices, repeats_needed)
        logger.info(f"  Tiled pricing data {repeats_needed}x to cover all sequences")
    else:
        tiled_prices = all_prices
    
    # Get thresholds
    if config.optimizer_params['buy_threshold_mwh'] is None:
        global_buy_threshold = np.percentile(all_prices * 1000.0, config.optimizer_params['buy_percentile'])
        logger.info(f"  Dynamic buy threshold: ${global_buy_threshold:.2f}/MWh (p{config.optimizer_params['buy_percentile']})")
    else:
        global_buy_threshold = config.optimizer_params['buy_threshold_mwh']
        logger.info(f"  Static buy threshold: ${global_buy_threshold:.2f}/MWh")
    
    if config.optimizer_params['sell_threshold_mwh'] is None:
        global_sell_threshold = np.percentile(all_prices * 1000.0, config.optimizer_params['sell_percentile'])
        logger.info(f"  Dynamic sell threshold: ${global_sell_threshold:.2f}/MWh (p{config.optimizer_params['sell_percentile']})")
    else:
        global_sell_threshold = config.optimizer_params['sell_threshold_mwh']
        logger.info(f"  Static sell threshold: ${global_sell_threshold:.2f}/MWh")
    
    for i in tqdm(range(len(consumption_predictions)), desc="Computing optimal decisions"):
        battery_state = {
            'current_charge_kwh': current_battery_charge,
            'capacity_kwh': 40.0,
            'min_soc': 0.20,
            'max_soc': 1.0,
            'max_charge_rate_kw': 10.0,
            'max_discharge_rate_kw': 8.0,
            'efficiency': 0.95
        }
        
        # Get price data for this sequence from tiled prices
        price_data = tiled_prices[i:i+48]
        
        # Calculate optimal decisions with global thresholds
        labels = calculate_optimal_trading_decisions(
            predicted_consumption=consumption_predictions[i],
            actual_prices=price_data,
            battery_state=battery_state,
            household_price_kwh=config.trading_loss_params['household_price_kwh'],
            preferred_buy_threshold_mwh=global_buy_threshold,
            preferred_sell_threshold_mwh=global_sell_threshold,
            min_soc_for_sell=config.optimizer_params['min_soc_for_sell'],
            target_soc_on_buy=config.optimizer_params['target_soc_on_buy'],
            reserve_hours=12.0
        )
        
        optimal_decisions.append(labels['optimal_decisions'])
        optimal_quantities.append(labels['optimal_quantities'])
        optimal_prices.append(price_data[0])
        
        # Update battery state based on FIRST decision in sequence (for continuity)
        decision = labels['optimal_decisions'][0]
        quantity = labels['optimal_quantities'][0]
        
        if decision == 0:  # Buy
            current_battery_charge += quantity * battery_state['efficiency']
        elif decision == 2:  # Sell
            current_battery_charge -= quantity
        
        # Apply consumption from first interval
        actual_consumption = consumption_predictions[i][0]
        current_battery_charge -= actual_consumption
        
        # Enforce battery constraints
        current_soc = current_battery_charge / 40.0
        current_soc = np.clip(current_soc, 0.20, 1.0)
        current_battery_charge = 40.0 * current_soc
    
    optimal_decisions = np.array(optimal_decisions)
    optimal_quantities = np.array(optimal_quantities)
    optimal_prices = np.array(optimal_prices)
    
    logger.info(f"Calculated {len(optimal_decisions):,} optimal labels")
    
    # Decision distribution
    unique, counts = np.unique(optimal_decisions[:, 0], return_counts=True)
    logger.info("Decision distribution:")
    for decision, count in zip(unique, counts):
        decision_name = ['Buy', 'Hold', 'Sell'][int(decision)]
        logger.info(f"  - {decision_name}: {count} ({count/len(optimal_decisions)*100:.1f}%)")
    
    # Calculate aggregate profit metrics
    total_buy_cost = 0.0
    total_sell_revenue = 0.0
    total_buy_energy = 0.0
    total_sell_energy = 0.0
    
    for i in range(len(optimal_decisions)):
        decision = optimal_decisions[i, 0]
        quantity = optimal_quantities[i, 0]
        price = optimal_prices[i]
        
        if decision == 0:  # Buy
            total_buy_cost += quantity * price
            total_buy_energy += quantity
        elif decision == 2:  # Sell
            total_sell_revenue += quantity * price
            total_sell_energy += quantity
    
    market_profit = total_sell_revenue - total_buy_cost
    logger.info(f"\nOptimizer profit analysis:")
    logger.info(f"  Total buy energy: {total_buy_energy:.2f} kWh @ cost ${total_buy_cost:.2f}")
    logger.info(f"  Total sell energy: {total_sell_energy:.2f} kWh @ revenue ${total_sell_revenue:.2f}")
    logger.info(f"  Market profit: ${market_profit:.2f}")
    logger.info(f"  Avg buy price: ${(total_buy_cost/total_buy_energy*1000 if total_buy_energy > 0 else 0):.2f}/MWh")
    logger.info(f"  Avg sell price: ${(total_sell_revenue/total_sell_energy*1000 if total_sell_energy > 0 else 0):.2f}/MWh")
    
    return optimal_decisions, optimal_quantities, optimal_prices


def prepare_trading_features(
    consumption_predictions: np.ndarray,
    df_complete: pd.DataFrame,
    optimal_decisions: np.ndarray,
    optimal_quantities: np.ndarray,
    optimal_prices: np.ndarray,
    config: TrainingConfig
):
    """Prepare trading features and create sequences."""
    logger.info("\n8. Preparing trading features...")
    
    engineer = TradingFeatureEngineer()
    features_trading = engineer.prepare_features(
        consumption_predictions,
        df_complete.iloc[:len(consumption_predictions)]
    )
    
    logger.info(f"Trading features: {features_trading.shape}")
    
    n_samples_trading = len(features_trading) - config.sequence_length
    X_trading = np.zeros((n_samples_trading, config.sequence_length, features_trading.shape[1]))
    
    for i in range(n_samples_trading):
        X_trading[i] = features_trading[i:i+config.sequence_length]
    
    y_trading = {
        'price': optimal_prices[:n_samples_trading],
        'decisions': optimal_decisions[:n_samples_trading, 0],
        'quantities': optimal_quantities[:n_samples_trading, 0],
        'consumption': consumption_predictions[:n_samples_trading, 0]
    }
    
    logger.info(f"Trading sequences: {X_trading.shape}")
    logger.info("Target shapes:")
    for key, value in y_trading.items():
        logger.info(f"  - {key}: {value.shape}")
    
    return X_trading, y_trading, engineer


def train_trading_transformer(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    y_keys: list,
    config: TrainingConfig
):
    """Train the trading transformer."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING TRADING TRANSFORMER")
    logger.info("="*70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'market_profit': [],
        'business_profit': []
    }
    
    for epoch in range(1, config.training_params['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config.training_params['num_epochs']}")
        logger.info("-" * 50)
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, config.device, y_keys
        )
        
        val_loss, val_metrics = validate(
            model, val_loader, criterion, config.device, y_keys
        )
        
        scheduler.step(val_loss)
        
        # Evaluate on full validation set for metrics
        model.eval()
        all_pred_decisions = []
        all_target_decisions = []
        all_profits = []
        all_target_profits = []
        
        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch[0].to(config.device)
                targets_batch = {key: batch[i+1].to(config.device) for i, key in enumerate(y_keys)}
                predictions_batch = model(X_batch)
                _, loss_dict_batch = criterion(predictions_batch, targets_batch)
                
                pred_decisions = torch.argmax(predictions_batch['trading_decisions'][:, 0, :], dim=1)
                all_pred_decisions.append(pred_decisions.cpu())
                all_target_decisions.append(targets_batch['decisions'].cpu())
                all_profits.append(loss_dict_batch['predicted_market_profit'])
                all_target_profits.append(loss_dict_batch['market_profit'])
        
        # Aggregate metrics
        all_pred_decisions = torch.cat(all_pred_decisions)
        all_target_decisions = torch.cat(all_target_decisions)
        decision_accuracy = (all_pred_decisions == all_target_decisions).float().mean().item() * 100
        avg_predicted_profit = np.mean(all_profits)
        avg_target_profit = np.mean(all_target_profits)
        
        # Get loss dict from last batch for other metrics
        loss_dict = loss_dict_batch
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['market_profit'].append(avg_predicted_profit)
        history['business_profit'].append(loss_dict.get('predicted_business_profit', 0))
        
        logger.info(f"Train Loss:      {train_loss:.4f}")
        logger.info(f"Val Loss:        {val_loss:.4f}")
        logger.info(f"Decision Accuracy:         {decision_accuracy:.2f}%")
        logger.info(f"Market Profit (target):    ${avg_target_profit:.4f} (optimal baseline)")
        logger.info(f"Market Profit (predicted): ${avg_predicted_profit:.4f} (model performance)")
        logger.info(f"Business Profit:           ${loss_dict.get('predicted_business_profit', 0):.4f}")
        logger.info(f"LR:                        {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = config.checkpoint_dir / 'trading_transformer_best_local.pt'
            save_checkpoint(
                model, optimizer, epoch,
                {'train_loss': train_loss, 'val_loss': val_loss},
                str(checkpoint_path)
            )
            logger.info(f"✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{config.training_params['patience']}")
            
            if patience_counter >= config.training_params['patience']:
                logger.info(f"\nEarly stopping after {epoch} epochs")
                break
        
        if epoch % 10 == 0:
            checkpoint_path = config.checkpoint_dir / f'trading_transformer_epoch_{epoch}_local.pt'
            save_checkpoint(
                model, optimizer, epoch,
                {'train_loss': train_loss, 'val_loss': val_loss},
                str(checkpoint_path)
            )
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("="*70)
    
    return history, best_val_loss


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("LOCAL TRADING TRANSFORMER TRAINING")
    logger.info("="*70)
    logger.info(f"Device: {TrainingConfig.device}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = TrainingConfig()
    
    try:
        df_complete, consumption_df, pricing_df = load_and_prepare_data(config)
        
        X_cons, y_cons, cons_engineer = prepare_consumption_features(df_complete, config)
        
        logger.info("\nLoading pre-trained consumption transformer...")
        model_consumption = ConsumptionTransformer(**config.consumption_model_params).to(config.device)
        load_checkpoint(str(config.consumption_model_checkpoint), model_consumption)
        model_consumption.eval()
        logger.info("✓ Consumption model loaded")
        
        consumption_predictions = generate_consumption_predictions(model_consumption, X_cons, config)
        
        optimal_decisions, optimal_quantities, optimal_prices = calculate_optimal_labels(
            consumption_predictions, pricing_df, config
        )
        
        X_trading, y_trading, trading_engineer = prepare_trading_features(
            consumption_predictions, df_complete, optimal_decisions, 
            optimal_quantities, optimal_prices, config
        )
        
        logger.info("\n9. Creating data loaders...")
        train_loader, val_loader, y_keys = create_data_loaders(
            X_trading,
            y_trading,
            batch_size=config.training_params['batch_size'],
            train_split=config.training_params['train_split']
        )
        
        logger.info("\n10. Initializing trading transformer...")
        model_trading = TradingTransformer(**config.trading_model_params).to(config.device)
        total_params = sum(p.numel() for p in model_trading.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB")
        
        logger.info("\n11. Setting up training...")
        class_weights = calculate_class_weights(optimal_decisions[:, 0])
        logger.info(f"Class weights (buy/hold/sell): {[f'{w:.2f}' for w in class_weights]}")
        
        criterion = TradingLossV2(
            **config.trading_loss_params,
            class_weights=class_weights
        ).to(config.device)
        
        optimizer = torch.optim.AdamW(
            model_trading.parameters(),
            lr=config.training_params['learning_rate'],
            weight_decay=config.training_params['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        history, best_val_loss = train_trading_transformer(
            model_trading, criterion, optimizer, scheduler,
            train_loader, val_loader, y_keys, config
        )
        
        logger.info("\n" + "="*70)
        logger.info("SUCCESS")
        logger.info("="*70)
        logger.info(f"Best model saved to: {config.checkpoint_dir / 'trading_transformer_best_local.pt'}")
        logger.info(f"Training history saved to: {config.checkpoint_dir / 'training_history_local.json'}")
        
        import json
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        with open(config.checkpoint_dir / 'training_history_local.json', 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
