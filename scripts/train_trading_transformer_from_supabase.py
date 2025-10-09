"""
Training Script Using Real Supabase Battery Data.

This script uses the actual optimal trading decisions from Supabase battery tables
instead of re-calculating them. This ensures we train on the corrected optimizer output.
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

from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import load_consumption_data, load_pricing_data
from src.models.trading_transformer_v2 import (
    TradingTransformer,
    TradingLossV2,
    calculate_class_weights
)
from src.models.feature_engineering_trading import TradingFeatureEngineer
from src.training.training_utils import (
    create_data_loaders,
    train_epoch,
    validate,
    save_checkpoint
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training."""
    
    device = torch.device('cpu')
    
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
        'decision_weight': 0.60,
        'profit_weight': 0.20,
        'household_price_kwh': 0.027,
        'profit_scale': 10.0
    }
    
    training_params = {
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'train_split': 0.8,
        'patience': 10
    }
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    sequence_length = 48
    prediction_horizon = 48


def load_battery_data_from_supabase(connector: SupabaseConnector, house_ids: list) -> pd.DataFrame:
    """Load battery data from Supabase tables."""
    logger.info("Loading battery data from Supabase...")
    
    all_battery_data = []
    
    for house_id in house_ids:
        table_name = f"house{house_id}_battery"
        logger.info(f"  Loading {table_name}...")
        
        try:
            response = connector.client.table(table_name).select("*").execute()
            if response.data:
                df = pd.DataFrame(response.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                all_battery_data.append(df)
                logger.info(f"    ✓ Loaded {len(df)} records")
        except Exception as e:
            logger.warning(f"    ⚠ Error loading {table_name}: {e}")
    
    if not all_battery_data:
        raise ValueError("No battery data loaded from Supabase!")
    
    battery_df = pd.concat(all_battery_data, ignore_index=True)
    battery_df = battery_df.sort_values(['house_id', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"  ✓ Total battery records: {len(battery_df):,}")
    logger.info(f"  ✓ Houses: {sorted(battery_df['house_id'].unique())}")
    logger.info(f"  ✓ Date range: {battery_df['timestamp'].min()} to {battery_df['timestamp'].max()}")
    
    return battery_df


def prepare_training_data(
    battery_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    config: TrainingConfig
) -> tuple:
    """Prepare training data from battery records."""
    logger.info("\nPreparing training data...")
    
    # Map decisions to integers
    action_map = {'buy': 0, 'hold': 1, 'sell': 2}
    battery_df['decision'] = battery_df['action'].map(action_map)
    
    # Create consumption predictions (use actual consumption from battery records)
    # Battery data already has consumption_kwh which is per 30-min
    # We need to create a day-ahead (48 intervals) prediction
    # For simplicity, use historical consumption pattern
    
    # Group by house and create sequences
    sequences = []
    
    for house_id in battery_df['house_id'].unique():
        house_data = battery_df[battery_df['house_id'] == house_id].copy()
        house_data = house_data.sort_values('timestamp').reset_index(drop=True)
        
        # Create 48-interval sequences
        for i in range(len(house_data) - config.sequence_length):
            sequence = house_data.iloc[i:i + config.sequence_length]
            
            sequences.append({
                'house_id': house_id,
                'timestamp': sequence.iloc[-1]['timestamp'],
                'consumption': sequence['consumption_kwh'].values,
                'prices': sequence['price_per_kwh'].values,
                'soc': sequence['battery_soc_percent'].values,
                'decisions': sequence['decision'].values,
                'quantities': sequence['trade_amount_kwh'].values,
                # Target: next interval
                'target_price': sequence.iloc[0]['price_per_kwh'],
                'target_decision': sequence.iloc[0]['decision'],
                'target_quantity': sequence.iloc[0]['trade_amount_kwh'],
                'target_consumption': sequence.iloc[0]['consumption_kwh']
            })
    
    logger.info(f"  ✓ Created {len(sequences)} sequences")
    
    # Create features using TradingFeatureEngineer
    engineer = TradingFeatureEngineer()
    
    # Build feature matrix
    X = []
    y = {
        'price': [],
        'decisions': [],
        'quantities': [],
        'consumption': []
    }
    
    for seq in tqdm(sequences, desc="Engineering features"):
        # Create basic features for each timestep
        features_timesteps = []
        for t in range(config.sequence_length):
            features_t = [
                seq['consumption'][t],
                seq['prices'][t],
                seq['soc'][t],
                # Add more features as needed
            ]
            # Pad to match expected feature count (30 features)
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
    
    logger.info(f"  ✓ X shape: {X.shape}")
    logger.info(f"  ✓ y shapes: {', '.join(f'{k}={v.shape}' for k, v in y.items())}")
    
    # Print decision distribution
    unique, counts = np.unique(y['decisions'], return_counts=True)
    logger.info("\n  Decision distribution:")
    for decision, count in zip(unique, counts):
        decision_name = ['Buy', 'Hold', 'Sell'][int(decision)]
        logger.info(f"    {decision_name}: {count} ({count/len(y['decisions'])*100:.1f}%)")
    
    return X, y, engineer


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("TRADING TRANSFORMER TRAINING - USING SUPABASE BATTERY DATA")
    logger.info("="*70)
    
    config = TrainingConfig()
    
    try:
        # 1. Load data from Supabase
        logger.info("\n1. Loading data from Supabase...")
        connector = SupabaseConnector()
        
        consumption_df = load_consumption_data(source='supabase')
        pricing_df = load_pricing_data()
        house_ids = sorted(consumption_df['house_id'].unique())
        
        battery_df = load_battery_data_from_supabase(connector, house_ids)
        
        # 2. Prepare training data
        X, y, engineer = prepare_training_data(battery_df, pricing_df, config)
        
        # 3. Create data loaders
        logger.info("\n2. Creating data loaders...")
        train_loader, val_loader, y_keys = create_data_loaders(
            X, y,
            batch_size=config.training_params['batch_size'],
            train_split=config.training_params['train_split']
        )
        
        # 4. Initialize model
        logger.info("\n3. Initializing trading transformer...")
        model = TradingTransformer(**config.trading_model_params).to(config.device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {total_params:,}")
        
        # 5. Setup training
        logger.info("\n4. Setting up training...")
        class_weights = calculate_class_weights(y['decisions'])
        logger.info(f"  Class weights: {[f'{w:.2f}' for w in class_weights]}")
        
        criterion = TradingLossV2(
            **config.trading_loss_params,
            class_weights=class_weights
        ).to(config.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
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
        
        # 6. Train
        logger.info("\n" + "="*70)
        logger.info("TRAINING")
        logger.info("="*70)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
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
            
            # Evaluate on full validation set
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
            
            all_pred_decisions = torch.cat(all_pred_decisions)
            all_target_decisions = torch.cat(all_target_decisions)
            decision_accuracy = (all_pred_decisions == all_target_decisions).float().mean().item() * 100
            avg_predicted_profit = np.mean(all_profits)
            avg_target_profit = np.mean(all_target_profits)
            
            logger.info(f"Train Loss:      {train_loss:.4f}")
            logger.info(f"Val Loss:        {val_loss:.4f}")
            logger.info(f"Decision Accuracy: {decision_accuracy:.2f}%")
            logger.info(f"Market Profit (target):    ${avg_target_profit:.4f}")
            logger.info(f"Market Profit (predicted): ${avg_predicted_profit:.4f}")
            logger.info(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint_path = config.checkpoint_dir / 'trading_transformer_supabase_best.pt'
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
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
