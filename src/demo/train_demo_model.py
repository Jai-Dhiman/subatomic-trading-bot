"""
Train a lightweight Transformer model for demo purposes.

Uses reduced architecture (4 layers, 4 heads, 256 d_model) for faster training.
Trains on multi-household synthetic data for 5 epochs.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.models.transformer_model import EnergyTransformer, MultiTaskLoss
from src.models.feature_engineering import FeatureEngineer


def load_multi_household_data(data_path: Path):
    """Load multi-household dataset from disk."""
    print("=" * 60)
    print("Loading Multi-Household Data")
    print("=" * 60)
    
    if data_path.suffix == '.npz':
        data = np.load(data_path, allow_pickle=True)
        df = pd.DataFrame(data['data'], columns=data['columns'].tolist())
        print(f"✓ Loaded {len(df)} samples from {data['num_households']} households")
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} samples")
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    return df


def prepare_training_data(df: pd.DataFrame, config: dict):
    """Prepare features and targets for all households."""
    print("\n" + "=" * 60)
    print("Feature Engineering")
    print("=" * 60)
    
    engineer = FeatureEngineer()
    
    # Process each household separately to maintain temporal consistency
    num_households = df['household_id'].nunique()
    all_X = []
    all_y_consumption_day = []
    all_y_price_day = []
    all_y_consumption_week = []
    all_y_price_week = []
    
    for hh_id in range(1, num_households + 1):
        hh_df = df[df['household_id'] == hh_id].copy()
        
        # Extract features
        features = engineer.prepare_features(hh_df, fit=(hh_id == 1))
        
        # Create sequences
        consumption_target = hh_df['total_consumption_kwh'].values
        price_target = hh_df['price_per_kwh'].values
        
        X, y = engineer.create_sequences(
            features,
            consumption_target,
            price_target,
            sequence_length=config['sequence_length'],
            horizons=config['horizons']
        )
        
        all_X.append(X)
        all_y_consumption_day.append(y['consumption_day'])
        all_y_price_day.append(y['price_day'])
        all_y_consumption_week.append(y['consumption_week'])
        all_y_price_week.append(y['price_week'])
    
    # Concatenate all households
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = {
        'consumption_day': np.concatenate(all_y_consumption_day, axis=0),
        'price_day': np.concatenate(all_y_price_day, axis=0),
        'consumption_week': np.concatenate(all_y_consumption_week, axis=0),
        'price_week': np.concatenate(all_y_price_week, axis=0)
    }
    
    print(f"✓ Combined features shape: {X_combined.shape}")
    for key, val in y_combined.items():
        print(f"  {key}: {val.shape}")
    
    return X_combined, y_combined, engineer


def create_data_loaders(X, y_dict, batch_size=16, train_split=0.8):
    """Create train and validation data loaders."""
    print("\n" + "=" * 60)
    print("Creating Data Loaders")
    print("=" * 60)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensors = {key: torch.FloatTensor(val) for key, val in y_dict.items()}
    
    print(f"  X shape: {X_tensor.shape}")
    for key, val in y_tensors.items():
        print(f"  {key} shape: {val.shape}")
    
    # Create dataset
    y_keys = sorted(y_tensors.keys())
    dataset = TensorDataset(X_tensor, *[y_tensors[k] for k in y_keys])
    
    # Split into train/val
    n_total = len(dataset)
    n_train = int(train_split * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n  Dataset split:")
    print(f"    Train: {n_train} samples ({n_train/n_total*100:.1f}%)")
    print(f"    Val:   {n_val} samples ({n_val/n_total*100:.1f}%)")
    print(f"    Batches: {len(train_loader)} train, {len(val_loader)} val")
    
    return train_loader, val_loader, y_keys


def train_epoch(model, train_loader, criterion, optimizer, device, y_keys, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # Unpack batch
        X_batch = batch[0].to(device)
        y_batch = {key: batch[i+1].to(device) for i, key in enumerate(y_keys)}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Calculate loss
        loss, _ = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, y_keys):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            X_batch = batch[0].to(device)
            y_batch = {key: batch[i+1].to(device) for i, key in enumerate(y_keys)}
            
            outputs = model(X_batch)
            loss, _ = criterion(outputs, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_demo_model(config: dict, data_path: Path, output_path: Path):
    """Train the demo model."""
    print("=" * 60)
    print("Training Demo Transformer Model")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {config['d_model']}d, {config['n_layers']}L, {config['n_heads']}H")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    
    device = torch.device('cpu')
    print(f"  Device: {device}")
    
    # Load data
    df = load_multi_household_data(data_path)
    
    # Prepare training data
    X, y, engineer = prepare_training_data(df, config)
    
    # Create data loaders
    train_loader, val_loader, y_keys = create_data_loaders(
        X, y,
        batch_size=config['batch_size'],
        train_split=0.8
    )
    
    # Initialize model
    print("\n" + "=" * 60)
    print("Initializing Model")
    print("=" * 60)
    
    model = EnergyTransformer(
        n_features=24,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        horizons=config['horizons']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized")
    print(f"  Parameters: {n_params:,}")
    print(f"  Model size: ~{n_params * 4 / 1e6:.1f} MB")
    
    # Initialize training components
    criterion = MultiTaskLoss(
        consumption_weight=0.6,
        price_weight=0.4,
        horizon_weights={'day': 1.0, 'week': 0.5}
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    total_start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 60)
        
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, y_keys, epoch)
        val_loss = validate(model, val_loader, criterion, device, y_keys)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Time:       {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save checkpoint
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'history': history,
                'feature_engineer': engineer
            }, output_path)
            
            print(f"    ✓ Best model saved!")
    
    total_time = time.time() - total_start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Initial train loss: {history['train_loss'][0]:.4f}")
    print(f"  Final train loss:   {history['train_loss'][-1]:.4f}")
    print(f"  Best val loss:      {best_val_loss:.4f}")
    print(f"  Total time:         {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Model saved to:     {output_path}")
    
    return model, engineer, history


if __name__ == "__main__":
    # Configuration for fast demo training
    config = {
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'sequence_length': 48,
        'horizons': {'day': 48, 'week': 336}
    }
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "demo" / "multi_household_training_data.npz"
    output_path = project_root / "checkpoints" / "demo_model.pt"
    
    # Train model
    model, engineer, history = train_demo_model(config, data_path, output_path)
    
    print("\n✅ Demo model ready for simulation!")
