"""
Local Training Script for Transformer Pre-Flight Verification.

This script trains the EnergyTransformer model on synthetic data
to verify the entire pipeline works before deploying to Google Colab.

Usage:
    python src/training/train_transformer_local.py
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

from src.data_integration.synthetic_data_generator import generate_synthetic_household_data
from src.models.feature_engineering import FeatureEngineer
from src.models.transformer_model import EnergyTransformer, MultiTaskLoss


def create_data_loaders(X, y_dict, batch_size=32, train_split=0.8):
    """
    Create train and validation data loaders.
    
    Args:
        X: Input sequences (n_samples, seq_len, n_features)
        y_dict: Dictionary of targets
        batch_size: Batch size for training
        train_split: Fraction of data for training
        
    Returns:
        train_loader, val_loader
    """
    print(f"\n{'='*60}")
    print("Creating Data Loaders")
    print(f"{'='*60}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensors = {key: torch.FloatTensor(val) for key, val in y_dict.items()}
    
    print(f"  X shape: {X_tensor.shape}")
    for key, val in y_tensors.items():
        print(f"  {key} shape: {val.shape}")
    
    # Create dataset
    # Order: X, then all y values in sorted key order
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
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"\n  Dataset split:")
    print(f"    Train: {n_train} samples ({n_train/n_total*100:.1f}%)")
    print(f"    Val:   {n_val} samples ({n_val/n_total*100:.1f}%)")
    print(f"    Batches per epoch: {len(train_loader)} train, {len(val_loader)} val")
    
    return train_loader, val_loader, y_keys


def train_epoch(model, train_loader, criterion, optimizer, device, y_keys, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {}
    
    for batch_idx, batch in enumerate(train_loader):
        # Unpack batch
        X_batch = batch[0].to(device)
        y_batch = {key: batch[i+1].to(device) for i, key in enumerate(y_keys)}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Calculate loss
        loss, loss_dict = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for key, val in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = 0.0
            loss_components[key] += val
        
        # Print progress every 5 batches
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] - Loss: {avg_loss:.4f}")
    
    # Average losses
    avg_loss = total_loss / len(train_loader)
    for key in loss_components:
        loss_components[key] /= len(train_loader)
    
    return avg_loss, loss_components


def validate(model, val_loader, criterion, device, y_keys):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    loss_components = {}
    
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch
            X_batch = batch[0].to(device)
            y_batch = {key: batch[i+1].to(device) for i, key in enumerate(y_keys)}
            
            # Forward pass
            outputs = model(X_batch)
            
            # Calculate loss
            loss, loss_dict = criterion(outputs, y_batch)
            
            # Track losses
            total_loss += loss.item()
            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += val
    
    # Average losses
    avg_loss = total_loss / len(val_loader)
    for key in loss_components:
        loss_components[key] /= len(val_loader)
    
    return avg_loss, loss_components


def main():
    print("="*60)
    print("Transformer Local Training - Pre-Flight Verification")
    print("="*60)
    
    # Configuration
    config = {
        'days': 10,
        'batch_size': 8,  # Small batch for local training
        'epochs': 10,
        'learning_rate': 1e-4,
        'device': 'cpu',  # Use CPU for pre-flight
        'horizons': {'day': 48, 'week': 336}  # Removed month for 10-day data
    }
    
    print("\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    
    # Set device
    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")
    
    # Step 1: Generate synthetic data
    print(f"\n{'='*60}")
    print("Step 1: Generating Synthetic Data")
    print(f"{'='*60}")
    
    df = generate_synthetic_household_data(days=config['days'])
    print(f"✓ Generated {len(df)} samples")
    
    # Step 2: Feature engineering
    print(f"\n{'='*60}")
    print("Step 2: Feature Engineering")
    print(f"{'='*60}")
    
    engineer = FeatureEngineer()
    features = engineer.prepare_features(df, fit=True)
    print(f"✓ Extracted features: {features.shape}")
    
    # Create sequences
    consumption_target = df['total_consumption_kwh'].values
    price_target = df['price_per_kwh'].values
    
    X, y = engineer.create_sequences(
        features,
        consumption_target,
        price_target,
        sequence_length=48,
        horizons=config['horizons']
    )
    
    print(f"✓ Created sequences: X={X.shape}")
    for key, val in y.items():
        print(f"  {key}: {val.shape}")
    
    # Step 3: Create data loaders
    train_loader, val_loader, y_keys = create_data_loaders(
        X, y, 
        batch_size=config['batch_size'],
        train_split=0.8
    )
    
    # Step 4: Initialize model
    print(f"\n{'='*60}")
    print("Step 3: Initializing Model")
    print(f"{'='*60}")
    
    model = EnergyTransformer(
        n_features=24,
        d_model=512,
        n_heads=8,
        n_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        horizons=config['horizons']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized")
    print(f"  Parameters: {n_params:,}")
    print(f"  Model size: ~{n_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Step 5: Initialize training components
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
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    print(f"✓ Optimizer: AdamW (lr={config['learning_rate']}, weight_decay=0.01)")
    print(f"✓ Scheduler: ReduceLROnPlateau (patience=3)")
    print(f"✓ Criterion: MultiTaskLoss (consumption=0.6, price=0.4)")
    
    # Step 6: Training loop
    print(f"\n{'='*60}")
    print("Step 4: Training")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 60)
        
        # Train
        start_time = time.time()
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, y_keys, epoch
        )
        train_time = time.time() - start_time
        
        # Validate
        val_loss, val_components = validate(
            model, val_loader, criterion, device, y_keys
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print epoch summary
        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Time:       {train_time:.1f}s")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_dir = project_root / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / "transformer_preflight.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'history': history
            }, checkpoint_path)
            
            print(f"    ✓ Best model saved! (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n  Early stopping after {epoch} epochs")
                break
    
    # Step 7: Load best model and test
    print(f"\n{'='*60}")
    print("Step 5: Final Evaluation")
    print(f"{'='*60}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['val_loss']:.4f}")
    
    # Make sample predictions
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        X_sample = sample_batch[0][:1].to(device)
        
        predictions = model(X_sample)
        
        print(f"\n  Sample predictions:")
        for key, val in predictions.items():
            print(f"    {key}: min={val.min():.4f}, max={val.max():.4f}, mean={val.mean():.4f}")
    
    # Print training summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    print(f"\nResults:")
    print(f"  Initial train loss: {history['train_loss'][0]:.4f}")
    print(f"  Final train loss:   {history['train_loss'][-1]:.4f}")
    print(f"  Best val loss:      {best_val_loss:.4f}")
    print(f"  Epochs completed:   {len(history['train_loss'])}")
    print(f"  Model saved to:     {checkpoint_path}")
    
    # Check if loss decreased
    if history['train_loss'][-1] < history['train_loss'][0]:
        print(f"\n✅ Training successful - loss decreased!")
    else:
        print(f"\n⚠️  Warning: loss did not decrease significantly")
    
    print(f"\n✅ Pre-flight verification complete!")
    print(f"✅ Architecture verified - ready for Colab GPU training")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
