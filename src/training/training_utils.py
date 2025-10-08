"""
Training Utilities for Dual-Transformer System.

Provides functions for:
- Creating data loaders (train/val split)
- Training epoch
- Validation
- Checkpoint saving/loading
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def create_data_loaders(
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create train and validation data loaders.
    
    Args:
        X: Input sequences (n_samples, seq_len, n_features)
        y: Dict of target arrays
        batch_size: Batch size for training
        train_split: Fraction of data for training (default 0.8)
        shuffle: Whether to shuffle data
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        y_keys: List of target keys
    """
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensors = [torch.FloatTensor(y[key]) for key in sorted(y.keys())]
    y_keys = sorted(y.keys())
    
    # Create dataset
    dataset = TensorDataset(X_tensor, *y_tensors)
    
    # Split into train and val
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Created data loaders:")
    print(f"  Train samples: {n_train:,}")
    print(f"  Val samples: {n_val:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, y_keys


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    y_keys: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        y_keys: List of target keys
    
    Returns:
        avg_loss: Average loss over epoch
        metrics: Dict of additional metrics
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        # Unpack batch
        X_batch = batch[0].to(device)
        y_batch = {key: batch[i+1].to(device) for i, key in enumerate(y_keys)}
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss, loss_dict = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    metrics = {'train_loss': avg_loss}
    
    return avg_loss, metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    y_keys: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        y_keys: List of target keys
    
    Returns:
        avg_loss: Average loss over validation set
        metrics: Dict of additional metrics
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch
            X_batch = batch[0].to(device)
            y_batch = {key: batch[i+1].to(device) for i, key in enumerate(y_keys)}
            
            # Forward pass
            predictions = model(X_batch)
            loss, loss_dict = criterion(predictions, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    metrics = {'val_loss': avg_loss}
    
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        metrics: Dict of metrics at this checkpoint
        path: Path to save checkpoint
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
    
    Returns:
        checkpoint: Dict with checkpoint information
    """
    # PyTorch 2.6+ requires weights_only=False to load our checkpoints
    # This is safe since we trust our own checkpoints
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Metrics: {checkpoint['metrics']}")
    
    return checkpoint


if __name__ == "__main__":
    print("="*70)
    print("TRAINING UTILITIES TEST")
    print("="*70)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    n_samples = 500
    seq_len = 48
    n_features = 17
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y = {
        'consumption_day': np.random.randn(n_samples, 48),
        'consumption_week': np.random.randn(n_samples, 336)
    }
    
    print(f"   X shape: {X.shape}")
    print(f"   y keys: {list(y.keys())}")
    
    # Test data loaders
    print("\n2. Creating data loaders...")
    train_loader, val_loader, y_keys = create_data_loaders(
        X, y,
        batch_size=32,
        train_split=0.8
    )
    
    # Create dummy model
    print("\n3. Creating dummy model...")
    from src.models.consumption_transformer import ConsumptionTransformer, ConsumptionLoss
    
    model = ConsumptionTransformer(n_features=n_features)
    criterion = ConsumptionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cpu')
    
    print(f"   ✓ Model created")
    
    # Test training epoch
    print("\n4. Testing training epoch...")
    train_loss, train_metrics = train_epoch(
        model, train_loader, criterion, optimizer, device, y_keys
    )
    print(f"   ✓ Train loss: {train_loss:.4f}")
    
    # Test validation
    print("\n5. Testing validation...")
    val_loss, val_metrics = validate(
        model, val_loader, criterion, device, y_keys
    )
    print(f"   ✓ Val loss: {val_loss:.4f}")
    
    # Test checkpoint saving
    print("\n6. Testing checkpoint saving...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
        
        save_checkpoint(
            model, optimizer, epoch=1,
            metrics={'train_loss': train_loss, 'val_loss': val_loss},
            path=str(checkpoint_path)
        )
        
        # Test checkpoint loading
        print("\n7. Testing checkpoint loading...")
        model2 = ConsumptionTransformer(n_features=n_features)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
        
        checkpoint = load_checkpoint(str(checkpoint_path), model2, optimizer2)
        print(f"   ✓ Loaded from epoch: {checkpoint['epoch']}")
    
    print("\n" + "="*70)
    print("✅ TRAINING UTILITIES TEST COMPLETE")
    print("="*70)
