"""
Consumption Transformer Model.

Multi-horizon transformer for predicting household energy consumption:
- Day-ahead predictions: 48 values (next 24 hours in 30-min intervals)
- Week-ahead predictions: 336 values (next 7 days)

Architecture:
- Input: (batch, 48, 17) - 48 timesteps of 17 features
- d_model: 384
- n_heads: 6
- n_layers: 5
- dim_feedforward: 1536
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        # x is (batch, seq_len, d_model), pe is (max_len, 1, d_model)
        # We need to transpose pe to match batch_first=True format
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        return self.dropout(x)


class ConsumptionTransformer(nn.Module):
    """
    Transformer model for multi-horizon consumption prediction.
    
    Predicts energy consumption for multiple time horizons:
    - Day: next 24 hours (48 half-hour intervals)
    - Week: next 7 days (336 half-hour intervals)
    """
    
    def __init__(
        self,
        n_features: int = 17,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 5,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        horizons: Dict[str, int] = None
    ):
        """
        Initialize Consumption Transformer.
        
        Args:
            n_features: Number of input features (default 17)
            d_model: Dimension of model embeddings (default 384)
            n_heads: Number of attention heads (default 6)
            n_layers: Number of transformer layers (default 5)
            dim_feedforward: Dimension of feedforward network (default 1536)
            dropout: Dropout rate (default 0.1)
            horizons: Dict of prediction horizons (default: day=48, week=336)
        """
        super().__init__()
        
        if horizons is None:
            horizons = {'day': 48, 'week': 336}
        
        self.n_features = n_features
        self.d_model = d_model
        self.horizons = horizons
        
        # Input projection: map input features to d_model dimensions
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Consumption prediction heads (one per horizon)
        self.consumption_heads = nn.ModuleDict()
        for horizon_name, horizon_len in horizons.items():
            self.consumption_heads[horizon_name] = nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 4, horizon_len)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, n_features)
               Default: (batch, 48, 17)
        
        Returns:
            Dictionary containing predictions for each horizon:
            {
                'consumption_day': (batch, 48),
                'consumption_week': (batch, 336)
            }
        """
        # Input projection: (batch, seq_len, n_features) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        # Output: (batch, seq_len, d_model)
        encoded = self.transformer_encoder(x)
        
        # Use the last timestep's encoding for prediction
        # Shape: (batch, d_model)
        last_encoding = encoded[:, -1, :]
        
        # Generate predictions for each horizon
        predictions = {}
        for horizon_name, head in self.consumption_heads.items():
            # Output: (batch, horizon_len)
            predictions[f'consumption_{horizon_name}'] = head(last_encoding)
        
        return predictions


class ConsumptionLoss(nn.Module):
    """
    Loss function for consumption prediction.
    
    Uses weighted MSE to balance multiple horizons:
    - Day-ahead: weight 1.0 (more important)
    - Week-ahead: weight 0.5 (less important)
    """
    
    def __init__(self, horizon_weights: Dict[str, float] = None):
        """
        Initialize loss function.
        
        Args:
            horizon_weights: Weights for each horizon (default: day=1.0, week=0.5)
        """
        super().__init__()
        if horizon_weights is None:
            horizon_weights = {'day': 1.0, 'week': 0.5}
        self.horizon_weights = horizon_weights
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate weighted MSE loss.
        
        Args:
            predictions: Dict of predictions from model
            targets: Dict of target values
        
        Returns:
            total_loss: Weighted sum of all horizon losses
            loss_dict: Individual losses for each horizon
        """
        total_loss = 0.0
        loss_dict = {}
        
        for key in predictions.keys():
            if key not in targets:
                raise ValueError(f"Target for {key} not found in targets dict")
            
            # Extract horizon name (e.g., 'consumption_day' -> 'day')
            horizon_name = key.replace('consumption_', '')
            weight = self.horizon_weights.get(horizon_name, 1.0)
            
            # Calculate MSE for this horizon
            mse_loss = self.mse(predictions[key], targets[key])
            
            # Apply weight
            weighted_loss = weight * mse_loss
            total_loss += weighted_loss
            
            loss_dict[key] = mse_loss.item()
        
        # Normalize by total weight
        total_weight = sum(self.horizon_weights.values())
        total_loss = total_loss / total_weight
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def calculate_mape(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-7) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        predictions: Predicted values
        targets: Target values
        epsilon: Small value to avoid division by zero
    
    Returns:
        MAPE as percentage
    """
    return np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100


if __name__ == "__main__":
    print("="*70)
    print("CONSUMPTION TRANSFORMER TEST")
    print("="*70)
    
    # Test configuration
    batch_size = 4
    sequence_length = 48
    n_features = 17
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Number of features: {n_features}")
    
    # Create model
    print(f"\n1. Creating Consumption Transformer...")
    model = ConsumptionTransformer(
        n_features=n_features,
        d_model=384,
        n_heads=6,
        n_layers=5,
        dim_feedforward=1536,
        dropout=0.1,
        horizons={'day': 48, 'week': 336}
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model created")
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    print(f"   ✓ Model size: ~{total_params * 4 / (1024**2):.1f} MB (float32)")
    
    # Create dummy input
    print(f"\n2. Testing forward pass...")
    x = torch.randn(batch_size, sequence_length, n_features)
    print(f"   Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"   ✓ Forward pass successful")
    print(f"   Output shapes:")
    for key, value in output.items():
        print(f"     - {key}: {value.shape}")
    
    # Test loss function
    print(f"\n3. Testing loss function...")
    criterion = ConsumptionLoss(horizon_weights={'day': 1.0, 'week': 0.5})
    
    # Create dummy targets
    targets = {
        'consumption_day': torch.randn(batch_size, 48),
        'consumption_week': torch.randn(batch_size, 336)
    }
    
    loss, loss_dict = criterion(output, targets)
    print(f"   ✓ Loss calculation successful")
    print(f"   Loss values:")
    for key, value in loss_dict.items():
        print(f"     - {key}: {value:.4f}")
    
    # Test MAPE calculation
    print(f"\n4. Testing MAPE calculation...")
    pred = np.random.randn(100)
    target = pred + np.random.randn(100) * 0.1
    mape = calculate_mape(pred, target)
    print(f"   ✓ MAPE: {mape:.2f}%")
    
    # Test gradient flow
    print(f"\n5. Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    x = torch.randn(batch_size, sequence_length, n_features)
    targets = {
        'consumption_day': torch.randn(batch_size, 48),
        'consumption_week': torch.randn(batch_size, 336)
    }
    
    optimizer.zero_grad()
    output = model(x)
    loss, _ = criterion(output, targets)
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Gradient flow successful")
    print(f"   ✓ Model can be trained")
    
    print("\n" + "="*70)
    print("✅ CONSUMPTION TRANSFORMER TEST COMPLETE")
    print("="*70)
