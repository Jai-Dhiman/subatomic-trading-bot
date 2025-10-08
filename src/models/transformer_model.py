"""
Transformer-based model for multi-horizon energy consumption and price prediction.

Architecture:
- Positional encoding for time series
- Multi-head self-attention (6 layers, 8 heads)
- Multi-task learning (consumption + price prediction)
- Multi-horizon outputs (1 day, 1 week, 1 month)
- Compatible with federated learning (FedAvg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EnergyTransformer(nn.Module):
    """
    Transformer model for multi-horizon consumption and price prediction.
    
    Input: (batch_size, sequence_length, n_features)
    Output: Dict with keys:
        - consumption_day: (batch_size, 48)
        - consumption_week: (batch_size, 336)
        - consumption_month: (batch_size, 1440)
        - price_day: (batch_size, 48)
        - price_week: (batch_size, 336)
        - price_month: (batch_size, 1440)
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        horizons: Dict[str, int] = None
    ):
        super().__init__()
        
        if horizons is None:
            horizons = {'day': 48, 'week': 336, 'month': 1440}
        
        self.horizons = horizons
        self.d_model = d_model
        self.n_features = n_features
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Multi-task prediction heads
        self.consumption_heads = nn.ModuleDict()
        self.price_heads = nn.ModuleDict()
        
        for horizon_name, horizon_len in horizons.items():
            # Consumption prediction head
            self.consumption_heads[horizon_name] = nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, horizon_len)
            )
            
            # Price prediction head
            self.price_heads[horizon_name] = nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, horizon_len)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, n_features)
            
        Returns:
            Dictionary with predictions for each horizon and task
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Use last token representation for predictions
        context = encoded[:, -1, :]  # (batch, d_model)
        
        # Generate predictions for each horizon
        outputs = {}
        
        for horizon_name in self.horizons.keys():
            outputs[f'consumption_{horizon_name}'] = self.consumption_heads[horizon_name](context)
            outputs[f'price_{horizon_name}'] = self.price_heads[horizon_name](context)
        
        return outputs
    
    def get_model_weights(self) -> dict:
        """Get model weights for federated learning."""
        return self.state_dict()
    
    def update_model_weights(self, weights: dict):
        """Update model weights from federated aggregation."""
        self.load_state_dict(weights)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for consumption and price prediction.
    
    Combines MSE for consumption and MAE for price prediction,
    weighted by task importance and horizon length.
    """
    
    def __init__(
        self,
        consumption_weight: float = 0.6,
        price_weight: float = 0.4,
        horizon_weights: Dict[str, float] = None
    ):
        super().__init__()
        
        self.consumption_weight = consumption_weight
        self.price_weight = price_weight
        
        if horizon_weights is None:
            horizon_weights = {'day': 1.0, 'week': 0.5, 'month': 0.25}
        self.horizon_weights = horizon_weights
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate multi-task loss.
        
        Args:
            predictions: Dict of predicted tensors
            targets: Dict of target tensors
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components for logging
        """
        total_loss = 0.0
        loss_dict = {}
        
        for horizon in self.horizon_weights.keys():
            # Consumption loss (MSE)
            cons_key = f'consumption_{horizon}'
            if cons_key in predictions and cons_key in targets:
                cons_loss = F.mse_loss(predictions[cons_key], targets[cons_key])
                weighted_cons_loss = (
                    cons_loss * 
                    self.consumption_weight * 
                    self.horizon_weights[horizon]
                )
                total_loss += weighted_cons_loss
                loss_dict[f'loss_{cons_key}'] = cons_loss.item()
            
            # Price loss (MAE)
            price_key = f'price_{horizon}'
            if price_key in predictions and price_key in targets:
                price_loss = F.l1_loss(predictions[price_key], targets[price_key])
                weighted_price_loss = (
                    price_loss * 
                    self.price_weight * 
                    self.horizon_weights[horizon]
                )
                total_loss += weighted_price_loss
                loss_dict[f'loss_{price_key}'] = price_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EnergyTransformer")
    print("=" * 60)
    
    # Model configuration
    config = {
        'n_features': 24,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'horizons': {'day': 48, 'week': 336, 'month': 1440}
    }
    
    print("\n1. Initializing model...")
    model = EnergyTransformer(**config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {n_params:,}")
    print(f"   Model size: ~{n_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Test input
    batch_size = 4
    seq_len = 48
    x = torch.randn(batch_size, seq_len, config['n_features'])
    
    print(f"\n2. Testing forward pass...")
    print(f"   Input shape: {x.shape}")
    
    # Forward pass
    outputs = model(x)
    
    print(f"\n3. Output shapes:")
    for key, val in outputs.items():
        print(f"     {key}: {val.shape}")
    
    # Verify output shapes
    expected_shapes = {
        'consumption_day': (batch_size, 48),
        'consumption_week': (batch_size, 336),
        'consumption_month': (batch_size, 1440),
        'price_day': (batch_size, 48),
        'price_week': (batch_size, 336),
        'price_month': (batch_size, 1440),
    }
    
    all_correct = True
    for key, expected_shape in expected_shapes.items():
        if outputs[key].shape != expected_shape:
            print(f"   ❌ ERROR: {key} has shape {outputs[key].shape}, expected {expected_shape}")
            all_correct = False
    
    if all_correct:
        print(f"\n   ✓ All output shapes correct!")
    
    # Test loss calculation
    print(f"\n4. Testing loss calculation...")
    targets = {key: torch.randn_like(val) for key, val in outputs.items()}
    
    criterion = MultiTaskLoss()
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"   Loss components:")
    for key, val in loss_dict.items():
        print(f"     {key}: {val:.4f}")
    
    # Test gradient flow
    print(f"\n5. Testing gradient flow...")
    loss.backward()
    
    has_grads = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"   ❌ No gradient for {name}")
            has_grads = False
            break
    
    if has_grads:
        print(f"   ✓ Gradients computed for all parameters!")
    
    # Test save/load
    print(f"\n6. Testing save/load...")
    weights = model.get_model_weights()
    print(f"   Saved {len(weights)} weight tensors")
    
    # Create new model and load weights
    model2 = EnergyTransformer(**config)
    model2.update_model_weights(weights)
    print(f"   ✓ Weights loaded successfully!")
    
    # Verify weights match
    outputs2 = model2(x)
    match = all(torch.allclose(outputs[k], outputs2[k]) for k in outputs.keys())
    if match:
        print(f"   ✓ Outputs match after loading weights!")
    else:
        print(f"   ❌ Outputs don't match after loading weights")
    
    print("\n" + "=" * 60)
    print("✅ EnergyTransformer test complete!")
    print(f"✅ Model has {n_params:,} parameters (~12M target)")
    print(f"✅ Ready for training (Task 4)")
    print("=" * 60)
