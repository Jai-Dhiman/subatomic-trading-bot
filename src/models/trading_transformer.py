"""
Trading Transformer Model.

Multi-task transformer for price prediction and trading decisions:
- Price predictions: 48 values (next 24 hours)
- Trading decisions: 48x3 (buy/sell/hold probabilities)
- Trade quantities: 48 values (kWh amounts)

Architecture:
- Input: (batch, 48, 20) - 48 timesteps of 20 features
- d_model: 512
- n_heads: 8
- n_layers: 6
- Multi-task loss: 30% price + 30% decision + 40% profitability
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
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        return self.dropout(x)


class TradingTransformer(nn.Module):
    """
    Multi-task transformer for price prediction and trading decisions.
    
    Predicts:
    - Electricity prices for next 24 hours
    - Trading decisions (Buy/Hold/Sell)
    - Trade quantities (kWh amounts)
    """
    
    def __init__(
        self,
        n_features: int = 20,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        prediction_horizon: int = 48
    ):
        """
        Initialize Trading Transformer.
        
        Args:
            n_features: Number of input features (default 20)
            d_model: Dimension of model embeddings (default 512)
            n_heads: Number of attention heads (default 8)
            n_layers: Number of transformer layers (default 6)
            dim_feedforward: Dimension of feedforward network (default 2048)
            dropout: Dropout rate (default 0.1)
            prediction_horizon: Number of intervals to predict (default 48)
        """
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        
        # Input projection
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
        
        # Multi-task prediction heads
        
        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, prediction_horizon)
        )
        
        # Trading decision head (Buy/Hold/Sell for each interval)
        self.decision_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, prediction_horizon * 3)
        )
        
        # Trade quantity head
        self.quantity_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, prediction_horizon),
            nn.ReLU()  # Ensure non-negative quantities
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
               Default: (batch, 48, 20)
        
        Returns:
            Dictionary containing:
            {
                'predicted_price': (batch, 48),
                'trading_decisions': (batch, 48, 3),  # logits for buy/hold/sell
                'trade_quantities': (batch, 48)
            }
        """
        # Input projection: (batch, seq_len, n_features) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Use last timestep's encoding for predictions
        last_encoding = encoded[:, -1, :]
        
        # Generate predictions from each head
        predicted_price = self.price_head(last_encoding)
        
        trading_decisions = self.decision_head(last_encoding)
        trading_decisions = trading_decisions.view(-1, self.prediction_horizon, 3)
        
        trade_quantities = self.quantity_head(last_encoding)
        
        return {
            'predicted_price': predicted_price,
            'trading_decisions': trading_decisions,
            'trade_quantities': trade_quantities
        }


class TradingLoss(nn.Module):
    """
    Multi-task loss for trading transformer.
    
    Business-Focused Loss:
    - 20% price prediction accuracy (MAE) - less critical
    - 20% trading decision correctness (CrossEntropy) - guidance
    - 60% profitability reward (negative profit = higher loss) - PRIMARY GOAL
    
    Profit Calculation:
    - Household revenue: consumption * $0.27/kWh (constant revenue)
    - Market profit: (sell revenue - buy costs)
    - Penalty for excess battery storage (holding costs)
    - Higher profit = lower loss (reward mechanism)
    """
    
    def __init__(
        self,
        price_weight: float = 0.20,
        decision_weight: float = 0.20,
        profit_weight: float = 0.60,
        household_price_kwh: float = 0.27,
        excess_storage_penalty: float = 0.10
    ):
        """
        Initialize multi-task loss.
        
        Args:
            price_weight: Weight for price prediction loss (default 0.20)
            decision_weight: Weight for trading decision loss (default 0.20)
            profit_weight: Weight for profitability reward (default 0.60)
            household_price_kwh: Price charged to households (default 0.27)
            excess_storage_penalty: Penalty for holding excess power (default 0.10)
        """
        super().__init__()
        self.price_weight = price_weight
        self.decision_weight = decision_weight
        self.profit_weight = profit_weight
        self.household_price_kwh = household_price_kwh
        self.excess_storage_penalty = excess_storage_penalty
        
        self.mae = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate multi-task loss.
        
        Args:
            predictions: Dict with model predictions
            targets: Dict with target values:
                - 'price': (batch,) actual prices
                - 'decisions': (batch,) optimal decisions [0=Buy, 1=Hold, 2=Sell]
                - 'quantities': (batch,) optimal quantities
        
        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Individual loss components
        """
        # 1. Price prediction loss (MAE)
        # Only use first prediction for next interval
        price_loss = self.mae(
            predictions['predicted_price'][:, 0],
            targets['price']
        )
        
        # 2. Trading decision loss (CrossEntropy)
        # Only use first decision for next interval
        decision_loss = self.ce(
            predictions['trading_decisions'][:, 0, :],
            targets['decisions'].long()
        )
        
        # 3. Profitability reward (PRIMARY OBJECTIVE)
        # Calculate profit from predicted decisions and quantities
        pred_decisions = torch.argmax(predictions['trading_decisions'][:, 0, :], dim=1)
        pred_quantities = predictions['trade_quantities'][:, 0]
        actual_prices = targets['price']  # Market prices
        
        # Get actual consumption for household revenue calculation
        actual_consumption = targets.get('consumption', torch.zeros_like(actual_prices))
        
        # Profit calculation:
        # Household revenue: consumption * $0.27/kWh (constant)
        household_revenue = actual_consumption * self.household_price_kwh
        
        # Market transactions:
        # Buy (0): cost = -quantity * market_price
        # Hold (1): zero profit/cost
        # Sell (2): revenue = quantity * market_price
        market_profit = torch.zeros_like(actual_prices)
        
        buy_mask = (pred_decisions == 0)
        sell_mask = (pred_decisions == 2)
        
        market_profit[buy_mask] = -pred_quantities[buy_mask] * actual_prices[buy_mask]  # Cost
        market_profit[sell_mask] = pred_quantities[sell_mask] * actual_prices[sell_mask]  # Revenue
        
        # Total profit = household revenue + market profit
        total_profit = household_revenue + market_profit
        
        # Penalty for holding excess power (opportunity cost)
        # If holding when could sell at good price, penalize
        hold_mask = (pred_decisions == 1)
        good_sell_price_mask = (actual_prices > 0.040)  # $40/MWh threshold
        excess_hold_penalty = torch.zeros_like(actual_prices)
        excess_hold_penalty[hold_mask & good_sell_price_mask] = self.excess_storage_penalty
        
        # Convert profit to loss (higher profit = lower loss)
        # Normalize by household price to keep scale consistent
        profit_loss = -total_profit.mean() / (self.household_price_kwh + 1e-7) + excess_hold_penalty.mean()
        
        # 4. Combine losses
        total_loss = (
            self.price_weight * price_loss +
            self.decision_weight * decision_loss +
            self.profit_weight * profit_loss
        )
        
        loss_dict = {
            'price_mae': price_loss.item(),
            'decision_ce': decision_loss.item(),
            'profit_loss': profit_loss.item(),
            'avg_profit': total_profit.mean().item() if 'total_profit' in locals() else 0.0,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("="*70)
    print("TRADING TRANSFORMER TEST")
    print("="*70)
    
    # Test configuration
    batch_size = 4
    sequence_length = 48
    n_features = 20
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Number of features: {n_features}")
    
    # Create model
    print(f"\n1. Creating Trading Transformer...")
    model = TradingTransformer(
        n_features=n_features,
        d_model=512,
        n_heads=8,
        n_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        prediction_horizon=48
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
    criterion = TradingLoss(price_weight=0.20, decision_weight=0.20, profit_weight=0.60)
    
    # Create dummy targets
    targets = {
        'price': torch.randn(batch_size) * 0.1 + 0.20,
        'decisions': torch.randint(0, 3, (batch_size,)),
        'quantities': torch.rand(batch_size) * 5,
        'consumption': torch.randn(batch_size) * 0.5 + 2.0  # Household consumption
    }
    
    loss, loss_dict = criterion(output, targets)
    print(f"   ✓ Loss calculation successful")
    print(f"   Loss values:")
    for key, value in loss_dict.items():
        print(f"     - {key}: {value:.4f}")
    
    # Test gradient flow
    print(f"\n4. Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    x = torch.randn(batch_size, sequence_length, n_features)
    targets = {
        'price': torch.randn(batch_size) * 0.1 + 0.20,
        'decisions': torch.randint(0, 3, (batch_size,)),
        'quantities': torch.rand(batch_size) * 5
    }
    
    optimizer.zero_grad()
    output = model(x)
    loss, _ = criterion(output, targets)
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Gradient flow successful")
    print(f"   ✓ Model can be trained")
    
    # Test decision extraction
    print(f"\n5. Testing decision extraction...")
    with torch.no_grad():
        output = model(x)
        decisions = torch.argmax(output['trading_decisions'][:, 0, :], dim=1)
        print(f"   ✓ Decisions extracted: {decisions}")
        print(f"   ✓ Decision distribution:")
        print(f"     - Buy (0): {(decisions == 0).sum().item()}")
        print(f"     - Hold (1): {(decisions == 1).sum().item()}")
        print(f"     - Sell (2): {(decisions == 2).sum().item()}")
    
    print("\n" + "="*70)
    print("✅ TRADING TRANSFORMER TEST COMPLETE")
    print("="*70)
