"""
Trading Transformer Model - V2 with Improved Loss Function.

Key improvements:
1. Focuses purely on market profit (excludes constant household revenue)
2. Better gradient scaling for profit loss
3. Class-balanced CrossEntropy loss
4. Removed unnecessary normalization
5. Clearer separation of training objective vs business metrics
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


class TradingLossV2(nn.Module):
    """
    IMPROVED Multi-task loss for trading transformer - V2.
    
    Business Model:
    - Household pays $0.27/kWh (CONSTANT revenue - not included in loss)
    - Company buys energy from market at variable prices (COST)
    - Company sells excess battery to market at variable prices (REVENUE)
    
    Loss Components:
    - 10% price prediction accuracy (MAE) - helps with decision timing
    - 10% trading decision correctness (Balanced CrossEntropy) - weak supervision
    - 80% MARKET PROFIT maximization (PRIMARY GOAL)
    
    Market Profit Formula (what the model learns to maximize):
        market_profit = sell_revenue - buy_costs
        sell_revenue = sum(sell_quantity * market_price)
        buy_costs = sum(buy_quantity * market_price)
    
    Business Profit (calculated AFTER training for reporting):
        business_profit = household_revenue + market_profit
        household_revenue = consumption * $0.27/kWh (constant)
    
    Key Improvements:
    1. Removed household revenue from loss (it's constant)
    2. Focus purely on maximizing market profit
    3. Increased profit weight to 80%
    4. Removed normalization that weakened gradients
    5. Added class balancing to CrossEntropy
    6. Better profit scaling for gradient flow
    """
    
    def __init__(
        self,
        price_weight: float = 0.10,
        decision_weight: float = 0.10,
        profit_weight: float = 0.80,
        household_price_kwh: float = 0.27,  # For business metrics only
        profit_scale: float = 100.0,  # Scale profit to match other losses
        class_weights: list = None  # [buy_weight, hold_weight, sell_weight]
    ):
        """
        Initialize improved multi-task loss.
        
        Args:
            price_weight: Weight for price prediction loss (default 0.10)
            decision_weight: Weight for trading decision loss (default 0.10)
            profit_weight: Weight for market profitability (default 0.80)
            household_price_kwh: Price charged to households (for metrics, not training)
            profit_scale: Multiplier to scale profit loss (default 100.0)
            class_weights: Optional class balancing weights [buy, hold, sell]
        """
        super().__init__()
        self.price_weight = price_weight
        self.decision_weight = decision_weight
        self.profit_weight = profit_weight
        self.household_price_kwh = household_price_kwh
        self.profit_scale = profit_scale
        
        self.mae = nn.L1Loss()
        
        # Class-balanced CrossEntropy if weights provided
        if class_weights is not None:
            class_weights_tensor = torch.FloatTensor(class_weights)
            self.ce = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate improved multi-task loss.
        
        Args:
            predictions: Dict with model predictions:
                - 'predicted_price': (batch, horizon) predicted prices
                - 'trading_decisions': (batch, horizon, 3) decision logits
                - 'trade_quantities': (batch, horizon) trade quantities
            targets: Dict with target values:
                - 'price': (batch,) actual prices for next interval
                - 'decisions': (batch,) optimal decisions [0=Buy, 1=Hold, 2=Sell]
                - 'quantities': (batch,) optimal quantities
                - 'consumption': (batch,) actual consumption (for business metrics)
        
        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Individual loss components and metrics
        """
        # 1. Price prediction loss (MAE) - 10%
        # Only use first prediction for next interval
        price_loss = self.mae(
            predictions['predicted_price'][:, 0],
            targets['price']
        )
        
        # 2. Trading decision loss (CrossEntropy) - 10%
        # Only use first decision for next interval
        decision_loss = self.ce(
            predictions['trading_decisions'][:, 0, :],
            targets['decisions'].long()
        )
        
        # 3. MARKET PROFIT MAXIMIZATION - 80% (PRIMARY OBJECTIVE)
        # Calculate profit ONLY from market transactions
        pred_decisions = torch.argmax(predictions['trading_decisions'][:, 0, :], dim=1)
        pred_quantities = predictions['trade_quantities'][:, 0]
        market_prices = targets['price']  # $/kWh
        
        # Market profit calculation (excludes constant household revenue):
        # Buy (0): cost = -quantity * market_price (negative profit)
        # Hold (1): zero market profit/cost
        # Sell (2): revenue = quantity * market_price (positive profit)
        market_profit = torch.zeros_like(market_prices)
        
        buy_mask = (pred_decisions == 0)
        sell_mask = (pred_decisions == 2)
        
        # Market transactions
        market_profit[buy_mask] = -pred_quantities[buy_mask] * market_prices[buy_mask]  # Cost
        market_profit[sell_mask] = pred_quantities[sell_mask] * market_prices[sell_mask]  # Revenue
        
        # Convert profit to loss (maximize profit = minimize negative profit)
        # Scale profit to be comparable to other loss terms
        profit_loss = -market_profit.mean() * self.profit_scale
        
        # 4. Combine losses with weights
        total_loss = (
            self.price_weight * price_loss +
            self.decision_weight * decision_loss +
            self.profit_weight * profit_loss
        )
        
        # 5. Calculate business metrics (for monitoring, not training)
        # These include the constant household revenue for reporting
        actual_consumption = targets.get('consumption', torch.zeros_like(market_prices))
        household_revenue = actual_consumption * self.household_price_kwh
        business_profit = household_revenue + market_profit
        
        # Build loss dictionary
        loss_dict = {
            'price_mae': price_loss.item(),
            'decision_ce': decision_loss.item(),
            'profit_loss': profit_loss.item(),
            'market_profit': market_profit.mean().item(),  # Training objective
            'business_profit': business_profit.mean().item(),  # Business metric
            'household_revenue': household_revenue.mean().item(),  # Reference
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


def calculate_class_weights(decisions: np.ndarray) -> list:
    """
    Calculate class weights for balanced training.
    
    Args:
        decisions: Array of decision labels [0=Buy, 1=Hold, 2=Sell]
    
    Returns:
        List of weights [buy_weight, hold_weight, sell_weight]
    """
    unique, counts = np.unique(decisions, return_counts=True)
    total = len(decisions)
    
    # Inverse frequency weights
    weights = np.zeros(3)
    for decision, count in zip(unique, counts):
        weights[int(decision)] = total / (3 * count)  # Normalized
    
    return weights.tolist()


if __name__ == "__main__":
    print("="*70)
    print("TRADING TRANSFORMER V2 TEST")
    print("="*70)
    
    # Test configuration
    batch_size = 4
    sequence_length = 48
    n_features = 30
    
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
    print(f"   Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB (float32)")
    
    # Create dummy input
    print(f"\n2. Testing forward pass...")
    x = torch.randn(batch_size, sequence_length, n_features)
    print(f"   Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"   Forward pass successful")
    print(f"   Output shapes:")
    for key, value in output.items():
        print(f"     - {key}: {value.shape}")
    
    # Test loss function V2
    print(f"\n3. Testing improved loss function (V2)...")
    
    # Simulate imbalanced training data (30% buy, 54% hold, 16% sell)
    decisions = np.random.choice([0, 1, 2], size=1000, p=[0.30, 0.54, 0.16])
    class_weights = calculate_class_weights(decisions)
    print(f"   Class distribution: Buy=30%, Hold=54%, Sell=16%")
    print(f"   Calculated class weights: {[f'{w:.2f}' for w in class_weights]}")
    
    # Create loss function with class balancing
    criterion = TradingLossV2(
        price_weight=0.10,
        decision_weight=0.10,
        profit_weight=0.80,
        class_weights=class_weights
    )
    
    # Create dummy targets
    targets = {
        'price': torch.rand(batch_size) * 0.05 + 0.02,  # $0.02-0.07/kWh
        'decisions': torch.randint(0, 3, (batch_size,)),
        'quantities': torch.rand(batch_size) * 5 + 1,  # 1-6 kWh
        'consumption': torch.rand(batch_size) * 2 + 1  # 1-3 kWh/30min
    }
    
    loss, loss_dict = criterion(output, targets)
    print(f"   Loss calculation successful")
    print(f"\n   Loss Components:")
    for key, value in loss_dict.items():
        print(f"     - {key}: {value:.4f}")
    
    # Compare old vs new loss approach
    print(f"\n4. Key improvements in V2:")
    print(f"   - Removed household revenue from training loss")
    print(f"   - Profit weight increased: 60% -> 80%")
    print(f"   - Price weight decreased: 20% -> 10%")
    print(f"   - Decision weight decreased: 20% -> 10%")
    print(f"   - Added class balancing for imbalanced data")
    print(f"   - Market profit scaled by {criterion.profit_scale}x for better gradients")
    print(f"   - Household revenue tracked separately for business metrics")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
