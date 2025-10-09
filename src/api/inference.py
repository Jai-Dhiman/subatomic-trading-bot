"""
Model inference engine with sequential execution pipeline.

CRITICAL: Trading model MUST be called after consumption model, as it uses
consumption predictions as input features.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from src.models.consumption_transformer import ConsumptionTransformer
from src.models.trading_transformer_v2 import TradingTransformer

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton model manager for loading and managing transformer models.
    
    Ensures models are loaded once and reused across requests.
    """
    
    _instance: Optional['ModelManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelManager._initialized:
            self.consumption_model: Optional[ConsumptionTransformer] = None
            self.trading_model: Optional[TradingTransformer] = None
            self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
            ModelManager._initialized = True
            logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_models(self) -> None:
        """Load both transformer models from checkpoints."""
        if self.consumption_model is not None and self.trading_model is not None:
            logger.info("Models already loaded, skipping...")
            return
        
        logger.info("Loading models from checkpoints...")
        
        consumption_checkpoint = self.checkpoint_dir / 'consumption_transformer_best.pt'
        trading_checkpoint = self.checkpoint_dir / 'trading_transformer_supabase_best.pt'
        
        if not consumption_checkpoint.exists():
            raise FileNotFoundError(
                f"Consumption model not found: {consumption_checkpoint}"
            )
        if not trading_checkpoint.exists():
            raise FileNotFoundError(
                f"Trading model not found: {trading_checkpoint}"
            )
        
        self.consumption_model = ConsumptionTransformer(
            n_features=17,
            d_model=384,
            n_heads=6,
            n_layers=5,
            dim_feedforward=1536,
            dropout=0.1,
            horizons={'day': 48, 'week': 336}
        ).to(self.device)
        
        self.trading_model = TradingTransformer(
            n_features=30,
            d_model=512,
            n_heads=8,
            n_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            prediction_horizon=48
        ).to(self.device)
        
        consumption_state = torch.load(consumption_checkpoint, map_location=self.device)
        if 'model_state_dict' in consumption_state:
            self.consumption_model.load_state_dict(consumption_state['model_state_dict'])
        else:
            self.consumption_model.load_state_dict(consumption_state)
        
        trading_state = torch.load(trading_checkpoint, map_location=self.device)
        if 'model_state_dict' in trading_state:
            self.trading_model.load_state_dict(trading_state['model_state_dict'])
        else:
            self.trading_model.load_state_dict(trading_state)
        
        self.consumption_model.eval()
        self.trading_model.eval()
        
        consumption_params = sum(p.numel() for p in self.consumption_model.parameters())
        trading_params = sum(p.numel() for p in self.trading_model.parameters())
        
        logger.info(f"Consumption model loaded: {consumption_params:,} parameters")
        logger.info(f"Trading model loaded: {trading_params:,} parameters")
        logger.info("All models ready for inference")
    
    def get_model_info(self) -> Dict:
        """Get model architecture information."""
        if self.consumption_model is None or self.trading_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        consumption_params = sum(p.numel() for p in self.consumption_model.parameters())
        trading_params = sum(p.numel() for p in self.trading_model.parameters())
        
        return {
            'consumption_transformer': {
                'parameters': consumption_params,
                'n_features': 17,
                'd_model': 384,
                'n_heads': 6,
                'n_layers': 5,
                'dim_feedforward': 1536,
                'horizons': {'day': 48, 'week': 336}
            },
            'trading_transformer': {
                'parameters': trading_params,
                'n_features': 30,
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'dim_feedforward': 2048,
                'prediction_horizon': 48
            },
            'device': str(self.device)
        }
    
    def predict_sequential(
        self,
        consumption_features: torch.Tensor,
        current_price: float,
        battery_soc: float
    ) -> Dict[str, torch.Tensor]:
        """
        Run sequential prediction pipeline: consumption -> trading.
        
        CRITICAL: This function implements the sequential execution that is
        mandatory for correct predictions. Trading model REQUIRES consumption
        predictions as input.
        
        Args:
            consumption_features: (batch, 48, 17) consumption input features
            current_price: Current electricity price in cents/kWh
            battery_soc: Battery state of charge percentage (0-100)
        
        Returns:
            Dictionary containing:
                - consumption_day: (batch, 48) consumption predictions
                - consumption_week: (batch, 336) consumption predictions
                - predicted_price: (batch, 48) price predictions
                - trading_decisions_logits: (batch, 48, 3) decision logits
                - trading_decisions: (batch, 48) decoded decisions (0=BUY, 1=HOLD, 2=SELL)
                - trade_quantities: (batch, 48) trade quantities in kWh
        """
        if self.consumption_model is None or self.trading_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        batch_size = consumption_features.shape[0]
        
        with torch.no_grad():
            logger.info(f"Step 1: Running consumption model with input shape: {consumption_features.shape}")
            consumption_output = self.consumption_model(consumption_features)
            consumption_day = consumption_output['consumption_day']
            consumption_week = consumption_output.get('consumption_week', None)
            
            logger.info(f"Step 2: Building trading model input (sequential dependency)")
            trading_features = self._build_trading_features(
                consumption_predictions=consumption_day,
                consumption_features=consumption_features,
                current_price=current_price,
                battery_soc=battery_soc
            )
            
            logger.info(f"Step 3: Running trading model with input shape: {trading_features.shape}")
            trading_output = self.trading_model(trading_features)
            
            logger.info(f"Step 4: Post-processing trading decisions")
            decisions_logits = trading_output['trading_decisions']
            decisions = torch.argmax(decisions_logits, dim=-1)
            
            result = {
                'consumption_day': consumption_day,
                'consumption_week': consumption_week,
                'predicted_price': trading_output['predicted_price'],
                'trading_decisions_logits': decisions_logits,
                'trading_decisions': decisions,
                'trade_quantities': trading_output['trade_quantities']
            }
            
            logger.info(f"Sequential prediction complete")
            return result
    
    def _build_trading_features(
        self,
        consumption_predictions: torch.Tensor,
        consumption_features: torch.Tensor,
        current_price: float,
        battery_soc: float
    ) -> torch.Tensor:
        """
        Build trading model input features including consumption predictions.
        
        Trading model expects (batch, 48, 30) features:
        - Features 0-2: Consumption-related (predicted consumption, current, historical)
        - Features 3-8: Pricing features (current, lags, historical)
        - Features 9-12: Battery state (SoC, available energy, capacity, SoH)
        - Features 13-29: Temporal and additional features
        
        Args:
            consumption_predictions: (batch, 48) consumption predictions from T1
            consumption_features: (batch, 48, 17) original consumption features
            current_price: Current price in cents/kWh
            battery_soc: Battery SoC percentage
        
        Returns:
            (batch, 48, 30) trading features tensor
        """
        batch_size, seq_len, _ = consumption_features.shape
        trading_features = torch.zeros(batch_size, seq_len, 30, device=consumption_features.device)
        
        for t in range(seq_len):
            predicted_consumption = consumption_predictions[:, t]
            
            trading_features[:, t, 0] = predicted_consumption
            
            if t < consumption_features.shape[1]:
                current_consumption = consumption_features[:, t, :9].sum(dim=1)
                trading_features[:, t, 1] = current_consumption
            
            if t > 0:
                trading_features[:, t, 2] = consumption_predictions[:, t-1]
            
            trading_features[:, t, 3] = current_price
            
            for lag in range(1, 5):
                if t >= lag:
                    trading_features[:, t, 3 + lag] = current_price * (1.0 + np.random.randn() * 0.05)
            
            trading_features[:, t, 8] = current_price * (1.0 + np.random.randn() * 0.1)
            
            trading_features[:, t, 9] = battery_soc
            available_energy = 13.5 * (battery_soc / 100.0)
            trading_features[:, t, 10] = available_energy
            available_capacity = 13.5 * (1.0 - battery_soc / 100.0)
            trading_features[:, t, 11] = available_capacity
            trading_features[:, t, 12] = 100.0
            
            hour = t * 0.5
            trading_features[:, t, 13] = np.sin(2 * np.pi * hour / 24)
            trading_features[:, t, 14] = np.cos(2 * np.pi * hour / 24)
            
            day_of_week = 0
            trading_features[:, t, 15] = np.sin(2 * np.pi * day_of_week / 7)
            trading_features[:, t, 16] = np.cos(2 * np.pi * day_of_week / 7)
            
            for i in range(13):
                trading_features[:, t, 17 + i] = 0.0
        
        return trading_features


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return ModelManager()
