"""
API routes for energy trading predictions.
"""

import torch
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
import logging

from src.api.models import (
    PredictionRequest, PredictionResponse, HealthResponse, ModelInfo
)
from src.api.inference import get_model_manager
from src.api.utils import (
    create_energy_trades_from_predictions,
    create_energy_transactions_from_predictions,
    create_dummy_consumption_features
)
from src.api.backtest_demo import run_backtest_demo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify models are loaded and ready.
    """
    try:
        manager = get_model_manager()
        
        consumption_loaded = manager.consumption_model is not None
        trading_loaded = manager.trading_model is not None
        
        if not (consumption_loaded and trading_loaded):
            return HealthResponse(
                status="initializing",
                consumption_model_loaded=consumption_loaded,
                trading_model_loaded=trading_loaded,
                device=str(manager.device),
                consumption_model_params=0,
                trading_model_params=0
            )
        
        consumption_params = sum(p.numel() for p in manager.consumption_model.parameters())
        trading_params = sum(p.numel() for p in manager.trading_model.parameters())
        
        return HealthResponse(
            status="ready",
            consumption_model_loaded=True,
            trading_model_loaded=True,
            device=str(manager.device),
            consumption_model_params=consumption_params,
            trading_model_params=trading_params
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/models/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get detailed model architecture information.
    """
    try:
        manager = get_model_manager()
        info = manager.get_model_info()
        return ModelInfo(
            consumption_transformer=info['consumption_transformer'],
            trading_transformer=info['trading_transformer']
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Raw prediction endpoint.
    
    Runs sequential model inference: consumption model -> trading model.
    Returns complete 24-hour forecast with 48 intervals (30 minutes each).
    """
    try:
        logger.info("Received prediction request")
        manager = get_model_manager()
        
        if len(request.consumption_features) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="consumption_features cannot be empty"
            )
        
        consumption_features = torch.tensor(
            request.consumption_features,
            dtype=torch.float32
        ).unsqueeze(0) if len(request.consumption_features[0]) == 17 else torch.tensor(
            request.consumption_features,
            dtype=torch.float32
        )
        
        if consumption_features.shape != (1, 48, 17):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected shape (1, 48, 17), got {consumption_features.shape}"
            )
        
        consumption_features = consumption_features.to(manager.device)
        
        logger.info("Running sequential prediction pipeline...")
        predictions = manager.predict_sequential(
            consumption_features=consumption_features,
            current_price=request.current_price,
            battery_soc=request.battery_soc_percent
        )
        
        start_time = datetime.fromisoformat(request.start_timestamp) if request.start_timestamp else datetime.utcnow()
        
        logger.info("Converting predictions to frontend format...")
        energy_trades = create_energy_trades_from_predictions(
            predictions=predictions,
            start_time=start_time,
            current_price=request.current_price
        )
        
        energy_transactions = create_energy_transactions_from_predictions(
            predictions=predictions,
            start_time=start_time
        )
        
        consumption_predictions = predictions['consumption_day'][0].cpu().tolist()
        price_predictions = predictions['predicted_price'][0].cpu().tolist()
        
        return PredictionResponse(
            energy_trades=energy_trades,
            energy_transactions=energy_transactions,
            consumption_predictions=consumption_predictions,
            price_predictions=price_predictions,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/generate-trades", response_model=PredictionResponse)
async def generate_trades():
    """
    Main endpoint for frontend to generate complete 24-hour energy trading forecast.
    
    Uses dummy/sample data for now. In production, this would accept current
    market state and generate predictions based on real-time data.
    
    Returns 48 intervals (30 minutes each) of:
    - Energy trades (BUY/SELL/HOLD decisions)
    - Energy transactions (grid purchases/sales)
    - Consumption predictions
    - Price predictions
    """
    try:
        logger.info("Generating trades with sample data")
        manager = get_model_manager()
        
        consumption_features = create_dummy_consumption_features(batch_size=1)
        consumption_features = consumption_features.to(manager.device)
        
        current_price = 5.0
        battery_soc = 50.0
        
        logger.info("Running sequential prediction pipeline...")
        predictions = manager.predict_sequential(
            consumption_features=consumption_features,
            current_price=current_price,
            battery_soc=battery_soc
        )
        
        start_time = datetime.utcnow()
        
        logger.info("Converting predictions to frontend format...")
        energy_trades = create_energy_trades_from_predictions(
            predictions=predictions,
            start_time=start_time,
            current_price=current_price
        )
        
        energy_transactions = create_energy_transactions_from_predictions(
            predictions=predictions,
            start_time=start_time
        )
        
        consumption_predictions = predictions['consumption_day'][0].cpu().tolist()
        price_predictions = predictions['predicted_price'][0].cpu().tolist()
        
        logger.info(f"Generated {len(energy_trades)} energy trades")
        
        return PredictionResponse(
            energy_trades=energy_trades,
            energy_transactions=energy_transactions,
            consumption_predictions=consumption_predictions,
            price_predictions=price_predictions,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Trade generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trade generation failed: {str(e)}"
        )


@router.get("/backtest-demo")
async def backtest_demo():
    """
    Backtest demonstration endpoint.
    
    Loads historical data (Oct 28 - Nov 4, 2024), runs it through
    trained transformer models, and compares predictions against
    actual data (Nov 4-11, 2024) for demonstration purposes.
    
    Returns:
        Dictionary containing:
        - metadata: Date ranges, model info
        - predictions: Model predictions for price, consumption, trading
        - actuals: Actual historical values for comparison
        - metrics: Accuracy metrics (MAE, RMSE, accuracy %)
    """
    try:
        logger.info("Running backtest demonstration...")
        result = run_backtest_demo()
        logger.info("Backtest demonstration complete")
        return result
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Backtest demonstration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest demonstration failed: {str(e)}"
        )
