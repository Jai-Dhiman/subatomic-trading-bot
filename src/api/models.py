"""
Pydantic models for API request/response matching TypeScript interfaces.

These models exactly match the TypeScript interfaces from the frontend
at: project-nucleus/src/types/vpp.ts
"""

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TradeSignal(str, Enum):
    """Trading signal enum matching TypeScript 'BUY' | 'SELL' | 'HOLD'."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RequestType(str, Enum):
    """House request type matching TypeScript 'BUY' | 'SELL'."""
    BUY = "BUY"
    SELL = "SELL"


class RequestStatus(str, Enum):
    """Request status matching TypeScript."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"


class PriceDataPoint(BaseModel):
    """Price data point model."""
    timestamp: str
    price_cents_kwh: float
    hour_of_day: float


class HouseRequest(BaseModel):
    """Individual house energy request."""
    house_id: str
    house_address: Optional[str] = None
    request_type: RequestType
    requested_kwh: float
    consumed_kwh: Optional[float] = None
    price_cents_kwh: float
    status: RequestStatus
    filled_kwh: Optional[float] = None
    execution_price_cents_kwh: Optional[float] = None


class EnergyTrade(BaseModel):
    """30-minute energy trading interval with predictions and execution."""
    timestamp: str
    interval_start: str
    interval_end: str
    signal: TradeSignal
    trade_quantity_kwh: float
    exact_price_cents_kwh: float
    trade_cost_usd: float
    day_ahead_predicted_kwh: float
    day_ahead_price_cents_kwh: float
    prediction_accuracy_pct: Optional[float] = None
    market_clearing_price_cents_kwh: Optional[float] = None
    profit_loss_usd: float
    house_requests: List[HouseRequest]
    total_house_requests_kwh: float
    total_house_consumption_kwh: float
    consumption_power_kw: float
    is_live: Optional[bool] = False
    last_updated: Optional[str] = None
    execution_timestamp: Optional[str] = None


class EnergyTransaction(BaseModel):
    """30-minute energy transaction with grid."""
    timestamp: str
    interval_start: str
    interval_end: str
    interval_number: int
    mwh_purchased: float
    mwh_sold: float
    net_mwh: float
    price_cents_kwh: float
    purchase_cost_usd: float
    sale_revenue_usd: float
    net_cost_usd: float


class VPPParameters(BaseModel):
    """Virtual Power Plant parameters."""
    fleet_size: int
    battery_capacity_kwh: float
    total_capacity_mwh: float
    import_power_mw: float
    export_power_mw: float
    interval_minutes: int


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    consumption_features: List[List[float]] = Field(
        description="Consumption features: (batch, 48, 17)"
    )
    current_price: float = Field(description="Current price in cents/kWh")
    battery_soc_percent: float = Field(default=50.0, description="Battery state of charge (%)")
    start_timestamp: Optional[str] = Field(
        default=None, 
        description="Start timestamp for predictions (ISO format)"
    )


class PredictionResponse(BaseModel):
    """Response model containing predictions."""
    energy_trades: List[EnergyTrade]
    energy_transactions: List[EnergyTransaction]
    consumption_predictions: List[float] = Field(description="48 consumption predictions (kWh)")
    price_predictions: List[float] = Field(description="48 price predictions (cents/kWh)")
    timestamp: str = Field(description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    consumption_model_loaded: bool
    trading_model_loaded: bool
    device: str
    consumption_model_params: int
    trading_model_params: int


class ModelInfo(BaseModel):
    """Model architecture information."""
    consumption_transformer: dict
    trading_transformer: dict
