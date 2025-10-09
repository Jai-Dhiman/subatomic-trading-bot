"""
Data transformation utilities for converting model outputs to frontend format.
"""

import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Dict
import random

from src.api.models import (
    EnergyTrade, EnergyTransaction, HouseRequest,
    TradeSignal, RequestType, RequestStatus
)


def generate_interval_timestamps(start_time: datetime, num_intervals: int = 48) -> List[tuple]:
    """
    Generate 30-minute interval timestamps.
    
    Args:
        start_time: Starting timestamp
        num_intervals: Number of intervals (default 48 for 24 hours)
    
    Returns:
        List of (timestamp, interval_start, interval_end) tuples
    """
    intervals = []
    for i in range(num_intervals):
        interval_start = start_time + timedelta(minutes=30*i)
        interval_end = interval_start + timedelta(minutes=30)
        intervals.append((
            interval_start.isoformat(),
            interval_start.isoformat(),
            interval_end.isoformat()
        ))
    return intervals


def decision_to_signal(decision: int) -> TradeSignal:
    """
    Convert decision integer to TradeSignal enum.
    
    Args:
        decision: 0=BUY, 1=HOLD, 2=SELL
    
    Returns:
        TradeSignal enum
    """
    mapping = {
        0: TradeSignal.BUY,
        1: TradeSignal.HOLD,
        2: TradeSignal.SELL
    }
    return mapping.get(decision, TradeSignal.HOLD)


def generate_house_requests(
    signal: TradeSignal,
    total_kwh: float,
    price_cents_kwh: float,
    num_houses: int = 10
) -> List[HouseRequest]:
    """
    Generate synthetic house requests for an interval.
    
    Args:
        signal: Trading signal for this interval
        total_kwh: Total energy traded
        price_cents_kwh: Price per kWh
        num_houses: Number of houses to generate
    
    Returns:
        List of HouseRequest objects
    """
    requests = []
    
    if signal == TradeSignal.HOLD or total_kwh <= 0:
        return requests
    
    request_type = RequestType.BUY if signal == TradeSignal.BUY else RequestType.SELL
    
    remaining_kwh = total_kwh
    for i in range(num_houses):
        if remaining_kwh <= 0:
            break
        
        if i == num_houses - 1:
            request_kwh = remaining_kwh
        else:
            request_kwh = random.uniform(0.1, min(remaining_kwh, total_kwh / num_houses * 1.5))
        
        consumed_kwh = request_kwh * random.uniform(0.9, 1.1) if request_type == RequestType.BUY else None
        
        requests.append(HouseRequest(
            house_id=f"house_{i+1:04d}",
            house_address=f"{random.randint(100, 9999)} Main St",
            request_type=request_type,
            requested_kwh=round(request_kwh, 2),
            consumed_kwh=round(consumed_kwh, 2) if consumed_kwh else None,
            price_cents_kwh=round(price_cents_kwh, 2),
            status=RequestStatus.FILLED,
            filled_kwh=round(request_kwh * random.uniform(0.95, 1.0), 2),
            execution_price_cents_kwh=round(price_cents_kwh * random.uniform(0.98, 1.02), 2)
        ))
        
        remaining_kwh -= request_kwh
    
    return requests


def calculate_trade_cost(signal: TradeSignal, quantity_kwh: float, price_cents_kwh: float) -> float:
    """
    Calculate trade cost/revenue in USD.
    
    Args:
        signal: Trading signal
        quantity_kwh: Trade quantity in kWh
        price_cents_kwh: Price in cents/kWh
    
    Returns:
        Cost (negative for buy, positive for sell) in USD
    """
    price_usd_kwh = price_cents_kwh / 100.0
    
    if signal == TradeSignal.BUY:
        return -quantity_kwh * price_usd_kwh
    elif signal == TradeSignal.SELL:
        return quantity_kwh * price_usd_kwh
    else:
        return 0.0


def calculate_profit_loss(
    signal: TradeSignal,
    quantity_kwh: float,
    execution_price_cents_kwh: float,
    day_ahead_price_cents_kwh: float
) -> float:
    """
    Calculate profit/loss compared to day-ahead prediction.
    
    Args:
        signal: Trading signal
        quantity_kwh: Trade quantity
        execution_price_cents_kwh: Actual execution price
        day_ahead_price_cents_kwh: Predicted price
    
    Returns:
        Profit/loss in USD
    """
    price_diff = (execution_price_cents_kwh - day_ahead_price_cents_kwh) / 100.0
    
    if signal == TradeSignal.BUY:
        return -quantity_kwh * price_diff
    elif signal == TradeSignal.SELL:
        return quantity_kwh * price_diff
    else:
        return 0.0


def create_energy_trades_from_predictions(
    predictions: Dict[str, torch.Tensor],
    start_time: datetime,
    current_price: float
) -> List[EnergyTrade]:
    """
    Convert model predictions to EnergyTrade objects.
    
    Args:
        predictions: Dictionary from ModelManager.predict_sequential()
        start_time: Start timestamp for predictions
        current_price: Current price in cents/kWh
    
    Returns:
        List of 48 EnergyTrade objects
    """
    consumption_day = predictions['consumption_day'][0].cpu().numpy()
    predicted_prices = predictions['predicted_price'][0].cpu().numpy()
    decisions = predictions['trading_decisions'][0].cpu().numpy()
    quantities = predictions['trade_quantities'][0].cpu().numpy()
    
    intervals = generate_interval_timestamps(start_time, num_intervals=48)
    
    energy_trades = []
    
    for i in range(48):
        timestamp, interval_start, interval_end = intervals[i]
        
        signal = decision_to_signal(int(decisions[i]))
        quantity_kwh = float(quantities[i])
        exact_price = float(predicted_prices[i])
        consumption_kwh = float(consumption_day[i])
        
        day_ahead_price = exact_price * random.uniform(0.9, 1.1)
        
        trade_cost = calculate_trade_cost(signal, quantity_kwh, exact_price)
        profit_loss = calculate_profit_loss(signal, quantity_kwh, exact_price, day_ahead_price)
        
        house_requests = generate_house_requests(signal, quantity_kwh, exact_price)
        total_house_requests = sum(r.requested_kwh for r in house_requests)
        total_house_consumption = sum(
            r.consumed_kwh for r in house_requests 
            if r.consumed_kwh is not None
        )
        
        consumption_power_kw = consumption_kwh * 2
        
        market_clearing_price = exact_price * random.uniform(0.95, 1.05)
        
        prediction_accuracy = random.uniform(85.0, 98.0)
        
        energy_trades.append(EnergyTrade(
            timestamp=timestamp,
            interval_start=interval_start,
            interval_end=interval_end,
            signal=signal,
            trade_quantity_kwh=round(quantity_kwh, 2),
            exact_price_cents_kwh=round(exact_price, 2),
            trade_cost_usd=round(trade_cost, 2),
            day_ahead_predicted_kwh=round(consumption_kwh, 2),
            day_ahead_price_cents_kwh=round(day_ahead_price, 2),
            prediction_accuracy_pct=round(prediction_accuracy, 1),
            market_clearing_price_cents_kwh=round(market_clearing_price, 2),
            profit_loss_usd=round(profit_loss, 2),
            house_requests=house_requests,
            total_house_requests_kwh=round(total_house_requests, 2),
            total_house_consumption_kwh=round(total_house_consumption, 2),
            consumption_power_kw=round(consumption_power_kw, 2),
            is_live=False,
            last_updated=datetime.utcnow().isoformat(),
            execution_timestamp=None
        ))
    
    return energy_trades


def create_energy_transactions_from_predictions(
    predictions: Dict[str, torch.Tensor],
    start_time: datetime
) -> List[EnergyTransaction]:
    """
    Convert model predictions to EnergyTransaction objects.
    
    Args:
        predictions: Dictionary from ModelManager.predict_sequential()
        start_time: Start timestamp for predictions
    
    Returns:
        List of 48 EnergyTransaction objects
    """
    predicted_prices = predictions['predicted_price'][0].cpu().numpy()
    decisions = predictions['trading_decisions'][0].cpu().numpy()
    quantities = predictions['trade_quantities'][0].cpu().numpy()
    
    intervals = generate_interval_timestamps(start_time, num_intervals=48)
    
    energy_transactions = []
    
    for i in range(48):
        timestamp, interval_start, interval_end = intervals[i]
        
        signal = decision_to_signal(int(decisions[i]))
        quantity_kwh = float(quantities[i])
        price_cents_kwh = float(predicted_prices[i])
        
        mwh_purchased = 0.0
        mwh_sold = 0.0
        
        if signal == TradeSignal.BUY:
            mwh_purchased = quantity_kwh / 1000.0
        elif signal == TradeSignal.SELL:
            mwh_sold = quantity_kwh / 1000.0
        
        net_mwh = mwh_purchased - mwh_sold
        
        price_usd_kwh = price_cents_kwh / 100.0
        purchase_cost_usd = (mwh_purchased * 1000) * price_usd_kwh
        sale_revenue_usd = (mwh_sold * 1000) * price_usd_kwh
        net_cost_usd = purchase_cost_usd - sale_revenue_usd
        
        energy_transactions.append(EnergyTransaction(
            timestamp=timestamp,
            interval_start=interval_start,
            interval_end=interval_end,
            interval_number=i,
            mwh_purchased=round(mwh_purchased, 4),
            mwh_sold=round(mwh_sold, 4),
            net_mwh=round(net_mwh, 4),
            price_cents_kwh=round(price_cents_kwh, 2),
            purchase_cost_usd=round(purchase_cost_usd, 2),
            sale_revenue_usd=round(sale_revenue_usd, 2),
            net_cost_usd=round(net_cost_usd, 2)
        ))
    
    return energy_transactions


def create_dummy_consumption_features(batch_size: int = 1) -> torch.Tensor:
    """
    Create dummy consumption features for testing.
    
    Args:
        batch_size: Batch size
    
    Returns:
        (batch_size, 48, 17) tensor of dummy features
    """
    features = torch.randn(batch_size, 48, 17) * 0.5 + 0.5
    features = torch.clamp(features, 0.0, 2.0)
    return features
