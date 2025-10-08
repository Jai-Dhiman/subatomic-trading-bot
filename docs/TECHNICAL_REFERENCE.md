# Technical Reference - Energy Trading System

**Last Updated:** 2025-10-08  
**Status:** Production-ready algorithms and formulas

## Table of Contents

1. [Prediction Model Details](#prediction-model-details)
2. [Battery Management Algorithms](#battery-management-algorithms)
3. [Trading Logic & Decision Making](#trading-logic--decision-making)
4. [Federated Learning Implementation](#federated-learning-implementation)
5. [Market Mechanism](#market-mechanism)
6. [Metrics & Evaluation](#metrics--evaluation)

---

## Prediction Model Details

### Transformer Architecture

```python
Input Features (24 total features per 30-min interval):
  Appliance Sensors (7):
    - fridge_consumption, microwave_consumption, dishwasher_consumption
    - hvac_consumption, water_heater_consumption, ev_charger_consumption
    - other_consumption
  
  Battery Sensors (3):
    - battery_charge_kwh, battery_soc, battery_temperature
  
  Weather Features (4):
    - temperature, humidity, cloud_cover, solar_irradiance
  
  Temporal Features (6):
    - hour_sin, hour_cos, day_of_week_sin, day_of_week_cos
    - month_sin, month_cos
  
  Pricing Features (4):
    - current_price, avg_price_24h, peak_price_today, is_peak_hour

Model Structure:
  Input Projection: Linear(24, 512)
  Positional Encoding: Sinusoidal
  Transformer Encoder: 6 layers × (
    Multi-Head Attention (8 heads, d_model=512, d_k=64)
    + Feed-Forward Network (d_ff=2048)
    + Layer Normalization + Residual Connections
  )
  Multi-Task Output Heads:
    - Day Prediction: Linear(512, 48)    # Next 24 hours
    - Week Prediction: Linear(512, 336)  # Next 7 days
    - Month Prediction: Linear(512, 1440) # Next 30 days

Total Parameters: ~12 million

Output:
  predicted_consumption: {
    'day': [48 intervals],     # Next 24 hours (30-min intervals)
    'week': [336 intervals],   # Next 7 days (30-min intervals)
    'month': [1440 intervals]  # Next 30 days (30-min intervals)
  }
```

### Training Configuration

```python
Loss Function: Multi-Task MSE Loss
  total_loss = λ_day × MSE(day) + λ_week × MSE(week) + λ_month × MSE(month)
  where λ_day=1.0, λ_week=0.5, λ_month=0.25

Optimizer: AdamW (lr=0.001, weight_decay=0.01)
Batch Size: 32
Epochs: 50 (for initial training)
Train/Val Split: 80/20
Early Stopping: patience=10
Learning Rate Schedule: ReduceLROnPlateau (factor=0.5, patience=5)

Data Normalization:
  All features: StandardScaler (mean=0, std=1)
  Applied per feature independently
```

---

## Battery Management Algorithms

### Battery State Tracking

```python
class BatteryState:
    capacity_kwh: float = 13.5
    usable_capacity_kwh: float = 10.8  # 80% of total
    current_charge_kwh: float
    efficiency: float = 0.90
    max_charge_rate: float = 5.0  # kW
    max_discharge_rate: float = 5.0  # kW
    min_reserve_percent: float = 0.10

    def charge_of_charge(self) -> float:
        return self.current_charge_kwh / self.usable_capacity_kwh

    def available_capacity(self) -> float:
        """Available space for charging"""
        return self.usable_capacity_kwh - self.current_charge_kwh

    def available_energy(self) -> float:
        """Available energy for discharging (above reserve)"""
        reserve = self.usable_capacity_kwh * self.min_reserve_percent
        return max(0, self.current_charge_kwh - reserve)
```

### Charge/Discharge Operations

```python
def charge_battery(energy_kwh: float, interval_hours: float = 0.5) -> float:
    """
    Charge battery with efficiency losses
    
    Args:
        energy_kwh: Amount of energy to store
        interval_hours: Time interval (default 0.5 for 30-min)
    
    Returns:
        actual_energy_stored: Accounting for efficiency and rate limits
    """
    max_charge_this_interval = max_charge_rate * interval_hours
    energy_to_charge = min(
        energy_kwh,
        available_capacity(),
        max_charge_this_interval
    )
    
    actual_stored = energy_to_charge * efficiency
    current_charge_kwh += actual_stored
    
    return actual_stored

def discharge_battery(energy_kwh: float, interval_hours: float = 0.5) -> float:
    """
    Discharge battery with efficiency losses
    
    Args:
        energy_kwh: Amount of energy requested
        interval_hours: Time interval
    
    Returns:
        actual_energy_delivered: Accounting for efficiency and rate limits
    """
    max_discharge_this_interval = max_discharge_rate * interval_hours
    energy_available = available_energy()
    
    energy_to_discharge = min(
        energy_kwh,
        energy_available,
        max_discharge_this_interval
    )
    
    current_charge_kwh -= energy_to_discharge
    actual_delivered = energy_to_discharge * efficiency
    
    return actual_delivered
```

---

## Trading Logic & Decision Making

### Node Trading Algorithm

```python
def make_trading_decision(
    predicted_consumption: List[float],  # Next 6 intervals
    current_battery_state: BatteryState,
    market_price: float,
    pge_price: float
) -> TradingDecision:
    """
    Autonomous trading decision for a single node
    
    Returns:
        action: 'buy', 'sell', 'hold'
        quantity: kWh to trade
        max_price: Willing to pay (for buy)
        min_price: Willing to accept (for sell)
    """
    
    # Calculate net position over next 3 hours
    total_predicted = sum(predicted_consumption)
    battery_available = current_battery_state.available_energy()
    
    # Net position: negative means need energy, positive means excess
    net_position = battery_available - total_predicted
    
    # Decision logic
    if net_position < -1.0:  # Need more than 1 kWh
        # BUY DECISION
        quantity = min(
            abs(net_position) * 0.5,  # Buy 50% of deficit
            current_battery_state.available_capacity(),
            2.0  # Max trade size
        )
        
        # Only buy if market price is better than PG&E
        if market_price < pge_price * 0.95:  # 5% margin
            return TradingDecision(
                action='buy',
                quantity=quantity,
                max_price=pge_price * 0.95
            )
    
    elif net_position > 1.5:  # Have excess > 1.5 kWh
        # SELL DECISION
        quantity = min(
            net_position * 0.3,  # Sell 30% of excess
            2.0  # Max trade size
        )
        
        # Only sell if market price is better than cost basis
        cost_basis = pge_price * 0.7  # Assume we got it at off-peak
        if market_price > cost_basis * 1.05:  # 5% profit margin
            return TradingDecision(
                action='sell',
                quantity=quantity,
                min_price=cost_basis * 1.05
            )
    
    # Default: HOLD
    return TradingDecision(action='hold', quantity=0)
```

### Risk Management

```python
class TradingConstraints:
    max_sell_per_day: float = 10.0  # kWh (California regulation)
    max_single_trade: float = 2.0  # kWh
    min_profit_margin: float = 0.05  # 5%
    min_battery_reserve: float = 0.10  # 10%
    
    sold_today: float = 0.0
    
    def can_sell(self, quantity: float) -> bool:
        return (
            self.sold_today + quantity <= self.max_sell_per_day
            and quantity <= self.max_single_trade
        )
    
    def update_sold(self, quantity: float):
        self.sold_today += quantity
    
    def reset_daily(self):
        self.sold_today = 0.0
```

---

## Federated Learning Implementation

### FedAvg Algorithm

```python
def federated_averaging(node_models: List[NodeModel]) -> Dict:
    """
    FedAvg: Simple weighted averaging of model parameters
    
    Args:
        node_models: List of node model instances
    
    Returns:
        global_weights: Averaged model parameters
    """
    
    total_samples = sum(len(node.training_data) for node in node_models)
    global_weights = {}
    
    # Initialize with zeros
    first_model = node_models[0].model.state_dict()
    for key in first_model.keys():
        global_weights[key] = torch.zeros_like(first_model[key])
    
    # Weighted average based on local dataset size
    for node in node_models:
        local_weights = node.model.state_dict()
        weight = len(node.training_data) / total_samples
        
        for key in local_weights.keys():
            global_weights[key] += local_weights[key] * weight
    
    return global_weights

def federated_update_cycle(
    nodes: List[Node],
    central_model: CentralModel
):
    """
    Complete federated learning cycle
    
    1. Nodes train locally
    2. Central aggregates weights
    3. Central distributes updated model
    """
    
    # Step 1: Local training
    for node in nodes:
        node.train_local(epochs=5)
    
    # Step 2: Aggregate
    global_weights = federated_averaging(nodes)
    central_model.update_weights(global_weights)
    
    # Step 3: Distribute
    for node in nodes:
        node.update_model(global_weights)
    
    # Step 4: Evaluate convergence
    convergence_metric = calculate_weight_divergence(nodes)
    
    return convergence_metric
```

### Update Schedule

```python
Federated Update Frequency: Every 6 intervals (3 hours)
  - Interval 0: Initial state
  - Interval 6: First update
  - Interval 12: Second update
  - Interval 18: Third update
  - Interval 24: Fourth update
  - etc.

Local Training:
  - Use last 24 hours of actual consumption data
  - 5 epochs per update cycle
  - Quick fine-tuning, not full retraining
```

---

## Market Mechanism

### Price Signal Generation

```python
def generate_price_signal(
    aggregate_demand: float,  # Total demand from all nodes
    aggregate_supply: float,  # Total available supply
    time_of_day: int,  # 0-23
    base_price: float = 0.35  # Base $/kWh
) -> float:
    """
    Dynamic pricing based on supply/demand balance
    
    Returns:
        market_price: $/kWh
    """
    
    # Time-of-day factor
    if 16 <= time_of_day < 20:  # Peak hours (4-8pm)
        tod_multiplier = 1.4
    elif 0 <= time_of_day < 6:  # Super off-peak (12-6am)
        tod_multiplier = 0.8
    else:  # Off-peak
        tod_multiplier = 1.0
    
    # Supply/demand imbalance factor
    imbalance = (aggregate_demand - aggregate_supply) / aggregate_demand
    
    if imbalance > 0.2:  # High demand
        imbalance_multiplier = 1.0 + (imbalance * 0.5)
    elif imbalance < -0.2:  # High supply
        imbalance_multiplier = 1.0 + (imbalance * 0.3)
    else:  # Balanced
        imbalance_multiplier = 1.0
    
    # Calculate final price
    price = base_price * tod_multiplier * imbalance_multiplier
    
    # Apply bounds
    price = max(0.10, min(1.00, price))
    
    return price
```

### Trade Matching Algorithm

```python
def match_trades(
    buy_orders: List[BuyOrder],
    sell_orders: List[SellOrder],
    market_price: float,
    transmission_efficiency: float = 0.95
) -> List[Transaction]:
    """
    Instant matching at market price
    
    Args:
        buy_orders: List of buy requests
        sell_orders: List of sell offers
        market_price: Current market price
        transmission_efficiency: Energy loss factor (95%)
    
    Returns:
        transactions: List of executed trades
    """
    
    transactions = []
    
    # Filter orders willing to trade at market price
    valid_buyers = [b for b in buy_orders if b.max_price >= market_price]
    valid_sellers = [s for s in sell_orders if s.min_price <= market_price]
    
    # Sort by quantity for efficient matching
    valid_buyers.sort(key=lambda x: x.quantity, reverse=True)
    valid_sellers.sort(key=lambda x: x.quantity, reverse=True)
    
    buyer_idx = 0
    seller_idx = 0
    
    while buyer_idx < len(valid_buyers) and seller_idx < len(valid_sellers):
        buyer = valid_buyers[buyer_idx]
        seller = valid_sellers[seller_idx]
        
        # Match quantity
        trade_quantity = min(buyer.quantity, seller.quantity)
        
        # Account for transmission loss
        delivered_quantity = trade_quantity * transmission_efficiency
        loss = trade_quantity - delivered_quantity
        
        transaction = Transaction(
            timestamp=current_time(),
            buyer_id=buyer.node_id,
            seller_id=seller.node_id,
            energy_kwh=trade_quantity,
            delivered_kwh=delivered_quantity,
            loss_kwh=loss,
            price_per_kwh=market_price,
            total_cost=delivered_quantity * market_price
        )
        
        transactions.append(transaction)
        
        # Update remaining quantities
        buyer.quantity -= trade_quantity
        seller.quantity -= trade_quantity
        
        # Move to next order if fulfilled
        if buyer.quantity <= 0.01:
            buyer_idx += 1
        if seller.quantity <= 0.01:
            seller_idx += 1
    
    return transactions
```

---

## Metrics & Evaluation

### Cost Calculation

```python
def calculate_baseline_cost(
    consumption_data: List[float],  # kWh per 30-min interval
    timestamps: List[datetime]
) -> float:
    """
    Calculate cost using standard PG&E TOU-C rates
    """
    total_cost = 0.0
    
    for consumption, timestamp in zip(consumption_data, timestamps):
        hour = timestamp.hour
        month = timestamp.month
        
        # Determine season
        is_summer = 6 <= month <= 9
        
        # Determine rate period
        if 16 <= hour < 20:  # Peak (4-8pm)
            rate = 0.51 if is_summer else 0.40
        elif 0 <= hour < 6:  # Super off-peak (12-6am)
            rate = 0.28
        else:  # Off-peak
            rate = 0.35 if is_summer else 0.33
        
        total_cost += consumption * rate
    
    return total_cost

def calculate_optimized_cost(
    trades: List[Transaction],
    battery_operations: List[BatteryOperation],
    grid_purchases: List[GridPurchase]
) -> float:
    """
    Calculate actual cost with optimization
    """
    total_cost = 0.0
    
    # Cost from peer-to-peer trades
    for trade in trades:
        if trade.buyer_id == self.node_id:
            total_cost += trade.total_cost
        elif trade.seller_id == self.node_id:
            total_cost -= trade.total_cost  # Revenue
    
    # Cost from grid purchases
    for purchase in grid_purchases:
        total_cost += purchase.energy_kwh * purchase.rate
    
    return total_cost

def calculate_savings(baseline_cost: float, optimized_cost: float) -> Dict:
    """
    Calculate savings metrics
    """
    absolute_savings = baseline_cost - optimized_cost
    percent_savings = (absolute_savings / baseline_cost) * 100
    
    return {
        'baseline_cost': baseline_cost,
        'optimized_cost': optimized_cost,
        'absolute_savings': absolute_savings,
        'percent_savings': percent_savings
    }
```

### Prediction Accuracy Metrics

```python
def calculate_mape(actual: List[float], predicted: List[float]) -> float:
    """Mean Absolute Percentage Error"""
    n = len(actual)
    mape = sum(abs((a - p) / a) for a, p in zip(actual, predicted)) / n
    return mape * 100

def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
    """Root Mean Squared Error"""
    n = len(actual)
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
    return math.sqrt(mse)
```

### System-Wide Metrics

```python
def calculate_system_metrics(all_households: List[Household]) -> Dict:
    """
    Aggregate metrics across all households
    """
    total_energy_traded = sum(
        sum(t.energy_kwh for t in h.trades)
        for h in all_households
    ) / 2  # Divide by 2 to avoid double counting
    
    total_baseline_cost = sum(h.baseline_cost for h in all_households)
    total_optimized_cost = sum(h.optimized_cost for h in all_households)
    total_savings = total_baseline_cost - total_optimized_cost
    
    avg_battery_utilization = sum(
        h.battery.avg_charge_percent for h in all_households
    ) / len(all_households)
    
    peak_demand_baseline = max(
        sum(h.consumption[i] for h in all_households)
        for i in range(16*2, 20*2)  # 4-8pm intervals
    )
    
    peak_demand_optimized = max(
        sum(h.net_consumption[i] for h in all_households)
        for i in range(16*2, 20*2)
    )
    
    peak_reduction = (
        (peak_demand_baseline - peak_demand_optimized) / peak_demand_baseline
    ) * 100
    
    return {
        'total_energy_traded_kwh': total_energy_traded,
        'total_system_savings': total_savings,
        'avg_household_savings_percent': (total_savings / total_baseline_cost) * 100,
        'avg_battery_utilization_percent': avg_battery_utilization,
        'peak_demand_reduction_percent': peak_reduction,
        'num_trades': sum(len(h.trades) for h in all_households)
    }
```

---

## Configuration Parameters

```yaml
simulation:
  num_households: 10
  num_intervals: 48
  interval_duration_minutes: 30
  random_seed: 42

battery:
  capacity_kwh: 13.5
  usable_capacity_percent: 0.80
  efficiency: 0.90
  max_charge_rate_kw: 5.0
  max_discharge_rate_kw: 5.0
  min_reserve_percent: 0.10

trading:
  max_sell_per_day_kwh: 10.0
  max_single_trade_kwh: 2.0
  min_profit_margin: 0.05
  transmission_efficiency: 0.95
  price_bounds:
    min: 0.10
    max: 1.00

federated_learning:
  update_frequency_intervals: 6
  local_epochs: 5
  learning_rate: 0.001

model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  dropout: 0.1
  batch_size: 32
  sequence_length: 48
  prediction_horizons:
    day: 48
    week: 336
    month: 1440

pricing:
  base_price: 0.35
  pge_rates:
    summer_peak: 0.51
    summer_off_peak: 0.35
    winter_peak: 0.40
    winter_off_peak: 0.33
    super_off_peak: 0.28
```

---

**Last Updated:** 2025-10-07
