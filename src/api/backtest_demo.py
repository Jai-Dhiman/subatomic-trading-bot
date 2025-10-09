"""
Backtesting Demo Module for Energy Trading System.

Loads aligned historical data, runs it through trained transformer models,
and compares predictions against actual data for demonstration purposes.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

from src.api.inference import get_model_manager

logger = logging.getLogger(__name__)


class BacktestDemo:
    """
    Backtesting demonstration using real historical data.
    
    Process:
    1. Load historical appliance + pricing data for Oct 28 - Nov 4, 2024
    2. Prepare features for consumption transformer (17 features)
    3. Run consumption transformer to get energy predictions
    4. Prepare features for trading transformer (30 features)
    5. Run trading transformer to get price predictions and trading decisions
    6. Load actual pricing data for Nov 4-11, 2024
    7. Calculate accuracy metrics (MAE, RMSE, accuracy %)
    8. Format response for frontend visualization
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'demo'
        self.model_manager = get_model_manager()
        
        # Ensure models are loaded
        if self.model_manager.consumption_model is None:
            self.model_manager.load_models()
    
    def load_aligned_data(self) -> pd.DataFrame:
        """Load aligned historical data from CSV files."""
        logger.info("Loading aligned demo data...")
        
        combined_path = self.data_dir / 'combined_demo_data.csv'
        
        if not combined_path.exists():
            raise FileNotFoundError(
                f"Combined data file not found: {combined_path}. "
                "Please run scripts/align_house_data_for_demo.py first."
            )
        
        df = pd.read_csv(combined_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def prepare_consumption_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare 17 features for consumption transformer.
        
        Features:
        - 9 appliance features (normalized)
        - 4 temporal features (hour_sin, hour_cos, day_sin, day_cos)
        - 4 historical patterns (last_week, weekday_avg, rolling_avg, seasonal)
        
        Args:
            df: DataFrame with appliance data and timestamps
            
        Returns:
            Tensor of shape (n_samples, 17)
        """
        logger.info("Preparing consumption features (17 features)...")
        
        # Appliance features (9) - normalize using min-max scaling
        appliance_cols = ['ac', 'fridge', 'washing_machine', 'ev_charging', 'dishwasher',
                          'computers', 'stove', 'water_heater', 'misc']
        
        appliance_data = df[appliance_cols].values
        
        # Normalize to [0, 1] range
        appliance_min = appliance_data.min(axis=0, keepdims=True)
        appliance_max = appliance_data.max(axis=0, keepdims=True)
        appliance_range = appliance_max - appliance_min
        appliance_range[appliance_range == 0] = 1.0  # Avoid division by zero
        appliance_normalized = (appliance_data - appliance_min) / appliance_range
        
        # Temporal features (4)
        timestamps = pd.to_datetime(df['timestamp'])
        hour = timestamps.dt.hour + timestamps.dt.minute / 60.0  # Include minutes for 30-min resolution
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        day = timestamps.dt.dayofweek
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        temporal_features = np.column_stack([hour_sin, hour_cos, day_sin, day_cos])
        
        # Historical patterns (4)
        consumption = df['total_kwh'].values
        n_samples = len(consumption)
        historical_features = np.zeros((n_samples, 4))
        
        # 1. Last week same time (336 intervals ago = 7 days * 48 intervals/day)
        for i in range(n_samples):
            if i >= 336:
                historical_features[i, 0] = consumption[i - 336]
            else:
                historical_features[i, 0] = consumption[:i+1].mean() if i > 0 else consumption[0]
        
        # 2. Weekday average for this hour
        df_temp = df.copy()
        df_temp['dayofweek'] = timestamps.dt.dayofweek
        df_temp['hour'] = timestamps.dt.hour
        
        for i in range(n_samples):
            dow = df_temp.iloc[i]['dayofweek']
            hour_int = df_temp.iloc[i]['hour']
            
            mask = (df_temp['dayofweek'] == dow) & (df_temp['hour'] == hour_int)
            mask.iloc[i:] = False
            
            if mask.sum() > 0:
                historical_features[i, 1] = df_temp.loc[mask, 'total_kwh'].mean()
            else:
                historical_features[i, 1] = consumption[:i+1].mean() if i > 0 else consumption[0]
        
        # 3. Rolling 7-day average (336 intervals)
        for i in range(n_samples):
            start_idx = max(0, i - 336)
            historical_features[i, 2] = consumption[start_idx:i+1].mean()
        
        # 4. Seasonal factor (based on month)
        month = timestamps.dt.month
        seasonal = np.where(month.isin([6, 7, 8]), 1.0,  # Summer
                   np.where(month.isin([12, 1, 2]), 1.2,  # Winter
                   1.1))  # Spring/Fall
        historical_features[:, 3] = seasonal
        
        # Normalize historical features
        hist_mean = historical_features.mean(axis=0, keepdims=True)
        hist_std = historical_features.std(axis=0, keepdims=True)
        hist_std[hist_std == 0] = 1.0
        historical_normalized = (historical_features - hist_mean) / hist_std
        
        # Concatenate all features: 9 + 4 + 4 = 17
        features = np.hstack([
            appliance_normalized,
            temporal_features,
            historical_normalized
        ])
        
        logger.info(f"  ✓ Feature shape: {features.shape}")
        logger.info(f"  ✓ NaN check: {np.isnan(features).sum()} NaN values")
        
        # Replace any remaining NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def prepare_trading_features(
        self,
        consumption_predictions: torch.Tensor,
        df: pd.DataFrame,
        current_price: float = 0.05,
        battery_soc: float = 50.0
    ) -> torch.Tensor:
        """
        Prepare 30 features for trading transformer.
        
        Features:
        - 1 consumption prediction feature (predicted_consumption_day)
        - 5 pricing features (current_price + lags)
        - 4 battery state features
        - 4 temporal features
        - 16 additional contextual features
        
        Args:
            consumption_predictions: Predicted consumption from T1
            df: DataFrame with historical pricing
            current_price: Current electricity price
            battery_soc: Battery state of charge
            
        Returns:
            Tensor of shape (n_samples, 30)
        """
        logger.info("Preparing trading features (30 features)...")
        
        n_samples = len(df)
        features = np.zeros((n_samples, 30))
        
        # Feature index tracker
        idx = 0
        
        # 1. Consumption prediction from T1 (1 feature)
        # We have 48 predictions but need to broadcast to n_samples
        cons_pred = consumption_predictions.cpu().numpy()
        if cons_pred.shape[0] < n_samples:
            # Tile/repeat predictions to match n_samples
            num_reps = int(np.ceil(n_samples / cons_pred.shape[0]))
            cons_pred_tiled = np.tile(cons_pred, num_reps)[:n_samples]
            features[:, idx] = cons_pred_tiled
        else:
            features[:, idx] = cons_pred[:n_samples]
        idx += 1
        
        # 2. Pricing features (5 features: current + 4 lags)
        prices = df['price_kwh'].values
        features[:, idx] = prices  # current price
        idx += 1
        
        for lag in [1, 2, 3, 4]:
            lagged = np.roll(prices, lag)
            lagged[:lag] = prices[0]  # Fill initial values
            features[:, idx] = lagged
            idx += 1
        
        # 3. Battery state (4 features)
        features[:, idx] = battery_soc / 100.0  # Normalize to [0, 1]
        idx += 1
        features[:, idx] = (battery_soc / 100.0) * 13.5  # Available kWh (assuming 13.5 kWh battery)
        idx += 1
        features[:, idx] = (1.0 - battery_soc / 100.0) * 13.5  # Available capacity
        idx += 1
        features[:, idx] = 1.0  # Battery health (100%)
        idx += 1
        
        # 4. Temporal features (4 features)
        timestamps = pd.to_datetime(df['timestamp'])
        hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day = timestamps.dt.dayofweek
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        features[:, idx:idx+4] = np.column_stack([hour_sin, hour_cos, day_sin, day_cos])
        idx += 4
        
        # 5. Additional contextual features (16 features)
        # Price statistics
        features[:, idx] = np.convolve(prices, np.ones(48)/48, mode='same')  # 24h moving avg
        idx += 1
        features[:, idx] = np.convolve(prices, np.ones(336)/336, mode='same')  # 7day moving avg
        idx += 1
        features[:, idx] = prices / (features[:, idx-1] + 1e-8)  # Price relative to 7day avg
        idx += 1
        
        # Price volatility
        rolling_std = pd.Series(prices).rolling(window=48, min_periods=1).std().fillna(0).values
        features[:, idx] = rolling_std
        idx += 1
        
        # Consumption patterns
        consumption = df['total_kwh'].values
        features[:, idx] = consumption / (consumption.mean() + 1e-8)  # Relative to mean
        idx += 1
        features[:, idx] = np.convolve(consumption, np.ones(48)/48, mode='same')  # 24h avg consumption
        idx += 1
        
        # Time-of-day indicators
        is_peak_hour = ((hour >= 16) & (hour < 21)).astype(float)  # 4-9 PM
        is_offpeak_hour = ((hour >= 0) & (hour < 6)).astype(float)  # Midnight-6 AM
        features[:, idx] = is_peak_hour
        idx += 1
        features[:, idx] = is_offpeak_hour
        idx += 1
        
        # Weekend indicator
        is_weekend = (day >= 5).astype(float)
        features[:, idx] = is_weekend
        idx += 1
        
        # Fill remaining features with useful metrics
        while idx < 30:
            # Use price-related features
            if idx < 30:
                features[:, idx] = prices * consumption / (prices.mean() * consumption.mean() + 1e-8)
                idx += 1
            if idx < 30:
                features[:, idx] = np.gradient(prices)  # Price change rate
                idx += 1
            if idx < 30:
                features[:, idx] = np.gradient(consumption)  # Consumption change rate
                idx += 1
            if idx < 30:
                features[:, idx] = hour / 24.0  # Normalized hour
                idx += 1
            if idx < 30:
                features[:, idx] = day / 7.0  # Normalized day
                idx += 1
            if idx < 30:
                # Pad with zeros if we still need more
                features[:, idx:30] = 0.0
                break
        
        # Normalize all features
        feature_mean = features.mean(axis=0, keepdims=True)
        feature_std = features.std(axis=0, keepdims=True)
        feature_std[feature_std == 0] = 1.0
        features_normalized = (features - feature_mean) / feature_std
        
        # Replace NaN with 0
        features_normalized = np.nan_to_num(features_normalized, nan=0.0)
        
        logger.info(f"  ✓ Trading feature shape: {features_normalized.shape}")
        
        return torch.tensor(features_normalized, dtype=torch.float32)
    
    def calculate_accuracy_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate prediction accuracy metrics.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            
        Returns:
            Dictionary with MAE, RMSE, and accuracy percentage
        """
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # Accuracy as percentage (1 - normalized error)
        relative_error = np.abs(predictions - actuals) / (np.abs(actuals) + 1e-8)
        accuracy = (1.0 - np.mean(relative_error)) * 100
        accuracy = max(0, min(100, accuracy))  # Clip to [0, 100]
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'accuracy_percent': float(accuracy)
        }
    
    def run_backtest(self) -> Dict:
        """
        Run complete backtesting demonstration.
        
        Returns:
            Dictionary containing:
            - metadata: Date ranges and model info
            - predictions: Model predictions for Nov 4-11, 2024
            - actuals: Actual values for comparison
            - metrics: Accuracy metrics
            - simulation_data: Data formatted for frontend
        """
        logger.info("="*70)
        logger.info("RUNNING BACKTEST DEMONSTRATION")
        logger.info("="*70)
        
        # Step 1: Load data
        df_all = self.load_aligned_data()
        
        # Step 2: Define date ranges
        import pytz
        input_start = datetime(2024, 10, 28, 0, 0, 0, tzinfo=pytz.UTC)
        input_end = datetime(2024, 11, 4, 0, 0, 0, tzinfo=pytz.UTC)
        pred_start = datetime(2024, 11, 4, 0, 0, 0, tzinfo=pytz.UTC)
        pred_end = datetime(2024, 11, 11, 0, 0, 0, tzinfo=pytz.UTC)
        
        # Split data
        df_input = df_all[(df_all['timestamp'] >= input_start) & (df_all['timestamp'] < input_end)].copy()
        df_pred = df_all[(df_all['timestamp'] >= pred_start) & (df_all['timestamp'] < pred_end)].copy()
        
        logger.info(f"Input period: {len(df_input)} samples ({input_start.date()} to {input_end.date()})")
        logger.info(f"Prediction period: {len(df_pred)} samples ({pred_start.date()} to {pred_end.date()})")
        
        # Step 3: Prepare features and run consumption model
        logger.info("\nRunning Consumption Transformer...")
        features_input = self.prepare_consumption_features(df_input)
        
        # Take last 48 timesteps as input sequence
        if len(features_input) >= 48:
            input_sequence = features_input[-48:].unsqueeze(0)  # (1, 48, 17)
        else:
            # Pad if needed
            padding = torch.zeros((48 - len(features_input), 17))
            input_sequence = torch.cat([padding, features_input]).unsqueeze(0)
        
        input_sequence = input_sequence.to(self.model_manager.device)
        
        with torch.no_grad():
            consumption_output = self.model_manager.consumption_model(input_sequence)
            consumption_pred = consumption_output['consumption_day']  # (1, 48)
        
        logger.info(f"  ✓ Consumption predictions shape: {consumption_pred.shape}")
        
        # Step 4: Prepare trading features and run trading model
        logger.info("\nRunning Trading Transformer...")
        
        # For trading, we need features for the prediction period
        features_pred = self.prepare_consumption_features(df_pred)
        trading_features = self.prepare_trading_features(
            consumption_predictions=consumption_pred[0],  # Use predictions
            df=df_pred,
            current_price=df_pred['price_kwh'].iloc[0],
            battery_soc=50.0
        )
        
        # Take sequence for trading model (use last 48)
        if len(trading_features) >= 48:
            trading_sequence = trading_features[-48:].unsqueeze(0)  # (1, 48, 30)
        else:
            padding = torch.zeros((48 - len(trading_features), 30))
            trading_sequence = torch.cat([padding, trading_features]).unsqueeze(0)
        
        trading_sequence = trading_sequence.to(self.model_manager.device)
        
        with torch.no_grad():
            trading_output = self.model_manager.trading_model(trading_sequence)
            price_pred = trading_output['predicted_price']  # (1, 48)
            trading_decisions = trading_output['trading_decisions']  # (1, 48, 3)
            trade_quantities = trading_output['trade_quantities']  # (1, 48)
        
        logger.info(f"  ✓ Price predictions shape: {price_pred.shape}")
        logger.info(f"  ✓ Trading decisions shape: {trading_decisions.shape}")
        
        # Step 5: Calculate accuracy metrics
        logger.info("\nCalculating Accuracy Metrics...")
        
        # Get actual prices for comparison (first 48 intervals of prediction period)
        actual_prices = df_pred['price_kwh'].values[:48]
        predicted_prices = price_pred[0].cpu().numpy()
        
        metrics = self.calculate_accuracy_metrics(predicted_prices, actual_prices)
        
        logger.info(f"  MAE: ${metrics['mae']:.4f}")
        logger.info(f"  RMSE: ${metrics['rmse']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy_percent']:.1f}%")
        
        # Step 6: Format response for frontend
        logger.info("\nFormatting response for frontend...")
        
        response = {
            'metadata': {
                'input_start': input_start.isoformat(),
                'input_end': input_end.isoformat(),
                'prediction_start': pred_start.isoformat(),
                'prediction_end': pred_end.isoformat(),
                'model_info': self.model_manager.get_model_info()
            },
            'metrics': metrics,
            'predictions': {
                'timestamps': [pred_start + timedelta(minutes=30*i) for i in range(48)],
                'prices': predicted_prices.tolist(),
                'consumption': consumption_pred[0].cpu().numpy().tolist(),
                'trading_decisions': torch.argmax(trading_decisions[0], dim=-1).cpu().numpy().tolist(),
                'trade_quantities': trade_quantities[0].cpu().numpy().tolist()
            },
            'actuals': {
                'timestamps': df_pred['timestamp'].iloc[:48].tolist(),
                'prices': actual_prices.tolist(),
                'consumption': df_pred['total_kwh'].iloc[:48].tolist()
            }
        }
        
        logger.info("="*70)
        logger.info("BACKTEST COMPLETE!")
        logger.info("="*70)
        
        return response


def run_backtest_demo() -> Dict:
    """
    Convenience function to run backtest demonstration.
    
    Returns:
        Dictionary with predictions, actuals, and metrics
    """
    demo = BacktestDemo()
    return demo.run_backtest()
