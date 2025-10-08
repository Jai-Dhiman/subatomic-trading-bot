"""
Node model with LSTM for energy consumption prediction.

This module implements the LSTM-based prediction model for household energy consumption.
Data comes from per-house sensor tables (house_{id}_data) parsed by consumption_parser.

Current Architecture (v1):
    Input Features (5):
        - consumption_kwh: Historical consumption from appliance sensors
        - temperature: From local weather sensors
        - solar_irradiance: From local weather sensors
        - hour_of_day: Time feature (0-23, normalized to 0-1)
        - day_of_week: Time feature (0-6, normalized to 0-1)
    
    Model: LSTM (2 layers, 64 hidden units)
    Output: 6 intervals of predicted consumption (3 hours ahead)

Future Enhancements (v2):
    Could add battery state as input features:
        - battery_soc_percent: Current state of charge from BMS
        - battery_soh_percent: Battery health from BMS
        - battery_cycle_count: Cumulative cycles from BMS
    
    This would allow the model to learn correlations between battery state
    and consumption patterns (e.g., EV charging behavior, battery cycling).
    However, v1 keeps the model simple and focused on consumption prediction.

Data Flow:
    1. Supabase: house_{id}_data table (sensor readings)
    2. ConsumptionParser: parse_house_data() → validated DataFrame
    3. NodeModel: train() with historical data
    4. NodeModel: predict() for next intervals
    5. Profitability: generate_node_signals() for central model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


class LSTMPredictor(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


class NodeModel:
    """Node model for a household with LSTM prediction and training.
    
    This model uses real consumption data from per-house sensor tables.
    NO synthetic data or fallbacks - all training data comes from actual
    appliance sensors via consumption_parser.parse_house_data().
    
    Architecture:
        - LSTM with 2 layers, 64 hidden units (default)
        - 5 input features: consumption, temperature, solar, hour, day
        - Predicts 6 intervals (3 hours) of future consumption
        - MinMax scaling for consumption, temperature, solar
        - 80/20 train/validation split with early stopping
    
    Usage:
        >>> from src.data_integration import supabase_connector, consumption_parser
        >>> connector = supabase_connector.SupabaseConnector()
        >>> data = consumption_parser.parse_house_data(6, connector)
        >>> 
        >>> model = NodeModel(household_id=6, config={...})
        >>> model.train(data, epochs=50, verbose=True)
        >>> predictions = model.predict(data.tail(48))  # Last 48 intervals
    """
    
    def __init__(self, household_id: int, config: dict):
        """
        Initialize node model.
        
        Args:
            household_id: Household identifier
            config: Configuration dictionary with model parameters
        """
        self.household_id = household_id
        self.config = config
        
        self.input_size = 5
        self.hidden_size = config.get('lstm_hidden_size', 64)
        self.num_layers = config.get('lstm_num_layers', 2)
        self.output_size = config.get('prediction_horizon_intervals', 6)
        self.dropout = config.get('dropout', 0.2)
        self.sequence_length = config.get('input_sequence_length', 48)
        
        self.model = LSTMPredictor(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout
        )
        
        self.consumption_scaler = MinMaxScaler(feature_range=(0, 1))
        self.temperature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.solar_scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.training_data = None
        self.is_trained = False
        
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from household consumption history.
        
        Input data comes from consumption_parser.parse_house_data() which reads
        sensor data from house_{id}_data table and validates all columns.
        
        Args:
            df: DataFrame with consumption, temperature, solar, etc.
                Must have columns: consumption_kwh, temperature, solar_irradiance,
                                  hour_of_day, day_of_week
            
        Returns:
            (X, y) training arrays where:
                X: shape (samples, sequence_length=48, features=5)
                y: shape (samples, output_size=6) - next 6 intervals
        """
        self.training_data = df
        
        consumption = df['consumption_kwh'].values.reshape(-1, 1)
        temperature = df['temperature'].values.reshape(-1, 1)
        solar = df['solar_irradiance'].values.reshape(-1, 1)
        
        consumption_scaled = self.consumption_scaler.fit_transform(consumption)
        temperature_scaled = self.temperature_scaler.fit_transform(temperature)
        solar_scaled = self.solar_scaler.fit_transform(solar)
        
        hour_of_day = (df['hour_of_day'].values / 24.0).reshape(-1, 1)
        day_of_week = (df['day_of_week'].values / 7.0).reshape(-1, 1)
        
        features = np.concatenate([
            consumption_scaled,
            temperature_scaled,
            solar_scaled,
            hour_of_day,
            day_of_week
        ], axis=1)
        
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.output_size):
            X.append(features[i:i + self.sequence_length])
            y.append(consumption_scaled[i + self.sequence_length:i + self.sequence_length + self.output_size].flatten())
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 0.001, verbose: bool = False):
        """
        Train the LSTM model on real consumption data from sensors.
        
        Training data must come from consumption_parser.parse_house_data() which
        ensures all required columns are present and validated. NO synthetic data.
        
        Args:
            df: Training data from parse_house_data()
                Must have: consumption_kwh, temperature, solar_irradiance,
                          hour_of_day, day_of_week
            epochs: Number of training epochs (default 50)
            batch_size: Batch size for training (default 32)
            learning_rate: Adam optimizer learning rate (default 0.001)
            verbose: Print training progress (default False)
            
        Raises:
            ValueError: If training data is invalid or insufficient
            
        Note:
            Minimum required data: sequence_length (48) + output_size (6) + 100 samples
            Recommended: At least 30 days of historical data (1440 intervals)
        """
        if df is None or df.empty:
            raise ValueError(
                f"Cannot train: Training data is empty for household {self.household_id}. "
                f"Provide historical consumption data with required features."
            )
        
        required_columns = ['consumption_kwh', 'temperature', 'solar_irradiance', 
                           'hour_of_day', 'day_of_week']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Cannot train: Missing required columns for household {self.household_id}: {missing_columns}. "
                f"Required columns: {required_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        min_training_samples = self.sequence_length + self.output_size + 100
        if len(df) < min_training_samples:
            raise ValueError(
                f"Cannot train: Insufficient training data for household {self.household_id}. "
                f"Need at least {min_training_samples} samples for effective training, got {len(df)}. "
                f"Recommendation: Provide at least 30 days (1440 intervals) of historical data."
            )
        
        if verbose:
            print(f"Training household {self.household_id} with {len(df)} samples...")
        
        X, y = self.prepare_training_data(df)
        
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {np.mean(train_losses):.6f}, Val Loss = {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        
        self.is_trained = True
        
        if verbose:
            print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    
    def predict(self, recent_data: pd.DataFrame) -> np.ndarray:
        """
        Predict future consumption.
        
        Args:
            recent_data: Recent data (at least sequence_length intervals)
            
        Returns:
            Predicted consumption for next intervals (in kWh)
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If recent_data is invalid or insufficient
        """
        if not self.is_trained:
            raise RuntimeError(
                f"Cannot predict: Model for household {self.household_id} is not trained. "
                f"Call train() with historical data before making predictions. "
                f"Model requires at least {self.sequence_length} historical intervals for training."
            )
        
        if recent_data is None or recent_data.empty:
            raise ValueError(
                f"Cannot predict: recent_data is empty for household {self.household_id}. "
                f"Provide at least {self.sequence_length} recent consumption intervals."
            )
        
        if len(recent_data) < self.sequence_length:
            raise ValueError(
                f"Cannot predict: Insufficient data for household {self.household_id}. "
                f"Need at least {self.sequence_length} intervals, got {len(recent_data)}. "
                f"Ensure consumption data covers the required lookback period."
            )
        
        required_columns = ['consumption_kwh', 'temperature', 'solar_irradiance', 
                           'hour_of_day', 'day_of_week']
        missing_columns = set(required_columns) - set(recent_data.columns)
        if missing_columns:
            raise ValueError(
                f"Cannot predict: Missing required columns for household {self.household_id}: {missing_columns}. "
                f"Required columns: {required_columns}"
            )
        
        consumption = recent_data['consumption_kwh'].values[-self.sequence_length:].reshape(-1, 1)
        temperature = recent_data['temperature'].values[-self.sequence_length:].reshape(-1, 1)
        solar = recent_data['solar_irradiance'].values[-self.sequence_length:].reshape(-1, 1)
        
        consumption_scaled = self.consumption_scaler.transform(consumption)
        temperature_scaled = self.temperature_scaler.transform(temperature)
        solar_scaled = self.solar_scaler.transform(solar)
        
        hour_of_day = (recent_data['hour_of_day'].values[-self.sequence_length:] / 24.0).reshape(-1, 1)
        day_of_week = (recent_data['day_of_week'].values[-self.sequence_length:] / 7.0).reshape(-1, 1)
        
        features = np.concatenate([
            consumption_scaled,
            temperature_scaled,
            solar_scaled,
            hour_of_day,
            day_of_week
        ], axis=1)
        
        X = torch.FloatTensor(features).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            prediction_scaled = self.model(X).numpy().flatten()
        
        prediction = self.consumption_scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).flatten()
        
        prediction = np.maximum(prediction, 0.1)
        
        return prediction
    
    def get_model_weights(self) -> dict:
        """Get model weights for federated learning."""
        return self.model.state_dict()
    
    def update_model_weights(self, weights: dict):
        """Update model weights from federated aggregation."""
        self.model.load_state_dict(weights)
    
    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'consumption_scaler': self.consumption_scaler,
            'temperature_scaler': self.temperature_scaler,
            'solar_scaler': self.solar_scaler,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.consumption_scaler = checkpoint['consumption_scaler']
        self.temperature_scaler = checkpoint['temperature_scaler']
        self.solar_scaler = checkpoint['solar_scaler']
        self.is_trained = True


if __name__ == "__main__":
    config = {
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'prediction_horizon_intervals': 6,
        'input_sequence_length': 48
    }
    
    model = NodeModel(household_id=1, config=config)
    print(f"Node model created for household {model.household_id}")
    print(f"Input size: {model.input_size}, Output size: {model.output_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters())}")
