"""
Data adapter to transform Supabase schema to simulation format.
"""

import pandas as pd
from typing import Optional
from datetime import datetime


class DataAdapter:
    """Adapts Supabase data schema to simulation format."""
    
    @staticmethod
    def adapt_consumption_data(raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt consumption data from Supabase schema to simulation format.
        
        Expected Supabase columns (adjust based on your actual schema):
        - timestamp
        - household_id  
        - consumption_kwh or energy_kwh or power_kw
        
        Required simulation columns:
        - timestamp
        - household_id
        - consumption_kwh
        - hour_of_day
        - day_of_week
        - is_weekend
        
        Args:
            raw_df: Raw DataFrame from Supabase
            
        Returns:
            Adapted DataFrame ready for simulation
        """
        if raw_df.empty:
            return pd.DataFrame()
        
        df = raw_df.copy()
        
        if 'timestamp' not in df.columns:
            raise ValueError("Missing 'timestamp' column in consumption data")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'consumption_kwh' not in df.columns:
            if 'energy_kwh' in df.columns:
                df['consumption_kwh'] = df['energy_kwh']
            elif 'power_kw' in df.columns:
                df['consumption_kwh'] = df['power_kw'] * 0.5
            else:
                raise ValueError(
                    "Missing energy column. Expected 'consumption_kwh', 'energy_kwh', or 'power_kw'"
                )
        
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'] >= 5
        
        return df
    
    @staticmethod
    def adapt_weather_data(raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt weather data from Supabase schema to simulation format.
        
        Expected Supabase columns (adjust based on your actual schema):
        - timestamp
        - temperature or temp_f or temp_c
        - solar_irradiance or solar or irradiance
        
        Required simulation columns:
        - timestamp
        - temperature (Fahrenheit)
        - solar_irradiance (W/m²)
        
        Args:
            raw_df: Raw DataFrame from Supabase
            
        Returns:
            Adapted DataFrame ready for simulation
        """
        if raw_df.empty:
            return pd.DataFrame()
        
        df = raw_df.copy()
        
        if 'timestamp' not in df.columns:
            raise ValueError("Missing 'timestamp' column in weather data")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'temperature' not in df.columns:
            if 'temp_f' in df.columns:
                df['temperature'] = df['temp_f']
            elif 'temp_c' in df.columns:
                df['temperature'] = df['temp_c'] * 9/5 + 32
            else:
                raise ValueError(
                    "Missing temperature column. Expected 'temperature', 'temp_f', or 'temp_c'"
                )
        
        if 'solar_irradiance' not in df.columns:
            if 'solar' in df.columns:
                df['solar_irradiance'] = df['solar']
            elif 'irradiance' in df.columns:
                df['solar_irradiance'] = df['irradiance']
            else:
                raise ValueError(
                    "Missing solar irradiance data. Weather data must include one of: "
                    "'solar_irradiance', 'solar', or 'irradiance' columns. "
                    f"Available columns: {df.columns.tolist()}. "
                    "Please ensure your Supabase weather table has solar irradiance data."
                )
        
        return df
    
    @staticmethod
    def adapt_household_consumption_with_weather(
        consumption_df: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge consumption data with weather data for a household.
        
        Args:
            consumption_df: Adapted consumption DataFrame
            weather_df: Adapted weather DataFrame
            
        Returns:
            Merged DataFrame with all required columns
        """
        if consumption_df.empty:
            return pd.DataFrame()
        
        if weather_df.empty:
            raise ValueError(
                "Cannot merge consumption with weather: Weather data is empty. "
                "Please ensure weather data is available in Supabase for the simulation period. "
                f"Consumption data date range: {consumption_df['timestamp'].min()} to {consumption_df['timestamp'].max()}"
            )
        
        consumption_df['timestamp_rounded'] = consumption_df['timestamp'].dt.floor('30min')
        weather_df['timestamp_rounded'] = weather_df['timestamp'].dt.floor('30min')
        
        merged_df = pd.merge(
            consumption_df,
            weather_df[['timestamp_rounded', 'temperature', 'solar_irradiance']],
            on='timestamp_rounded',
            how='left'
        )
        
        # Forward fill for short gaps (acceptable for up to 2 hours)
        merged_df['temperature'] = merged_df['temperature'].ffill(limit=4)
        merged_df['solar_irradiance'] = merged_df['solar_irradiance'].ffill(limit=4)
        
        # Check for remaining nulls and fail if found
        if merged_df['temperature'].isna().any():
            missing_count = merged_df['temperature'].isna().sum()
            missing_timestamps = merged_df[merged_df['temperature'].isna()]['timestamp'].tolist()[:5]
            raise ValueError(
                f"Missing temperature data for {missing_count} timestamps after merge and forward-fill. "
                f"First 5 missing timestamps: {missing_timestamps}. "
                "Weather data must cover all consumption timestamps. "
                "Check that weather data is complete in Supabase."
            )
        
        if merged_df['solar_irradiance'].isna().any():
            missing_count = merged_df['solar_irradiance'].isna().sum()
            missing_timestamps = merged_df[merged_df['solar_irradiance'].isna()]['timestamp'].tolist()[:5]
            raise ValueError(
                f"Missing solar irradiance data for {missing_count} timestamps after merge and forward-fill. "
                f"First 5 missing timestamps: {missing_timestamps}. "
                "Weather data must cover all consumption timestamps. "
                "Check that weather data is complete in Supabase."
            )
        
        merged_df.drop('timestamp_rounded', axis=1, inplace=True)
        
        return merged_df
    
    @staticmethod
    def validate_data_format(df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, raises ValueError if not
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
    
    @staticmethod
    def adapt_pricing_data(raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt California pricing data from Supabase schema to simulation format.
        
        Expected Supabase columns:
        - INTERVALSTARTTIME_GMT (timestamp)
        - Price KWH (numeric) - price per kWh
        
        Required simulation columns:
        - timestamp
        - price_per_kwh
        
        Args:
            raw_df: Raw DataFrame from cabuyingpricehistoryseptember2025 table
            
        Returns:
            Adapted DataFrame with pricing data
        """
        if raw_df.empty:
            return pd.DataFrame()
        
        df = raw_df.copy()
        
        if 'INTERVALSTARTTIME_GMT' not in df.columns:
            raise ValueError("Missing 'INTERVALSTARTTIME_GMT' column in pricing data")
        
        df['timestamp'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])
        
        if 'Price KWH' in df.columns:
            df['price_per_kwh'] = pd.to_numeric(df['Price KWH'], errors='coerce')
        elif 'Price MWH' in df.columns:
            df['price_per_kwh'] = pd.to_numeric(df['Price MWH'], errors='coerce') / 1000.0
        else:
            raise ValueError("Missing price column. Expected 'Price KWH' or 'Price MWH'")
        
        # Forward fill for short gaps (acceptable for up to 4 intervals = 2 hours)
        df['price_per_kwh'] = df['price_per_kwh'].ffill(limit=4)
        
        # Check for remaining nulls and fail if found
        if df['price_per_kwh'].isna().any():
            missing_count = df['price_per_kwh'].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            first_missing = df[df['price_per_kwh'].isna()]['timestamp'].iloc[0] if 'timestamp' in df.columns else 'unknown'
            raise ValueError(
                f"Missing pricing data for {missing_count} intervals ({missing_pct:.1f}%) after forward-fill. "
                f"First missing timestamp: {first_missing}. "
                "CA pricing data must be complete. "
                "Check Supabase table 'cabuyingpricehistoryseptember2025' for gaps in data."
            )
        
        df = df[['timestamp', 'price_per_kwh']].copy()
        
        return df
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> dict:
        """
        Get summary information about a DataFrame.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data info
        """
        if df.empty:
            return {'empty': True}
        
        info = {
            'num_records': len(df),
            'columns': df.columns.tolist(),
            'date_range': None,
            'null_counts': df.isnull().sum().to_dict()
        }
        
        if 'timestamp' in df.columns:
            info['date_range'] = {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        
        return info


if __name__ == "__main__":
    print("Data Adapter Test")
    print("="*50)
    
    test_consumption = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=48, freq='30min'),
        'household_id': [1] * 48,
        'energy_kwh': [0.5 + i*0.1 for i in range(48)]
    })
    
    print("\n1. Testing consumption data adaptation...")
    adapted_consumption = DataAdapter.adapt_consumption_data(test_consumption)
    print(f"   Columns: {adapted_consumption.columns.tolist()}")
    print(f"   Shape: {adapted_consumption.shape}")
    
    test_weather = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=48, freq='30min'),
        'temp_f': [70 + i for i in range(48)],
        'solar': [100 * i for i in range(48)]
    })
    
    print("\n2. Testing weather data adaptation...")
    adapted_weather = DataAdapter.adapt_weather_data(test_weather)
    print(f"   Columns: {adapted_weather.columns.tolist()}")
    print(f"   Shape: {adapted_weather.shape}")
    
    print("\n3. Testing merge...")
    merged = DataAdapter.adapt_household_consumption_with_weather(
        adapted_consumption, adapted_weather
    )
    print(f"   Columns: {merged.columns.tolist()}")
    print(f"   Shape: {merged.shape}")
    
    print("\n4. Getting data info...")
    info = DataAdapter.get_data_info(merged)
    print(f"   Records: {info['num_records']}")
    print(f"   Date range: {info['date_range']}")
    
    print("\n✓ All adapter tests passed!")
