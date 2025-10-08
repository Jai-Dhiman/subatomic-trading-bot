"""
Supabase database connector for energy trading system.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client


class SupabaseConnector:
    """Connector for Supabase database with read-only access."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize Supabase connector.
        
        Args:
            url: Supabase URL (defaults to SUPABASE_URL env var)
            key: Supabase key (defaults to SUPABASE_KEY env var)
        """
        load_dotenv()
        
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found. "
                "Please set SUPABASE_URL and SUPABASE_KEY in .env file"
            )
        
        self.client: Client = create_client(self.url, self.key)
        
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        response = self.client.table('households').select('*').limit(1).execute()
            
        return True
    
    def get_households(self) -> pd.DataFrame:
        """
        Fetch all households.
        
        Returns:
            DataFrame with household information
        """
        response = self.client.table('households').select('*').execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    
    def get_household_by_id(self, household_id: int) -> Dict:
        """
        Fetch a specific household.
        
        Args:
            household_id: Household ID
            
        Returns:
            Household data dictionary
        """
        response = (
            self.client
            .table('households')
            .select('*')
            .eq('id', household_id)
            .single()
            .execute()
        )
        
        return response.data if response.data else {}
    
    def get_consumption_data(
        self,
        household_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch consumption data for a household.
        
        Args:
            household_id: Household ID
            start_date: Start date (optional)
            end_date: End date (optional)
            limit: Max number of records (optional)
            
        Returns:
            DataFrame with consumption data
        """
        query = (
            self.client
            .table('consumption')
            .select('*')
            .eq('household_id', household_id)
            .order('timestamp', desc=False)
        )
        
        if start_date:
            query = query.gte('timestamp', start_date.isoformat())
        
        if end_date:
            query = query.lte('timestamp', end_date.isoformat())
        
        if limit:
            query = query.limit(limit)
        
        response = query.execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        return pd.DataFrame()
    
    def get_weather_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch weather data.
        
        Args:
            start_date: Start date (optional)
            end_date: End date (optional)
            limit: Max number of records (optional)
            
        Returns:
            DataFrame with weather data
        """
        query = (
            self.client
            .table('weather')
            .select('*')
            .order('timestamp', desc=False)
        )
        
        if start_date:
            query = query.gte('timestamp', start_date.isoformat())
        
        if end_date:
            query = query.lte('timestamp', end_date.isoformat())
        
        if limit:
            query = query.limit(limit)
        
        response = query.execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        return pd.DataFrame()
    
    def get_pricing_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch pricing data from CA electricity market.
        
        Filters for LMP_TYPE='LMP' (Locational Marginal Price) only.
        Uses 'Price MWH' column as the primary price field.
        
        Args:
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with pricing data (hourly intervals)
        """
        query = (
            self.client
            .table('cabuyingpricehistoryseptember2025')
            .select('*')
            .eq('LMP_TYPE', 'LMP')  # Filter for LMP type only
            .order('INTERVALSTARTTIME_GMT', desc=False)
        )
        
        if start_date:
            query = query.gte('INTERVALSTARTTIME_GMT', start_date.isoformat())
        
        if end_date:
            query = query.lte('INTERVALSTARTTIME_GMT', end_date.isoformat())
        
        response = query.execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            # Convert timestamp columns to datetime
            if 'INTERVALSTARTTIME_GMT' in df.columns:
                df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])
            if 'INTERVALENDTIME_GMT' in df.columns:
                df['INTERVALENDTIME_GMT'] = pd.to_datetime(df['INTERVALENDTIME_GMT'])
            
            # Add price_per_kwh column from 'Price MWH'
            if 'Price MWH' in df.columns:
                df['Price KWH'] = df['Price MWH'] / 1000.0  # Convert $/MWh to $/kWh
            
            return df
        
        return pd.DataFrame()
    
    def get_shared_pricing(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch shared pricing data (market prices for P2P trading).
        
        This queries the shared pricing_data table that all households use.
        In production, this would contain real-time P2P market prices.
        
        Args:
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with pricing data
            
        Raises:
            ValueError: If pricing_data table doesn't exist or has no data
        """
        try:
            query = (
                self.client
                .table('pricing_data')
                .select('*')
                .order('timestamp', desc=False)
            )
            
            if start_date:
                query = query.gte('timestamp', start_date.isoformat())
            
            if end_date:
                query = query.lte('timestamp', end_date.isoformat())
            
            response = query.execute()
            
            if not response.data:
                raise ValueError(
                    "No pricing data found in pricing_data table. "
                    f"Date range: {start_date} to {end_date}. "
                    "Ensure market pricing data is populated."
                )
            
            df = pd.DataFrame(response.data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            if "relation" in str(e) and "does not exist" in str(e):
                raise ValueError(
                    "Table 'pricing_data' does not exist in database. "
                    "Create this shared table for P2P market prices."
                )
            raise
    
    def query_house_data(
        self,
        house_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Query per-house data table with sensor readings.
        
        This is a convenience method that wraps consumption_parser.parse_house_data().
        Returns raw sensor data including appliances, weather, and battery sensors.
        
        Args:
            house_id: Household ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with all sensor readings from house_{id}_data table
            
        Raises:
            ValueError: If table doesn't exist or data is missing
            
        Note:
            For validated and processed data, use consumption_parser.parse_house_data()
            which includes appliance validation and time feature extraction.
        """
        from . import consumption_parser
        
        return consumption_parser.parse_house_data(
            house_id=house_id,
            connector=self,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_battery_config(self, house_id: int) -> Dict:
        """
        Fetch battery configuration for a household.
        
        Args:
            house_id: Household ID
            
        Returns:
            Dictionary with battery specs ready for BatteryManager
            
        Raises:
            ValueError: If config table doesn't exist or config not found
        """
        from . import consumption_parser
        
        return consumption_parser.load_battery_config(house_id, self)
    
    def get_grid_config(self, house_id: int) -> Dict:
        """
        Fetch grid connection configuration for a household.
        
        Args:
            house_id: Household ID
            
        Returns:
            Dictionary with grid constraints ready for GridConstraintsManager
            
        Raises:
            ValueError: If config table doesn't exist or config not found
        """
        from . import consumption_parser
        
        return consumption_parser.load_grid_config(house_id, self)
    
    def get_trading_config(self, house_id: int) -> Dict:
        """
        Fetch trading configuration for a household.
        
        Args:
            house_id: Household ID
            
        Returns:
            Dictionary with trading rules ready for TradingStrategy
            
        Raises:
            ValueError: If config table doesn't exist or config not found
        """
        from . import consumption_parser
        
        return consumption_parser.load_trading_config(house_id, self)
    
    def get_data_summary(self) -> Dict:
        """
        Get summary of available data.
        
        Returns:
            Dictionary with data counts and date ranges
        """
        summary = {}
        
        households_df = self.get_households()
        summary['num_households'] = len(households_df)
        summary['household_ids'] = households_df['id'].tolist() if 'id' in households_df.columns else []
        
        consumption_response = (
            self.client
            .table('consumption')
            .select('timestamp', count='exact')
            .execute()
        )
        summary['total_consumption_records'] = consumption_response.count if consumption_response.count else 0
        
        weather_response = (
            self.client
            .table('weather')
            .select('timestamp', count='exact')
            .execute()
        )
        summary['total_weather_records'] = weather_response.count if weather_response.count else 0
        
        first_consumption = (
            self.client
            .table('consumption')
            .select('timestamp')
            .order('timestamp', desc=False)
            .limit(1)
            .execute()
        )
        
        last_consumption = (
            self.client
            .table('consumption')
            .select('timestamp')
            .order('timestamp', desc=True)
            .limit(1)
            .execute()
        )
        
        if first_consumption.data and last_consumption.data:
            summary['consumption_date_range'] = {
                'start': first_consumption.data[0]['timestamp'],
                'end': last_consumption.data[0]['timestamp']
            }
        
        return summary


if __name__ == "__main__":
    print("Testing Supabase connection...")
    
    connector = SupabaseConnector()
        
    print("✓ Connection established")
    
    print("\nFetching data summary...")
    summary = connector.get_data_summary()
    
    print(f"\nData Summary:")
    print(f"  Households: {summary.get('num_households', 0)}")
    print(f"  Pricing records: {summary.get('total_pricing_records', 0):,}")
    
    if 'pricing_date_range' in summary:
        print(f"  Date range: {summary['pricing_date_range']['start']} to {summary['pricing_date_range']['end']}")
    
    print("\n✓ All tests passed!")
