"""
Main simulation runner for energy trading demo.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.config_loader import load_config
from src.data_integration.supabase_connector import SupabaseConnector
from src.data_integration.data_adapter import DataAdapter
from src.simulation.household_node import HouseholdNode
from src.models.central_model import CentralModel
from src.trading.market_mechanism import MarketMechanism, BuyOrder, SellOrder
from src.federated.federated_aggregator import federated_update_cycle


def load_real_data_from_supabase(config: dict):
    """
    Load real data from Supabase for simulation.
    
    Raises:
        ValueError: If required data tables are not populated
        ConnectionError: If Supabase connection fails
    """
    print("\n" + "="*60)
    print("LOADING REAL DATA FROM SUPABASE")
    print("="*60)
    
    try:
        connector = SupabaseConnector()
        print("✓ Connected to Supabase")
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Supabase: {e}. "
            "Check your .env file has SUPABASE_URL and SUPABASE_KEY set."
        )
    
    supabase_config = config.get('supabase', {})
    
    # Load pricing data
    print("\nFetching pricing data...")
    pricing_table = supabase_config.get('pricing_table', 'cabuyingpricehistoryseptember2025')
    raw_pricing = connector.get_pricing_data()
    
    if raw_pricing.empty:
        raise ValueError(
            f"No pricing data found in Supabase table '{pricing_table}'. "
            "Please populate the pricing table with CA electricity market data."
        )
    
    pricing_df = DataAdapter.adapt_pricing_data(raw_pricing)
    print(f"✓ Loaded {len(pricing_df)} pricing records")
    print(f"  Date range: {pricing_df['timestamp'].min()} to {pricing_df['timestamp'].max()}")
    print(f"  Price range: ${pricing_df['price_per_kwh'].min():.3f} - ${pricing_df['price_per_kwh'].max():.3f}/kWh")
    
    # Load household data
    print("\nFetching household consumption data...")
    households_table = supabase_config.get('households_table', 'households')
    consumption_table = supabase_config.get('consumption_table', 'consumption')
    
    households = connector.get_households()
    if households.empty:
        raise ValueError(
            f"No households found in Supabase table '{households_table}'. "
            "Please populate the households table. Expected schema: House # | ..."
        )
    
    print(f"✓ Found {len(households)} households")
    
    # For now, we'll raise an error for consumption data since it's not ready
    # TODO: Once consumption table is populated, load it here
    raise NotImplementedError(
        f"Household consumption data loading not yet implemented. "
        f"Please populate Supabase table '{consumption_table}' with historical consumption data. "
        f"Expected schema: House # | TimeStamp | Usage (json). "
        f"Usage JSON should contain: {{appliance1: kwh, appliance2: kwh, ...}}"
    )


def run_simulation(config: dict):
    """Run the energy trading simulation."""
    start_time = time.time()
    
    print("\n" + "="*60)
    print("FEDERATED ENERGY TRADING SYSTEM - SIMULATION")
    print("="*60)
    
    sim_config = config['simulation']
    num_households = sim_config['num_households']
    num_intervals = sim_config['num_intervals']
    start_date = datetime.strptime(sim_config['start_date'], "%Y-%m-%d")
    
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / "data" / "generated"
    
    # Check data source configuration
    data_source = config.get('data_source', {})
    if data_source.get('type') != 'supabase':
        raise ValueError(
            "Configuration error: data_source.type must be 'supabase' for production POC. "
            f"Current value: '{data_source.get('type')}'. "
            "Update config/config.yaml to use real data."
        )
    
    print("\nLoading household data from Supabase...")
    
    # This will raise appropriate errors if data isn't ready
    load_real_data_from_supabase(config)
    
    # TODO: Once load_real_data_from_supabase is complete, implement node loading
    raise NotImplementedError(
        "Household node initialization from Supabase data not yet complete. "
        "Next steps: "
        "1. Populate Supabase 'consumption' table with historical data. "
        "2. Implement consumption data parsing and transformation. "
        "3. Implement weather data integration (add later). "
        "4. Create household nodes with real data."
    )
    
    print(f"Loaded {len(nodes)} households")
    
    print("\nTraining initial models...")
    for i, node in enumerate(nodes, 1):
        print(f"  Training household {i}/{len(nodes)}...", end=" ")
        node.train_model(epochs=30, verbose=False)
        print("Done")
    
    print("\nInitializing market and central model...")
    central_model = CentralModel(config)
    market = MarketMechanism(config['trading'])
    
    print("\nStarting simulation...")
    print(f"  Duration: {num_intervals} intervals ({num_intervals/48:.1f} days)")
    print(f"  Federated updates: Every {config['federated_learning']['update_frequency_intervals']} intervals")
    
    simulation_start = start_date + timedelta(days=365)
    
    # TODO: Load real weather data from Supabase
    raise NotImplementedError(
        "Weather data loading from Supabase not yet implemented. "
        "Weather table should contain: timestamp, temperature (F), solar_irradiance (W/m²). "
        "This will be added later per the plan."
    )
    
    for interval in range(num_intervals):
        timestamp = simulation_start + timedelta(minutes=30*interval)
        
        if interval % 10 == 0:
            print(f"  Interval {interval}/{num_intervals} ({timestamp.strftime('%Y-%m-%d %H:%M')})")
        
        current_weather_idx = interval
        if current_weather_idx >= len(weather_sim):
            current_weather_idx = len(weather_sim) - 1
        
        temp = weather_sim.iloc[current_weather_idx]['temperature']
        solar = weather_sim.iloc[current_weather_idx]['solar_irradiance']
        
        pge_price = central_model.get_pge_rate(timestamp)
        
        buy_orders = []
        sell_orders = []
        aggregate_demand = 0.0
        aggregate_supply = 0.0
        
        for node in nodes:
            historical_start = max(0, len(node.consumption_data) - 48)
            recent_data = node.consumption_data.iloc[historical_start:historical_start + 48].copy()
            
            if len(recent_data) < 48:
                recent_data = node.consumption_data.iloc[-48:].copy()
            
            predicted = node.predict_consumption(recent_data)
            
            # Calculate preliminary supply/demand for this node
            battery_available = node.battery_manager.battery.available_energy()
            net_position = battery_available - predicted.sum()
            
            if net_position < -1.0:
                aggregate_demand += abs(net_position) * 0.5
            elif net_position > 1.5:
                aggregate_supply += net_position * 0.3
        
        # Calculate market price based on aggregate supply/demand
        market_price = central_model.generate_price_signal(
            aggregate_demand=aggregate_demand,
            aggregate_supply=aggregate_supply,
            timestamp=timestamp
        )
        
        # Now make trading decisions with actual market price
        for node in nodes:
            historical_start = max(0, len(node.consumption_data) - 48)
            recent_data = node.consumption_data.iloc[historical_start:historical_start + 48].copy()
            
            if len(recent_data) < 48:
                recent_data = node.consumption_data.iloc[-48:].copy()
            
            predicted = node.predict_consumption(recent_data)
            
            decision = node.make_trading_decision(predicted, market_price, pge_price)
            
            if decision.action == 'buy':
                buy_orders.append(BuyOrder(
                    node_id=node.household_id,
                    quantity=decision.quantity,
                    max_price=decision.max_price
                ))
                aggregate_demand += decision.quantity
                
            elif decision.action == 'sell':
                sell_orders.append(SellOrder(
                    node_id=node.household_id,
                    quantity=decision.quantity,
                    min_price=decision.min_price
                ))
                aggregate_supply += decision.quantity
        
        market_price = central_model.generate_price_signal(
            aggregate_demand=aggregate_demand,
            aggregate_supply=aggregate_supply,
            timestamp=timestamp
        )
        
        transactions = market.match_trades(
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            market_price=market_price,
            timestamp=timestamp,
            interval=interval
        )
        
        for transaction in transactions:
            for node in nodes:
                if node.household_id in [transaction.buyer_id, transaction.seller_id]:
                    node.execute_transaction(transaction)
        
        for node in nodes:
            hist_idx = len(node.consumption_data) - num_intervals + interval
            if hist_idx < 0 or hist_idx >= len(node.consumption_data):
                raise IndexError(
                    f"Consumption data index {hist_idx} out of bounds for household {node.household_id}. "
                    f"Available data: {len(node.consumption_data)} records. "
                    f"Simulation interval: {interval}/{num_intervals}. "
                    f"Check that training data covers the full simulation period."
                )
            actual_consumption = node.consumption_data.iloc[hist_idx]['consumption_kwh']
            
            node.consume_energy(actual_consumption, pge_price, market_price)
            
            decision_action = 'hold'
            for order in buy_orders:
                if order.node_id == node.household_id:
                    decision_action = 'buy'
            for order in sell_orders:
                if order.node_id == node.household_id:
                    decision_action = 'sell'
            
            recent_data_pred = node.consumption_data.iloc[-48:].copy()
            predicted_val = node.predict_consumption(recent_data_pred)[0]
            
            node.record_interval(
                interval=interval,
                timestamp=timestamp,
                predicted=predicted_val,
                actual=actual_consumption,
                market_price=market_price,
                pge_price=pge_price,
                action=decision_action
            )
        
        if interval > 0 and interval % config['federated_learning']['update_frequency_intervals'] == 0:
            print(f"\n  ** Federated learning update at interval {interval} **")
            federated_update_cycle(
                [node.model for node in nodes],
                epochs=config['federated_learning']['local_epochs'],
                verbose=True
            )
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    print("\nCollecting results...")
    results = []
    total_baseline = 0
    total_optimized = 0
    
    for node in nodes:
        node_results = node.get_results()
        results.append(node_results)
        total_baseline += node_results['costs']['baseline_pge_total']
        total_optimized += node_results['costs']['optimized_total']
        
        print(f"\nHousehold {node.household_id}:")
        print(f"  Baseline cost: ${node_results['costs']['baseline_pge_total']:.2f}")
        print(f"  Optimized cost: ${node_results['costs']['optimized_total']:.2f}")
        print(f"  Savings: ${node_results['costs']['savings']:.2f} ({node_results['costs']['savings_percent']:.1f}%)")
        print(f"  Trades: {node_results['num_trades']}")
    
    total_savings = total_baseline - total_optimized
    avg_savings_percent = (total_savings / total_baseline * 100) if total_baseline > 0 else 0
    
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    print(f"Total baseline cost: ${total_baseline:.2f}")
    print(f"Total optimized cost: ${total_optimized:.2f}")
    print(f"Total savings: ${total_savings:.2f}")
    print(f"Average savings: {avg_savings_percent:.1f}%")
    print(f"Total trades: {market.get_market_stats()['total_trades']}")
    print(f"Total energy traded: {market.get_market_stats()['total_energy_traded_kwh']:.2f} kWh")
    
    output = {
        'simulation_metadata': {
            'date': datetime.now().isoformat(),
            'num_households': num_households,
            'intervals': num_intervals,
            'simulation_duration_seconds': time.time() - start_time
        },
        'households': results,
        'aggregate_metrics': {
            'total_baseline_cost': total_baseline,
            'total_optimized_cost': total_optimized,
            'total_savings': total_savings,
            'avg_household_savings_percent': avg_savings_percent,
            **market.get_market_stats()
        }
    }
    
    output_dir = root_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "simulation_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Simulation completed in {time.time() - start_time:.1f} seconds")
    
    return output


if __name__ == "__main__":
    print("Loading configuration...")
    config = load_config()
    
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / "data" / "generated"
    
    if not (data_dir / "household_1_data.csv").exists():
        generate_data(config)
    
    results = run_simulation(config)
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"\nTarget: 20-40% savings")
    print(f"Achieved: {results['aggregate_metrics']['avg_household_savings_percent']:.1f}% savings")
    
    if results['aggregate_metrics']['avg_household_savings_percent'] >= 20:
        print("\nTARGET MET!")
    else:
        print("\nClose to target - may need tuning")
