"""
Complete Energy Trading Demo - Orchestrator Script

Runs the complete demo flow:
1. Generate multi-household training data
2. Train lightweight Transformer model
3. Generate simulation data
4. Run 48-interval P2P trading simulation
5. Output comprehensive JSON results

Usage:
    python src/demo/run_complete_demo.py
"""

import sys
from pathlib import Path
import time
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch

# Import demo components
from src.demo.generate_multi_household_data import generate_multi_household_dataset, save_dataset
from src.demo.train_demo_model import train_demo_model
from src.demo.demo_household_agent import DemoHouseholdAgent, TradeDecision
from src.models.transformer_model import EnergyTransformer


def generate_simulation_data(num_households: int, num_intervals: int, training_df: pd.DataFrame):
    """
    Generate simulation data based on training patterns.
    
    Creates 48 intervals of actual consumption with some variation from training.
    """
    print("\n" + "=" * 60)
    print("Generating Simulation Data")
    print("=" * 60)
    print(f"  Intervals: {num_intervals}")
    print(f"  Households: {num_households}")
    
    # Use last day of training data as baseline
    sim_data = {}
    
    for hh_id in range(1, num_households + 1):
        hh_train = training_df[training_df['household_id'] == hh_id]
        
        # Take last 48 intervals as baseline and add 10% noise
        last_48 = hh_train.iloc[-48:].copy()
        
        sim_data[hh_id] = {
            'consumption': last_48['total_consumption_kwh'].values * (0.95 + np.random.rand(48) * 0.1),
            'prices': last_48['price_per_kwh'].values,
            'features': last_48[[col for col in last_48.columns if col not in ['household_id', 'timestamp', 'total_consumption_kwh']]].values
        }
    
    print(f"✓ Generated simulation data for {num_households} households")
    return sim_data


def calculate_market_price(buy_orders: list, sell_orders: list, pge_price: float) -> float:
    """
    Calculate market clearing price based on supply/demand.
    
    Simple mechanism: average of bids and asks, bounded by PGE price.
    """
    if not buy_orders and not sell_orders:
        return pge_price * 0.85  # Default to 85% of PGE price
    
    total_demand = sum(o.quantity for o in buy_orders)
    total_supply = sum(o.quantity for o in sell_orders)
    
    if total_demand == 0 and total_supply == 0:
        return pge_price * 0.85
    
    # Price based on supply/demand ratio
    if total_supply > 0:
        ratio = total_demand / total_supply
        if ratio > 1.5:  # High demand
            price = pge_price * 0.90
        elif ratio < 0.5:  # High supply
            price = pge_price * 0.75
        else:  # Balanced
            price = pge_price * 0.82
    else:
        price = pge_price * 0.88
    
    # Ensure price is reasonable
    return max(0.15, min(price, pge_price * 0.95))


def match_trades(buy_orders: list, sell_orders: list, market_price: float, interval: int):
    """
    Match buy and sell orders at market price.
    
    Returns list of matched trades.
    """
    transactions = []
    
    # Filter orders that accept the market price
    valid_buyers = [b for b in buy_orders if b.max_price >= market_price]
    valid_sellers = [s for s in sell_orders if s.min_price <= market_price]
    
    # Sort by quantity (larger trades first for efficiency)
    valid_buyers.sort(key=lambda x: x.quantity, reverse=True)
    valid_sellers.sort(key=lambda x: x.quantity, reverse=True)
    
    buyer_idx = 0
    seller_idx = 0
    
    while buyer_idx < len(valid_buyers) and seller_idx < len(valid_sellers):
        buyer = valid_buyers[buyer_idx]
        seller = valid_sellers[seller_idx]
        
        # Match trades
        trade_quantity = min(buyer.quantity, seller.quantity)
        
        if trade_quantity > 0.01:  # Minimum trade size
            transactions.append({
                'interval': interval,
                'buyer_id': buyer.household_id,
                'seller_id': seller.household_id,
                'quantity_kwh': round(trade_quantity, 3),
                'price_per_kwh': round(market_price, 4),
                'total_cost': round(trade_quantity * market_price, 2)
            })
            
            # Update remaining quantities
            buyer.quantity -= trade_quantity
            seller.quantity -= trade_quantity
        
        # Move to next buyer/seller if quantities depleted
        if buyer.quantity <= 0.01:
            buyer_idx += 1
        if seller.quantity <= 0.01:
            seller_idx += 1
    
    return transactions


def run_simulation(agents: list, sim_data: dict, num_intervals: int, start_time: datetime):
    """
    Run the P2P trading simulation for 48 intervals.
    """
    print("\n" + "=" * 60)
    print("Running P2P Trading Simulation")
    print("=" * 60)
    print(f"  Duration: {num_intervals} intervals (24 hours)")
    print(f"  Households: {len(agents)}")
    
    all_transactions = []
    
    for interval in range(num_intervals):
        timestamp = start_time + timedelta(minutes=30 * interval)
        
        if interval % 10 == 0:
            print(f"\n  Processing interval {interval}/{num_intervals} ({timestamp.strftime('%H:%M')})")
        
        # Get PGE price for this interval
        pge_price = sim_data[1]['prices'][interval]
        
        # Collect predictions and trading decisions from all households
        buy_orders = []
        sell_orders = []
        
        for agent in agents:
            hh_id = agent.household_id
            
            # Get recent features for prediction (need 48 timesteps)
            if interval < 48:
                # Use last 48 from training data
                recent_features = sim_data[hh_id]['features'][-48:]
            else:
                # Use recent simulation features (simplified - use last 48 from training)
                recent_features = sim_data[hh_id]['features'][-48:]
            
            # Make prediction
            predictions = agent.predict_consumption(recent_features, interval)
            predicted_next = predictions['consumption_day'][0]  # Next interval prediction
            
            # Make trading decision
            decision = agent.make_trading_decision(
                predicted_consumption_next=predicted_next,
                market_price=pge_price * 0.85,  # Initial estimate
                pge_price=pge_price,
                battery_soc=agent.battery.soc_percent
            )
            
            # Add to order book
            if decision.action == 'buy':
                decision.household_id = hh_id
                buy_orders.append(decision)
            elif decision.action == 'sell':
                decision.household_id = hh_id
                sell_orders.append(decision)
        
        # Calculate market price
        market_price = calculate_market_price(buy_orders, sell_orders, pge_price)
        
        # Match trades
        transactions = match_trades(buy_orders, sell_orders, market_price, interval)
        all_transactions.extend(transactions)
        
        # Execute trades
        for txn in transactions:
            buyer = next(a for a in agents if a.household_id == txn['buyer_id'])
            seller = next(a for a in agents if a.household_id == txn['seller_id'])
            
            buyer.execute_trade('buy', txn['quantity_kwh'], txn['price_per_kwh'], txn['seller_id'], interval)
            seller.execute_trade('sell', txn['quantity_kwh'], txn['price_per_kwh'], txn['buyer_id'], interval)
        
        # Handle energy consumption for all households
        for agent in agents:
            hh_id = agent.household_id
            actual_consumption = sim_data[hh_id]['consumption'][interval]
            
            # Consume energy
            agent.consume_energy(actual_consumption, pge_price, interval)
            
            # Determine action for recording
            action = 'hold'
            for txn in transactions:
                if txn['buyer_id'] == hh_id:
                    action = 'buy'
                    break
                elif txn['seller_id'] == hh_id:
                    action = 'sell'
                    break
            
            # Get prediction for recording
            if interval < 48:
                recent_features = sim_data[hh_id]['features'][-48:]
            else:
                recent_features = sim_data[hh_id]['features'][-48:]
            predictions = agent.predict_consumption(recent_features, interval)
            predicted = predictions['consumption_day'][0]
            
            # Record interval
            agent.record_interval(
                interval=interval,
                timestamp=timestamp,
                predicted=predicted,
                actual=actual_consumption,
                market_price=market_price,
                pge_price=pge_price,
                action=action
            )
        
        if interval % 10 == 0:
            print(f"    Trades: {len(transactions)}, Market price: ${market_price:.4f}/kWh")
    
    print(f"\n✓ Simulation complete!")
    print(f"  Total transactions: {len(all_transactions)}")
    
    return all_transactions


def format_output(agents: list, transactions: list, execution_time: float, config: dict) -> dict:
    """
    Format all results into JSON output.
    """
    print("\n" + "=" * 60)
    print("Formatting Results")
    print("=" * 60)
    
    # Collect household results
    households_results = []
    total_baseline = 0.0
    total_optimized = 0.0
    
    for agent in agents:
        result = agent.get_results()
        households_results.append(result)
        total_baseline += result['costs']['baseline_pge_total']
        total_optimized += result['costs']['optimized_total']
    
    # Calculate aggregate metrics
    total_savings = total_baseline - total_optimized
    avg_savings_percent = (total_savings / total_baseline * 100) if total_baseline > 0 else 0
    
    # Market summary
    total_energy_traded = sum(t['quantity_kwh'] for t in transactions)
    avg_trade_size = total_energy_traded / len(transactions) if transactions else 0
    avg_price = sum(t['price_per_kwh'] for t in transactions) / len(transactions) if transactions else 0
    price_range = [min(t['price_per_kwh'] for t in transactions), max(t['price_per_kwh'] for t in transactions)] if transactions else [0, 0]
    
    # Sample predictions from first household
    sample_agent = agents[0]
    sample_predictions = sample_agent.predictions_cache.get(0, {})
    
    output = {
        'metadata': {
            'demo_date': datetime.now().isoformat(),
            'num_households': len(agents),
            'num_intervals': config['num_intervals'],
            'duration_hours': config['num_intervals'] / 2,
            'execution_time_seconds': round(execution_time, 1),
            'model_config': {
                'd_model': config['d_model'],
                'n_layers': config['n_layers'],
                'n_heads': config['n_heads']
            }
        },
        'households': households_results,
        'sample_predictions': {
            'household_id': 1,
            'day_ahead_consumption': sample_predictions.get('consumption_day', []).tolist()[:10] if 'consumption_day' in sample_predictions else [],
            'day_ahead_price': sample_predictions.get('price_day', []).tolist()[:10] if 'price_day' in sample_predictions else []
        },
        'market_transactions': transactions,
        'market_summary': {
            'total_trades': len(transactions),
            'total_energy_traded_kwh': round(total_energy_traded, 2),
            'avg_trade_size_kwh': round(avg_trade_size, 3),
            'avg_market_price': round(avg_price, 4),
            'price_range': [round(p, 4) for p in price_range]
        },
        'aggregate_metrics': {
            'total_baseline_cost': round(total_baseline, 2),
            'total_optimized_cost': round(total_optimized, 2),
            'total_savings': round(total_savings, 2),
            'avg_household_savings_percent': round(avg_savings_percent, 1),
            'target_savings_percent_range': [20, 40],
            'target_met': avg_savings_percent >= 20
        }
    }
    
    print(f"✓ Results formatted")
    print(f"  Households: {len(households_results)}")
    print(f"  Transactions: {len(transactions)}")
    print(f"  Total savings: ${total_savings:.2f} ({avg_savings_percent:.1f}%)")
    
    return output


def main():
    """Main demo execution."""
    print("=" * 60)
    print("ENERGY TRADING DEMO - COMPLETE SYSTEM")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    demo_start_time = time.time()
    
    # Configuration
    config = {
        'num_households': 10,
        'num_intervals': 48,
        'training_days': 10,
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'sequence_length': 48,
        'horizons': {'day': 48, 'week': 336}
    }
    
    # Paths
    data_path = project_root / "data" / "demo" / "multi_household_training_data.npz"
    model_path = project_root / "checkpoints" / "demo_model.pt"
    output_path = project_root / "data" / "output" / "demo_results.json"
    
    # Step 1: Generate training data (if not exists)
    if not data_path.exists():
        print("\n" + "=" * 60)
        print("STEP 1: Generate Training Data")
        print("=" * 60)
        dataset = generate_multi_household_dataset(
            num_households=config['num_households'],
            days=config['training_days']
        )
        save_dataset(dataset, data_path)
    else:
        print(f"\n✓ Training data exists: {data_path}")
    
    # Step 2: Train model (if not exists)
    if not model_path.exists():
        print("\n" + "=" * 60)
        print("STEP 2: Train Model")
        print("=" * 60)
        train_demo_model(config, data_path, model_path)
    else:
        print(f"\n✓ Trained model exists: {model_path}")
    
    # Step 3: Load trained model
    print("\n" + "=" * 60)
    print("STEP 3: Load Trained Model")
    print("=" * 60)
    
    checkpoint = torch.load(model_path)
    model = EnergyTransformer(
        n_features=24,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        horizons=config['horizons']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    feature_engineer = checkpoint['feature_engineer']
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    
    # Step 4: Load training data for simulation baseline
    print("\n" + "=" * 60)
    print("STEP 4: Load Data for Simulation")
    print("=" * 60)
    
    data = np.load(data_path, allow_pickle=True)
    training_df = pd.DataFrame(data['data'], columns=data['columns'].tolist())
    print(f"✓ Loaded {len(training_df)} training samples")
    
    # Step 5: Generate simulation data
    sim_data = generate_simulation_data(config['num_households'], config['num_intervals'], training_df)
    
    # Step 6: Initialize household agents
    print("\n" + "=" * 60)
    print("STEP 5: Initialize Household Agents")
    print("=" * 60)
    
    agents = []
    for hh_id in range(1, config['num_households'] + 1):
        agent = DemoHouseholdAgent(
            household_id=hh_id,
            model=model,
            feature_engineer=feature_engineer
        )
        agents.append(agent)
    
    print(f"✓ Initialized {len(agents)} household agents")
    
    # Step 7: Run simulation
    sim_start_time = datetime(2024, 10, 11, 0, 0)  # Start at midnight
    transactions = run_simulation(agents, sim_data, config['num_intervals'], sim_start_time)
    
    # Step 8: Format and save results
    execution_time = time.time() - demo_start_time
    results = format_output(agents, transactions, execution_time, config)
    
    # Save to JSON
    print("\n" + "=" * 60)
    print("STEP 6: Save Results")
    print("=" * 60)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print(f"\nExecution Time: {execution_time:.1f}s ({execution_time/60:.1f}m)")
    print(f"\nKey Results:")
    print(f"  Total baseline cost: ${results['aggregate_metrics']['total_baseline_cost']}")
    print(f"  Total optimized cost: ${results['aggregate_metrics']['total_optimized_cost']}")
    print(f"  Total savings: ${results['aggregate_metrics']['total_savings']}")
    print(f"  Average savings: {results['aggregate_metrics']['avg_household_savings_percent']}%")
    print(f"  Target met: {'YES' if results['aggregate_metrics']['target_met'] else 'NO'} (target: 20-40%)")
    print(f"  Total trades: {results['market_summary']['total_trades']}")
    print(f"  Energy traded: {results['market_summary']['total_energy_traded_kwh']} kWh")
    print(f"\nOutput JSON: {output_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
