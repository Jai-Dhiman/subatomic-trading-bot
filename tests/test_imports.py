"""
Test that all modules can be imported successfully.
"""

def test_data_generation_imports():
    """Test data generation module imports."""
    from src.data_generation import household_generator
    from src.data_generation import weather_generator
    from src.data_generation import pricing_generator
    assert household_generator is not None
    assert weather_generator is not None
    assert pricing_generator is not None


def test_models_imports():
    """Test models module imports."""
    from src.models import battery_manager
    from src.models import node_model
    from src.models import central_model
    assert battery_manager is not None
    assert node_model is not None
    assert central_model is not None


def test_federated_imports():
    """Test federated module imports."""
    from src.federated import federated_aggregator
    assert federated_aggregator is not None


def test_trading_imports():
    """Test trading module imports."""
    from src.trading import trading_logic
    from src.trading import market_mechanism
    assert trading_logic is not None
    assert market_mechanism is not None


def test_simulation_imports():
    """Test simulation module imports."""
    from src.simulation import household_node
    from src.simulation import config_loader
    from src.simulation import run_demo
    assert household_node is not None
    assert config_loader is not None
    assert run_demo is not None


def test_config_loading():
    """Test configuration loading."""
    from src.simulation.config_loader import load_config
    config = load_config()
    assert config is not None
    assert 'simulation' in config
    assert 'battery' in config
    assert 'trading' in config
    assert 'model' in config


if __name__ == "__main__":
    print("Running import tests...")
    
    test_data_generation_imports()
    print("✓ Data generation imports OK")
    
    test_models_imports()
    print("✓ Models imports OK")
    
    test_federated_imports()
    print("✓ Federated imports OK")
    
    test_trading_imports()
    print("✓ Trading imports OK")
    
    test_simulation_imports()
    print("✓ Simulation imports OK")
    
    test_config_loading()
    print("✓ Config loading OK")
    
    print("\nAll import tests passed!")
