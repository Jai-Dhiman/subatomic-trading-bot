"""
Import smoke tests for current modules.

This test intentionally imports only the modules that are part of the
current training-focused codebase.
"""

def test_models_imports():
    from src.models import battery_manager
    from src.models import node_model
    from src.models import consumption_transformer
    from src.models import trading_transformer_v2
    from src.models import feature_engineering_consumption
    from src.models import feature_engineering_trading
    assert battery_manager is not None
    assert node_model is not None
    assert consumption_transformer is not None
    assert trading_transformer_v2 is not None
    assert feature_engineering_consumption is not None
    assert feature_engineering_trading is not None


def test_training_utils_imports():
    from src.training import training_utils
    assert training_utils is not None


def test_data_integration_imports():
    from src.data_integration import data_adapter
    from src.data_integration import supabase_connector
    assert data_adapter is not None
    assert supabase_connector is not None


def test_simulation_config_loader():
    from src.simulation import config_loader
    assert config_loader is not None


def test_config_loading():
    from src.simulation.config_loader import load_config
    config = load_config()
    assert config is not None
    assert 'simulation' in config
    assert 'battery' in config
    assert 'trading' in config
    assert 'model' in config


if __name__ == "__main__":
    print("Running import tests...")

    test_models_imports()
    print("✓ Models imports OK")

    test_training_utils_imports()
    print("✓ Training utils imports OK")

    test_data_integration_imports()
    print("✓ Data integration imports OK")

    test_simulation_config_loader()
    print("✓ Simulation config loader import OK")

    test_config_loading()
    print("✓ Config loading OK")

    print("\nAll import tests passed!")
