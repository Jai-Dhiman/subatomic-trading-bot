"""
Test that model properly fails when untrained or given invalid data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from src.models.node_model import NodeModel


def test_untrained_model_prediction():
    """Test that untrained model raises RuntimeError instead of returning fallback."""
    print("\n" + "="*60)
    print("TEST 1: Untrained Model Prediction")
    print("="*60)
    
    config = {
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'prediction_horizon_intervals': 6,
        'input_sequence_length': 48
    }
    
    model = NodeModel(household_id=1, config=config)
    
    recent_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=48, freq='30min'),
        'consumption_kwh': np.random.uniform(0.5, 2.0, 48),
        'temperature': np.random.uniform(60, 80, 48),
        'solar_irradiance': np.random.uniform(0, 1000, 48),
        'hour_of_day': [i // 2 for i in range(48)],
        'day_of_week': [0] * 48
    })
    
    try:
        prediction = model.predict(recent_data)
        print("❌ FAILED: Model should have raised RuntimeError but returned prediction")
        print(f"   Got prediction: {prediction}")
        return False
    except RuntimeError as e:
        print("✅ PASSED: Model correctly raised RuntimeError")
        print(f"   Error message: {str(e)[:100]}...")
        return True
    except Exception as e:
        print(f"❌ FAILED: Unexpected exception type: {type(e).__name__}")
        print(f"   Error: {e}")
        return False


def test_insufficient_recent_data():
    """Test that model raises ValueError when given insufficient recent data."""
    print("\n" + "="*60)
    print("TEST 2: Insufficient Recent Data")
    print("="*60)
    
    config = {
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'prediction_horizon_intervals': 6,
        'input_sequence_length': 48
    }
    
    model = NodeModel(household_id=2, config=config)
    model.is_trained = True  # Fake trained state for this test
    
    insufficient_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=20, freq='30min'),
        'consumption_kwh': np.random.uniform(0.5, 2.0, 20),
        'temperature': np.random.uniform(60, 80, 20),
        'solar_irradiance': np.random.uniform(0, 1000, 20),
        'hour_of_day': [i // 2 for i in range(20)],
        'day_of_week': [0] * 20
    })
    
    try:
        prediction = model.predict(insufficient_data)
        print("❌ FAILED: Model should have raised ValueError but returned prediction")
        return False
    except ValueError as e:
        print("✅ PASSED: Model correctly raised ValueError")
        print(f"   Error message: {str(e)[:100]}...")
        return True
    except Exception as e:
        print(f"❌ FAILED: Unexpected exception type: {type(e).__name__}")
        print(f"   Error: {e}")
        return False


def test_missing_columns():
    """Test that model raises ValueError when required columns are missing."""
    print("\n" + "="*60)
    print("TEST 3: Missing Required Columns")
    print("="*60)
    
    config = {
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'prediction_horizon_intervals': 6,
        'input_sequence_length': 48
    }
    
    model = NodeModel(household_id=3, config=config)
    model.is_trained = True
    
    incomplete_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=48, freq='30min'),
        'consumption_kwh': np.random.uniform(0.5, 2.0, 48),
        'hour_of_day': [i // 2 for i in range(48)]
    })
    
    try:
        prediction = model.predict(incomplete_data)
        print("❌ FAILED: Model should have raised ValueError but returned prediction")
        return False
    except ValueError as e:
        print("✅ PASSED: Model correctly raised ValueError")
        print(f"   Error message: {str(e)[:120]}...")
        return True
    except Exception as e:
        print(f"❌ FAILED: Unexpected exception type: {type(e).__name__}")
        print(f"   Error: {e}")
        return False


def test_insufficient_training_data():
    """Test that model raises ValueError when training data is insufficient."""
    print("\n" + "="*60)
    print("TEST 4: Insufficient Training Data")
    print("="*60)
    
    config = {
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'prediction_horizon_intervals': 6,
        'input_sequence_length': 48
    }
    
    model = NodeModel(household_id=4, config=config)
    
    insufficient_training = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='30min'),
        'consumption_kwh': np.random.uniform(0.5, 2.0, 50),
        'temperature': np.random.uniform(60, 80, 50),
        'solar_irradiance': np.random.uniform(0, 1000, 50),
        'hour_of_day': [i // 2 for i in range(50)],
        'day_of_week': [i // 48 for i in range(50)]
    })
    
    try:
        model.train(insufficient_training, epochs=1, verbose=False)
        print("❌ FAILED: Model should have raised ValueError but training succeeded")
        return False
    except ValueError as e:
        print("✅ PASSED: Model correctly raised ValueError")
        print(f"   Error message: {str(e)[:120]}...")
        return True
    except Exception as e:
        print(f"❌ FAILED: Unexpected exception type: {type(e).__name__}")
        print(f"   Error: {e}")
        return False


def test_empty_training_data():
    """Test that model raises ValueError when training data is empty."""
    print("\n" + "="*60)
    print("TEST 5: Empty Training Data")
    print("="*60)
    
    config = {
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'prediction_horizon_intervals': 6,
        'input_sequence_length': 48
    }
    
    model = NodeModel(household_id=5, config=config)
    
    empty_data = pd.DataFrame()
    
    try:
        model.train(empty_data, epochs=1, verbose=False)
        print("❌ FAILED: Model should have raised ValueError but training succeeded")
        return False
    except ValueError as e:
        print("✅ PASSED: Model correctly raised ValueError")
        print(f"   Error message: {str(e)[:100]}...")
        return True
    except Exception as e:
        print(f"❌ FAILED: Unexpected exception type: {type(e).__name__}")
        print(f"   Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL VALIDATION TESTS")
    print("Testing fail-fast behavior (no silent fallbacks)")
    print("="*60)
    
    tests = [
        test_untrained_model_prediction,
        test_insufficient_recent_data,
        test_missing_columns,
        test_insufficient_training_data,
        test_empty_training_data
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ TEST CRASHED: {test.__name__}")
            print(f"   Exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Fail-fast behavior working correctly!")
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED - Review failures above")
    
    print("="*60)
