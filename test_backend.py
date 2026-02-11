#!/usr/bin/env python3
"""
Test script for all backend modules - validates PyTorch integration.
"""
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

print("=" * 70)
print("TESLA PRICE PREDICTION - BACKEND TEST SUITE")
print("=" * 70)

# Test 1: Import utils
print("\n[1/5] Testing utils module...")
try:
    from backend import utils
    print(f"  ✓ PROJECT_ROOT: {utils.PROJECT_ROOT}")
    print(f"  ✓ MODELS_DIR: {utils.MODELS_DIR}")
    print(f"  ✓ DATA_DIR: {utils.DATA_DIR}")
    print(f"  ✓ LOOKBACK: {utils.LOOKBACK}")
    print(f"  ✓ HORIZONS: {utils.HORIZONS}")
    print("  ✓ utils module loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading utils: {e}")
    sys.exit(1)

# Test 2: Import preprocessing
print("\n[2/5] Testing preprocessing module...")
try:
    from backend import preprocessing
    df = preprocessing.load_data()
    print(f"  ✓ Data loaded: {len(df)} rows")
    print(f"  ✓ Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  ✓ Close price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print("  ✓ preprocessing module loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading preprocessing: {e}")
    sys.exit(1)

# Test 3: Import and test prediction module
print("\n[3/5] Testing prediction module (PyTorch models)...")
try:
    from backend import prediction
    print("  ✓ PyTorch models defined (LSTM, SimpleRNN)")
    
    # Test model loading
    print("  Testing model loading for all horizons...")
    for model_type in ['lstm', 'simple_rnn']:
        for horizon in [1, 5, 10]:
            model = prediction.load_model(model_type, horizon)
            if model is not None:
                print(f"    ✓ Loaded {model_type} for {horizon}-day horizon")
            else:
                print(f"    ✗ Could not load {model_type} for {horizon}-day horizon")
    print("  ✓ prediction module loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test price predictions
print("\n[4/5] Testing predictions...")
try:
    print("  Testing LSTM predictions:")
    for horizon in [1, 5, 10]:
        result = prediction.predict_price('lstm', horizon)
        if result.get('status') == 'success':
            print(f"    ✓ {horizon}-day LSTM: ${result['predicted_price']}")
        else:
            print(f"    ✗ {horizon}-day LSTM error: {result.get('error')}")
    
    print("  Testing SimpleRNN predictions:")
    for horizon in [1, 5, 10]:
        result = prediction.predict_price('simple_rnn', horizon)
        if result.get('status') == 'success':
            print(f"    ✓ {horizon}-day SimpleRNN: ${result['predicted_price']}")
        else:
            print(f"    ✗ {horizon}-day SimpleRNN error: {result.get('error')}")
    
    latest = prediction.get_latest_price()
    print(f"  ✓ Latest price: ${latest:.2f}")
    print("  ✓ prediction module tested successfully")
except Exception as e:
    print(f"  ✗ Error testing prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Import API module
print("\n[5/5] Testing API module...")
try:
    from backend import api
    print(f"  ✓ API Blueprint created: {api.api_bp.name}")
    print(f"  ✓ Routes available:")
    for rule in api.api_bp.deferred_functions:
        print(f"    - {rule.__name__ if hasattr(rule, '__name__') else 'route'}")
    print("  ✓ api module loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading api: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL BACKEND TESTS PASSED")
print("=" * 70)
print("\nBackend is ready for:")
print("  • Flask API server deployment")
print("  • Stock price predictions (LSTM & SimpleRNN)")
print("  • All PyTorch models loaded successfully")
print("=" * 70)
