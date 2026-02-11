#!/usr/bin/env python3
"""
Test script for Flask API routes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'app'))

print("=" * 70)
print("TESLA PRICE PREDICTION - FLASK API TEST")
print("=" * 70)

# Test Flask app creation
print("\n[1/2] Creating Flask app...")
try:
    from flask import Flask
    from backend.api import api_bp
    
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    
    print("  ✓ Flask app created")
    print("  ✓ API blueprint registered")
    print("  ✓ Routes configured:")
    for rule in app.url_map.iter_rules():
        if 'api' in rule.endpoint or 'health' in rule.endpoint:
            print(f"    - {rule.rule} [{','.join(rule.methods-{'OPTIONS','HEAD'})}]")
except Exception as e:
    print(f"  ✗ Error creating Flask app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test API endpoints (without running server)
print("\n[2/2] Testing API endpoints with test client...")
try:
    client = app.test_client()
    
    # Test health endpoint
    print("  Testing /health endpoint...")
    response = client.get('/health')
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert data['status'] == 'ok', f"Expected status='ok', got {data['status']}"
    print(f"    ✓ Health check: {data}")
    
    # Test latest price endpoint
    print("  Testing /api/latest endpoint...")
    response = client.get('/api/latest')
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert data['status'] == 'success', f"Expected success, got {data['status']}"
    print(f"    ✓ Latest price: ${data['latest_price']:.2f}")
    
    # Test predictions with GET
    print("  Testing /api/predict endpoint (GET parameters)...")
    response = client.get('/api/predict?model=lstm&horizon=1')
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert data['status'] == 'success', f"Expected success, got {data['status']}"
    print(f"    ✓ 1-day LSTM prediction: ${data['predicted_price']}")
    
    # Test predictions with POST
    print("  Testing /api/predict endpoint (POST JSON)...")
    response = client.post('/api/predict', 
                           json={'model': 'simple_rnn', 'horizon': 5},
                           content_type='application/json')
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert data['status'] == 'success', f"Expected success, got {data['status']}"
    print(f"    ✓ 5-day SimpleRNN prediction: ${data['predicted_price']}")
    
    # Test error handling
    print("  Testing error handling...")
    response = client.get('/api/predict?model=lstm&horizon=99')
    assert response.status_code == 400, f"Expected 400 for invalid horizon, got {response.status_code}"
    print(f"    ✓ Invalid horizon properly rejected")
    
    print("  ✓ All API endpoints tested successfully")
except Exception as e:
    print(f"  ✗ Error testing API: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ FLASK API TESTS PASSED")
print("=" * 70)
print("\nAvailable endpoints:")
print("  GET  /health                 - Health check")
print("  GET  /api/latest             - Latest closing price")
print("  GET  /api/predict             - Get prediction (query params)")
print("  POST /api/predict             - Get prediction (JSON body)")
print("\nExample usage:")
print("  curl 'http://localhost:5000/api/predict?model=lstm&horizon=1'")
print("  curl -X POST http://localhost:5000/api/predict \\")
print("       -H 'Content-Type: application/json' \\")
print("       -d '{\"model\": \"lstm\", \"horizon\": 5}'")
print("=" * 70)
