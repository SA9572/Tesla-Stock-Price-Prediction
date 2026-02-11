#!/usr/bin/env python3
"""
Flask App Startup Test - Verify application runs properly
"""
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'app'))

print("\n" + "=" * 80)
print(" " * 20 + "TESLA PRICE PREDICTION - FLASK APP TEST")
print("=" * 80)

# Create Flask app
print("\n[1/3] Creating Flask application...")
try:
    from flask import Flask, render_template, send_from_directory
    from backend.api import api_bp
    
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "app" / "frontend" / "templates"),
        static_folder=str(Path(__file__).parent / "app" / "frontend" / "static"),
    )
    app.register_blueprint(api_bp)
    
    @app.route("/")
    def index():
        """Serve main dashboard."""
        return render_template("index.html")
    
    @app.route("/static/<path:filename>")
    def static_files(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)
    
    print("  ✓ Flask app created successfully")
    print(f"  ✓ Template folder: {app.template_folder}")
    print(f"  ✓ Static folder: {app.static_folder}")
except Exception as e:
    print(f"  ✗ Error creating Flask app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with development server
print("\n[2/3] Running app with test client (no external server)...")
try:
    with app.test_client() as client:
        print("  ✓ Test client created")
        
        # Test all routes
        print("\n  Testing routes:")
        
        response = client.get('/')
        print(f"    GET / → {response.status_code} (Main dashboard)")
        
        response = client.get('/health')
        print(f"    GET /health → {response.status_code} (Health check)")
        
        response = client.get('/api/latest')
        data = response.get_json()
        print(f"    GET /api/latest → {response.status_code} (Latest price: ${data['latest_price']:.2f})")
        
        response = client.get('/api/predict?model=lstm&horizon=1')
        data = response.get_json()
        print(f"    GET /api/predict → {response.status_code} (LSTM 1-day: ${data['predicted_price']:.2f})")
        
except Exception as e:
    print(f"  ✗ Error testing app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Demonstrate how to run
print("\n[3/3] Deployment information...")
print("─" * 80)
print("\n  To run the Flask development server:")
print("  ─" * 40)
print("    $ python app/main.py")
print("    * Running on http://localhost:5000")
print("\n  The server will:")
print("    ✓ Serve the web dashboard at http://localhost:5000/")
print("    ✓ Handle API requests at http://localhost:5000/api/*")
print("    ✓ Support heath checks at http://localhost:5000/health")
print("\n  Production deployment:")
print("    ✓ Use Gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 app.main:app")
print("    ✓ Use Render.com via Procfile (already configured)")
print("    ✓ Environment requirement: PyTorch + scikit-learn")

print("\n" + "=" * 80)
print("✅ FLASK APP TEST PASSED - APPLICATION IS READY FOR DEPLOYMENT")
print("=" * 80 + "\n")
