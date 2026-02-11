"""
Flask API routes for Tesla Stock Price Prediction.
"""
from flask import Blueprint, jsonify, request

from .prediction import predict_price, get_latest_price

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render."""
    return jsonify({"status": "ok", "service": "Tesla Stock Price Prediction"})


@api_bp.route("/api/predict", methods=["GET", "POST"])
def predict():
    """
    Predict Tesla stock price.
    
    Query params (GET) or JSON body (POST):
    - model: 'simple_rnn' or 'lstm'
    - horizon: 1, 5, or 10 (days ahead)
    """
    if request.method == "POST" and request.is_json:
        data = request.get_json()
        model_type = data.get("model", "lstm")
        horizon = int(data.get("horizon", 1))
    else:
        model_type = request.args.get("model", "lstm")
        try:
            horizon = int(request.args.get("horizon", 1))
        except ValueError:
            horizon = 1

    result = predict_price(model_type, horizon)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@api_bp.route("/api/latest", methods=["GET"])
def latest():
    """Get latest closing price from dataset."""
    price = get_latest_price()
    return jsonify({"latest_price": price, "status": "success"})
