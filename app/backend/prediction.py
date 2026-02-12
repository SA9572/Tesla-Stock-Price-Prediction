"""
Model inference logic for Tesla stock price prediction.
"""
import numpy as np
from pathlib import Path

# Guarded import of torch so the app can start even if torch isn't installed
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from .utils import get_model_path, get_scaler_path, LOOKBACK, HORIZONS
from .preprocessing import (
    load_data,
    create_sequence_from_prices,
    get_scaler,
    scale_prices,
    inverse_scale_prices,
)

# Device configuration (only if torch is available)
device = None
if TORCH_AVAILABLE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model architectures (defined only if nn is available)
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, output_size=1, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            lstm_out = self.dropout(lstm_out)
            predictions = self.fc(lstm_out[:, -1, :])
            return predictions

    class SimpleRNNModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, output_size=1, dropout=0.2):
            super(SimpleRNNModel, self).__init__()
            self.hidden_size = hidden_size
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            rnn_out, _ = self.rnn(x)
            rnn_out = self.dropout(rnn_out)
            predictions = self.fc(rnn_out[:, -1, :])
            return predictions


def load_model(model_type: str, horizon: int):
    """Load SimpleRNN or LSTM model from disk using PyTorch.

    Returns None if torch is unavailable or model file missing.
    """
    if not TORCH_AVAILABLE:
        return None

    path = get_model_path(model_type, horizon)
    if not path.exists():
        return None

    # Create model based on type
    if model_type == 'lstm':
        model = LSTMModel(input_size=1, hidden_size=64, output_size=1, dropout=0.2)
    elif model_type == 'simple_rnn':
        model = SimpleRNNModel(input_size=1, hidden_size=50, output_size=1, dropout=0.2)
    else:
        return None

    # Load weights
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_price(model_type: str, horizon: int, use_last_n_days: int = None) -> dict:
    """
    Predict stock price for given horizon using specified model.
    
    Args:
        model_type: 'simple_rnn' or 'lstm'
        horizon: 1, 5, or 10 days ahead
        use_last_n_days: Optional custom number of days; defaults to LOOKBACK from latest data
    
    Returns:
        dict with 'predicted_price', 'model', 'horizon', 'status'
    """
    if horizon not in HORIZONS:
        return {"error": f"Invalid horizon. Use {HORIZONS}", "status": "error"}

    if not TORCH_AVAILABLE:
        return {"error": "PyTorch is not installed in the environment.", "status": "error"}

    model = load_model(model_type, horizon)
    if model is None:
        return {"error": f"Model not found: {model_type}_{horizon}day_best.pt or torch unavailable", "status": "error"}

    scaler = get_scaler()
    if scaler is None:
        return {"error": "Scaler not found. Run feature engineering notebook first.", "status": "error"}

    df = load_data()
    if df is None or df.empty:
        return {"error": "Data not found.", "status": "error"}

    close_prices = df["Close"].values
    n_days = use_last_n_days or min(LOOKBACK, len(close_prices))

    if len(close_prices) < n_days:
        return {"error": f"Not enough data. Need at least {n_days} days.", "status": "error"}

    prices = close_prices[-n_days:]
    scaled = scale_prices(prices, scaler)
    seq = create_sequence_from_prices(scaled, lookback=LOOKBACK)

    if seq is None:
        return {"error": f"Could not create sequence. Need {LOOKBACK} days.", "status": "error"}

    # PyTorch prediction
    X_tensor = torch.FloatTensor(seq).to(device)
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()
    
    pred_price = float(inverse_scale_prices(pred_scaled, scaler)[0, 0])

    return {
        "predicted_price": round(pred_price, 2),
        "model": model_type,
        "horizon": horizon,
        "status": "success",
    }


def get_latest_price() -> float:
    """Get the most recent closing price from data."""
    df = load_data()
    if df is None or df.empty:
        return 0.0
    return float(df["Close"].iloc[-1])
