"""
Utility functions and model paths for Tesla Stock Price Prediction.
"""
from pathlib import Path

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Model file names
LOOKBACK = 60
HORIZONS = [1, 5, 10]

def get_model_path(model_type: str, horizon: int) -> Path:
    """Get path to saved PyTorch model file."""
    return MODELS_DIR / f"{model_type}_{horizon}day_best.pt"

def get_scaler_path() -> Path:
    """Get path to saved scaler."""
    return MODELS_DIR / "scaler.pkl"
