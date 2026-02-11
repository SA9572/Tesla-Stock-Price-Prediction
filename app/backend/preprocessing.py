"""
Data scaling and sequence creation for Tesla stock price prediction.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from .utils import get_scaler_path, LOOKBACK, PROJECT_ROOT, DATA_DIR


def load_data():
    """Load Tesla stock data from raw or processed."""
    raw_path = DATA_DIR / "raw" / "TSLA.csv"
    processed_path = DATA_DIR / "processed" / "tsla_cleaned.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path, index_col="Date", parse_dates=True)
    else:
        df = pd.read_csv(raw_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df.ffill().bfill()

    return df


def create_sequence_from_prices(prices: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
    """
    Create input sequence from last N closing prices.
    Returns shape (1, lookback, 1) for model input.
    """
    if len(prices) < lookback:
        return None
    seq = prices[-lookback:].reshape(1, lookback, 1)
    return seq.astype(np.float32)


def get_scaler():
    """Load MinMaxScaler from disk."""
    scaler_path = get_scaler_path()
    if not scaler_path.exists():
        return None
    return joblib.load(scaler_path)


def scale_prices(prices: np.ndarray, scaler) -> np.ndarray:
    """Scale prices using fitted MinMaxScaler."""
    return scaler.transform(prices.reshape(-1, 1))


def inverse_scale_prices(scaled: np.ndarray, scaler) -> np.ndarray:
    """Inverse scale predictions back to original price range."""
    return scaler.inverse_transform(scaled.reshape(-1, 1))
