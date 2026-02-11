"""
Standalone script to train SimpleRNN and LSTM models.
Run from project root: python train_models.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "TSLA.csv"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
LOOKBACK = 60
TEST_RATIO = 0.2
HORIZONS = [1, 5, 10]

DATA_PROC.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


def create_sequences(data, lookback, forecast_horizon):
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i - lookback : i])
        y.append(data[i + forecast_horizon - 1])
    return np.array(X), np.array(y)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_RAW)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.ffill().bfill()

    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    joblib.dump(scaler, MODELS / "scaler.pkl")
    print("Scaler saved.")

    for horizon in HORIZONS:
        X, y = create_sequences(scaled_data, LOOKBACK, horizon)
        split_idx = int(len(X) * (1 - TEST_RATIO))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        np.save(DATA_PROC / f"X_train_{horizon}d.npy", X_train)
        np.save(DATA_PROC / f"X_test_{horizon}d.npy", X_test)
        np.save(DATA_PROC / f"y_train_{horizon}d.npy", y_train)
        np.save(DATA_PROC / f"y_test_{horizon}d.npy", y_test)

    print("Training SimpleRNN models...")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    for horizon in HORIZONS:
        X_train = np.load(DATA_PROC / f"X_train_{horizon}d.npy")
        y_train = np.load(DATA_PROC / f"y_train_{horizon}d.npy")
        X_test = np.load(DATA_PROC / f"X_test_{horizon}d.npy")
        y_test = np.load(DATA_PROC / f"y_test_{horizon}d.npy")

        model = Sequential([
            SimpleRNN(50, return_sequences=False, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=100,
            batch_size=32,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ],
            verbose=0,
        )
        model.save(MODELS / f"simple_rnn_{horizon}day.h5")
        pred = model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, pred)
        print(f"  SimpleRNN {horizon}-day MSE: {mse:.6f}")

    print("Training LSTM models...")
    from tensorflow.keras.layers import LSTM

    for horizon in HORIZONS:
        X_train = np.load(DATA_PROC / f"X_train_{horizon}d.npy")
        y_train = np.load(DATA_PROC / f"y_train_{horizon}d.npy")
        X_test = np.load(DATA_PROC / f"X_test_{horizon}d.npy")
        y_test = np.load(DATA_PROC / f"y_test_{horizon}d.npy")

        model = Sequential([
            LSTM(64, return_sequences=False, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=100,
            batch_size=32,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ],
            verbose=0,
        )
        model.save(MODELS / f"lstm_{horizon}day.h5")
        pred = model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, pred)
        print(f"  LSTM {horizon}-day MSE: {mse:.6f}")

    print("All models trained and saved.")


if __name__ == "__main__":
    main()
