# Tesla Stock Price Prediction - Project Report

## Executive Summary

This project implements deep learning models (SimpleRNN and LSTM) to predict Tesla's closing stock price for 1-day, 5-day, and 10-day horizons. The solution includes data cleaning, exploratory analysis, feature engineering, model training with hyperparameter tuning, and deployment via a Flask web application on Render.

---

## 1. Problem Understanding

**Goal:** Predict Tesla stock closing prices using historical sequential data.

**Approach:** Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are well-suited for time-series data due to their ability to capture temporal dependencies.

**Deliverables:**
- Models for 1-day, 5-day, and 10-day price predictions
- Comparison of SimpleRNN vs LSTM performance
- Proper handling of missing values in time-series context
- GridSearchCV (Keras Tuner) for hyperparameter optimization

---

## 2. Data Preprocessing

### 2.1 Dataset
- **Source:** TSLA.csv (Date, Open, High, Low, Close, Adj Close, Volume)
- **Target:** Closing price (as per problem statement)
- **Time Range:** 2010-06-29 to 2020-02-03

### 2.2 Missing Values Handling (Time-Series Specific)

For time-series data, we **do not** drop rows with missing values because:
- Dropping breaks temporal continuity
- Sequences for LSTM/RNN require uninterrupted time steps

**Strategy:**
- **Forward fill (ffill):** Use previous day's value – preserves temporal order
- **Backward fill (bfill):** For leading NaN values
- **Alternative:** Linear interpolation for small gaps

### 2.3 Feature Engineering
- **Scaling:** MinMaxScaler (0–1) for better model convergence
- **Sequence Creation:** Sliding window of 60 days (lookback) to predict t+1, t+5, or t+10
- **Train/Test Split:** 80/20 with temporal order preserved (no shuffling)

---

## 3. Model Development

### 3.1 Architecture

**SimpleRNN:**
```
SimpleRNN(50 units) → Dropout(0.2) → Dense(1)
```

**LSTM:**
```
LSTM(64 units) → Dropout(0.2) → Dense(1)
```

### 3.2 Training
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Callbacks:** EarlyStopping (patience=10), ModelCheckpoint
- **Validation:** 10% holdout from training set

### 3.3 Hyperparameter Tuning (GridSearchCV / Keras Tuner)

For LSTM, we tune:
- **LSTM units:** 32, 64, 128
- **Dropout rate:** 0.1, 0.2, 0.3, 0.4
- **Learning rate:** 1e-4 to 1e-2 (log scale)

---

## 4. Model Evaluation

### Metrics
- **MSE (Mean Squared Error):** Primary metric for regression

### Comparison
- SimpleRNN vs LSTM for each horizon (1, 5, 10 days)
- Visualization of actual vs predicted prices

### Expected Insights
- LSTM typically outperforms SimpleRNN on longer sequences
- Shorter horizons (1-day) easier to predict than longer (10-day)
- Stock prices are inherently noisy; model captures trends, not exact values

---

## 5. Business Use Cases

1. **Automated Trading:** Algorithmic buy/sell signals based on predictions
2. **Risk Management:** Portfolio allocation and hedging
3. **Long-Term Planning:** Retirement, ETF, mutual fund decisions
4. **Corporate Use:** Revenue forecasting, competitor analysis
5. **Research:** Compare with GRU, Transformers, ARIMA; add news sentiment

---

## 6. Limitations

- **Market Volatility:** Black swan events not captured
- **External Factors:** News, macro factors not included
- **Data Recency:** Historical data; live data requires pipeline
- **Regulatory:** Not financial advice; educational use

---

## 7. Improvements

- Add news sentiment (NLP)
- Include Volume, technical indicators (RSI, MACD)
- Compare GRU, Transformer, ARIMA
- Real-time data integration

---

## 8. Project Timeline

| Milestone              | Deadline   |
|------------------------|------------|
| Data cleaning & EDA    | Week 1     |
| Feature engineering    | Week 2     |
| SimpleRNN training     | Week 2     |
| LSTM training + tuning | Week 3     |
| Model comparison      | Week 3     |
| Flask app & deployment | Week 4     |
| **Submission**         | **Jan 12, 2026** |

---

## 9. Technical Stack

- **Python 3.11+**
- **TensorFlow/Keras** – Deep learning
- **pandas, numpy** – Data handling
- **scikit-learn** – Preprocessing, metrics
- **Flask** – Web API
- **Render** – Deployment
