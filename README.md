# Tesla Stock Price Prediction

Deep learning models (SimpleRNN and LSTM) to predict Tesla stock closing prices for 1-day, 5-day, and 10-day horizons.

## Project Structure

```
Tesla-Stock-Price-Prediction/
├── app/                    # Flask web application
│   ├── backend/            # API, prediction, preprocessing
│   ├── frontend/           # Templates, CSS, JS
│   └── main.py             # Entry point
├── models/                 # Pre-trained model weights
├── notebooks/              # Jupyter notebooks
├── data/
│   ├── raw/TSLA.csv
│   └── processed/
├── requirements.txt
├── Procfile                # Render deployment
├── render.yaml
└── train_models.py         # Train all models
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

Run the training script to generate models and scaler:

```bash
python train_models.py
```

Or run the notebooks in order:
- `01_data_cleaning_eda.ipynb`
- `02_feature_engineering.ipynb`
- `03_simple_rnn_model.ipynb`
- `04_lstm_model.ipynb`
- `05_model_comparison.ipynb`

### 3. Run Flask App

```bash
python -m app.main
# or
gunicorn app.main:app
```

Open http://localhost:5000

## API Endpoints

- `GET /api/predict?model=lstm&horizon=1` - Predict price
- `GET /api/latest` - Latest closing price
- `GET /health` - Health check

## Deployment (Render)

1. Connect your GitHub repo to Render
2. Configure as a Web Service
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app.main:app`

## Evaluation Metrics

- MSE (Mean Squared Error) for model comparison
- GridSearchCV/Keras Tuner for hyperparameter tuning (LSTM units, dropout, learning rate)

## Domain

Financial Services • Stock Market Trading & Investment Strategies
