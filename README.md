# Churn MLOps

An end-to-end MLOps pipeline for predicting customer churn. Trains a Random Forest classifier with MLflow experiment tracking and serves predictions via a FastAPI REST API, containerised with Docker and tested through GitHub Actions CI.

## Project Structure

```
churn-mlops/
├── app/
│   ├── train.py        # Model training + MLflow tracking
│   ├── predict.py      # FastAPI prediction API
│   └── preprocess.py   # Data preprocessing utilities
├── data/
│   └── Churn.csv       # Training dataset
├── tests/
│   └── test_api.py     # API endpoint tests
├── .github/
│   └── workflows/
│       └── ci.yml      # GitHub Actions CI pipeline
├── Dockerfile
└── requirements.txt
```

## Setup

**Prerequisites:** Python 3.12, Docker

```bash
git clone <repo-url>
cd churn-mlops

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Trains a `RandomForestClassifier` on `data/Churn.csv` and logs parameters, metrics, and the model artifact to MLflow. The trained model is also saved to `app/model.pkl` for the API to load.

```bash
python app/train.py
```

To view MLflow experiment results:

```bash
mlflow ui
# Open http://localhost:5000
```

## Running the API

**Locally:**

```bash
uvicorn app.predict:app --host 0.0.0.0 --port 8000 --reload
```

**With Docker:**

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

API is available at `http://localhost:8000`.

## API Endpoints

### `GET /`
Health check.

```json
{"message": "Customer Churn Prediction API is running"}
```

### `POST /predict`
Predicts whether a customer will churn.

**Request body:**
```json
{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 12,
  "PhoneService": 1,
  "MultipleLines": 0,
  "InternetService": 1,
  "OnlineSecurity": 0,
  "OnlineBackup": 1,
  "DeviceProtection": 0,
  "TechSupport": 1,
  "StreamingTV": 1,
  "StreamingMovies": 0,
  "Contract": 2,
  "PaperlessBilling": 1,
  "PaymentMethod": 1,
  "MonthlyCharges": 70.5,
  "TotalCharges": 850.2
}
```

**Response:**
```json
{"churn_prediction": 0}
```

`churn_prediction`: `0` = no churn, `1` = churn.

## Tests

```bash
pytest tests/
```

Tests cover the `/` and `/predict` endpoints using FastAPI's `TestClient`. The model must be trained before running tests locally (`python app/train.py`).

## CI Pipeline

GitHub Actions runs on every push and pull request to `main`:

1. Checks out code
2. Sets up Python 3.12
3. Installs dependencies from `requirements.txt`
4. Trains the model (`app/train.py`)
5. Runs the test suite (`pytest tests/`)

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
