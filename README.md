# Churn MLOps

An end-to-end MLOps pipeline for predicting customer churn. Trains a Random Forest classifier with MLflow experiment tracking and serves predictions via a FastAPI REST API, containerised with Docker, tested through GitHub Actions CI, and deployed on Render.

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

Trains a `RandomForestClassifier` on `data/Churn.csv` with `class_weight="balanced"` to handle the class imbalance (~73% no-churn). Logs parameters, metrics, and the model artifact to MLflow. The trained model is saved to `app/model.pkl`.

```bash
python app/train.py
```

**Metrics logged to MLflow:** accuracy, precision, recall, F1, ROC-AUC, and the decision threshold.

To view experiment results:

```bash
mlflow ui
# Open http://localhost:5000
```

**Model performance** (threshold = 0.35):

| Metric | Value |
|---|---|
| Recall (churn) | 0.83 |
| Precision (churn) | 0.49 |
| F1 Score | 0.62 |
| ROC-AUC | 0.83 |

A threshold of 0.35 is used instead of the default 0.5 to maximise recall — catching churners that would otherwise be missed matters more than avoiding false alarms.

## Running the API

**Locally:**

```bash
python app/train.py   # generates app/model.pkl first

uvicorn app.predict:app --host 0.0.0.0 --port 10000 --reload
```

**With Docker** (training runs automatically during build):

```bash
docker build -t churn-api .
docker run -p 10000:10000 churn-api
```

API is available at `http://localhost:10000`.

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
{
  "churn_prediction": 1,
  "churn_probability": 0.6183,
  "threshold": 0.35
}
```

- `churn_prediction`: `0` = no churn, `1` = churn
- `churn_probability`: raw model probability (0–1)
- `threshold`: the cutoff used to make the binary decision

## Tests

```bash
pytest tests/
```

Tests cover the `/` and `/predict` endpoints using FastAPI's `TestClient`. Run `python app/train.py` first to generate the model file.

## CI Pipeline

GitHub Actions runs on every push and pull request to `main`:

1. Checks out code
2. Sets up Python 3.12
3. Caches pip dependencies
4. Installs dependencies from `requirements.txt`
5. Trains the model (`app/train.py`)
6. Runs the test suite (`pytest tests/`)

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Deployment

Deployed on **Render** via Docker. The model is trained during the Docker image build (`RUN python app/train.py` in the Dockerfile), so no pre-built model file needs to be committed to the repository. The service runs on port `10000`.
