# Churn MLOps

An end-to-end MLOps pipeline for predicting customer churn — from training to a monitored, deployed REST API.

**Live API:** `https://<your-render-service>.onrender.com`

## Tech Stack

| Layer | Tool |
|---|---|
| Model | scikit-learn RandomForestClassifier |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Monitoring | Python logging + CSV audit trail |

## Architecture

```
data/Churn.csv
      │
      ▼
app/train.py  ──►  MLflow (metrics + params)
      │
      ▼
app/model.pkl
      │
      ▼
app/predict.py (FastAPI)
      │
      ├──► logs/predictions.csv  (audit trail)
      └──► stdout (structured logging)
```

**CI/CD flow:**

```
git push main
      │
      ▼
GitHub Actions (ci.yml)        GitHub Actions (deploy.yml)
  install deps                   needs: test job to pass
  train model          ──────►   curl Render deploy hook
  pytest                               │
                                       ▼
                                 Render builds Docker image
                                 (trains model at build time)
                                 deploys new container
```

## Project Structure

```
churn-mlops/
├── app/
│   ├── train.py        # Model training + MLflow tracking
│   ├── predict.py      # FastAPI prediction API + logging
│   └── preprocess.py   # Data preprocessing utilities
├── data/
│   └── Churn.csv       # Training dataset
├── logs/
│   └── predictions.csv # Prediction audit trail (auto-created)
├── tests/
│   └── test_api.py     # API endpoint tests
├── .github/
│   └── workflows/
│       ├── ci.yml      # Run tests on push + PR
│       └── deploy.yml  # Test then deploy to Render on push to main
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

Trains a `RandomForestClassifier` on `data/Churn.csv` with `class_weight="balanced"` to handle the imbalanced dataset (~73% no-churn). Logs parameters, metrics, and the model to MLflow. Saves the model and decision threshold to `app/model.pkl`.

```bash
python app/train.py
```

**Model performance** (threshold = 0.35):

| Metric | Value |
|---|---|
| Recall (churn) | 0.83 |
| Precision (churn) | 0.49 |
| F1 Score | 0.62 |
| ROC-AUC | 0.83 |

A threshold of 0.35 is used instead of the default 0.5 — catching churners matters more than minimising false alarms in a retention use case.

To view MLflow experiment results:

```bash
mlflow ui
# Open http://localhost:5000
```

## Running the API

**Locally** (train first to generate `app/model.pkl`):

```bash
python app/train.py
uvicorn app.predict:app --host 0.0.0.0 --port 10000 --reload
```

**With Docker** (model is trained automatically during image build):

```bash
docker build -t churn-api .
docker run -p 10000:10000 churn-api
```

API available at `http://localhost:10000`.

## API Endpoints

### `GET /`

Health check.

**Response:**
```json
{"message": "Customer Churn Prediction API is running"}
```

### `POST /predict`

Predicts whether a customer will churn.

**Sample request:**
```bash
curl -X POST http://localhost:10000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "tenure": 12, "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
    "OnlineSecurity": 0, "OnlineBackup": 1, "DeviceProtection": 0,
    "TechSupport": 1, "StreamingTV": 1, "StreamingMovies": 0,
    "Contract": 2, "PaperlessBilling": 1, "PaymentMethod": 1,
    "MonthlyCharges": 70.5, "TotalCharges": 850.2
  }'
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
- `churn_probability`: raw model score (0–1)
- `threshold`: cutoff used for the binary decision

Every request is logged to stdout and appended to `logs/predictions.csv` with a UTC timestamp for audit and monitoring.

## Tests

```bash
pytest tests/
```

Tests cover the `/` and `/predict` endpoints using FastAPI's `TestClient`. Run `python app/train.py` first to generate the model file locally.

## CI/CD

Two GitHub Actions workflows:

**`ci.yml`** — runs on every push and pull request to `main`:
- Install dependencies → train model → run tests

**`deploy.yml`** — runs on push to `main`, after tests pass:
- Same test steps, then triggers a Render deploy via webhook

To enable deployment, add your Render deploy hook URL as a repository secret named `RENDER_DEPLOY_HOOK_URL`:
> GitHub repo → Settings → Secrets and variables → Actions → New repository secret

## Monitoring

Every prediction logs two things:

**Stdout (structured logs):**
```
2026-06-29 04:00:00 INFO Incoming prediction request: {'gender': 1, ...}
2026-06-29 04:00:00 INFO Prediction result: churn=1, probability=0.6183, threshold=0.35
```

**`logs/predictions.csv` (audit trail):**
```
timestamp,gender,SeniorCitizen,...,churn_probability,churn_prediction
2026-06-29T04:00:00.123456,1,0,...,0.6183,1
```
