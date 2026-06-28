from pathlib import Path
import sys

# Ensure project root is on sys.path so `app` package is importable during tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.predict import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "churn_prediction" in body
    assert "churn_probability" in body
    assert "threshold" in body
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0