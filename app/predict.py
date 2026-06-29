import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = artifact["threshold"]

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "predictions.csv"
LOG_PATH.parent.mkdir(exist_ok=True)
_csv_lock = threading.Lock()

if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges",
            "churn_probability", "churn_prediction",
        ])


class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    logger.info(f"Incoming prediction request: {data.model_dump()}")

    input_data = pd.DataFrame([data.model_dump()])
    churn_probability = model.predict_proba(input_data)[0][1]
    churn_prediction = int(churn_probability >= threshold)

    logger.info(
        f"Prediction result: churn={churn_prediction}, "
        f"probability={churn_probability:.4f}, threshold={threshold}"
    )

    with _csv_lock:
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                *data.model_dump().values(),
                round(float(churn_probability), 4),
                churn_prediction,
            ])

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(float(churn_probability), 4),
        "threshold": threshold,
    }
