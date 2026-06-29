import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from . import database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(lifespan=lifespan)

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = artifact["threshold"]


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

    database.log_prediction(
        timestamp=datetime.now(timezone.utc),
        input_data=data.model_dump(),
        churn_probability=round(float(churn_probability), 4),
        churn_prediction=churn_prediction,
    )

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(float(churn_probability), 4),
        "threshold": threshold,
    }
