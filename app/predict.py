from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

artifact = joblib.load("app/model.pkl")
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
    input_data = pd.DataFrame([data.model_dump()])
    churn_probability = model.predict_proba(input_data)[0][1]
    churn_prediction = int(churn_probability >= threshold)

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(float(churn_probability), 4),
        "threshold": threshold,
    }
