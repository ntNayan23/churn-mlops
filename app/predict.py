from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("app/model.pkl")


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
    # Use Pydantic V2 API `model_dump()` instead of deprecated `dict()`
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)

    return {
        "churn_prediction": int(prediction[0])
    }