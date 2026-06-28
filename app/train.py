import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score
)
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Churn.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
CHURN_THRESHOLD = 0.35

df = pd.read_csv(DATA_PATH)

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df.dropna(inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("customer-churn")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= CHURN_THRESHOLD).astype(int)

    accuracy  = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall    = recall_score(y_test, preds)
    f1        = f1_score(y_test, preds)
    roc_auc   = roc_auc_score(y_test, proba)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("churn_threshold", CHURN_THRESHOLD)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(model, name="model")

    joblib.dump({"model": model, "threshold": CHURN_THRESHOLD}, MODEL_PATH)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"Threshold: {CHURN_THRESHOLD}")
print(f"Model saved to {MODEL_PATH}")
