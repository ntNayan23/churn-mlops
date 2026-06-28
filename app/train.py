import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

# Load data
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Churn.csv"
df = pd.read_csv(DATA_PATH)

# Drop unnecessary column
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow tracking
mlflow.set_experiment("customer-churn")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    # Save model to a file next to this script so path is correct
    MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
    joblib.dump(model, MODEL_PATH)

print(f"Model Accuracy: {accuracy}")
print(f"Model saved successfully to {MODEL_PATH}")