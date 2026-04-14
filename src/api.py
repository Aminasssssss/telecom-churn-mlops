import joblib
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Production ML API for predicting customer churn in telecom companies",
    version="1.0.0"
)

model = None
scaler = None
feature_names = None


def load_and_preprocess_reference():
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines"
    ]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    multi_cols = [
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    X = df.drop("Churn", axis=1)
    return X.columns.tolist(), X.iloc[0:1]


@app.on_event("startup")
def load_model():
    global model, scaler, feature_names
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names, _ = load_and_preprocess_reference()
    print(f"Model loaded: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")


class CustomerRaw(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: float = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 65.0
    TotalCharges: float = 780.0


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    recommendation: str


def preprocess_customer(customer: CustomerRaw) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder

    df = pd.DataFrame([customer.dict()])

    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                  "No phone service": 0, "No internet service": 0}

    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "MultipleLines"]:
        df[col] = df[col].map(lambda x: binary_map.get(x, 0))

    multi_cols = [
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    ref_cols, _ = load_and_preprocess_reference()
    for col in ref_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[ref_cols]
    return df


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Telecom Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "none",
        "features_count": len(feature_names) if feature_names else 0
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerRaw):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = preprocess_customer(customer)
    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0][1]
    prediction = bool(proba >= 0.5)

    if proba < 0.3:
        risk_level = "LOW"
        recommendation = "Customer is likely to stay. Continue standard engagement."
    elif proba < 0.6:
        risk_level = "MEDIUM"
        recommendation = "Customer shows churn signals. Consider a loyalty offer."
    else:
        risk_level = "HIGH"
        recommendation = "High churn risk. Immediate retention action required."

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": prediction,
        "risk_level": risk_level,
        "recommendation": recommendation
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "features_count": len(feature_names),
        "feature_names": feature_names
    }