import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app, load_model

load_model()

client = TestClient(app)

sample_customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0
}

loyal_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.0,
    "TotalCharges": 2700.0
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features_count" in data
    assert data["features_count"] == 29


def test_predict_returns_valid_response():
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "recommendation" in data


def test_predict_probability_range():
    response = client.post("/predict", json=sample_customer)
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_risk_levels():
    response = client.post("/predict", json=sample_customer)
    data = response.json()
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_loyal_customer_low_risk():
    response = client.post("/predict", json=loyal_customer)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_probability"] < 0.5
    assert data["churn_prediction"] is False


def test_high_risk_customer():
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["churn_prediction"], bool)