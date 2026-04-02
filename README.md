# Telecom Customer Churn Prediction — MLOps Pipeline

Production-grade machine learning system for predicting customer churn in telecom companies.
Built with full MLOps stack: experiment tracking, REST API, containerization, and CI/CD.

## Business Impact

Telecom companies lose significant revenue to customer churn. This system predicts which customers
are likely to leave, enabling proactive retention actions.

| Metric | Value |
|--------|-------|
| Dataset | 7,043 telecom customers |
| Churn Rate | 26.5% |
| Monthly Revenue at Risk | ~$130,000 |
| Annual Revenue at Risk | ~$1,560,000 |
| Model reduction of 20% churn saves | ~$312,000/year |

## Results

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Gradient Boosting (champion) | 75.2% | 0.583 | **0.829** |
| LightGBM | 76.9% | 0.595 | 0.826 |
| XGBoost | 76.9% | 0.589 | 0.816 |
| Random Forest | 76.7% | 0.580 | 0.816 |
| Logistic Regression | 74.3% | 0.570 | 0.805 |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML | scikit-learn, XGBoost, LightGBM, SHAP |
| Data | pandas, NumPy, imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Visualization | Matplotlib, Seaborn |

## Project Structure

```
telecom-churn-mlops/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── tests/
│   └── test_api.py
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Aminasssssss/telecom-churn-mlops.git
cd telecom-churn-mlops
pip install -r requirements.txt
```

### 2. Train models

```bash
python src/train.py
```

### 3. Run API

```bash
uvicorn src.api:app --reload --port 8000
```

### 4. Open docs

```
http://localhost:8000/docs
```

### 5. Run tests

```bash
pytest tests/ -v
```

### 6. Docker

```bash
docker-compose up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| GET | `/model-info` | Model details |
| POST | `/predict` | Single prediction |

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Example Response

```json
{
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "recommendation": "High churn risk. Immediate retention action required."
}
```

## MLflow Tracking

```bash
mlflow ui --port 5000
```

Open http://localhost:5000 to view all experiments, metrics, and model versions.

## Key Insights from EDA

- Month-to-month contracts have 42% churn rate vs 3% for 2-year contracts
- New customers (low tenure) churn most — first 12 months are critical
- High monthly charges correlate strongly with churn
- Fiber optic internet users churn more than DSL users
- Senior citizens churn at higher rate than non-seniors

## Author

Zhumatayeva Amina — 2nd year Information Systems student, KBTU
<!-- v1 -->
<!-- v2 -->
<!-- done -->
<!-- v1 -->
<!-- v2 -->
