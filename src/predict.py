import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    customer_ids = None
    if "customerID" in df.columns:
        customer_ids = df["customerID"].values
        df.drop("customerID", axis=1, inplace=True)

    if "Churn" in df.columns:
        df.drop("Churn", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)

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

    return df, customer_ids


def align_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else df.columns.tolist()

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]
    return df


def predict_churn(input_path: str, output_path: str = None):
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"Loading data from {input_path}...")
    df, customer_ids = load_and_preprocess(input_path)

    df = align_features(df, scaler)
    df_scaled = scaler.transform(df)

    probas = model.predict_proba(df_scaled)[:, 1]
    predictions = (probas >= 0.5).astype(int)

    results = pd.DataFrame({
        "customer_id": customer_ids if customer_ids is not None else range(len(df)),
        "churn_probability": probas.round(4),
        "churn_prediction": predictions,
        "risk_level": pd.cut(
            probas,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"]
        )
    })

    print("\n=== PREDICTION RESULTS ===")
    print(f"Total customers: {len(results)}")
    print(f"High risk:   {(results['risk_level'] == 'HIGH').sum()}")
    print(f"Medium risk: {(results['risk_level'] == 'MEDIUM').sum()}")
    print(f"Low risk:    {(results['risk_level'] == 'LOW').sum()}")
    print(f"Predicted churn rate: {predictions.mean():.1%}")

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("\nTop 10 highest risk customers:")
        print(results.sort_values("churn_probability", ascending=False).head(10).to_string(index=False))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch churn prediction")
    parser.add_argument("--input", type=str, default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                        help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/predictions.csv",
                        help="Path to output CSV file")
    args = parser.parse_args()

    predict_churn(args.input, args.output)