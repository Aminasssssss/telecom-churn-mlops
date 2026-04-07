import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "models"
MLFLOW_EXPERIMENT = "telecom-churn-mlops"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_preprocess(path: str):
    df = pd.read_csv(path)

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
    y = df["Churn"]

    return X, y


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            verbose=-1
        ),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


def train_and_log(model_name, model, X_train, y_train, X_test, y_test, feature_names):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, artifact_path="model")

        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)

        print(f"\n{model_name}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

        return metrics


def save_shap(model, X_test, model_name, feature_names):
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test[:100])

        shap_df = pd.DataFrame(
            shap_values.values,
            columns=feature_names
        )
        shap_path = os.path.join(MODEL_DIR, f"{model_name}_shap.csv")
        shap_df.to_csv(shap_path, index=False)
        print(f"  SHAP values saved to {shap_path}")
    except Exception as e:
        print(f"  SHAP skipped for {model_name}: {e}")


def main():
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess(DATA_PATH)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Churn rate in train: {y_train.mean():.3f}")

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    models = get_models()
    results = {}

    print("\nTraining models...")
    for name, model in models.items():
        metrics = train_and_log(
            name, model,
            X_train_scaled, y_train_res,
            X_test_scaled, y_test,
            feature_names
        )
        results[name] = metrics

    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = joblib.load(os.path.join(MODEL_DIR, f"{best_model_name}.pkl"))

    print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
    print("\nGenerating SHAP explanations for best model...")
    save_shap(best_model, X_test_scaled, best_model_name, feature_names)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    print(f"\nBest model saved to {MODEL_DIR}/best_model.pkl")

    print("\nAll results:")
    results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    print(results_df.round(4))


if __name__ == "__main__":
    main()