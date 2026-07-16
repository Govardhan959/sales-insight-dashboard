"""
evaluate_model.py

Purpose:
    Evaluate trained models on the held-out (most recent) test period
    using MAE, RMSE, and MAPE, and compare against the baseline.

Input:
    models/sales_forecast_model.pkl
    data/processed/daily_sales_features.csv

Output:
    Printed comparison table of model performance.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_model import (
    load_features, time_based_split, get_feature_target_columns,
    train_baseline_model, train_random_forest
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sales_forecast_model.pkl"


def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE — expressed as a percentage, which is the metric most
    business stakeholders actually understand ("we're off by 8%"
    means more to a manager than "MAE is 412.5").
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero on zero-sales days
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(model, X_test, y_test, model_name: str):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)

    print(f"\n{model_name}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")

    return {"model": model_name, "mae": mae, "rmse": rmse, "mape": mape}


if __name__ == "__main__":
    df = load_features(FEATURES_PATH)
    train_df, test_df = time_based_split(df)
    feature_cols, target_col = get_feature_target_columns(df)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    results = []

    baseline_model = train_baseline_model(X_train, y_train)
    results.append(evaluate(baseline_model, X_test, y_test, "Linear Regression (baseline)"))

    rf_model = train_random_forest(X_train, y_train)
    results.append(evaluate(rf_model, X_test, y_test, "Random Forest"))

    xgb_model = joblib.load(MODEL_PATH)
    results.append(evaluate(xgb_model, X_test, y_test, "XGBoost (tuned, saved model)"))

    print("\n--- Comparison Summary ---")
    results_df = pd.DataFrame(results)
    print(results_df.sort_values("mae"))