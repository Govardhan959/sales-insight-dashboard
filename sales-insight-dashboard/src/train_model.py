"""
train_model.py

Purpose:
    Train and tune a sales forecasting model using time-aware
    cross-validation, then save the best model to disk.

Input:
    data/processed/daily_sales_features.csv

Output:
    models/sales_forecast_model.pkl
    models/feature_columns.pkl   (list of features, needed at prediction time)

Functions:
    - load_features(path) -> pd.DataFrame
    - time_based_split(df, test_size) -> tuple
    - get_feature_target_columns(df) -> tuple
    - train_baseline_model(X_train, y_train) -> model
    - train_random_forest(X_train, y_train) -> model
    - tune_xgboost(X_train, y_train) -> model
    - save_model(model, feature_cols, model_path, features_path)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sales_forecast_model.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

TARGET_COL = "total_sales"


def load_features(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    print(f"Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split chronologically: earliest (1 - test_size) rows for training,
    the most recent test_size rows for testing.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train period: {train_df['order_date'].min().date()} to {train_df['order_date'].max().date()} "
          f"({len(train_df)} rows)")
    print(f"Test period:  {test_df['order_date'].min().date()} to {test_df['order_date'].max().date()} "
          f"({len(test_df)} rows)")

    return train_df, test_df


def get_feature_target_columns(df: pd.DataFrame):
    """
    Feature columns = everything except the raw date and the target itself.
    We keep this as an explicit list so the Flask app can reconstruct the
    same input shape at prediction time.
    """
    
    exclude_cols = ["order_date", TARGET_COL]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols, TARGET_COL


def train_baseline_model(X_train, y_train):
    """Linear Regression — establishes the floor any real model must beat."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Random Forest — a strong, low-effort non-linear baseline."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def tune_xgboost(X_train, y_train):
    """
    Hyperparameter tuning using GridSearchCV with TimeSeriesSplit —
    NOT regular KFold, since regular KFold would shuffle time order
    across folds and leak future information into training folds.
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    base_model = XGBRegressor(random_state=42, objective="reg:squarederror")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best XGBoost params: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f}")

    return grid_search.best_estimator_


def save_model(model, feature_cols, model_path=MODEL_PATH, features_path=FEATURE_COLS_PATH):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, features_path)

    print(f"Model saved to: {model_path}")
    print(f"Feature column list saved to: {features_path}")


if __name__ == "__main__":
    df = load_features(FEATURES_PATH)
    train_df, test_df = time_based_split(df)
    feature_cols, target_col = get_feature_target_columns(df)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print("\n--- Training baseline (Linear Regression) ---")
    baseline_model = train_baseline_model(X_train, y_train)

    print("\n--- Training Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)

    print("\n--- Tuning XGBoost (this may take a minute) ---")
    xgb_model = tune_xgboost(X_train, y_train)

    # Final model selection happens in evaluate_model.py based on test-set metrics
    save_model(xgb_model, feature_cols)