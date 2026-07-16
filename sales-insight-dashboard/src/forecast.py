"""
forecast.py — matches feature set with total_orders removed
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sales_forecast_model.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
FORECAST_OUTPUT_PATH = PROJECT_ROOT / "reports" / "forecast_output.csv"

TARGET_COL = "total_sales"


def load_model_artifacts(model_path=MODEL_PATH, features_path=FEATURE_COLS_PATH):
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    print(f"Model loaded. Expects {len(feature_cols)} features: {feature_cols}")
    return model, feature_cols


def build_calendar_features(date: pd.Timestamp) -> dict:
    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.dayofweek,
        "day_of_year": date.dayofyear,
        "week_of_year": int(date.isocalendar().week),
        "quarter": date.quarter,
        "is_weekend": int(date.dayofweek in [5, 6]),
        "is_month_start": int(date.is_month_start),
        "is_month_end": int(date.is_month_end),
        "is_holiday_season": int(date.month in [11, 12]),
    }


def recursive_forecast(model, feature_cols: list, history_df: pd.DataFrame,
                        target_col: str = TARGET_COL, n_days: int = 30) -> pd.DataFrame:
    # Only ever track order_date + target_col — nothing else exists in this feature set
    history = history_df[["order_date", target_col]].copy()
    history = history.sort_values("order_date").reset_index(drop=True)

    last_date = history["order_date"].max()
    forecasts = []

    lags = [1, 7, 14, 30]
    windows = [7, 30]

    for step in range(1, n_days + 1):
        next_date = last_date + timedelta(days=step)
        row = build_calendar_features(next_date)

        for lag in lags:
            lag_date = next_date - timedelta(days=lag)
            match = history.loc[history["order_date"] == lag_date, target_col]
            row[f"{target_col}_lag_{lag}"] = match.values[0] if len(match) else np.nan

        for window in windows:
            window_start = next_date - timedelta(days=window)
            window_data = history.loc[
                (history["order_date"] >= window_start) & (history["order_date"] < next_date),
                target_col
            ]
            row[f"{target_col}_roll_mean_{window}"] = window_data.mean() if len(window_data) else np.nan
            row[f"{target_col}_roll_std_{window}"] = window_data.std() if len(window_data) else np.nan

        X_next = pd.DataFrame([row])[feature_cols]

        if X_next.isnull().any(axis=1).values[0]:
            raise ValueError(
                f"Missing feature values for {next_date.date()} — history window "
                f"doesn't cover enough prior days."
            )

        prediction = model.predict(X_next)[0]
        forecasts.append({"order_date": next_date, target_col: prediction})

        history = pd.concat(
            [history, pd.DataFrame([{"order_date": next_date, target_col: prediction}])],
            ignore_index=True
        )

    forecast_df = pd.DataFrame(forecasts)
    print(f"Generated {n_days}-day forecast from {forecast_df['order_date'].min().date()} "
          f"to {forecast_df['order_date'].max().date()}")
    return forecast_df


if __name__ == "__main__":
    model, feature_cols = load_model_artifacts()
    history_df = pd.read_csv(FEATURES_DATA_PATH, parse_dates=["order_date"])

    forecast_df = recursive_forecast(model, feature_cols, history_df, n_days=30)

    FORECAST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(FORECAST_OUTPUT_PATH, index=False)
    print(f"\nForecast saved to: {FORECAST_OUTPUT_PATH}")
    print(forecast_df.head(10))