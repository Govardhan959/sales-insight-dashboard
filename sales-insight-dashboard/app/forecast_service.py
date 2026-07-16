"""
forecast_service.py

Purpose:
    Load the trained model once at app startup (not per-request, which
    would be slow and wasteful), and expose a simple forecast() function
    for app.py to call.

Input:
    models/sales_forecast_model.pkl
    models/feature_columns.pkl
    data/processed/daily_sales_features.csv

Output:
    In-memory forecast results (returned to app.py, not saved to disk here)
"""

import sys
from pathlib import Path

# Allow importing from src/ regardless of where Flask is launched from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
import joblib
from forecast import recursive_forecast, build_calendar_features 

MODEL_PATH = PROJECT_ROOT / "models" / "sales_forecast_model.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"

# Load once at import time — this module is imported exactly once by app.py,
# so the model stays in memory across all incoming requests instead of
# being reloaded from disk on every single prediction (which would be slow).
_model = joblib.load(MODEL_PATH)
_feature_cols = joblib.load(FEATURE_COLS_PATH)
_history_df = pd.read_csv(FEATURES_DATA_PATH, parse_dates=["order_date"])

print(f"[forecast_service] Model and {len(_feature_cols)} features loaded into memory.")


def get_forecast(n_days: int = 30) -> pd.DataFrame:
    """
    Run the recursive forecast for n_days ahead using the pre-loaded
    model and history. Returns a DataFrame with order_date and total_sales.
    """
    if n_days < 1 or n_days > 90:
        raise ValueError("n_days must be between 1 and 90")

    forecast_df = recursive_forecast(
        model=_model,
        feature_cols=_feature_cols,
        history_df=_history_df,
        n_days=n_days
    )
    return forecast_df