"""
feature_engineering.py

Purpose:
    Build the full feature set for sales forecasting from the clean
    daily sales time series produced in Module 3.

Input:
    data/processed/daily_sales.csv

Output:
    data/processed/daily_sales_features.csv

Functions:
    - load_daily_sales(path) -> pd.DataFrame
    - add_calendar_features(df) -> pd.DataFrame
    - add_lag_features(df, target_col, lags) -> pd.DataFrame
    - add_rolling_features(df, target_col, windows) -> pd.DataFrame
    - handle_feature_nulls(df) -> pd.DataFrame
    - feature_engineering_pipeline(input_path, output_path) -> pd.DataFrame
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_SALES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales.csv"
FEATURES_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"

TARGET_COL = "total_sales"


def load_daily_sales(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    print(f"Loaded daily sales: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar-based features from order_date.

    Why these matter:
        Retail sales are strongly driven by calendar effects — weekends,
        month-end payroll cycles, and holiday seasons (Nov-Dec spikes,
        seen in Module 4's EDA) are all captured through these features
        without needing external data.
    """
    df = df.copy()
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day"] = df["order_date"].dt.day
    df["day_of_week"] = df["order_date"].dt.dayofweek       # 0=Monday, 6=Sunday
    df["day_of_year"] = df["order_date"].dt.dayofyear
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["order_date"].dt.quarter

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["order_date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["order_date"].dt.is_month_end.astype(int)

    # Holiday season flag — informed directly by the Q4 spike seen in EDA (Module 4)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    print("Added calendar features: year, month, day, day_of_week, day_of_year, "
          "week_of_year, quarter, is_weekend, is_month_start, is_month_end, is_holiday_season")
    return df



def add_lag_features(df, target_col="total_sales", lags=[1,7,14,30]):

    df = df.copy()

    # Lag sales
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # Lag total orders only
    if "total_orders" in df.columns:
        for lag in lags:
            df[f"total_orders_lag_{lag}"] = df["total_orders"].shift(lag)

    print("Lag features added.")

    return df

def add_rolling_features(df: pd.DataFrame, target_col: str = TARGET_COL, windows: list = [7, 30]) -> pd.DataFrame:
    """
    Create rolling mean and std features.

    Why rolling features matter:
        A single lag value can be noisy (e.g., one unusually high day).
        A rolling mean smooths that noise and captures the recent trend
        level, while rolling std captures recent volatility — both are
        useful signals a model can't get from a single lag alone.

    IMPORTANT: shift(1) before rolling — this ensures the rolling window
    only uses PAST data relative to the current row, never including
    today's own value. Without this shift, the feature would leak the
    target into itself (data leakage), producing artificially perfect
    but meaningless model performance.
    """
    df = df.copy()
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df[target_col].shift(1).rolling(window=window).mean()
        )
        df[f"{target_col}_roll_std_{window}"] = (
            df[target_col].shift(1).rolling(window=window).std()
        )

    print(f"Added rolling mean/std features for windows: {windows}")
    return df


def handle_feature_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag and rolling features create NaNs at the start of the series
    (e.g., lag_30 has no value for the first 30 rows — there's no data
    30 days before day 1). We drop these rows rather than fill them,
    since fabricating early lag values would introduce fake signal.
    """
    df = df.copy()
    before = df.shape[0]
    df = df.dropna().reset_index(drop=True)
    after = df.shape[0]
    print(f"Dropped {before - after} rows with NaN (from lag/rolling window warm-up period)")
    return df


def feature_engineering_pipeline(input_path=DAILY_SALES_PATH, output_path=FEATURES_OUTPUT_PATH) -> pd.DataFrame:
    df = load_daily_sales(input_path)

    # Drop total_orders entirely — not needed for sales forecasting,
    # and keeping it would require forecasting order counts too, which
    # is out of scope for this project's goal.
    if "total_orders" in df.columns:
        df = df.drop(columns=["total_orders"])
        print("Dropped 'total_orders' column — out of scope for sales forecasting")

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = handle_feature_nulls(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nFeature-engineered data saved to: {output_path}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFinal columns: {df.columns.tolist()}")
    return df
    df = handle_feature_nulls(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFeature-engineered data saved to: {output_path}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    feature_engineering_pipeline()