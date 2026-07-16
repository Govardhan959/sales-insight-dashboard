"""
preprocessing.py

Purpose:
    Prepare the cleaned Superstore dataset for analysis and forecasting.

Input:
    data/processed/superstore_cleaned.csv

Output:
    data/processed/superstore_preprocessed.csv
    data/processed/daily_sales.csv
"""

from pathlib import Path
import pandas as pd


def load_cleaned_data(path):
    """Load cleaned dataset."""

    df = pd.read_csv(
        path,
        parse_dates=["order_date", "ship_date"]
    )

    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def treat_outliers(df):
    """Cap outliers in Sales using the IQR method."""

    df = df.copy()

    q1 = df["sales"].quantile(0.25)
    q3 = df["sales"].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((df["sales"] < lower) | (df["sales"] > upper)).sum()

    print(f"Sales Outliers Capped : {outliers}")

    df["sales"] = df["sales"].clip(lower=lower, upper=upper)

    return df


def standardize_categorical_columns(df):
    """Standardize text columns."""

    df = df.copy()

    columns = [
        "segment",
        "country",
        "city",
        "state",
        "region",
        "category",
        "sub_category",
        "ship_mode"
    ]

    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
            )

    return df


def create_daily_sales(df):
    """Aggregate daily sales."""

    daily = (
        df.groupby("order_date")
        .agg(
            total_sales=("sales", "sum"),
            total_orders=("order_id", "nunique")
        )
        .reset_index()
        .sort_values("order_date")
    )

    all_dates = pd.date_range(
        daily["order_date"].min(),
        daily["order_date"].max(),
        freq="D"
    )

    daily = (
        daily
        .set_index("order_date")
        .reindex(all_dates)
        .fillna(0)
        .rename_axis("order_date")
        .reset_index()
    )

    print(f"Daily Sales Table : {daily.shape[0]} Days")

    return daily


def preprocessing_pipeline(input_path, output_path, daily_output_path):
    """Complete preprocessing pipeline."""

    df = load_cleaned_data(input_path)

    df = treat_outliers(df)

    df = standardize_categorical_columns(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nSaved : {output_path}")

    daily = create_daily_sales(df)

    daily.to_csv(daily_output_path, index=False)

    print(f"Saved : {daily_output_path}")

    return df, daily


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    input_file = project_root / "data" / "processed" / "superstore_cleaned.csv"

    output_file = project_root / "data" / "processed" / "superstore_preprocessed.csv"

    daily_file = project_root / "data" / "processed" / "daily_sales.csv"

    preprocessing_pipeline(
        input_path=input_file,
        output_path=output_file,
        daily_output_path=daily_file
    )