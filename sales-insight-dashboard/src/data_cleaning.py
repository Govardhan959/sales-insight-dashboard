"""
data_cleaning.py

Purpose:
    Load the raw Superstore dataset and clean it for analysis.

Input:
    data/raw/superstore_sales.csv

Output:
    data/processed/superstore_cleaned.csv
"""

from pathlib import Path
import pandas as pd


def load_raw_data(path):
    """Load the raw CSV file."""
    df = pd.read_csv(path, encoding="latin-1")
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def standardize_column_names(df):
    """Convert column names to lowercase with underscores."""
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["order_date", "ship_date"]:
        if col in df.columns:
            # Confirmed via diagnostic: raw dates are DD/MM/YYYY (day-first),
            # e.g. '15/04/2018' = 15th April 2018, not month 15.
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")
            n_failed = df[col].isna().sum()
            if n_failed > 0:
                print(f"Warning: {n_failed} rows in '{col}' failed to parse and became NaT")
    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Removed {before-after} duplicate rows")

    return df


def handle_missing_values(df):
    """Fill missing values."""

    print("\nMissing Values:")
    print(df.isnull().sum())

    if "postal_code" in df.columns:
        df["postal_code"] = (
            df["postal_code"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.replace(".0", "", regex=False)
        )

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def fix_data_types(df):
    """Correct data types."""

    df = df.copy()

    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].astype(str)

    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    return df


def remove_invalid_rows(df):
    """Remove rows having negative sales."""

    if "sales" in df.columns:

        before = len(df)

        df = df[df["sales"] >= 0]

        after = len(df)

        print(f"Removed {before-after} invalid sales rows")

    return df


def clean_pipeline(input_path, output_path):
    """Run the complete cleaning pipeline."""

    df = load_raw_data(input_path)

    df = standardize_column_names(df)

    df = parse_dates(df)

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = fix_data_types(df)

    df = remove_invalid_rows(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nCleaning Completed Successfully")
    print(f"Final Shape : {df.shape}")

    print(f"Saved to : {output_path}")

    return df


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    input_file = project_root / "data" / "raw" / "superstore_sales.csv"

    output_file = project_root / "data" / "processed" / "superstore_cleaned.csv"

    clean_pipeline(
        input_path=input_file,
        output_path=output_file
    )