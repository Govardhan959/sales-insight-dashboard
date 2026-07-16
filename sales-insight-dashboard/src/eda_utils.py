"""
eda_utils.py

Purpose:
    Reusable plotting and summary functions for exploratory data analysis
    on the Superstore sales dataset.

Input:
    data/processed/superstore_preprocessed.csv (order-level, for categorical analysis)
    data/processed/daily_sales.csv (daily time series, for trend/seasonality analysis)

Functions:
    - plot_sales_distribution(df)
    - plot_daily_trend(daily_df)
    - plot_monthly_trend(daily_df)
    - plot_sales_by_category(df, group_col)
    - plot_yoy_comparison(daily_df)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_sales_distribution(df: pd.DataFrame):
    """Histogram + boxplot of order-level sales values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["sales"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution of Order Sales")
    axes[0].set_xlabel("Sales")

    axes[1].boxplot(df["sales"], vert=False)
    axes[1].set_title("Sales Boxplot (post outlier-capping)")
    axes[1].set_xlabel("Sales")

    plt.tight_layout()
    plt.show()


def plot_daily_trend(daily_df: pd.DataFrame):
    """Line plot of daily sales over the full date range."""
    plt.figure(figsize=(14, 5))
    plt.plot(daily_df["order_date"], daily_df["total_sales"], linewidth=0.8, color="teal")
    plt.title("Daily Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()


def plot_monthly_trend(daily_df: pd.DataFrame):
    """Aggregate to monthly totals and plot — smooths daily noise, reveals seasonality."""
    monthly = (
        daily_df.set_index("order_date")["total_sales"]
        .resample("ME")
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 5))
    plt.plot(monthly["order_date"], monthly["total_sales"], marker="o", color="darkorange")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

    return monthly


def plot_sales_by_category(df: pd.DataFrame, group_col: str):
    """Bar chart of total sales grouped by a categorical column (region, category, segment)."""
    summary = df.groupby(group_col)["sales"].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    summary.plot(kind="bar", color="slateblue")
    plt.title(f"Total Sales by {group_col.replace('_', ' ').title()}")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return summary


def plot_yoy_comparison(daily_df: pd.DataFrame):
    """Compare total sales across years — reveals growth or decline trend."""
    df = daily_df.copy()
    df["year"] = df["order_date"].dt.year
    yearly = df.groupby("year")["total_sales"].sum()

    plt.figure(figsize=(8, 5))
    yearly.plot(kind="bar", color="seagreen")
    plt.title("Year-over-Year Total Sales")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

    return yearly
if __name__ == "__main__":

    df = pd.read_csv("data/processed/superstore_preprocessed.csv")
    daily_df = pd.read_csv("data/processed/daily_sales.csv")

    daily_df["order_date"] = pd.to_datetime(daily_df["order_date"])

    plot_sales_distribution(df)
    plot_daily_trend(daily_df)
    plot_monthly_trend(daily_df)
    plot_sales_by_category(df, "category")
    plot_sales_by_category(df, "region")
    plot_sales_by_category(df, "segment")
    plot_yoy_comparison(daily_df)