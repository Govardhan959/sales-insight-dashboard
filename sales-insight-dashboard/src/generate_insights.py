"""
generate_insights.py

Purpose:
    Generate a quantified business insights report by combining:
    - Model feature importances (what drives sales)
    - Forecast output (what's expected next)
    - Category/region trend analysis (growth or decline direction)

Input:
    models/sales_forecast_model.pkl
    models/feature_columns.pkl
    data/processed/superstore_preprocessed.csv
    reports/forecast_output.csv

Output:
    reports/business_insights.md

Functions:
    - get_feature_importance_summary(model, feature_cols) -> pd.DataFrame
    - get_forecast_summary(forecast_df) -> dict
    - get_category_trends(df) -> pd.DataFrame
    - generate_report(...)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sales_forecast_model.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
PREPROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "superstore_preprocessed.csv"
FORECAST_PATH = PROJECT_ROOT / "reports" / "forecast_output.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "business_insights.md"


def get_feature_importance_summary(model, feature_cols: list, top_n: int = 8) -> pd.DataFrame:
    """Rank features by importance — tells us WHAT drives sales, quantitatively."""
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    top = importance.head(top_n).reset_index()
    top.columns = ["feature", "importance"]
    top["importance_pct"] = (top["importance"] / top["importance"].sum() * 100).round(1)
    return top


def get_forecast_summary(forecast_df: pd.DataFrame) -> dict:
    """Summarize the 30-day forecast into headline numbers."""
    total_forecast = forecast_df["total_sales"].sum()
    avg_daily_forecast = forecast_df["total_sales"].mean()
    peak_day = forecast_df.loc[forecast_df["total_sales"].idxmax()]
    trough_day = forecast_df.loc[forecast_df["total_sales"].idxmin()]

    return {
        "total_forecast_30d": total_forecast,
        "avg_daily_forecast": avg_daily_forecast,
        "peak_date": peak_day["order_date"],
        "peak_value": peak_day["total_sales"],
        "trough_date": trough_day["order_date"],
        "trough_value": trough_day["total_sales"],
    }


def get_category_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each category's sales in the most recent 6 months vs the prior
    6 months — reveals growth/decline direction, not just totals.
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    max_date = df["order_date"].max()

    recent_start = max_date - pd.DateOffset(months=6)
    prior_start = max_date - pd.DateOffset(months=12)

    recent = df[df["order_date"] > recent_start].groupby("category")["sales"].sum()
    prior = df[(df["order_date"] > prior_start) & (df["order_date"] <= recent_start)].groupby("category")["sales"].sum()

    trend = pd.DataFrame({"recent_6mo": recent, "prior_6mo": prior}).fillna(0)
    trend["change_pct"] = ((trend["recent_6mo"] - trend["prior_6mo"]) / trend["prior_6mo"].replace(0, np.nan) * 100).round(1)
    trend = trend.sort_values("change_pct", ascending=False)

    return trend


def generate_report():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    df = pd.read_csv(PREPROCESSED_PATH, parse_dates=["order_date"])
    forecast_df = pd.read_csv(FORECAST_PATH, parse_dates=["order_date"])

    importance_summary = get_feature_importance_summary(model, feature_cols)
    forecast_summary = get_forecast_summary(forecast_df)
    category_trends = get_category_trends(df)

    report_lines = []
    report_lines.append("# Business Insights Report\n")
    report_lines.append(f"_Generated from model predictions and historical trend analysis._\n")

    report_lines.append("## 1. What Drives Sales (Model Feature Importance)\n")
    report_lines.append("| Feature | Importance (%) |")
    report_lines.append("|---|---|")
    for _, row in importance_summary.iterrows():
        report_lines.append(f"| {row['feature']} | {row['importance_pct']}% |")
    report_lines.append("")

    report_lines.append("## 2. 30-Day Sales Forecast Summary\n")
    report_lines.append(f"- **Total forecasted sales (next 30 days):** {forecast_summary['total_forecast_30d']:,.2f}")
    report_lines.append(f"- **Average daily forecast:** {forecast_summary['avg_daily_forecast']:,.2f}")
    report_lines.append(f"- **Peak day:** {forecast_summary['peak_date'].date()} "
                         f"({forecast_summary['peak_value']:,.2f})")
    report_lines.append(f"- **Lowest day:** {forecast_summary['trough_date'].date()} "
                         f"({forecast_summary['trough_value']:,.2f})")
    report_lines.append("")

    report_lines.append("## 3. Category Trends (Recent 6 Months vs Prior 6 Months)\n")
    report_lines.append("| Category | Recent 6mo Sales | Prior 6mo Sales | Change % |")
    report_lines.append("|---|---|---|---|")
    for cat, row in category_trends.iterrows():
        report_lines.append(f"| {cat} | {row['recent_6mo']:,.2f} | {row['prior_6mo']:,.2f} | {row['change_pct']}% |")
    report_lines.append("")

    report_lines.append("## 4. Recommendations\n")
    top_growth_cat = category_trends.index[0]
    top_decline_cat = category_trends.index[-1]
    report_lines.append(f"- **{top_growth_cat}** shows the strongest recent growth — consider prioritizing "
                         f"inventory and marketing focus here.")
    report_lines.append(f"- **{top_decline_cat}** shows the weakest recent trend — worth investigating "
                         f"whether this reflects market demand shifts or execution issues.")
    report_lines.append(f"- The top predictive driver of daily sales is **{importance_summary.iloc[0]['feature']}**, "
                         f"suggesting recent sales momentum matters more than calendar effects alone.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Business insights report saved to: {REPORT_PATH}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    generate_report()