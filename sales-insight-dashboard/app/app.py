"""
app.py

Purpose:
    Flask entry point. Serves:
    - GET  /                 : form page to request a forecast
    - POST /predict          : handles form submission, renders results page
    - GET  /api/forecast     : JSON API endpoint for programmatic access

Input:
    Form data (n_days) or query param (?days=N for the API)

Output:
    Rendered HTML pages, or JSON for the API route

Functions:
    - index()
    - predict()
    - api_forecast()
"""

from flask import Flask, render_template, request, jsonify
from forecast_service import get_forecast

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render the form where users pick a forecast horizon."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission: run forecast, render results page."""
    try:
        n_days = int(request.form.get("n_days", 30))
        forecast_df = get_forecast(n_days)

        # Convert to a list of dicts for easy template rendering
        forecast_records = forecast_df.to_dict(orient="records")
        for record in forecast_records:
            record["order_date"] = record["order_date"].strftime("%Y-%m-%d")
            record["total_sales"] = round(record["total_sales"], 2)

        total = round(forecast_df["total_sales"].sum(), 2)
        average = round(forecast_df["total_sales"].mean(), 2)

        return render_template(
            "result.html",
            forecast=forecast_records,
            n_days=n_days,
            total=total,
            average=average
        )
    except ValueError as e:
        return render_template("index.html", error=str(e))


@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    """JSON API endpoint — e.g. /api/forecast?days=14"""
    try:
        n_days = int(request.args.get("days", 30))
        forecast_df = get_forecast(n_days)

        result = [
            {"date": row["order_date"].strftime("%Y-%m-%d"), "predicted_sales": round(row["total_sales"], 2)}
            for _, row in forecast_df.iterrows()
        ]
        return jsonify({"n_days": n_days, "forecast": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)