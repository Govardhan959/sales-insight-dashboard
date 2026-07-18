# Project Report: Sales Insight Dashboard with ML Forecasting

**Author:** _Your Name_
**Program:** B.Tech Artificial Intelligence and Machine Learning, 7th Semester
**Project Type:** End-to-end Data Science / ML Engineering Portfolio Project

---

## 1. Executive Summary

This project builds a complete pipeline that takes raw retail transaction data and turns it into two usable products: a **Power BI dashboard** for descriptive analytics (what happened, broken down by region, category, and time), and a **Flask web application** serving a trained **XGBoost forecasting model** (what's expected to happen next, for a 1–90 day horizon).

The project was built to demonstrate the full data science lifecycle expected in an industry role — not just model training, but data cleaning, feature engineering, evaluation methodology, deployment, and documentation — using the Kaggle Superstore Sales dataset.

---

## 2. Problem Statement

Retail businesses generate large volumes of transactional data but often lack two things:
1. A self-service way for non-technical stakeholders to explore sales performance across dimensions (region, category, time).
2. A forward-looking view — what sales are likely to look like in the near future, to support inventory and staffing decisions.

**Objective:** Build a system that addresses both gaps using a single, clean data pipeline as the shared foundation.

**Success criteria defined at the outset:**
- A forecasting model that meaningfully outperforms a naive baseline (Linear Regression) on held-out, chronologically later data.
- A dashboard that lets a business user answer "how are we doing, by region/category/time" without writing any code or queries.
- A deployed, working application — not just a notebook — that any hiring manager could actually open and use.

---

## 3. Dataset

- **Source:** Kaggle Superstore Sales Dataset
- **Rows:** 9,800 order-level transactions
- **Date range:** 2015–2018 (approx.)
- **Columns:** Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name, Segment, Country, City, State, Postal Code, Region, Category, Sub-Category, Product ID, Product Name, Sales

**Important scoping note:** This particular dataset variant does **not** include Profit, Discount, or Quantity columns, which some other Superstore variants do. This was discovered during development (documented in Section 6) and the project scope was adjusted accordingly — all analysis, feature engineering, and forecasting are built around `Sales` as the sole target/quantity metric. This is disclosed here explicitly rather than glossed over, since an interviewer cross-referencing the well-known dataset would otherwise notice the mismatch.

---

## 4. Methodology

### 4.1 Data Cleaning
- Fixed file encoding (`latin-1`, not the default `utf-8`).
- Corrected date parsing: raw dates were in `DD/MM/YYYY` format, not `MM/DD/YYYY` — discovered via diagnostic inspection after an initial parse produced ~60% failures (see Section 6).
- Removed duplicate rows, standardized column naming (snake_case), handled missing postal codes.

### 4.2 Preprocessing
- Treated outliers in `Sales` using the IQR method (capping, not deletion — preserves total revenue signal).
- Aggregated order-level rows into a **daily time series**, since forecasting requires one row per time step, not a transaction log.
- Reindexed the daily series to a continuous, gap-free calendar range (missing days filled with zero sales), which is required for reliable lag/rolling feature computation and for most forecasting methods generally.

### 4.3 Exploratory Data Analysis
- Confirmed a right-skewed sales distribution, typical of retail data.
- Identified Q4 (Nov–Dec) seasonality, consistent with expected holiday retail patterns.
- Broke down sales by Region, Category, and Segment to identify top and bottom performers.

### 4.4 Feature Engineering
- **Calendar features:** year, month, day, day-of-week, quarter, is_weekend, is_month_start/end, is_holiday_season.
- **Lag features:** sales from 1, 7, 14, and 30 days prior — the single strongest predictor class in most retail forecasting problems.
- **Rolling window features:** 7-day and 30-day rolling mean/std, computed with `.shift(1)` applied before `.rolling()` to strictly prevent same-day data leakage into the feature itself.

### 4.5 Model Development
- **Baseline:** Linear Regression, to establish a performance floor.
- **Candidate models:** Random Forest, XGBoost (hyperparameter-tuned via `GridSearchCV`).
- **Validation strategy:** `TimeSeriesSplit` used throughout — standard k-fold cross-validation was deliberately avoided since it would shuffle time order across folds and leak future information into training data, which does not reflect how the model would ever actually be used.
- **Train/test split:** chronological — the model trains only on the earlier ~80% of the timeline and is evaluated on the most recent ~20%, mirroring real deployment conditions.

### 4.6 Forecasting
- Implemented **recursive multi-step forecasting**: since lag/rolling features for future dates cannot reference real data that hasn't happened yet, each day's prediction is fed back into the feature set as if it were an actual value, to generate the next day's prediction.
- **Backtesting:** validated the recursive approach against the last known 30 days (treated as if unknown), explicitly measuring how prediction error grows with forecast horizon — reported honestly rather than presenting a single blended accuracy number.

### 4.7 Business Insight Generation
- Automated extraction of model feature importances (what actually drives predicted sales).
- Automated category-level trend comparison (recent 6 months vs. prior 6 months) to surface growth/decline direction, not just totals.
- Combined into a structured, non-technical report intended to be read by business stakeholders, not just engineers.

### 4.8 Power BI Dashboard
- Built a dedicated Date table with `CALENDAR()` and marked it explicitly as a date table, required for time-intelligence DAX functions like `SAMEPERIODLASTYEAR`.
- Built measures using `DIVIDE()` throughout (not the raw `/` operator) to avoid divide-by-zero errors across filtered contexts.
- Four pages: Overview (KPIs + monthly trend), Regional Analysis (map + bar), Category Analysis (treemap + top sub-categories), Trend Analysis (daily trend + YoY growth).
- Slicers (Region, Category, Segment, Date) synced across all pages.

### 4.9 Flask Web Application
- Model loaded once at application startup (not per-request) for performance.
- Two consumption modes: an HTML form/results UI for human users, and a `/api/forecast` JSON endpoint for programmatic access.
- Input validation on forecast horizon (1–90 days) with clear error handling.

### 4.10 Deployment
- Local: served via **Waitress** (a production WSGI server), not Flask's built-in development server, which is explicitly unsuitable for anything beyond local testing.
- Public: deployed to **Render** using a `Procfile`, with all file paths built relative to `Path(__file__)` rather than hardcoded, so the same code runs identically across local Windows and Render's Linux environment.

---

## 5. Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression (baseline) | _fill in from your evaluate_model.py output_ | _fill in_ | _fill in_ |
| Random Forest | _fill in_ | _fill in_ | _fill in_ |
| XGBoost (tuned) | _fill in_ | _fill in_ | _fill in_ |

**Top predictive features** (from feature importance analysis): _fill in from generate_insights.py output — typically dominated by short-term lag and rolling mean features, with calendar/seasonality features contributing secondarily._

**Forecast horizon behavior:** Backtesting showed prediction error _increasing / stable_ (fill in based on your actual backtest plot) as the forecast horizon extended from 1 to 30 days, consistent with the expected behavior of recursive forecasting where early-step errors compound into later steps.

---

## 6. Challenges Encountered and Resolutions

Documenting real problems and how they were solved — this is often more valuable in an interview than a project that "just worked."

| Challenge | Root Cause | Resolution |
|---|---|---|
| `FileNotFoundError` when running scripts via VS Code's Run button | Code executed with a different working directory than expected, breaking relative paths | Rebuilt all path references using `Path(__file__).resolve().parent`, making scripts location-independent |
| Wrong Kaggle dataset variant initially used (13 columns, no dates) | Downloaded a reduced/different Superstore variant without checking columns first | Re-downloaded the correct variant with `Order Date` and `Ship Date`, added an explicit column-validation check at load time to fail fast and clearly in the future |
| ~60% of dates failed to parse (`NaT`) | Assumed `MM/DD/YYYY` format; actual data was `DD/MM/YYYY` (confirmed via direct inspection of raw values) | Switched to explicit `format="%d/%m/%Y"` after diagnosing actual raw string values rather than guessing |
| `KeyError: 'total_orders'` during recursive forecasting | An extra aggregated column (`total_orders`) was included as a raw same-day feature during training, but forecasting code had no way to predict its future value | Removed `total_orders` from the feature set entirely — it wasn't essential to the sales forecasting objective, and keeping it would have required either forecasting it independently or accepting a same-day data leakage risk |
| Flask app failed with `ModuleNotFoundError: No module named 'flask'` via one execution method but succeeded via another | VS Code's Code Runner extension used a different Python interpreter than the activated virtual environment | Standardized on running Flask apps from an activated terminal rather than a generic "Run" button — the correct practice for any long-running server process |

---

## 7. Key Design Decisions (Why, Not Just What)

- **Capping outliers instead of deleting them:** preserves genuine high-value transactions rather than discarding real revenue signal.
- **Separate order-level and daily-aggregated tables:** Power BI needs order-level granularity for drill-down; forecasting needs a clean daily time index. Conflating the two would compromise both.
- **`shift(1)` before every rolling computation:** the single most important anti-leakage safeguard in the entire feature engineering step; without it, the model would appear artificially accurate during training/testing while being unusable in real forecasting.
- **`TimeSeriesSplit` over standard k-fold everywhere:** any temporal data must never be validated with a method that can train on "future" folds relative to a test fold.
- **Waitress over Flask's dev server for anything beyond local testing:** the development server is explicitly documented as unsuitable for production and lacks proper concurrency handling.

---

## 8. Limitations

- The dataset lacks Profit, Discount, and Quantity — this project could not include margin-based insights, which a complete retail analytics solution would ideally cover.
- Recursive forecasting error compounds over longer horizons — the model is more reliable for near-term (1–7 day) forecasts than for the full 30–90 day range, and this is reported transparently rather than hidden.
- No external features (e.g., holiday calendars, promotional campaign data, macroeconomic indicators) were available or incorporated — these would likely improve forecast accuracy if accessible.
- The Flask API has no authentication layer, which would be a requirement before any real production use beyond a portfolio demonstration.

---

## 9. Future Enhancements

- Incorporate external holiday/promotion calendars as additional features.
- Add prediction intervals (not just point forecasts) to communicate model uncertainty explicitly.
- Automate scheduled model retraining as new transaction data becomes available.
- Add basic authentication/rate-limiting to the Flask API before any production-adjacent use.
- Extend Power BI with a drill-through page connecting directly to the ML forecast output for a unified view.

---

## 10. Conclusion

This project demonstrates a complete, defensible data science workflow: careful data cleaning grounded in actual diagnostic investigation (not assumptions), leakage-aware feature engineering, time-series-appropriate model validation, honest reporting of forecast reliability limits, and a working deployed application accessible to non-technical users. The debugging process documented in Section 6 reflects genuine engineering practice — diagnosing root causes from actual error output and raw data inspection, rather than guessing at fixes.
