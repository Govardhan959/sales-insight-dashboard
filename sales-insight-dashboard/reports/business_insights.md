# Business Insights Report

_Generated from model predictions and historical trend analysis._

## 1. What Drives Sales (Model Feature Importance)

| Feature | Importance (%) |
|---|---|
| day_of_year | 29.600000381469727% |
| total_sales_roll_mean_7 | 14.899999618530273% |
| total_sales_lag_7 | 14.300000190734863% |
| total_sales_lag_14 | 12.0% |
| month | 7.900000095367432% |
| total_sales_roll_std_7 | 7.800000190734863% |
| total_sales_lag_30 | 7.199999809265137% |
| day | 6.300000190734863% |

## 2. 30-Day Sales Forecast Summary

- **Total forecasted sales (next 30 days):** 21,960.78
- **Average daily forecast:** 732.03
- **Peak day:** 2018-12-31 (1,803.32)
- **Lowest day:** 2019-01-10 (539.77)

## 3. Category Trends (Recent 6 Months vs Prior 6 Months)

| Category | Recent 6mo Sales | Prior 6mo Sales | Change % |
|---|---|---|---|
| Furniture | 95,158.77 | 49,786.82 | 91.1% |
| Office Supplies | 105,716.20 | 59,462.48 | 77.8% |
| Technology | 86,549.10 | 49,479.36 | 74.9% |

## 4. Recommendations

- **Furniture** shows the strongest recent growth — consider prioritizing inventory and marketing focus here.
- **Technology** shows the weakest recent trend — worth investigating whether this reflects market demand shifts or execution issues.
- The top predictive driver of daily sales is **day_of_year**, suggesting recent sales momentum matters more than calendar effects alone.