# Restaurant Traffic Forecasting

## Overview

Restaurant Traffic Forecasting is a time-series forecasting pipeline for predicting daily guest count at the restaurant level for the Golden Gate (GGG) restaurant chain. The project was developed by **CADS - FPT** as part of a demand forecasting engagement.

The pipeline combines sales-data preprocessing, restaurant eligibility filtering, feature engineering, restaurant behavior clustering, and XGBoost-based forecasting to produce a next-month daily traffic forecast for each eligible restaurant.

> **Note:** This repository was revised from an internal forecasting project and cleaned for portfolio sharing. Internal data, HDFS paths, and production configurations are not included.

## Project Background

- **Client:** Golden Gate (GGG) — a major restaurant chain in Vietnam
- **Developer:** CADS - FPT
- **Problem:** Golden Gate needed daily guest-count forecasts across 450+ restaurants to support operational planning (staffing, inventory, revenue projection)
- **Data scope:** 19 internal data tables spanning sales, restaurant metadata, customers, bookings, promotions, and ratings; external data sources include weather and social ratings
- **Forecast horizon:** Next-month daily prediction (D+1 to D+30/31) per restaurant


## Problem Statement

Predict the number of customers visiting each Golden Gate restaurant in the next month.

- **Input:** Daily restaurant sales and traffic data from January 2021 onward, including guest count, total sales, loyalty sales, bill count, restaurant metadata, calendar, lunar calendar, and holiday features.
- **Output:** Daily guest-count forecast for each eligible restaurant in the next forecast month.
- **Forecasting level:** Restaurant-level daily prediction (D+1 to D+30/31).


## What This Project Does

- Loads raw sales, restaurant, calendar, lunar-calendar, and holiday-related data.
- Cleans and preprocesses restaurant-level daily guest-count records.
- Removes invalid and abnormal traffic observations.
- Fills missing guest-count labels using same-weekday historical values and annual restaurant-level averages.
- Creates time-series, calendar, holiday, lunar-date, restaurant, lag, and rolling-statistics features.
- Clusters restaurants by weekly seasonality and traffic volatility into 5 behavior groups.
- Splits restaurants into two forecasting groups:
  - **Global:** lower-complexity restaurants with clearer weekly patterns and lower traffic volatility.
  - **Local:** higher-complexity restaurants with weaker weekly patterns and higher traffic volatility.
- Trains and applies XGBoost models for each group.
- Aggregates Global and Local predictions into a final output file.

## Forecasting Approach

### Rolling Monthly Training

For each forecast month `T`, the model trains on historical data from January 2022 through month `T-1` and predicts daily guest count for month `T`. The validation period used for model selection and parameter tuning is July 2022 – December 2023 (18 months).

### Restaurant Segmentation

Restaurants are clustered into 5 behavior groups based on two dimensions:

1. **Weekly periodicity (`weekly_pc`)** — measures how strongly guest traffic follows a weekly pattern, calculated via Fourier analysis.
2. **Traffic fluctuation score** — measures traffic volatility using statistical properties (standard deviation, mean).

The 5 clusters are then grouped into two model families:

| Model | Characteristics |
|---|---|
| **Global** | Clearer weekly seasonality, lower traffic volatility, lower complexity |
| **Local** | Weaker weekly patterns, higher volatility, higher complexity |

### Model

**XGBoost Regressor** for structured time-series forecasting with two configurations:

#### Global Model

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.5
)
```

#### Local Model

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.6
)
```

Other parameters use XGBoost defaults.

### Model Optimization

Compared to the initial pilot model, the optimized pipeline achieved:

- **90% reduction** in the number of features used
- **70% reduction** in model training time
- Improved preprocessing logic: updated restaurant eligibility rules, better handling of missing and abnormal data
- Restaurant clustering replaced manual grouping for data-driven segmentation

## Feature Engineering

### Restaurant Features

- `cluster`: restaurant behavior cluster (weekly periodicity + traffic volatility).

### Calendar & Time Features

- `dayofweek`, `weekend`, `month`, `year`, day of month, and standard calendar attributes.

### Holiday Features

- Holiday indicator, pre-holiday indicator, and restaurant-level holiday weight.

`holiday_weight` quantifies how much a specific holiday affects guest count per restaurant. It is computed by comparing holiday traffic against average traffic on comparable weekdays, removing outliers via z-score filtering, and filling missing values with brand-level medians.

### Lunar Calendar Features

- Beginning-of-month and mid-month lunar date indicators.

### Lag Features

Historical guest-count values: previous 1–9 days and longer-cycle lags (14, 21, 28, 35, 42, 49, 56, 64 days).

### Statistical Features

Mean, median, minimum, and maximum of guest count computed over recent historical windows before the forecast month.

## Data Preprocessing

1. **Invalid values** — removes rows with negative, zero, or null `guest_count`.
2. **Abnormal values** — detects and removes outlier observations per restaurant using Prophet-based anomaly handling.
3. **Missing observations** — builds a complete restaurant-date frame, fills missing `guest_count` with the same weekday from the previous week, then falls back to the restaurant's annual average if still missing.
4. **Restaurant eligibility** — filters by operating status, open/close dates, location, brand, and historical data sufficiency.

## Evaluation

### Metrics

Absolute Percentage Error-based metrics, with **WAPE as the primary evaluation metric**:

```
MAPE = mean(|y_true - y_pred| / y_true) × 100
WAPE = sum(|y_true - y_pred|) / sum(y_true) × 100
```

### Evaluation Scope

- **Period:** July 2022 – December 2023 (18 months)
- **Scale:** 450+ restaurants per month (470+ in some months)
- **Comparison:** Three-way benchmark — GGG internal forecast vs. FPT pilot (old model) vs. FPT opt (new model)

### Restaurant Groups

- **Baseline:** restaurants open ≥ 1 year as of February 28, 2023.
- **Non-baseline:** restaurants open < 1 year as of February 28, 2023.

## Performance

### Overall WAPE Results (18-month evaluation period)

| Forecast Level | GGG | FPT Pilot | FPT Opt |
|---|---:|---:|---:|
| Daily | 27.1% | 25.3% | **22.0%** |
| Weekly | 16.4% | 17.5% | **13.6%** |
| Monthly | 12.2% | 15.5% | **10.9%** |

### WAPE by Restaurant Group

| Forecast Level | Group | GGG | FPT Pilot | FPT Opt |
|---|---|---|---|---|
| Daily | Baseline | 25.6% | 24.6% | **21.3%** |
| Daily | Non-baseline | 33.8% | 29.0% | **25.3%** |
| Weekly | Baseline | 15.5% | 16.7% | **12.8%** |
| Weekly | Non-baseline | 20.3% | 21.0% | **16.9%** |
| Monthly | Baseline | 11.5% | 14.9% | **10.3%** |
| Monthly | Non-baseline | 15.5% | 18.2% | **13.6%** |

**Key findings:**

- The optimized model improved daily WAPE by **~13%** vs. the old model (3 percentage points) and by **~19%** vs. GGG's internal forecast (5 percentage points).
- The largest gains were on the **non-baseline** group (~25% improvement over GGG), where sparse history made forecasting harder.
- On weekly and monthly aggregates, the optimized model consistently outperformed both GGG and the pilot model.

### Performance by Day Type (Daily WAPE)

| Day Type | GGG | FPT Pilot | FPT Opt | Improvement vs. GGG |
|---|---|---|---|---|
| Holiday | 37.3% | 27.4% | **23.9%** | ↓ 13.4 pp |
| Pre-holiday | 31.0% | 32.9% | **26.5%** | ↓ 4.5 pp |
| Weekend | 21.6% | 20.7% | **17.3%** | ↓ 4.3 pp |
| Weekday | 28.7% | 27.6% | **24.6%** | ↓ 4.1 pp |

The largest accuracy gains came on **holidays** (13.4 percentage point reduction in WAPE), driven by the holiday-weight feature and pre-holiday indicators. Pre-holiday accuracy also improved significantly over the pilot model due to added pre-holiday features.

### Performance Across Trial Phases (Daily WAPE — All Restaurants)

| Phase | Period | GGG | FPT Pilot | FPT Opt |
|---|---|---|---|---|
| 1 — Experiment | 05–09/2023 | 26.8% | 25.6% | **22.5%** |
| 2 — Pilot 1.0 | 10–12/2023 | 27.1% | 25.0% | **22.1%** |
| 3 — Pilot 2.0 | 01–04/2024 | — | 31.3% | **28.9%** |



### Feature Importance

The most influential features were historical lag features, historical traffic statistics, and holiday-related features — confirming the value of engineered time-series and calendar features over raw data inputs.




## Repository Structure

```text
.
├── 1_set_config.py              # Generate/update runtime configuration
├── 2_preprocess_data.py         # Data preprocessing & feature preparation
├── 3_cluster_restaurant_id.py   # Restaurant clustering (seasonality + volatility)
├── 4_main_predict_global.py     # Global model forecasting pipeline
├── 5_main_predict_local.py      # Local model forecasting pipeline
├── 6_aggregate_predict_result.py # Merge Global + Local predictions
├── Config/
│   └── config.ini               # Main configuration file
├── Data_Processing/
│   ├── raw/                     # Input data directory
│   └── result/                  # Output results directory
└── README.md
```

### Main Scripts

| Script | Description |
|---|---|
| `1_set_config.py` | Generates/updates runtime configuration in `Config/config.ini` |
| `2_preprocess_data.py` | Data preprocessing, anomaly handling, missing-value treatment, feature generation |
| `3_cluster_restaurant_id.py` | Clusters restaurants by weekly periodicity and traffic fluctuation |
| `4_main_predict_global.py` | Global forecasting pipeline for lower-complexity restaurants |
| `5_main_predict_local.py` | Local forecasting pipeline for higher-complexity restaurants |
| `6_aggregate_predict_result.py` | Aggregates Global and Local outputs into the final forecast |

## Input Data

Place input files in `Data_Processing/raw/`. Expected files:

| Dataset | Description |
|---|---|
| `sales_store` | Daily restaurant sales and traffic: `shift_date`, `restaurant_id`, `guest_count`, `total_sale`, `total_loyalty_sale`, `total_bill` |
| `dim_restaurant` | Restaurant metadata: operating info, capacity, open/close dates, brand, location |
| `dim_date` | Calendar table: dates, weekdays, holiday flags (2000–2050) |
| `moon_date` | Lunar calendar data (2020–2025) |
| `pre_holiday` | Pre-holiday indicator data |
| `holiday_weight` | Restaurant-level holiday impact weights |
| `cluster_km` | Restaurant behavior-cluster labels (if pre-computed) |

## Configuration

Main config file: `Config/config.ini`

Key fields to review before running:

- `final_date` — last date of the forecast month
- `path_sale_store` — path to sales data
- `path_dim_restaurant` — path to restaurant metadata
- `path_dim_date` — path to calendar data
- `path_holiday` — path to holiday weight data
- `path_cluster_km` — path to cluster labels
- `path_moon_date` — path to lunar calendar data
- `path_pre_holiday` — path to pre-holiday data

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure paths and forecast dates

```bash
python 1_set_config.py
```

Review and update `Config/config.ini` as needed.

### 3. Run preprocessing

```bash
python 2_preprocess_data.py
```

### 4. Cluster restaurants

```bash
python 3_cluster_restaurant_id.py
```

### 5. Run Global forecast

```bash
python 4_main_predict_global.py --time 2024-05
```

### 6. Run Local forecast

```bash
python 5_main_predict_local.py --time 2024-05
```

### 7. Aggregate results

```bash
python 6_aggregate_predict_result.py
```

## Output

The final prediction is a parquet file aggregating Global and Local model results. Example output path:

```text
data_forecasting/predict_result.parquet
```

The actual output path depends on your local configuration.

## Notes

- Private production data and internal HDFS paths are not included.
- Local filesystem and HDFS settings may need adaptation for your environment.
- Some script names and cluster IDs may differ depending on the cleaned repository version.
- **This repository was revised from original code used in an internal project**.
