# Restaurant Traffic Forecasting

## Overview

Restaurant Traffic Forecasting is a time-series forecasting pipeline for predicting daily restaurant guest count (`guest_count`) at the restaurant level for the Golden Gate restaurant chain.

The project combines sales-data preprocessing, restaurant eligibility filtering, feature engineering, restaurant behavior segmentation, and XGBoost-based forecasting. The final output is a next-month daily traffic forecast for each eligible restaurant.

This repository was revised from an internal forecasting project and cleaned for portfolio/repository sharing.

## Problem statement

The goal is to predict the number of customers visiting each Golden Gate restaurant in the next month.

- **Input:** Daily restaurant sales and traffic data from 2021 onward, including guest count, total sales, loyalty sales, bill count, restaurant metadata, calendar information, lunar calendar data, and holiday-related features.
- **Output:** Daily guest-count forecast for each eligible restaurant in the next forecast month.
- **Forecasting level:** Restaurant-level daily prediction, equivalent to forecasting `D+1` to `D+30/31` for each restaurant.

## Scope

The forecasting pipeline is designed for Golden Gate restaurants that satisfy the following conditions:

- Restaurant is currently active.
- Restaurant is not located in Vinh, Nghe An.
- Restaurant brand is not `I-cook`, `Icook Store`, or `G-delivery`.
- Restaurant has at least 42 historical days with available guest-count data before the first day of the forecast month.

## What this project does

- Loads raw sales, restaurant, calendar, lunar-calendar, and holiday-related data.
- Cleans and preprocesses restaurant-level daily guest-count records.
- Removes invalid and abnormal traffic observations.
- Fills missing guest-count labels using same-weekday historical values and annual restaurant-level averages.
- Creates time-series, calendar, holiday, lunar-date, restaurant, lag, and rolling-statistics features.
- Clusters restaurants by weekly seasonality and traffic volatility.
- Splits restaurants into two forecasting groups:
  - **Global:** lower-complexity restaurants with clearer weekly patterns and lower traffic volatility.
  - **Local:** higher-complexity restaurants with weaker weekly patterns and higher traffic volatility.
- Trains and applies XGBoost forecasting models for the Global and Local groups.
- Aggregates Global and Local predictions into a final prediction output file.

## Forecasting approach

### Rolling monthly training

For each forecast month `T`, the model uses historical data from January 2022 through month `T-1` as training data, then predicts daily guest count for month `T`.

The validation period used for model selection and parameter tuning is July 2022 to December 2023.

### Restaurant segmentation

Restaurants are clustered into 5 behavior groups based on two dimensions:

1. **Weekly periodicity (`weekly_pc`)**  
   Measures how strongly guest traffic repeats by weekly pattern. This is calculated using a Fourier-based approach.

2. **Traffic fluctuation score**  
   Measures how volatile restaurant traffic is over time using statistical properties such as standard deviation and mean.

The 5 clusters are then grouped into two model families:

- **Global model:** restaurants with clearer weekly seasonality and lower complexity.
- **Local model:** restaurants with unclear weekly seasonality and higher volatility.

### Model

The project uses **XGBoost Regressor** for structured time-series forecasting.

Two model configurations are used:

#### Global model

Used for lower-complexity restaurant groups.

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

#### Local model

Used for higher-complexity restaurant groups.

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

Other XGBoost parameters use default values.

## Feature engineering

### Restaurant features

- `cluster`: restaurant behavior cluster based on weekly periodicity and traffic volatility.

### Calendar and time features

- `dayofweek`
- `weekend`
- `month`
- `year`
- day of month
- regular calendar date attributes

### Holiday features

- holiday indicator
- pre-holiday indicator
- restaurant-level holiday weight

`holiday_weight` represents how much a specific holiday affects guest count for each restaurant. It is calculated by comparing holiday traffic with average traffic on comparable weekdays, removing outliers with z-score filtering, and filling missing values using brand-level medians where available.

### Lunar calendar features

- beginning-of-month lunar date indicator
- mid-month lunar date indicator

### Lag features

Historical guest-count values from previous days, including short-term and longer-cycle lags such as:

- previous 1–9 days
- previous 14, 21, 28, 35, 42, 49, 56, and 64 days

### Statistical features

Historical traffic statistics from prior periods, including:

- mean
- median
- minimum
- maximum

These features are calculated from previous guest-count observations, especially recent historical windows before the forecast month.

## Data preprocessing

The preprocessing pipeline handles the following issues:

1. **Invalid guest-count values**
   - Removes rows where `guest_count` is negative, zero, or null.

2. **Abnormal traffic values**
   - Detects and removes unusually high or low guest-count observations for each restaurant using Prophet-based anomaly handling.

3. **Missing daily observations**
   - Builds a complete restaurant-date frame for the forecast period.
   - Joins actual guest-count data into the complete frame.
   - Fills missing `guest_count` using the same weekday from the previous week.
   - If still missing, fills using the restaurant's average guest count in the same year.

4. **Restaurant eligibility**
   - Filters restaurants based on operating status, opening/closing dates, location, brand, and historical data availability.

## Evaluation

The model is evaluated using Absolute Percentage Error-based metrics:

### MAPE

```text
MAPE = mean(abs(y_true - y_pred) / y_true) * 100
```

### WAPE

```text
WAPE = sum(abs(y_true - y_pred)) / sum(y_true) * 100
```

WAPE is the main evaluation metric.

## Performance

Evaluation period: **July 2022 to December 2023**

Scope: **440+ restaurants per month**, with 470+ restaurants in 12 months. Restaurants are evaluated across two groups:

- **Baseline:** restaurants open for at least 1 year as of February 28, 2023.
- **Non-baseline:** restaurants open for less than 1 year as of February 28, 2023.

### WAPE results

| Forecast level | Old model - Baseline | Old model - Non-baseline | Old model - Total | New model - Baseline | New model - Non-baseline | New model - Total |
|---|---:|---:|---:|---:|---:|---:|
| Daily | 24.6 | 29.0 | 25.3 | 21.3 | 25.3 | 22.0 |
| Weekly | 16.7 | 21.0 | 17.5 | 12.8 | 16.9 | 13.6 |
| Monthly | 14.9 | 18.2 | 15.5 | 10.3 | 13.6 | 10.9 |

The new forecasting flow improved WAPE by approximately **3–4 percentage points** across daily, weekly, and monthly forecast levels.

Feature-importance analysis showed that the most influential features were mainly historical lag features, historical traffic statistics, and holiday-related features.

## Repository structure

```text
.
├── 1_set_config.py
├── 2_preprocess_data.py
├── 3_cluster_restaurant_id.py
├── 4_main_predict_global.py
├── 5_main_predict_local.py
├── 6_aggregate_predict_result.py
├── Config/
│   └── config.ini
├── Data_Processing/
│   ├── raw/
│   └── result/
└── README.md
```

### Main scripts

- `1_set_config.py`  
  Generates or updates runtime configuration in `Config/config.ini`.

- `2_preprocess_data.py`  
  Runs data preprocessing, anomaly handling, missing-value treatment, feature preparation, and dataset generation.

- `3_cluster_restaurant_id.py`  
  Clusters restaurants based on weekly periodicity and traffic fluctuation.

- `4_main_predict_global.py`  
  Runs the Global forecasting pipeline for lower-complexity restaurant groups.

- `5_main_predict_local.py`  
  Runs the Local forecasting pipeline for higher-complexity restaurant groups.

- `6_aggregate_predict_result.py`  
  Aggregates Global and Local prediction outputs into the final forecast result.

## Input data

Place private input files in:

```text
Data_Processing/raw/
```

Expected input files include:

- `sales_store.parquet`
- `dim_restaurant.parquet`
- `dim_date.parquet`
- `moon_date.parquet`
- `pre_holiday.parquet`
- holiday-weight data, if available
- restaurant-cluster data, if available

### Main data sources

| Dataset | Description |
|---|---|
| `sales_store` | Daily restaurant sales and traffic data, including `shift_date`, `restaurant_id`, `guest_count`, `total_sale`, `total_loyalty_sale`, and `total_bill`. |
| `d_date` / `dim_date` | Calendar table containing dates, weekdays, and holiday information. |
| `d_restaurant` / `dim_restaurant` | Restaurant metadata, including operating information and restaurant attributes. |
| `moon_calendar` | Lunar calendar data covering January 2020 to December 2025. |
| `holiday_weight` | Restaurant-level holiday impact weights. |
| `cluster_km` | Restaurant behavior-cluster labels. |

## Configuration

Main configuration file:

```text
Config/config.ini
```

Key fields to review before running:

- `final_date`
- `path_sale_store`
- `path_dim_restaurant`
- `path_dim_date`
- `path_holiday`
- `path_cluster_km`
- `path_moon_date`
- `path_pre_holiday`

The configuration should be updated to match the latest available data paths before each forecast run.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure input paths and forecast dates

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

Depending on your repository version, this step may be implemented as a notebook such as `3_cluster_restaurant_id.ipynb`.

### 5. Run Global forecast

```bash
python 4_main_predict_global.py --time 2024-05
```

### 6. Run Local forecast

```bash
python 5_main_predict_local.py --time 2024-05
```

### 7. Aggregate prediction results

```bash
python 6_aggregate_predict_result.py
```

## Output

The final prediction output is generated after aggregating Global and Local model results.

Example output path in the original internal environment:

```text
data_forecasting/predict_result.parquet
```

The actual output path depends on your local or production configuration.

## Notes

- The repository may include lightweight sample parquet files for sharing and testing.
- Private production data and internal HDFS paths are not included.
- Local filesystem and HDFS settings may need to be adapted for your environment.
- Some script names and cluster IDs may differ depending on the cleaned repository version. Check the current config and script arguments before running.
- This repository was revised from original code used in an internal project at a previous company.