# Restaurant Traffic Forecasting

> **Note:** This repository was revised from an internal forecasting project and cleaned for portfolio sharing. Internal data, HDFS paths, and production configurations are not included.

## Highlights

- **450+ restaurants** forecast daily across a major Vietnamese restaurant chain (Golden Gate)
- **22.0% daily WAPE** — a **13% improvement** over the previous model and **19% better** than the client's internal forecast
- **90% feature reduction** and **70% faster training** compared to the initial pilot
- **13.4 percentage point WAPE reduction** on holidays — the hardest days to predict
- Deployed and validated over **18 months** across 5 trial phases in production

---

## Results

*Evaluation period: July 2022 – December 2023 (18 months) · 450+ restaurants*

### Daily WAPE — Three-Way Benchmark

| | GGG (Client) | FPT Pilot (Old) | **FPT Opt (Ours)** |
|---|---|---|---|
| All restaurants | 27.1% | 25.3% | **22.0%** |
| Baseline (≥1 yr history) | 25.6% | 24.6% | **21.3%** |
| Non-baseline (<1 yr history) | 33.8% | 29.0% | **25.3%** |

### Performance by Day Type (Daily WAPE)

| Day Type | GGG (Client) | FPT Pilot (Old) | **FPT Opt (Ours)** | Gain vs. Client |
|---|---|---|---|---|
| **Holiday** | 37.3% | 27.4% | **23.9%** | ⬇ 13.4 pp |
| Pre-holiday | 31.0% | 32.9% | **26.5%** | ⬇ 4.5 pp |
| Weekend | 21.6% | 20.7% | **17.3%** | ⬇ 4.3 pp |
| Weekday | 28.7% | 27.6% | **24.6%** | ⬇ 4.1 pp |

The hardest prediction targets — holidays — saw the biggest gains, driven by custom holiday-weight features and pre-holiday indicators.

### Stability Across Trial Phases (Daily WAPE — All Restaurants)

| Phase | Period | Context | GGG | FPT Pilot | **FPT Opt** |
|---|---|---|---|---|---|
| 1 | 06–12/2022 | Stable period | 25.3% | 21.5% | **19.2%** |
| 2 | 01–04/2023 | Volatile (Tet holiday) | 30.5% | 31.8% | **26.0%** |
| 3 | 05–09/2023 | Experimentation | 26.8% | 25.6% | **22.5%** |
| 4 | 10–12/2023 | Trial 1.0 | 27.1% | 25.0% | **22.1%** |
| 5 | 01–04/2024 | Trial 2.0 | — | 31.3% | **28.9%** |

The optimized model maintained consistent accuracy across all phases. During the volatile Tet period, it outperformed the pilot model by **5.8 percentage points**.

### Weekly & Monthly WAPE

| Level | Group | GGG | FPT Pilot | **FPT Opt** |
|---|---|---|---|---|
| Weekly | All | 16.4% | 17.5% | **13.6%** |
| Weekly | Baseline | 15.5% | 16.7% | **12.8%** |
| Weekly | Non-baseline | 20.3% | 21.0% | **16.9%** |
| Monthly | All | 12.2% | 15.5% | **10.9%** |
| Monthly | Baseline | 11.5% | 14.9% | **10.3%** |
| Monthly | Non-baseline | 15.5% | 18.2% | **13.6%** |

---

## Problem & Context

**Client:** Golden Gate (GGG) — one of Vietnam's largest restaurant chains  
**Developer:** CADS — FPT  
**Goal:** Daily guest-count forecasts for every restaurant, one month ahead, to drive staffing, inventory, and revenue planning.

**Scale:**

- 19 internal data tables (sales, metadata, customers, bookings, promotions, ratings)
- External data sources: weather, social ratings
- Training data from January 2021 onward
- Forecast window: D+1 to D+30/31 per restaurant

**Eligibility:** active restaurants, excluding Vinh (Nghe An), excluding `I-cook`/`Icook Store`/`G-delivery` brands, requiring ≥ 42 days of guest-count history.

---

## Approach

### 1. Data Preprocessing

- Remove invalid observations (negative, zero, or null guest count)
- Detect and filter anomalies per restaurant (Prophet-based)
- Impute missing values: same-weekday historical → annual restaurant average
- Filter to eligible restaurants by status, location, brand, and data sufficiency

### 2. Feature Engineering

| Category | Examples |
|---|---|
| Calendar & time | `dayofweek`, `weekend`, `month`, `year`, day of month |
| Holiday | Holiday indicator, pre-holiday indicator, restaurant-level `holiday_weight` |
| Lunar calendar | Beginning-of-month and mid-month lunar date flags |
| Lag | Guest count from previous 1–9 days and long-cycle lags (14, 21, 28, 35, 42, 49, 56, 64 days) |
| Rolling statistics | Mean, median, min, max over recent historical windows |

**`holiday_weight`** — a per-restaurant, per-holiday impact score computed by comparing holiday traffic against comparable weekday averages, with z-score outlier removal and brand-level median fallback.

### 3. Restaurant Segmentation

Restaurants are clustered into 5 groups using two dimensions:

- **Weekly periodicity** — Fourier-based measure of weekly pattern strength
- **Traffic fluctuation** — statistical volatility (std dev, mean)

The 5 clusters are mapped to two model strategies:

| Strategy | Profile | Model Config |
|---|---|---|
| **Global** | Strong weekly patterns, low volatility | Shallower trees, lower LR, fewer estimators |
| **Local** | Weak/noisy patterns, high volatility | Deeper trees, higher LR, more estimators |

### 4. Model

**XGBoost Regressor** with rolling monthly training: for each forecast month `T`, train on January 2022 through `T-1`, predict month `T`.

**Global model:**

```python
XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=8,
             min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0.5)
```

**Local model:**

```python
XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=6,
             min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0.6)
```

### 5. Pipeline Optimizations

| Metric | Before | After |
|---|---|---|
| Features used | Baseline set | **90% fewer** |
| Training time | Baseline | **70% faster** |
| Restaurant grouping | Manual rules | **Data-driven clustering** |
| Eligibility filtering | Basic rules | **Refined logic with edge cases** |

---

## Evaluation

**Primary metric:** WAPE (Weighted Absolute Percentage Error)

```
WAPE = Σ|actual − predicted| / Σ actual × 100
```

**Scope:** 450–470+ restaurants per month · July 2022 – December 2023 (18 months) · Three-way benchmark (client / old model / new model)

**Groups:** Baseline (≥1 year history) vs. Non-baseline (<1 year history, harder to forecast)

Feature importance analysis confirmed that lag features, historical statistics, and holiday features were the strongest predictors.

---

## Repository Structure

```text
.
├── 1_set_config.py              # Runtime configuration
├── 2_preprocess_data.py         # Preprocessing & feature engineering
├── 3_cluster_restaurant_id.py   # Restaurant behavior clustering
├── 4_main_predict_global.py     # Global model pipeline
├── 5_main_predict_local.py      # Local model pipeline
├── 6_aggregate_predict_result.py # Merge predictions → final output
├── Config/
│   └── config.ini
├── Data_Processing/
│   ├── raw/                     # Input data
│   └── result/                  # Output results
└── README.md
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
python 1_set_config.py
# → Edit Config/config.ini with your data paths

# 3. Preprocess
python 2_preprocess_data.py

# 4. Cluster restaurants
python 3_cluster_restaurant_id.py

# 5. Forecast
python 4_main_predict_global.py --time 2024-05
python 5_main_predict_local.py --time 2024-05

# 6. Aggregate
python 6_aggregate_predict_result.py
```

### Input Data

Place these files in `Data_Processing/raw/`:

| File | Content |
|---|---|
| `sales_store` | Daily sales: `shift_date`, `restaurant_id`, `guest_count`, `total_sale`, `total_loyalty_sale`, `total_bill` |
| `dim_restaurant` | Restaurant metadata: brand, location, capacity, open/close dates |
| `dim_date` | Calendar with holiday flags (2000–2050) |
| `moon_date` | Lunar calendar (2020–2025) |
| `pre_holiday` | Pre-holiday indicators |
| `holiday_weight` | Per-restaurant holiday impact scores |
| `cluster_km` | Restaurant cluster labels (or generated by step 4) |

### Configuration

Key fields in `Config/config.ini`: `final_date`, `path_sale_store`, `path_dim_restaurant`, `path_dim_date`, `path_holiday`, `path_cluster_km`, `path_moon_date`, `path_pre_holiday`

---