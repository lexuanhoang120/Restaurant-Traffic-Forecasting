# Restaurant Traffic Forecasting

## Overview
Restaurant Traffic Forecasting is a time-series forecasting pipeline for daily guest-count prediction at restaurant level.  
The project combines data preprocessing, restaurant segmentation, and model-based forecasting to generate final prediction outputs.

## What this project does
- Preprocesses raw sales and restaurant metadata
- Creates forecasting features
- Clusters restaurants by behavior pattern
- Runs two forecasting strategies:
  - `Global`: one strategy for lower-complexity patterns
  - `Local`: one strategy for higher-complexity patterns
- Merges outputs into a final prediction file

## Repository structure
- `1_set_config.py`: generate/update runtime config (`Config/config.ini`)
- `2_preprocess_data.py`: data preprocessing and feature preparation
- `3_cluster_restaurant_id.py`: restaurant clustering
- `4_main_predict_global.py`: global forecasting pipeline
- `5_main_predict_local.py`: local forecasting pipeline
- `6_aggregate_predict_result.py`: final aggregation of prediction outputs

## Quick start
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Configure paths and dates
```bash
python 1_set_config.py
```

3. Run the pipeline
```bash
python 2_preprocess_data.py
python 3_cluster_restaurant_id.py
python 4_main_predict_global.py --time 2024-05
python 5_main_predict_local.py --time 2024-05
python 6_aggregate_predict_result.py
```

## Input data
Place your private input files in `Data_Processing/raw/`:
- `sales_store.parquet`
- `dim_restaurant.parquet`
- `dim_date.parquet`
- `moon_date.parquet`
- `pre_holiday.parquet`

See [README.md](/home/vli3/Documents/Restaurant-Traffic-Forecasting/Data_Processing/raw/README.md) in that folder for details.

## Configuration
Main config file: [config.ini](/home/vli3/Documents/Restaurant-Traffic-Forecasting/Config/config.ini)

Key paths to check before running:
- `path_sale_store`
- `path_dim_restaurant`
- `path_dim_date`
- `path_moon_date`
- `path_pre_holiday`

## Notes
- Included parquet artifacts are lightweight sample files for repository sharing.
- Depending on your environment, you may need to adapt local filesystem and HDFS settings.

---
This repository was revised from original code used in an internal project at a previous company.
