# Restaurant Traffic Forecasting (Public Sanitized Revision)

This repository is a **revised version of original internal company code**.

## Important note
- Private/internal paths, user names, and infrastructure references were removed.
- Large artifacts were reduced to lightweight samples.
- Included parquet files are anonymized and downsized for sharing.

## Repository structure
- `1_set_config.py`: generate runtime config at `Config/config.ini`
- `2_preprocess_data.py`: preprocessing + feature pipeline
- `3_cluster_restaurant_id.py`: restaurant clustering helper
- `4_main_predict_global.py`: global model prediction
- `5_main_predict_local.py`: local model prediction
- `6_aggregate_predict_result.py`: merge prediction outputs

## Data policy for this public repo
- `Data_Processing/raw/` is intentionally ignored in git.
- Use your own input data for production runs.
- Tracked parquet files under `Data_Processing/`, `Forecasting_Results/`, `Holiday_Weight/`, and `Model/` are sample-sized and anonymized.

## Quick start
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Generate config
```bash
python 1_set_config.py
```

3. Run pipeline
```bash
python 2_preprocess_data.py
python 3_cluster_restaurant_id.py
python 4_main_predict_global.py --time 2024-05
python 5_main_predict_local.py --time 2024-05
python 6_aggregate_predict_result.py
```

## Configure paths
Edit `Config/config.ini` (or `1_set_config.py`) for your environment:
- `path_sale_store`
- `path_dim_restaurant`
- `path_dim_date`
- `path_moon_date`
- `path_pre_holiday`

## Disclaimer
This public revision is intended for reference and experimentation.
You may need to adapt I/O logic (local filesystem vs HDFS) depending on your environment.
