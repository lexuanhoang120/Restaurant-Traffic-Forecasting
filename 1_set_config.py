import configparser
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize configparser
config = configparser.ConfigParser()

# Add the 'config' section
config.add_section('config')

# Define parameters for predicting the next month
# Có data tháng 4 => chạy predict tháng 5 => start-date và end_date theo tháng 5
start_date = '2024-05-01'
end_date = '2024-05-31'

# final_date = '2024-12-31'

path_sale_store = "./Data_Processing/raw/sales_store.parquet"
path_dim_restaurant = "./Data_Processing/raw/dim_restaurant.parquet"
path_dim_date = "./Data_Processing/raw/dim_date.parquet"

path_cluster = "./Data_Processing/result/cluster_km.parquet"
path_holiday = "./Data_Processing/result/holiday_weight.parquet"
path_cluster_km = "./Data_Processing/result/cluster_km.parquet"
path_moon_date = "./Data_Processing/raw/moon_date.parquet"
path_pre_holiday = "./Data_Processing/raw/pre_holiday.parquet"
path_output_data = "./Forecasting_Results/hybrid_result.parquet"

# Set paths and parameters in the config
logging.info("Setting paths and parameters in the config file...")
config_paths = {
    'path_cluster': path_cluster,
    'path_holiday': path_holiday,
    'path_sale_store': path_sale_store,
    'path_dim_restaurant': path_dim_restaurant,
    'path_dim_date': path_dim_date,
    'path_moon_date': path_moon_date,
    'path_cluster_km': path_cluster_km,
    'path_pre_holiday': path_pre_holiday,
    'path_output_data': path_output_data,
    # 'final_date': final_date,
    'start_date': start_date,
    'end_date': end_date,
    'path_output_processed_data': r'./Data_Processing/result/',
    'path_first_month': r'./Data_Processing/result/',
    'path_processed_data': r'./Data_Processing/result/processed_data.parquet',
    'path_feature_data': r'./Data_Processing/result/feature_data.parquet',
    'path_prediction_data': r'./Forecasting_Results/',
    'path_global_result': r'./Forecasting_Results/global_result.parquet',
    'path_local_result': r'./Forecasting_Results/local_result.parquet',
    'path_hybrid_result': r'./Forecasting_Results/hybrid_result.parquet',
    'path_train_set': r'./Data_Processing/data_train/',
    'path_test_set': r'./Data_Processing/data_test/',
    'path_dataset': r'./Data_Processing/dataset/'
}

# Add the paths and parameters to the config section
for key, value in config_paths.items():
    config.set('config', key, value)
    logging.info(f"Set {key} to {value}")

# Write the updated config to a file
config_file_path = "./Config/config.ini"
logging.info(f"Writing the configuration to {config_file_path}")
with open(config_file_path, 'w') as configfile:
    config.write(configfile)

logging.info("Configuration file created successfully.")
