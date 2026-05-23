import logging
import os
import sys
import subprocess
import matplotlib.pyplot as plt
from pyarrow import fs
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import configparser
from pathlib import Path
from warnings import simplefilter
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from scipy.signal import periodogram
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log start of the script
logging.info("Starting the script...")

# Environment setup
os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf/"
os.environ["JAVA_HOME"] = "/usr/jdk64/jdk1.8.0_112"
os.environ["HADOOP_HOME"] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ["ARROW_LIBHDFS_DIR"] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ["CLASSPATH"] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True
).decode("utf-8")

hdfs = fs.HadoopFileSystem(
    host=os.getenv("HDFS_HOST", "localhost"), port=int(os.getenv("HDFS_PORT", "8020"))
)

def check_early_sale_store(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(by=['shift_date'], inplace=True)
    df = df.drop_duplicates(subset=['shift_date', 'restaurant_id'], keep='last')
    df['shift_date'] = df['shift_date'].astype('datetime64[ns]')
    return df

def create_features_old(df, label=None):
    X = df.drop(columns=['y', 'date', 'restaurant_id'])
    if label:
        y = df[label]
        return X, y
    return X

# Load config
configFilePath = "./Config/config.ini"
logging.info('Loading metadata path from config file')

config = configparser.RawConfigParser()
config.read(configFilePath)
config_dict = dict(config.items('config'))

path_sale_store = config_dict['path_sale_store']
path_dim_restaurant = config_dict['path_dim_restaurant']
path_processed_data = config_dict['path_processed_data']
path_feature_data = config_dict['path_feature_data']
path_holiday = config_dict['path_holiday']
path_cluster = config_dict['path_cluster']
path_first_month = config_dict['path_first_month']
final_date = config_dict['end_date']
output_dir = config_dict['path_output_processed_data']
path_prediction_data = config_dict['path_prediction_data']
end_date = config_dict['start_date']
path_cluster_km = config_dict['path_cluster_km']

start_date = '2022-1-1'

# Log file paths
logging.info(f"path_sale_store: {path_sale_store}")
logging.info(f"path_dim_restaurant: {path_dim_restaurant}")
logging.info(f"path_processed_data: {path_processed_data}")
logging.info(f"path_feature_data: {path_feature_data}")
logging.info(f"path_holiday: {path_holiday}")
logging.info(f"path_cluster: {path_cluster}")

# Load data
logging.info("Loading data...")
try:
    df_dims = pd.read_parquet(path_dim_restaurant, filesystem=hdfs)
    df_sale_stores = pd.read_parquet(path_sale_store, filesystem=hdfs)
    df_cluster = pd.read_parquet(path_cluster, filesystem=hdfs)
    df_holidays = pd.read_parquet(path_holiday)
    df_processed = pd.read_parquet(path_processed_data)
    logging.info("Data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    sys.exit(1)

# Process data
logging.info("Processing data...")
df_processed = df_processed.rename(columns={"date": "shift_date"})
df_processed['shift_date'] = pd.to_datetime(df_processed.shift_date)
df_processed.sort_values('shift_date', ascending=True, inplace=True)
df_processed.set_index('shift_date', inplace=True)
df_processed = df_processed[['guest_count', 'restaurant_id']]
df_processed['month'] = df_processed.index.month

# Function to get periodogram
def get_periodogram(ts, detrend='linear'):
    DAYS_IN_MONTH = 30.44
    DAYS_IN_YEAR = 365.25
    months_in_days = lambda months: int(months * DAYS_IN_MONTH)
    years_in_days = lambda years: int(years * DAYS_IN_YEAR)
    # fs = datetime.timedelta(days=DAYS_IN_YEAR)
    frequency, amplitude = periodogram(
        ts,
        fs=365,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    df_periodogram = pd.DataFrame({'frequency': frequency, 'amplitude': amplitude})
    df_periodogram['r_frequency'] = df_periodogram.frequency.apply(lambda row: int(row))
    return df_periodogram.groupby('r_frequency').agg({'amplitude': 'sum'}).reset_index()

df_weekly_pattern = pd.DataFrame()

for restaurant_id in tqdm(df_processed.restaurant_id.unique()):
    df_id = df_processed[df_processed.restaurant_id == restaurant_id]
    df_id = df_id.to_period('D')
    try:
        df_id = df_id[(df_id.index < end_date) & (df_id.index > start_date)]
        df_periodogram = get_periodogram(df_id.guest_count).sort_values('amplitude', ascending=False)
        sum_amplitude = df_periodogram.amplitude.sum()
        weekly = df_periodogram.loc[df_periodogram.r_frequency.isin([50, 51, 52, 53, 54]), 'amplitude'].sum()
        df_weekly_pattern = pd.concat([df_weekly_pattern, pd.DataFrame({
            'restaurant_id': [restaurant_id],
            'weekly': [weekly],
            'sum_amplitude': [sum_amplitude],
            'weekly_pc': [weekly / sum_amplitude],
            'start_date': [df_id.index.min()],
            'end_date': [df_id.index.max()],
        })])
        # logging.info(f"Processed restaurant_id: {restaurant_id}")
    except Exception as e:
        logging.error(f"Error processing restaurant_id {restaurant_id}: {e}")
        df_weekly_pattern = pd.concat([df_weekly_pattern, pd.DataFrame({
            'restaurant_id': [restaurant_id],
            'weekly': [None],
            'sum_amplitude': [None],
            'weekly_pc': [None],
            'start_date': [df_id.index.min()],
            'end_date': [df_id.index.max()],
        })])

# Filter and process additional data
logging.info("Filtering and processing additional data...")
df_processed['q99'] = df_processed.groupby('restaurant_id').guest_count.transform(lambda x: x.quantile(0.95))
df_processed['q1'] = df_processed.groupby('restaurant_id').guest_count.transform(lambda x: x.quantile(0.05))

df_score = df_processed.loc[(df_processed['guest_count'] > df_processed['q1']) & (df_processed['guest_count'] < df_processed['q99'])] \
    .groupby('restaurant_id').agg({'guest_count': ['mean', 'std']}) \
    .rename(columns={'mean': 'avg_y', 'std': 'std_y'}) \
    .droplevel(0, axis=1).reset_index()

df_score['fluctuation_score'] = df_score.apply(lambda row: row.std_y / row.avg_y, axis=1)

df_merge_dims = df_weekly_pattern.join(df_score.set_index('restaurant_id'), on='restaurant_id', how='left')
df_merge_dims = df_merge_dims.join(df_dims.set_index('restaurant_id'), on='restaurant_id', how='left')
df_merge_dims = df_merge_dims.join(df_cluster.set_index('restaurant_id'), on='restaurant_id', how='left')

df_merge_dims.fillna(0, inplace=True)

dr1 = 'fluctuation_score'
dr2 = 'weekly_pc'
df_merge_dims_filter = df_merge_dims.loc[
    (df_merge_dims.weekly_pc < df_merge_dims.weekly_pc.quantile(0.99)) & (df_merge_dims.weekly_pc > df_merge_dims.weekly_pc.quantile(0.01))
    & (df_merge_dims['fluctuation_score'] < df_merge_dims['fluctuation_score'].quantile(0.99)) & (df_merge_dims['fluctuation_score'] > df_merge_dims['fluctuation_score'].quantile(0.01))
]

# Fit KMeans model
logging.info("Fitting KMeans model...")
model = KMeans(n_clusters=6, random_state=0)
model.fit(df_merge_dims_filter[['fluctuation_score', dr2]])
df_merge_dims['cluster_km'] = model.predict(df_merge_dims[['fluctuation_score', dr2]])

# Final clustering and saving the results
logging.info("Final clustering and saving the results...")
df_cluster_final = df_merge_dims[['restaurant_id', 'cluster_km', dr2, dr1]] \
    .groupby('cluster_km').agg({'restaurant_id': 'count', dr2: 'mean', dr1: 'mean'}) \
    .sort_values(dr2, ascending=False).reset_index()

df_cluster_final['cluster_final'] = df_cluster_final.index < 3
df_cluster_final['cluster_final'] = df_cluster_final['cluster_final'].map({True: 'global', False: 'local'})
df_merge_dims = df_merge_dims.merge(df_cluster_final[['cluster_final', 'cluster_km']], on='cluster_km', how='left')
df_merge_dims = df_merge_dims.drop(columns=['cluster_km']).rename(columns={'cluster_final': 'cluster_km'})

# Save result
output_file = path_cluster_km
df_merge_dims[['restaurant_id', 'cluster_km', 'cluster', dr2, dr1]].to_parquet(output_file,)
logging.info(f"Result saved to {output_file}")
