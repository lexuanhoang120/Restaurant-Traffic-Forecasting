import calendar
import configparser

import numpy as np
import json
# import torch
from tqdm import tqdm
import argparse
import os
import pickle
import glob
from datetime import datetime

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyarrow import fs
import pyarrow.parquet as pq

import pandas as pd
import argparse
from datetime import datetime
import shutil
##########################################
from Holiday import link_withHoliday
from Features import Basic_Features, Profile_features
from Config import split_dataset
from Metric import Evaluation_metric

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from xgboost import plot_tree
from matplotlib import pyplot

import pandas as pd

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

# from adtk.data import validate_series
# from adtk.visualization import plot
# from adtk.detector import *

# from adtk.detector import MinClusterDetector, OutlierDetector, AutoregressionAD, LevelShiftAD, PersistAD, \
    # VolatilityShiftAD, SeasonalAD
from sklearn.neighbors import LocalOutlierFactor

# outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")

from feature_engine.encoding import OneHotEncoder

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


def last_day_of_month(year, month):
    _, last_day = calendar.monthrange(year, month)
    return f"{year}-{month}-{last_day}"


def check_early_sale_store(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(by=['shift_date'], inplace=True)
    df = df.drop_duplicates(subset=['shift_date', 'restaurant_id'], keep='last')
    df['shift_date'] = df['shift_date'].astype('datetime64[ns]')
    return df


def check_first_open_month(df_sales: pd.DataFrame, number_month: int) -> pd.DataFrame:
    '''
    Check first open month to pre-process input data for modeling.

    Parameters:
    df_sales: sales_store data frame, including restaurant id, date, daily guest count and other information

    Returns:
    (dataframe): data frame containing first log date and first log month of each restaurant.
    '''

    # calculate first log date of each restaurant
    df_first_month = df_sales.groupby('restaurant_id')['shift_date'].min() \
        .reset_index() \
        .rename(columns={'shift_date': 'first_log_date'})

    # format date to 'year-month' to get first log month
    df_first_month['first_log_month'] = df_first_month['first_log_date'].dt.strftime('%Y-%m')

    # create first_month category
    df_first_month['is_first_month'] = 1

    return df_first_month


def process_data_prediction_time(df: pd.DataFrame, year_month: str) -> pd.DataFrame:
    """
    Xử lí các giá trị "Guest_count" theo từng restaurant_id
    - Merge với dataframe trắng (đủ ngày) cho từng restaurant_id
    - Fill các giá trị mising của từng restaurant_id bằng mean
      của tuần trước đó
    - Fill các giá trị missing của từng restaurant_id bằng mean
      của năm đó (nếu không thể fill được bằng mean của tuần)
    - Xử lí các giá trị bị âm của từng restaurant_id bằng mean
      của tuần trước đó hoặc bằng mean của năm đó

    Tham số:
    - df: Là dataframe gốc từ bảng "Sale_Store"

    Trả về:
    - merged_data: Là dataframe đã được xử lí không có missing
                  ở giữa các ngày và xử lí các giá trị âm
    """

    # Chuyển cột 'shift_date' sang kiểu dữ liệu ngày tháng
    df['shift_date'] = pd.to_datetime(df['shift_date'])
    df = df.sort_values(by=['restaurant_id', 'shift_date'], ascending=[True, True])
    year, month = year_month.split('-')

    day = last_day_of_month(int(year), int(month))
    # Giới hạn dữ liệu
    start_date = pd.to_datetime(df['shift_date'].min())
    end_date = pd.to_datetime(day)

    filtered_data = df[(df['shift_date'] >= start_date) & (df['shift_date'] <= end_date)]
    # Tạo cột 'week' và 'year' để lưu số tuần và năm của mỗi ngày
    filtered_data['year'] = filtered_data['shift_date'].dt.year
    filtered_data['dow'] = filtered_data['shift_date'].dt.dayofweek
    filtered_data_1 = filtered_data.copy()
    filtered_data_2 = filtered_data.copy()
    # fill na for guest_count in predicted month
    filtered_data_1.loc[filtered_data_1['shift_date'].dt.strftime('%Y-%m') == f'{year}-{month}', 'guest_count'] = None
    filtered_data_1['guest_count'] = filtered_data_1.groupby(['restaurant_id', 'dow'])['guest_count'].transform(
        lambda x: x.ffill())

    if int(month) == 1:
        # filtered_data_2.loc[df['shift_date'].dt.strftime('%Y-%m').isin(['2023-01','2023-02']), 'guest_count'] = None
        filtered_data_2.loc[
            (filtered_data_2['shift_date'] >= f'{int(year)}-12-20'), 'guest_count'] = None
        filtered_data_2['guest_count'] = filtered_data_2.groupby(['restaurant_id', 'dow'])['guest_count'].transform(
            lambda x: x.ffill())
        filtered_data_2 = filtered_data_2.set_index(['restaurant_id', 'shift_date']).combine_first(
            filtered_data_1.set_index(['restaurant_id', 'shift_date'])).reset_index()
        final_data = pd.concat(
            [filtered_data_1.loc[filtered_data_1['shift_date'].dt.strftime('%Y-%m') < f'{year}-{month}'],
             filtered_data_2.loc[filtered_data_2['shift_date'].dt.strftime('%Y-%m') == f'{year}-{month}']], axis=0,
            ignore_index=True)
        final_data.drop(columns=['year', 'dow'], inplace=True)
        return final_data

    elif year_month=='2024-03':
        # filtered_data_2.loc[df['shift_date'].dt.strftime('%Y-%m').isin(['2023-01','2023-02']), 'guest_count'] = None
        filtered_data_2.loc[
            (filtered_data_2['shift_date'] >= f'{int(year)}-01-29'), 'guest_count'] = None
        filtered_data_2['guest_count'] = filtered_data_2.groupby(['restaurant_id', 'dow'])['guest_count'].transform(

            lambda x: x.ffill())
        filtered_data_2 = filtered_data_2.set_index(['restaurant_id', 'shift_date']).combine_first(
            filtered_data_1.set_index(['restaurant_id', 'shift_date'])).reset_index()
        final_data = pd.concat(
            [filtered_data_1.loc[filtered_data_1['shift_date'].dt.strftime('%Y-%m') < f'{year}-{month}'],
             filtered_data_2.loc[filtered_data_2['shift_date'].dt.strftime('%Y-%m') == f'{year}-{month}']], axis=0,
            ignore_index=True)
        final_data.drop(columns=['year', 'dow'], inplace=True)
        return final_data
    elif year_month=='2023-02':
        # filtered_data_2.loc[df['shift_date'].dt.strftime('%Y-%m').isin(['2023-01','2023-02']), 'guest_count'] = None
        filtered_data_2.loc[
            (filtered_data_2['shift_date'] >= f'{int(year) - 1}-12-20'), 'guest_count'] = None
        filtered_data_2['guest_count'] = filtered_data_2.groupby(['restaurant_id', 'dow'])['guest_count'].transform(
            lambda x: x.ffill())
        filtered_data_2 = filtered_data_2.set_index(['restaurant_id', 'shift_date']).combine_first(
            filtered_data_1.set_index(['restaurant_id', 'shift_date'])).reset_index()
        final_data = pd.concat(
            [filtered_data_1.loc[filtered_data_1['shift_date'].dt.strftime('%Y-%m') < f'{year}-{month}'],
             filtered_data_2.loc[filtered_data_2['shift_date'].dt.strftime('%Y-%m') == f'{year}-{month}']], axis=0,
            ignore_index=True)
        final_data.drop(columns=['year', 'dow'], inplace=True)
        return final_data

    elif year_month=='2024-05':
        # filtered_data_2.loc[df['shift_date'].dt.strftime('%Y-%m').isin(['2023-01','2023-02']), 'guest_count'] = None
        filtered_data_2.loc[
            (filtered_data_2['shift_date'] >='2024-04-15'), 'guest_count'] = None
        filtered_data_2['guest_count'] = filtered_data_2.groupby(['restaurant_id', 'dow'])['guest_count'].transform(
            lambda x: x.ffill())
        filtered_data_2 = filtered_data_2.set_index(['restaurant_id', 'shift_date']).combine_first(
            filtered_data_1.set_index(['restaurant_id', 'shift_date'])).reset_index()
        final_data = pd.concat(
            [filtered_data_1.loc[filtered_data_1['shift_date'].dt.strftime('%Y-%m') < f'{year}-{month}'],
             filtered_data_2.loc[filtered_data_2['shift_date'].dt.strftime('%Y-%m') == f'{year}-{month}']], axis=0,
            ignore_index=True)
        final_data.drop(columns=['year', 'dow'], inplace=True)
        return final_data
    

    # elif int(month) == 5:
    #     # filtered_data_2.loc[df['shift_date'].dt.strftime('%Y-%m').isin(['2023-01','2023-02']), 'guest_count'] = None
    #     filtered_data_2.loc[
    #         (filtered_data_2['shift_date'] >= f'{year}-04-26'), 'guest_count'] = None
    #     filtered_data_2['guest_count'] = filtered_data_2.groupby(['restaurant_id', 'dow'])['guest_count'].transform(
    #         lambda x: x.ffill())
    #     filtered_data_2 = filtered_data_2.set_index(['restaurant_id', 'shift_date']).combine_first(
    #         filtered_data_1.set_index(['restaurant_id', 'shift_date'])).reset_index()
    #     final_data = pd.concat(
    #         [filtered_data_1.loc[filtered_data_1['shift_date'].dt.strftime('%Y-%m') < f'{year}-{month}'],
    #          filtered_data_2.loc[filtered_data_2['shift_date'].dt.strftime('%Y-%m') == f'{year}-{month}']], axis=0,
    #         ignore_index=True)
    #     final_data.drop(columns=['year', 'dow'], inplace=True)
    #     return final_data

    # elif int(month) == 9:
    #     # filtered_data_2.loc[df['shift_date'].dt.strftime('%Y-%m').isin(['2023-01','2023-02']), 'guest_count'] = None
    #     filtered_data_2.loc[
    #         (filtered_data_2['shift_date'] >= f'{year}-08-29'), 'guest_count'] = None
    #     filtered_data_2['guest_count'] = filtered_data_2.groupby(['restaurant_id', 'dow'])['guest_count'].transform(
    #         lambda x: x.ffill())
    #     filtered_data_2 = filtered_data_2.set_index(['restaurant_id', 'shift_date']).combine_first(
    #         filtered_data_1.set_index(['restaurant_id', 'shift_date'])).reset_index()
    #     final_data = pd.concat(
    #         [filtered_data_1.loc[filtered_data_1['shift_date'].dt.strftime('%Y-%m') < f'{year}-{month}'],
    #          filtered_data_2.loc[filtered_data_2['shift_date'].dt.strftime('%Y-%m') == f'{year}-{month}']], axis=0,
    #         ignore_index=True)
    #     final_data.drop(columns=['year', 'dow'], inplace=True)
    #     return final_data

    else:
        filtered_data_1.drop(columns=['year', 'dow'], inplace=True)
        return filtered_data_1

    # if year_month != '2023-02':
    #     filtered_data_1.drop(columns=['year', 'dow'], inplace=True)
    #     return filtered_data_1
    # else:
    #     filtered_data_2.loc[df['shift_date'].dt.strftime('%Y-%m').isin(['2023-01', '2023-02']), 'guest_count'] = None
    #     filtered_data_2['guest_count'] = filtered_data_2.groupby(['restaurant_id', 'dow'])['guest_count'].transform(
    #         lambda x: x.ffill())
    #     final_data = filtered_data_2.set_index(['restaurant_id', 'shift_date']).combine_first(
    #         filtered_data_1.set_index(['restaurant_id', 'shift_date'])).reset_index()
    #     final_data.drop(columns=['year', 'dow'], inplace=True)
    #     return final_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--filtered_data_fill", default="./Data_Processing/result/Data_Processing.parquet", type=str)
    # parser.add_argument("--df_holiday_weight",
    #                     default="./<private_path_removed>",
    #                     type=str)
    # parser.add_argument("--dim_restaurant",
    #                     default="./<private_path_removed>",
    #                     type=str)
    parser.add_argument("--feature_data", default=None,
                        type=str)
    # parser.add_argument("--time", default="2023-08", type=str)
    parser.add_argument('--time', nargs='+', help='A list of time values')
    # parser.add_argument("--output_dir", default="./Forecasting_Result", type=str, help="for saving results")
    args = parser.parse_args()
    # print("Loading arguments:")
    list_prediction = args.time
    path_feature_data = args.feature_data
    # filtered_data_fill = args.filtered_data_fill
    # df_holiday_path = args.df_holiday_weight
    # df_dataset_path = args.dataset
    # path_dim_restaurant = args.dim_restaurant

    configFilePath = "./Config/config.ini"
    print('Loading metadata path from config file')
    config = configparser.RawConfigParser()
    config.read(configFilePath)
    config_dict = dict(config.items('config'))
    path_sale_store = config_dict['path_sale_store']
    path_dim_restaurant = config_dict['path_dim_restaurant']
    path_processed_data = config_dict['path_processed_data']
    path_output_processed_data = config_dict['path_output_processed_data']
    # path_feature_data = config_dict['path_feature_data']
    path_holiday = config_dict['path_holiday']
    path_cluster = config_dict['path_cluster']
    path_cluster_km = config_dict['path_cluster_km']
    path_first_month = config_dict['path_first_month']
    final_date = config_dict['end_date']
    end_date = config_dict['end_date']
    start_date = config_dict['start_date']
    path_prediction_data = config_dict['path_prediction_data']
    path_local_result = config_dict['path_local_result']
    path_global_result = config_dict['path_global_result']
    path_train_set = config_dict['path_train_set']
    path_test_set = config_dict['path_test_set']
    path_dataset = config_dict['path_dataset']
    path_pre_holiday = config_dict['path_pre_holiday']


    print("Successfully for loading metadata path")
    print("{:<30}{}".format("path_sale_store:", path_sale_store))
    # print('path_sale_store: ', path_sale_store)
    # print('path_processed_data: ', path_processed_data)
    print("{:<30}{}".format("path_processed_data:", path_processed_data))
    # print('path_feature_data: ', path_feature_data)
    print("{:<30}{}".format("path_feature_data:", path_feature_data))
    # print('path_dim_restaurant: ', path_dim_restaurant)
    print("{:<30}{}".format("path_dim_restaurant:", path_dim_restaurant))
    # print('df_holiday_weight: ', path_holiday)
    print("{:<30}{}".format("path_holiday:", path_holiday))
    # print('path_cluster: ', path_cluster)
    print("{:<30}{}".format("path_cluster:", path_cluster))
    # print('path_cluster_km: ', path_cluster_km)
    # print('path_first_month: ', path_first_month)
    print("{:<30}{}".format("path_first_month:", path_first_month))
    # print('final_date: ', final_date)
    print("{:<30}{}".format("start_date:", start_date))
    print("{:<30}{}".format("end_date:", end_date))
    print("{:<30}{}".format("final_date:", final_date))

    # print('list_prediction: ', list_prediction)
    print("{:<30}{}".format("list_prediction:", list_prediction))
    # print('path_global_result: ', path_global_result)
    print("{:<30}{}".format("path_global_result:", path_global_result))
    print("{:<30}{}".format("path_train_set:", path_train_set))
    print("{:<30}{}".format("path_test_set:", path_test_set))
    print("{:<30}{}".format("path_dataset:", path_dataset))
    # print('path_prediction_data: ', path_prediction_data)
    print('Please check the metadata path before running the program')

    # # print loading agruments and print agruments to check before running
    # print("Processed data: ", path_processed_data)
    # print("Holiday_path: ", path_holiday)
    # print("Dim restaurant: ", path_dim_restaurant)
    # print("Feature dataset: ", path_feature_data)
    # print("Prediction list: ", list_prediction)
    # print("Output folder: ", output_dir)
    # Create a loop for waiting the user press to continue

    while True:
        user_input = input("Press any key to continue...")
        if user_input == "":
            continue
        else:
            break

    df_dims = pd.read_parquet(
        path_dim_restaurant,
        filesystem=hdfs,
    )
    df_cluster_km = pd.read_parquet(
        path_cluster_km,
        # filesystem=hdfs,
    )

    # load data from "Preprocessing dataset"
    df_processed_data = pd.read_parquet(
        path_processed_data, )
    df_processed_data = df_processed_data.rename(columns={"date": "shift_date"})
    df_processed_data = df_processed_data.merge(df_cluster_km[['restaurant_id', 'cluster_km']], on='restaurant_id',
                                                how='left')
    print("{:<80}{}".format("Loading processed dataset", "SUCCESS"))
    

    ######################
    df_holiday = pd.read_parquet(
        path_holiday,
        # filesystem=hdfs,
    )

    # Link with holiday dataset
    df_holiday = df_holiday.rename(columns={"date": "shift_date"})
    print("{:<80}{}".format("Loading holiday data", "SUCCESS"))

    df_pre_holiday = pd.read_parquet(
        path_pre_holiday,
        filesystem=hdfs,
    )
    # Link Holiday
    df_processed_data = link_withHoliday.Link_Holiday_updated(df_processed_data, df_holiday, df_pre_holiday)
    print("{:<80}{}".format("Link holiday + holiday_weight", "SUCCESS"))

    print("List time to predict: ", list_prediction)
    # df_time = ["2023-05", "2023-06", "2023-07"]

    dfs = []
    dfs_test = []
    prediction_cluster = ['global']
    df_processed_data = df_processed_data[
        (df_processed_data['cluster_km'].isin(prediction_cluster))]
    print("Number of cluster: ", prediction_cluster)

    for time in list_prediction:
        print("Prediction time: ", time)

        if path_feature_data is None:
            print("Create dataset")
            dataset = df_processed_data.copy()

            # Filter restaurant active by time
            res_id_active = df_dims.loc[(df_dims.open_date < pd.to_datetime(time)) & (
                    pd.to_datetime(time) < df_dims.close_date), 'restaurant_id'].unique()
            print("Restaurant active: ", len(res_id_active))
            dataset = dataset.loc[dataset['restaurant_id'].isin(res_id_active)]
            print("Restaurant active after filter: ", len(dataset['restaurant_id'].unique()))
            dataset = process_data_prediction_time(dataset, time)

            # Create global features
            print("Create global features")
            data_feature = Basic_Features.create_features_glob(dataset)

            """
            # Create specific statistic features
            print("Create specific statistic features")
            df_cal_fourier = dataset[["shift_date", "total_sale", "total_bill", "guest_count", "restaurant_id"]]
            df_cal_fourier = df_cal_fourier.set_index("shift_date")
            tqdm.pandas(desc="Progress")
            X_lag_fourier_window = df_cal_fourier.groupby(["restaurant_id"], group_keys=True).progress_apply(
                Statistic_features.create_features_lag_window)
            del X_lag_fourier_window["restaurant_id"]
            X_lag_fourier_window = X_lag_fourier_window.reset_index()
            X_lag_fourier_window = X_lag_fourier_window.rename(columns={"shift_date": "date"})
            print("{:<80}{}".format("Created specific statistic features", "SUCCESS"))

            # Create profile features
            # Create category features (profile)
            data_feature = Profile_features.label_brand(data_feature)
            data_feature = Profile_features.label_city(data_feature)
            data_feature = Profile_features.label_region(data_feature)
            data_feature = Profile_features.label_concept_detail(data_feature)
            data_feature = Profile_features.label_is_mall(data_feature)
            data_feature = pd.merge(data_feature, X_lag_fourier_window, on=['date', 'restaurant_id'], how='left')
            """

            # Create category features (profile)
            one_hot_features = [
                # 'brand', 'city', 'concept_detail','sbu',
                # 'dayofweek', 'holiday', 'pre_holiday', 'weekend',
                'cluster_km',
                # 'moon_day','moon_month',
                # 'month',
                # 'dayofweek',
                # 'dayofmonth',
            ]

            # Make sure to convert non-categorical columns to 'category' or 'object' type
            data_feature[one_hot_features] = data_feature[one_hot_features].astype('category')
            encoder = OneHotEncoder(variables=one_hot_features,
                                    drop_last=True)

            data_feature = encoder.fit_transform(data_feature)
            print(f"Transform one hot encoder for category features {one_hot_features}")

            # Save feature dataset
            path_dataset_time = path_dataset + f"{time}.parquet"
            data_feature.to_parquet(path_dataset_time, index=False)
            print("{:<80}{}".format(f"Save feature dataset to {path_dataset_time}", "SUCCESS"))

        else:
            print("Load dataset")
            path_dataset_time = path_dataset + f"{time}.parquet"
            data_feature = pd.read_parquet(path_dataset_time)
            print("{:<80}{}".format(f"Load feature dataset from {path_dataset_time}", "SUCCESS"))

        # Split dataset
        train_data, val_data, test_data = split_dataset.Config_Trainning_updated_V2(data_feature, time)

        # Save train, val, test set
        file_train_time = path_train_set + f"train-{time}.parquet"
        pd.concat([train_data, val_data], axis=0).to_parquet(file_train_time, index=True)
        print("{:<80}{}".format(f"Save train set to {file_train_time}", "SUCCESS"))

        file_test_time = path_test_set + f"test-{time}.parquet"
        test_data.to_parquet(file_test_time, index=True)
        print("{:<80}{}".format(f"Save test set to {file_test_time}", "SUCCESS"))

        X_train, y_train = Basic_Features.create_features_old(train_data, label='y')
        X_val, y_val = Basic_Features.create_features_old(val_data, label='y')
        X_test, y_test = Basic_Features.create_features_old(test_data, label='y')

        # reg = xgb.XGBRegressor(n_estimators=50)
        # reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.02, max_depth=8, verbosity=0, silent=True,
        #                        random_state=42)

        reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=8, verbosity=0, silent=False,
                               random_state=42, min_child_weight=1, gamma=0.5, subsample=0.8, colsample_bytree=0.8,
                               n_jobs=-1, )

        # Gộp tập huấn luyện và tập xác thực để tạo tập dữ liệu huấn luyện mới
        X_train_combined = pd.concat([X_train, X_val], axis=0)
        y_train_combined = pd.concat([y_train, y_val], axis=0)

        X_train_combined['holiday_weight'].replace([np.inf, np.nan], 1, inplace=True)
        X_test['holiday_weight'].replace([np.inf, np.nan], 1, inplace=True)

        reg.fit(X_train_combined, y_train_combined)

        data_list = data_feature['restaurant_id'].unique()
        print("Number of restaurant: ", len(data_list))

        for res_id in tqdm(data_list):
            # Cluster specific restaurant
            test_data_subset = test_data.loc[(test_data['restaurant_id'] == res_id)].copy()
            X_test, y_test = Basic_Features.create_features_old(test_data_subset, label='y')
            test_data_subset['number_Prediction'] = reg.predict(X_test)
            dfs_test.append(test_data_subset[['date', 'restaurant_id', 'y', 'number_Prediction']])

            try:
                y_true = test_data_subset['y']
                y_pred = test_data_subset['number_Prediction']
            except:
                continue

            try:
                wape = Evaluation_metric.weighted_absolute_percentage_error_v2(y_true, y_pred)
            except:
                wape = np.nan
            mape_values, ave_mape = Evaluation_metric.m_absolute_percentage_error(y_true, y_pred)
            positive_mape_values, negative_mape_values, positive_mape, negative_mape = Evaluation_metric.m_absolute_percentage_error_positive_negative(
                y_true, y_pred)

            start_date = pd.to_datetime(time)

            data_analysis = {
                'Restaurant_ID': [res_id],
                'Time': [start_date],
                'Pred': [test_data_subset['number_Prediction'].values.tolist()],
                'Actual': [test_data_subset['y'].values.tolist()],
                'Average_WAPE %': [wape],
                'Average_MAPE %': [ave_mape],
                'Positive_MAPE %': [positive_mape],
                'Negative_MAPE %': [negative_mape],
            }

            df_analysis = pd.DataFrame(data_analysis)
            dfs.append(df_analysis)

    df_final_test = pd.concat(dfs_test, ignore_index=True)
    df_total_analysis = pd.concat(dfs, ignore_index=True)

    # Save prediction
    df_final_test.reset_index().to_parquet(path_global_result, index=False)
    print("{:<80}{}".format(f"Save prediction to {path_global_result}", "SUCCESS"))

    # Save performance
    path_performance_global = path_prediction_data + "global_performance.parquet"
    df_total_analysis.reset_index().to_parquet(path_performance_global, index=False)
    print("{:<80}{}".format(f"Save performance to {path_performance_global}", "SUCCESS"))
