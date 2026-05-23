import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats

import numpy as np
import json
import torch
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

import warnings

warnings.filterwarnings('ignore')  # setting ignore as a parameter

import logging

logging.getLogger('fbprophet').setLevel(logging.WARNING)
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


############################
# data preparation

def prep_data_for_holiday_feature(
        df_sales,
        df_date,
        start_date=dt.datetime(2022, 1, 1),
        end_date=dt.datetime(2022, 12, 31)
) -> pd.DataFrame:
    '''
    Prep data to calculate holiday weight feature.
    
    Parameters:
    df_sales: sales_store data frame, including restaurant id, date, daily guest count and other information
    df_date: date data frame, including date, holiday and other information about date
    start_date: Start date of based period used to calculate holiday weight.
    end_date: End date of based period used to calculate holiday weight.
    
    Returns:
    (dataframe): Prepared sales table, including date, restaurant id, guest count, weekday, holiday, month.
    '''

    # sales tbl ----- use dim_date table from prep step
    df_sales_hw = df_sales.copy()

    # date tbl ----- use dim_date table from prep step
    df_date_hw = df_date.copy()

    # prep sales tbl
    df_sales_hw.drop_duplicates(inplace=True)  # remove duplicates
    df_sales_hw.loc[df_sales_hw['guest_count'] < 0, 'guest_count'] = np.nan  # replace negative guest_count
    df_sales_hw['shift_date'] = df_sales_hw['shift_date'].astype('datetime64[ns]')  # change data type

    # mapping sales tbl vs date tbl to get date info
    df_sales_date = pd.merge(
        df_sales_hw[df_sales_hw['shift_date'].between(start_date, end_date)][
            ['shift_date', 'restaurant_id', 'guest_count']],
        df_date_hw[['date', 'weekday', 'holiday']].rename(columns={'date': 'shift_date'}),
        on='shift_date',
        how='left',
        validate='many_to_one'
    )

    # add month columns
    df_sales_date['month'] = df_sales_date['shift_date'].dt.month

    return df_sales_date


############################
# calculate holiday weight

def cal_holiday_weight(df_sales_date: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate weight of each holiday by restaurant.
    
    Parameters:
    df_sales_date: Data frame prepared for calculate holiday weight, including restaurant, guest count and date information.
    
    Returns:
    (dataframe): Weight of holidays by restaurant, including holiday, restaurant id, holiday weight.
    '''

    # create a sales tbl excluding holiday
    df_sales_excl_holiday = df_sales_date[(df_sales_date['holiday'] == 'Non_holiday')]

    # calculate mean tc of non-holiday date by day of week, month and restaurant
    df_exclhld_tc = df_sales_excl_holiday.groupby([
        df_sales_excl_holiday['month'],
        df_sales_excl_holiday['restaurant_id'],
        df_sales_excl_holiday['weekday']
    ])['guest_count'].mean().reset_index() \
        .rename(columns=({'guest_count': 'exclhld_guest_count'}))

    # mapping tc of non-holiday date by day of week, month and restaurant to sales tbl
    df_holiday_weight = pd.merge(
        df_sales_date[df_sales_date['holiday'] != 'Non_holiday'],
        df_exclhld_tc,
        on=['month', 'restaurant_id', 'weekday'],
        how='left',
        validate='many_to_one'
    )

    # Calcualate holiday weight
    df_holiday_weight['holiday_weight'] = df_holiday_weight['guest_count'].values / (
                df_holiday_weight['exclhld_guest_count'].values + 0.001)

    # post-processing
    df_holiday_weight.loc[df_holiday_weight['holiday_weight'].isnull(), 'holiday_weight'] = 1
    df_holiday_weight = df_holiday_weight.groupby(['restaurant_id', 'holiday']).holiday_weight.mean().reset_index()

    # keep columns
    df_holiday_weight = df_holiday_weight[['holiday', 'restaurant_id', 'holiday_weight']]

    return df_holiday_weight


############################
# remove ineffective holiday

def remove_ineffective_holiday(
        df_holiday_weight: pd.DataFrame,
        lower_bound=0.9,
        upper_bound=1.1
) -> pd.DataFrame:
    '''
    Calculate median weight of each holiday to remove the ineffective
    >>> Default rule to determine effective holiday: 
        + holiday_weight > 1.1 (upper bound)
        + or holiday_weight < 0.9 (lower bound)
    
    Parameters:
    df_holiday_weight: Holiday weight dataframe, including holiday, restaurant id, holiday weight.
    lower_bound: lower bound of holiday weight to determine whether holiday is effective or not
    upper_bound: upper bound of holiday weight to determine whether holiday is effective or not
    
    Returns:
    (dataframe): Dataframe weight of holidays by restaurant after removing ineffective holidays, including holiday, restaurant id, holiday weight.
    '''

    # calculate mean holiday weight
    df_hw_gb_hld = df_holiday_weight.groupby('holiday').holiday_weight.median().reset_index().rename(
        columns={'holiday_weight': 'median_hw'})

    # list ineffective holiday
    ineffective_hld = list(df_hw_gb_hld[df_hw_gb_hld['median_hw'].between(lower_bound, upper_bound)].holiday.unique())

    # heuristics removed holiday list
    heu_ineffective_hld = [
        'Ngày Thầy thuốc Việt Nam',
        'Ngày Truyền Thống Bộ đội Biên phòng',
        "Valentine's Black Day",
        'Ngày thành lập Đoàn Thanh niên Cộng sản Hồ Chí Minh',
        'Ngày Thành lập Mặt trận Dân tộc Thống nhất Việt Nam',
        'Ngày Quốc tế Hạnh phúc',
        'Ngày Quốc Tế Nam Giới',
        'Ngày Thể thao Việt Nam'
    ]

    # remove ineffective holiday
    df_eff_hw = df_holiday_weight[
        ~(df_holiday_weight['holiday'].isin(ineffective_hld))
        & ~(df_holiday_weight['holiday'].isin(heu_ineffective_hld))
        ].reset_index()

    return df_eff_hw


############################
# remove outlier holiday weight

def remove_outlier_holiday(
        df_eff_hw: pd.DataFrame,
        threshold=2
) -> pd.DataFrame:
    '''
    Remove outlier holiday weight by Z-score
    >>> Default threshold: 2
    
    Parameters:
    df_eff_hw: Dataframe weight of holidays by restaurant after removing ineffective holidays, including holiday, restaurant id, holiday weight.
    threshold: z-score threshold to determine outlier holiday weight
    
    Returns:
    (dataframe): Dataframe weight of holidays by restaurant after removing outlier, including holiday, restaurant id, holiday weight.
    '''

    # z-score for each record by holiday
    df_eff_hw['z_score'] = df_eff_hw.groupby('holiday')['holiday_weight'].transform(lambda x: np.abs(stats.zscore(x)))

    # remove records having z-score above threshold
    df_nooutlier_hw = df_eff_hw[df_eff_hw['z_score'] <= threshold].copy()

    # remove records with hw = 0 excl Tet
    list_tet_holiday = ["Vietnamese New Year's Eve", 'Vietnamese New Year', \
                        'The second day of Tet Holiday', 'The third day of Tet Holiday', \
                        'The forth day of Tet Holiday', 'The fifth day of Tet Holiday']
    df_nooutlier_hw = df_nooutlier_hw[
        (df_nooutlier_hw['holiday'].isin(list_tet_holiday))
        | ((~df_nooutlier_hw['holiday'].isin(list_tet_holiday)) & (df_nooutlier_hw['holiday_weight'] > 0))
        ].reset_index()

    # post-processing
    df_nooutlier_hw = df_nooutlier_hw.drop(columns='z_score')

    return df_nooutlier_hw


############################
# add missing holiday weight for restaurants

def add_missing_hw(df_nooutlier_hw: pd.DataFrame) -> pd.DataFrame:
    '''
    Fill missing holiday weight of restaurant by median holiday weight of other restaurants in the same brand (missing due to missing sales, abnormal sales, not opening...)
    >>> Note: If holiday weight of brand is null, fill missing value with 1
    
    Parameters:
    df_nooutlier_hw: Dataframe weight of holidays by restaurant after removing outlier, including holiday, restaurant id, holiday weight.
    
    Returns:
    (dataframe): Dataframe full weight of holidays by restaurant, including holiday, restaurant id, holiday weight.
    '''

    # list effective holiday
    hld_list = list(df_nooutlier_hw.holiday.unique())

    # list restaurant
    res_from_dim = set(
        df_res_prep[df_res_prep['status'] == 'active'].restaurant_id.unique())  # restaurant list from dim_res tbl
    res_from_sales = set(df_sales.restaurant_id.unique())  # restaurant list from sales_store tbl
    res_list = list(res_from_dim | res_from_sales)  # full list restaurant id

    # create a blank df combine res_id vs holiday (full res + full holiday)
    data = []
    for res in res_list:
        for hld in hld_list:
            data.append([res, hld])
    df_full_hw = pd.DataFrame(data, columns=['restaurant_id', 'holiday'])

    # mapping holiday weight to blank dataframe
    df_full_hw = pd.merge(
        df_full_hw,
        df_nooutlier_hw,
        on=['holiday', 'restaurant_id'],
        how='left',
        validate='one_to_one'
    )

    # mapping restaurant info
    df_full_hw = pd.merge(
        df_full_hw,
        df_res_prep[['restaurant_id', 'brand']],
        on='restaurant_id',
        how='left',
        validate='many_to_one'
    )

    # mapping median holiday weight by brand
    df_median_hw = df_full_hw[df_full_hw['holiday_weight'].isnull() == False].groupby(['brand', 'holiday'])[
        'holiday_weight'].median().reset_index()
    df_full_hw = pd.merge(
        df_full_hw,
        df_median_hw.rename(columns={'holiday_weight': 'median_hw'}),
        on=['holiday', 'brand'],
        how='left',
        validate='many_to_one'
    )

    # fill missing holiday weight of restaurant by mean holiday weight of other restaurants in the same brand
    df_full_hw['hw'] = df_full_hw['holiday_weight']
    df_full_hw.loc[df_full_hw['hw'].isnull(), 'hw'] = df_full_hw['median_hw']
    df_full_hw.loc[df_full_hw['hw'].isnull(), 'hw'] = 1
    df_full_hw = df_full_hw[['holiday', 'restaurant_id', 'hw']].rename(columns={'hw': 'holiday_weight'}).reset_index()

    return df_full_hw


############################
# mapping date to holiday dataframe
def add_date_to_holiday_df(
        df_full_hw: pd.DataFrame,
        start_date=dt.datetime(2021, 1, 1),
        end_date=dt.datetime(2024, 12, 31)
) -> pd.DataFrame:
    '''
    Map date values to holiday dataframe (from start date to end date defined in param).
    
    Parameters:
    df_full_hw: Dataframe full weight of holidays by restaurant, including holiday, restaurant id, holiday weight.
    start_date: Start date of period defined holiday weight.
    end_date: End date of period defined holiday weight.
    
    Returns:
    (dataframe): Prepared sales table, including date, restaurant id, guest count, weekday, holiday, month.
    '''

    # map date
    df_full_date_hw = pd.merge(
        df_full_hw,
        df_date[(df_date['date'].between(start_date, end_date))][['date', 'holiday']],
        on='holiday',
        how='left',
        validate='many_to_many'
    )

    # reorder columns
    df_full_date_hw = df_full_date_hw[['date', 'restaurant_id', 'holiday', 'holiday_weight']]

    return df_full_date_hw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--filtered_data_fill", default="./<private_path_removed>", type=str)
    parser.add_argument("--sale",
                        default="./Data_Processing/raw/sales_store.parquet",
                        type=str)
    parser.add_argument("--date",
                        default="./Data_Processing/raw/dim_date.parquet",
                        type=str)
    parser.add_argument("--res_prep",
                        default="./Data_Processing/raw/dim_restaurant.parquet",
                        type=str)
    parser.add_argument("--output_dir", default="./result_holiday", type=str, help="for saving results")
    args = parser.parse_args()

    path_sale = args.sale
    path_date = args.date
    path_res_prep = args.res_prep
    path_save = args.output_dir

    # Read data (--> replace by data from step prep after combine source)
    df_sales = pd.read_parquet(
        path_sale,
        filesystem=hdfs
    )

    df_date = pd.read_parquet(
        path_date,
        filesystem=hdfs,
    )

    df_res_prep = pd.read_parquet(
        path_res_prep,
        filesystem=hdfs,
    )

    #####################
    # Extract feature: Holiday_weight

    # Step 1: Prepare data to calculate holiday weight
    df_sales_date = prep_data_for_holiday_feature(
        df_sales,
        df_date,
        start_date=dt.datetime(2022, 1, 1),
        end_date=dt.datetime(2022, 12, 31)
    )
    print('>>> Finish step 1: Prepare data to calculate holiday weight')

    # Step 2: Calculate holiday weight
    df_holiday_weight = cal_holiday_weight(df_sales_date)
    print('>>> Finish step 2: Calculate holiday weight')

    # Step 3: Remove ineffective holiday
    df_eff_hw = remove_ineffective_holiday(
        df_holiday_weight,
        lower_bound=0.9,
        upper_bound=1.1
    )
    print('>>> Finish step 3: Remove ineffective holiday')

    # Step 4: Remove outlier holiday weight
    df_nooutlier_hw = remove_outlier_holiday(df_eff_hw, threshold=2)
    print('>>> Finish step 4: Remove outlier holiday weight')

    # Step 5: Add missing holiday weight for restaurants
    df_full_hw = add_missing_hw(df_nooutlier_hw)
    print('>>> Finish step 5: Add missing holiday weight for restaurants')

    # Step 6: Mapping date to holiday dataframe
    df_full_date_hw = add_date_to_holiday_df(
        df_full_hw,
        start_date=dt.datetime(2021, 1, 1),
        end_date=dt.datetime(2025, 12, 31)
    )
    print('>>> Finish step 6: Mapping date to holiday dataframe')

    # Save result
    dt = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    # path_to_save_1 = f"./<private_path_removed>"
    path_to_save = f"./result_holiday/holiday_weight.parquet"
    try:
        df_full_date_hw.reset_index().to_parquet(path_to_save, index=False)
        print("Saved result successfully")
    except Exception as e:
        print(f"Failed to save result: {str(e)}")
