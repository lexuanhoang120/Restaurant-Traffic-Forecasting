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


def Link_Holiday_updated(df, df_holiday, df_pre_holiday):
    """
    Kết hợp dữ liệu Holiday và tiền xử lí dữ liệu về định dạng chuẩn

    Tham số:
    - df: Dữ liệu từ bảng "Sale_Store" sau khi đã tiền xử lí
                   missing và các phần bị âm

    Trả về:
    - merged_df_next: Là dataframe sau khi đã kết hợp với dữ liệu Holiday
                    và tiền xử lí
    """

    # Nhóm với 2 bảng dữ liệu holiday
    merged_df_next = pd.merge(df, df_holiday[
        ['shift_date', 'restaurant_id', 'holiday', 'holiday_weight']] \
                              , on=['shift_date', 'restaurant_id'], how='left')
    merged_df_next['holiday_weight'].fillna(1, inplace=True)
    merged_df_next['holiday_weight'].replace({0: 1}, inplace=True)
    # Tiền xử lí các giá trị bị Nan và
    merged_df_next['holiday'].fillna(0, inplace=True)
    merged_df_next['holiday'] = merged_df_next['holiday'].apply(lambda x: True if x != 0 else False)
    merged_df_next = pd.merge(merged_df_next, df_pre_holiday, on=['shift_date', 'restaurant_id'], how='left')
    merged_df_next['pre_holiday'].fillna(0, inplace=True)
    merged_df_next['pre_holiday'] = merged_df_next['pre_holiday'].apply(lambda x: True if x != 0 else False)
    merged_df_next['pre_holiday'] = merged_df_next['pre_holiday'].astype('bool')

    # Tạo cột mới "new_weekday" dựa trên cột "weekday"
    merged_df_next['dayofweek'] = merged_df_next['shift_date'].dt.dayofweek
    # Cast sang kiểu dữ liệu bool
    merged_df_next['weekend'] = merged_df_next['dayofweek'].apply(lambda x: 1 if x in [5, 6] else 0)
    merged_df_next['weekend'] = merged_df_next['weekend'].astype('bool')
    # Cast sang kiểu dữ liệu bool
    merged_df_next['is_Saturday'] = merged_df_next['dayofweek'].apply(lambda x: 1 if x == 5 else 0)
    merged_df_next['is_Saturday'] = merged_df_next['is_Saturday'].astype('bool')
    # Cast sang kiểu dữ liệu bool
    merged_df_next['is_Sunday'] = merged_df_next['dayofweek'].apply(lambda x: 1 if x == 6 else 0)
    merged_df_next['is_Sunday'] = merged_df_next['is_Sunday'].astype('bool')
    merged_df_next['dayofmonth'] = merged_df_next['shift_date'].dt.day
    merged_df_next['is_first_moon_day'] = merged_df_next['moon_day'].apply(lambda x: 1 if int(x) == 1 else 0)
    merged_df_next['is_first_moon_day'] = merged_df_next['is_first_moon_day'].astype('bool')
    merged_df_next['is_half_moon_day'] = merged_df_next['moon_day'].apply(lambda x: 1 if int(x) == 15 else 0)
    merged_df_next['is_half_moon_day'] = merged_df_next['is_half_moon_day'].astype('bool')
    

    merged_df_next.drop(columns=['dayofweek'], inplace=True)

    return merged_df_next
