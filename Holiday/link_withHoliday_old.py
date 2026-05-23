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


def Link_Holiday_updated(df_FromSeries, df_holiday_weight_newupdated):
    """
    Kết hợp dữ liệu Holiday và tiền xử lí dữ liệu về định dạng chuẩn 

    Tham số:
    - df_FromSeries: Dữ liệu từ bảng "Sale_Store" sau khi đã tiền xử lí 
                   missing và các phần bị âm

    Trả về:
    - merged_df_next: Là dataframe sau khi đã kết hợp với dữ liệu Holiday
                    và tiền xử lí
    """
    
    # Nhóm với 2 bảng dữ liệu holiday
    try:
        merged_df_next = pd.merge(df_FromSeries, df_holiday_weight_newupdated[['shift_date', 'restaurant_id', 'holiday', 'holiday_weight']] \
                              , on=['shift_date', 'restaurant_id'], how = 'left')
        merged_df_next['holiday_weight'].fillna(1, inplace = True)
        merged_df_next['holiday_weight'].replace({0: 1}, inplace = True)
    
        # Tiền xử lí các giá trị bị Nan và 
        merged_df_next['holiday'].fillna(0, inplace = True)
        merged_df_next['holiday'] = merged_df_next['holiday'].apply(lambda x: 1 if x != 0 else 0)
    
        # Tạo cột mới "new_weekday" dựa trên cột "weekday"
        merged_df_next['dayofweek'] = merged_df_next['shift_date'].dt.dayofweek
        merged_df_next['new_weekday'] = merged_df_next['dayofweek'].apply(lambda x: 0 if x in [5, 6] else 1)
    
        merged_df_next.drop(columns=['dayofweek'], inplace=True)
    except Exception as e:
        print(f"Failed to link holiday detail: {str(e)}")
    
    return merged_df_next