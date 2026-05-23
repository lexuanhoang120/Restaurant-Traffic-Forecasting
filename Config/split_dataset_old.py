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

from Holiday import link_withHoliday
from Features import Basic_Features, Profile_features
from Config import split_dataset



def Config_Trainning_updated_V2(df_FromSeries, time):
    """
    Thiết lập mốc thời gian cho quá trình train/val/test    

    Tham số:
    - df_FromSeries: là dataframe cần được chia trước trong quá trình train/val/test
    - time: là tháng cần dự báo 

    Trả về:
    - train_data, val_data, test_data: 3 tập dữ liệu ứng với 3 phần train/val/test
    """
    df = df_FromSeries

    start_date = pd.to_datetime(time)
    start_date_plus_before_month = start_date - pd.DateOffset(months=1)
    start_date_plus_after_month = start_date + pd.DateOffset(months=1)

    #Chia theo tháng
    train_start_date = pd.to_datetime('01/2022')
    val_start_date = pd.to_datetime(start_date_plus_before_month)
    test_start_date = pd.to_datetime(time)
    test_end_date = pd.to_datetime(start_date_plus_after_month)
    
    train_data = df[(df['date'] >= train_start_date) & (df['date'] < val_start_date)]
    # print(train_data)
    val_data = df[(df['date'] >= val_start_date) & (df['date'] < test_start_date)]
    # print(val_data)
    test_data = df[(df['date'] >= test_start_date) & (df['date'] < test_end_date)]
    # print(test_data)
    
    
    return train_data, val_data, test_data