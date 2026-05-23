import calendar
import os
import sys
import subprocess
import numpy as np
# import matplotlib.pyplot as plt

from tqdm import tqdm
from pyarrow import fs
import pyarrow.parquet as pq

import pandas as pd
import argparse
from datetime import datetime
import shutil
import logging
import os
import sys
import subprocess

from pyarrow import fs
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

#---

import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')  # setting ignore as a parameter
import configparser
import logging

logging.getLogger('xgboost').setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True

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
    return last_day




def check_early_sale_store(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(by=['shift_date'], inplace=True)
    df = df.drop_duplicates(subset=['shift_date', 'restaurant_id'], keep='last')
    df['shift_date'] = df['shift_date'].astype('datetime64[ns]')
    return df


def prep_data_for_holiday_feature(
    df_sales,
    df_date,
    df_res_prep,
    start_date=dt.datetime(2023,1,1),
    end_date=dt.datetime(2023,12,31)
) -> pd.DataFrame:
    '''
    Prep data to calculate holiday weight feature.

    Parameters:
        >>> df_sales: sales_store data frame, including restaurant id, date, daily guest count and other information
        >>> df_date: date data frame, including date information
        >>> df_res: restaurant data frame, including restaurant_id, and other restaurant profile information
        >>> start_date: Start date of based period used to calculate holiday weight.
        >>> end_date: End date of based period used to calculate holiday weight.

    Returns:
    df_sales_date: Prepared sales table, including date, restaurant id, guest count, weekday, holiday, month.
    '''

    # sales tbl
    df_sales_hw = df_sales.copy()

    # date tbl
    df_date_hw = df_date.copy()

    # prep sales tbl
    df_sales_hw = df_sales_hw[df_sales_hw['restaurant_id'].isnull()==False]
    df_sales_hw.drop_duplicates(subset=['shift_date', 'restaurant_id'], inplace=True) # remove duplicates
    df_sales_hw.loc[df_sales_hw['guest_count']<0, 'guest_count'] = np.nan # replace negative guest_count
    df_sales_hw['shift_date'] = df_sales_hw['shift_date'].astype('datetime64[ns]') # change data type

    # add missing date for restaurants
    ## create full date period
    full_date_period = pd.date_range(start = start_date, end = end_date, freq = 'D')
    ## create full restaurant list
    res_from_dim = set(df_res_prep[df_res_prep['status']=='active'].restaurant_id.unique()) # restaurant list from dim_res tbl
    res_from_sales = set(df_sales[df_sales['restaurant_id'].isnull()==False].restaurant_id.unique()) # restaurant list from sales_store tbl
    res_list = list(res_from_dim|res_from_sales) # full list restaurant id
    ## create continuous date range for each restaurant
    dfr = []
    for res in res_list:
        df_sales_by_res = df_sales_hw[df_sales_hw['restaurant_id'] == res][['shift_date', 'restaurant_id', 'guest_count']].copy()
        df_full_sales_by_res = pd.DataFrame({'shift_date': full_date_period})
        df_full_sales_by_res['restaurant_id'] = res
        # merge sales data
        df_full_sales_by_res = df_full_sales_by_res.merge(
            df_sales_by_res,
            on=['shift_date', 'restaurant_id'],
            how='left',
            validate='1:1'
        )
        # df_full_sales_by_res['guest_count'] = df_full_sales_by_res['guest_count'].fillna(0)
        dfr.append(df_full_sales_by_res)
    df_full_sales = pd.concat(dfr).reset_index(drop=True)

    # mapping sales tbl vs date tbl to get date info
    df_sales_date = pd.merge(
        df_full_sales[df_full_sales['shift_date'].between(start_date, end_date)][['shift_date', 'restaurant_id', 'guest_count']],
        df_date_hw[['date', 'weekday', 'holiday']].rename(columns={'date': 'shift_date'}),
        on='shift_date',
        how='left',
        validate='many_to_one'
    ).reset_index(drop=True)

    # fill missing guest_count for holiday
    df_sales_date.loc[(df_sales_date['holiday']!='Non_holiday') & (df_sales_date['guest_count'].isnull()), 'guest_count'] = 0

    # add month columns
    df_sales_date['month'] = df_sales_date['shift_date'].dt.month

    return df_sales_date

def cal_holiday_weight(df_sales_date: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate weight of each holiday by restaurant.

    Parameters:
        >>> df_sales_date: Data frame prepared for calculate holiday weight, including restaurant, guest count and date information.

    Returns:
    df_holiday_weight: Weight of holidays by restaurant, including holiday, restaurant id, holiday weight.
    '''

    # create a sales tbl excluding holiday
    df_sales_excl_holiday = df_sales_date[(df_sales_date['holiday'] == 'Non_holiday')]

    # calculate mean tc of non-holiday date by day of week, month and restaurant
    df_exclhld_tc = df_sales_excl_holiday.groupby([
        df_sales_excl_holiday['month'],
        df_sales_excl_holiday['restaurant_id'],
        df_sales_excl_holiday['weekday']
    ])['guest_count'].mean().reset_index()\
    .rename(columns = ({'guest_count': 'exclhld_guest_count'}))

    # mapping tc of non-holiday date by day of week, month and restaurant to sales tbl
    df_holiday_weight = pd.merge(
        df_sales_date[df_sales_date['holiday'] != 'Non_holiday'],
        df_exclhld_tc,
        on=['month', 'restaurant_id', 'weekday'],
        how='left',
        validate='many_to_one'
    )

    # Calcualate holiday weight
    df_holiday_weight['holiday_weight'] = df_holiday_weight['guest_count'].values / (df_holiday_weight['exclhld_guest_count'].values)

    # post-processing
    df_holiday_weight = df_holiday_weight.loc[~(df_holiday_weight['holiday_weight'] == np.inf)]
    df_holiday_weight = df_holiday_weight.loc[~(df_holiday_weight['holiday_weight'].isnull())]
    df_holiday_weight = df_holiday_weight.groupby(['restaurant_id', 'holiday']).holiday_weight.mean().reset_index()

    # keep columns
    df_holiday_weight = df_holiday_weight[['holiday', 'restaurant_id', 'holiday_weight']]

    return df_holiday_weight

def remove_outlier_holiday(
    df_holiday_weight: pd.DataFrame,
    threshold=3
) -> pd.DataFrame:
    '''
    Remove outlier holiday weight by Z-score
    >>> Default threshold: 3

    Parameters:
        >>> df_eff_hw: Dataframe weight of holidays by restaurant after removing ineffective holidays, including holiday, restaurant id, holiday weight.
        >>> threshold: z-score threshold to determine outlier holiday weight

    Returns:
    df_nooutlier_hw: Dataframe weight of holidays by restaurant after removing outlier, including holiday, restaurant id, holiday weight.
    '''

    # z-score for each record by holiday
    df_holiday_weight['z_score'] = df_holiday_weight.groupby('holiday')['holiday_weight'].transform(lambda x: np.abs(stats.zscore(x)))

    # remove records having z-score above threshold
    df_nooutlier_hw = df_holiday_weight[df_holiday_weight['z_score'] <= threshold].copy()

    # remove records with hw = 0 excl Tet
    list_tet_holiday = ["Vietnamese New Year's Eve", 'Vietnamese New Year',\
                        'The second day of Tet Holiday', 'The third day of Tet Holiday',\
                        'The forth day of Tet Holiday', 'The fifth day of Tet Holiday']
    df_nooutlier_hw = df_nooutlier_hw[
        (df_nooutlier_hw['holiday'].isin(list_tet_holiday))
        | ((~df_nooutlier_hw['holiday'].isin(list_tet_holiday)) & (df_nooutlier_hw['holiday_weight']>0))
    ].reset_index(drop=True)

    # post-processing
    df_nooutlier_hw = df_nooutlier_hw.drop(columns='z_score')

    return df_nooutlier_hw

def remove_ineffective_holiday(
    df_nooutlier_hw: pd.DataFrame,
    lower_bound=0.9,
    upper_bound=1.1
) -> pd.DataFrame:
    '''
    Calculate median weight of each holiday to remove the ineffective
    >>> Default rule to determine effective holiday:
        + holiday_weight > 1.1 (upper bound)
        + or holiday_weight < 0.9 (lower bound)
    >>> Exception:
        + keep all Tet holiday, ignore holiday weight rule due to difference in effect on different restaurants
        + remove ineffective holidays by heuristics, ignore holiday weight rule due to effect of other factors

    Parameters:
        >>> df_holiday_weight: Holiday weight dataframe, including holiday, restaurant id, holiday weight.
        >>> lower_bound: lower bound of holiday weight to determine whether holiday is effective or not
        >>> upper_bound: upper bound of holiday weight to determine whether holiday is effective or not

    Returns:
    df_eff_hw: Dataframe weight of holidays by restaurant after removing ineffective holidays, including holiday, restaurant id, holiday weight.
    '''

    # calculate mean holiday weight
    df_hw_gb_hld = df_nooutlier_hw.groupby('holiday').holiday_weight.median()\
                    .reset_index()\
                    .rename(columns={'holiday_weight': 'median_hw'})

    # list of ineffective holidays by rule
    ls_ineffective_hld = list(df_hw_gb_hld[df_hw_gb_hld['median_hw'].between(lower_bound, upper_bound)].holiday.unique())

    # list of Tet holidays to keep
    ls_tet = [
        "Vietnamese New Year's Eve",
        'Vietnamese New Year',
        'The second day of Tet Holiday',
        'The third day of Tet Holiday',
        'The forth day of Tet Holiday',
        'The fifth day of Tet Holiday'
    ]

    # list of ineffective holidays to remove by heuristics
    ls_heu_ineffective_hld = [
        'Ngày Thầy thuốc Việt Nam','Ngày Truyền Thống Bộ đội Biên phòng',"Valentine's Black Day","Valentine's White Day",'Ngày thành lập Đoàn Thanh niên Cộng sản Hồ Chí Minh',
        'Ngày Thành lập Mặt trận Dân tộc Thống nhất Việt Nam','Ngày Quốc tế Hạnh phúc','Ngày Quốc Tế Nam Giới','Ngày chiến thắng Điện Biên Phủ','Ngày sách Việt Nam',
        'Ngày Thể thao Việt Nam','Ngày Bảo hiểm Y tế Việt Nam','Ngày Khuyến học Việt Nam','Ngày Pháp luật Việt Nam','Ngày sinh của Chủ tịch Hồ Chí Minh',
        'Ngày thành lập Hội Nông dân Việt Nam','Ngày thành lập Công đoàn Việt Nam','Ngày Thành lập Hội Cựu Chiến binh Việt Nam','Ngày thành lập Lực Lượng Dân Quân Tự Vệ',
        'Ngày thành lập Đội Thiếu niên Tiền phong Hồ Chí Minh','Ngày Nước sạch Thế giới','Ngày giải phóng thủ đô','Ngày Thành lập Mặt trận Dân tộc Giải phóng miền Nam',
        'Ngày thành lập Quân đội Nhân dân Việt Nam','Ngày thành lập Đảng Cộng sản Việt Nam','Ngày Truyền Thống Học sinh, Sinh viên Việt Nam','Lễ Vu-lan'
    ]

    # final list of ineffective holidays to remove --- [ineffective holidays by rule] + [ineffective holidays to remove by heuristics] - [Tet holidays]
    ls_ineff_hld = list(set(ls_ineffective_hld + ls_heu_ineffective_hld))
    ls_final_remove = [x for x in ls_ineff_hld if x not in ls_tet]

    # remove ineffective holiday
    df_eff_hw = df_nooutlier_hw[
        ~(df_nooutlier_hw['holiday'].isin(ls_final_remove))
    ].reset_index(drop=True)

    return df_eff_hw

def add_missing_hw(df_eff_hw: pd.DataFrame,df_res_prep: pd.DataFrame, df_sales: pd.DataFrame) -> pd.DataFrame:
    '''
    Fill missing holiday weight of restaurant by median holiday weight of other restaurants in the same brand (missing due to missing sales, abnormal sales, not opening...)
    >>> Note: If holiday weight of brand is null, fill missing value with 1

    Parameters:
        >>> df_nooutlier_hw: Dataframe weight of holidays by restaurant after removing outlier, including holiday, restaurant id, holiday weight.

    Returns:
    df_full_hw: Dataframe full weight of holidays by restaurant, including holiday, restaurant id, holiday weight.
    '''

    # list effective holiday
    hld_list = list(df_eff_hw.holiday.unique())

    # list restaurant
    res_from_dim = set(df_res_prep[df_res_prep['status']=='active'].restaurant_id.unique()) # restaurant list from dim_res tbl
    res_from_sales = set(df_sales.restaurant_id.unique()) # restaurant list from sales_store tbl
    res_list = list(res_from_dim|res_from_sales) # full list restaurant id

    # create a blank df combine res_id vs holiday (full res + full holiday)
    data = []
    for res in res_list:
        for hld in hld_list:
            data.append([res, hld])
    df_full_hw = pd.DataFrame(data, columns=['restaurant_id', 'holiday'])

    # mapping holiday weight to blank dataframe
    df_full_hw = pd.merge(
        df_full_hw,
        df_eff_hw,
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
    df_median_hw = df_full_hw[df_full_hw['holiday_weight'].isnull()==False].groupby(['brand', 'holiday'])['holiday_weight'].median().reset_index()
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
    df_full_hw = df_full_hw[['holiday', 'restaurant_id', 'hw']].rename(columns={'hw': 'holiday_weight'}).reset_index(drop=True)

    return df_full_hw
    
def add_date_to_holiday_df(
    df_full_hw: pd.DataFrame,
    start_date=dt.datetime(2021, 1, 1),
    end_date=dt.datetime(2024, 12, 31)
) -> pd.DataFrame:

    '''
    Map date values to holiday dataframe (from start date to end date defined in param).

    Parameters:
        >>> df_full_hw: Dataframe full weight of holidays by restaurant, including holiday, restaurant id, holiday weight.
        >>> start_date: Start date of period defined holiday weight.
        >>> end_date: End date of period defined holiday weight.

    Returns:
    df_full_date_hw: Prepared sales table, including date, restaurant id, guest count, weekday, holiday, month.
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


def output_holiday_weight(
    df_sales: pd.DataFrame,
    df_date: pd.DataFrame,
    df_res_prep: pd.DataFrame,
    start_date=dt.datetime(2023,1,1),
    end_date=dt.datetime(2023,12,31)
) -> pd.DataFrame:
    '''
    Create valid holiday weight for restaurants.

    Parameters:
        >>> df_sales: sales_store data frame, including restaurant id, date, daily guest count and other information
        >>> df_date: date data frame, including date information
        >>> df_res: restaurant data frame, including restaurant_id, and other restaurant profile information
        >>> start_date: Start date of based period used to calculate holiday weight.
        >>> end_date: End date of based period used to calculate holiday weight.

    Returns:
    df_full_date_hw: Prepared sales table, including date, restaurant id, guest count, weekday, holiday, month.
    '''

    # Step 1: Prepare data
    df_sales_date = prep_data_for_holiday_feature(
        df_sales,
        df_date,
        df_res_prep,
        start_date=start_date,
        end_date=end_date
    )
    logging.info('>>> Finish Step 1: Prepare data')

    # Step 2: Calculate holiday weight
    df_holiday_weight = cal_holiday_weight(df_sales_date)
    logging.info('>>> Finish Step 2: Calculate holiday weight')

    # Step 3: Remove outlier values
    df_nooutlier_hw = remove_outlier_holiday(df_holiday_weight)
    logging.info('>>> Finish Step 3: Remove outlier values')

    # Step 4: Remove ineffective holidays
    df_eff_hw = remove_ineffective_holiday(df_nooutlier_hw)
    logging.info('>>> Finish Step 4: Remove ineffective holidays')

    # Step 5: Add missing holidays for restaurants
    df_full_hw = add_missing_hw(df_eff_hw,df_res_prep,df_sales)
    logging.info('>>> Finish Step 5: Add missing holidays for restaurants')

    # Step 6: Create holiday dataframe
    df_full_date_hw = add_date_to_holiday_df(df_full_hw)
    logging.info('>>> Finish Step 5: Add missing holidays for restaurants')

    return df_full_date_hw

def pad_full_date_and_fill_missing(df: pd.DataFrame, day: str) -> pd.DataFrame:
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

    # Giới hạn dữ liệu 
    start_date = pd.to_datetime(df['shift_date'].min())
    end_date = pd.to_datetime(day)

    filtered_data = df[(df['shift_date'] >= start_date) & (df['shift_date'] <= end_date)]

    merged_data_list = []
    for res_id in filtered_data['restaurant_id'].unique():
        # Lọc dữ liệu theo từng restaurant_id
        filtered_subset = filtered_data[filtered_data['restaurant_id'] == res_id]

        start_date_format = pd.to_datetime(filtered_subset['shift_date'].min())
        all_dates = pd.date_range(start=start_date_format, end=end_date, freq='D')
        all_dates_df = pd.DataFrame({'shift_date': all_dates})

        # Thực hiện merge giữa all_dates_df và filtered_subset dựa trên cột 'shift_date'
        merged_subset = pd.merge(all_dates_df, filtered_subset, on='shift_date', how='left')

        # Fill the nan of restaurant ID after MERGE
        merged_subset['restaurant_id'].fillna(res_id, inplace=True)
        # nan_rows_B = merged_subset[merged_subset['restaurant_id'].isnull()]
        # logging(nan_rows_B)

        # Thêm dữ liệu sau khi merge vào danh sách
        merged_data_list.append(merged_subset)

    # Gộp các phần dữ liệu lại thành một DataFrame duy nhất
    merged_data = pd.concat(merged_data_list, ignore_index=True)

    merged_data['shift_date'] = pd.to_datetime(merged_data['shift_date'])
    # Tạo cột 'week' và 'year' để lưu số tuần và năm của mỗi ngày
    merged_data['week'] = merged_data['shift_date'].dt.isocalendar().week
    merged_data['month'] = merged_data['shift_date'].dt.month
    merged_data['year'] = merged_data['shift_date'].dt.year
    merged_data['dow'] = merged_data['shift_date'].dt.dayofweek

    # Điền các giá trị bị thiếu trong cột 'guest_count' bằng giá trị trung bình của tuần trước đó
    def fill_na_with_previous_week_mean(group):
        filled_group = group.copy()
        for i in range(1, len(group)):
            if pd.isna(filled_group.iloc[i]['guest_count']):
                prev_week_mean = filled_group.iloc[i - 1]['guest_count']
                filled_group.at[filled_group.index[i], 'guest_count'] = prev_week_mean
        return filled_group

    def fill_negative_values_with_previous_week_mean(group):
        filled_group = group.copy()
        for i in range(1, len(group)):
            if filled_group.iloc[i]['guest_count'] < 0:
                prev_week_mean = filled_group.iloc[i - 1]['guest_count']
                filled_group.at[filled_group.index[i], 'guest_count'] = prev_week_mean
        return filled_group

    x_date = ['2022-01-31',
              '2022-02-01',
              '2022-02-02',
              '2022-02-03',
              '2022-02-04',
              '2022-02-05',
              '2023-01-21',
              '2023-01-22',
              '2023-01-23',
              '2023-01-24',
              '2023-01-25',
              '2023-01-26',
              # '2024-02-09',
              # '2024-02-10',
              # '2024-02-11',
              # '2024-02-12',
              # '2024-02-13'
             ]

    merged_data.loc[merged_data['shift_date'].isin(x_date) & merged_data['guest_count'].isna(), 'guest_count'] = 0

    # Áp dụng fill_na_with_previous_week_mean cho từng nhóm tuần và năm
    # merged_data = merged_data.groupby(['year', 'week', 'restaurant_id'], group_keys=False).apply(
    #     fill_na_with_previous_week_mean)
    # merged_data = merged_data.groupby(['year', 'week', 'restaurant_id'], group_keys=False).apply(
    #     fill_negative_values_with_previous_week_mean)

    # merged_data['guest_count'] = merged_data.groupby(['year', 'restaurant_id'])['guest_count'].transform(
    #     lambda group: group.fillna(group.mean()))

    # Replace negative values with null values
    merged_data.loc[merged_data['guest_count'] < 0, 'guest_count'] = None
    merged_data = merged_data.sort_values(by=['restaurant_id', 'shift_date'], ascending=[True, True])

    # merged_data['guest_count'] = merged_data.groupby(['restaurant_id', 'year', 'dow', ])['guest_count'].ffill()
    merged_data['guest_count'] = merged_data.groupby(['restaurant_id', 'dow'])['guest_count'].transform(
        lambda x: x.ffill())

    merged_data['total_sale'] = merged_data.groupby(['restaurant_id', 'dow'])['total_sale'].transform(
        lambda x: x.ffill())

    merged_data['total_bill'] = merged_data.groupby(['restaurant_id', 'dow'])['total_bill'].transform(
        lambda x: x.ffill())

    # def replace_negative_with_group_mean(group):
    #     group['guest_count'] = group['guest_count'].apply(lambda x: x if x >= 0 else group['guest_count'].mean())
    #     return group
    def replace_negative_with_group_mean(group):
        group['guest_count'] = group['guest_count'].apply(lambda x: x if x >= 0 else group['guest_count'].mean())
        return group

    # Sử dụng transform để áp dụng hàm trên cho mỗi nhóm
    # merged_data = merged_data.groupby(['year', 'restaurant_id'], group_keys=False).apply(
    #     replace_negative_with_group_mean)

    merged_data.drop(columns=['week', 'month', 'year', 'dow'], inplace=True)

    # Fill các trị chung bằng giá trị đầu đủ ở dòng cuối cùng
    # logging(merged_data.guest_count.isnull().count(axis=0))
    # merged_data.fillna(method='ffill', inplace=True)
    # logging(merged_data.isnull().count(axis=0))

    return merged_data


def check_first_open_month(df_sales: pd.DataFrame) -> pd.DataFrame:
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


if __name__ == '__main__':

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
    path_cluster_km = config_dict['path_cluster_km']
    path_moon_date = config_dict['path_moon_date']
    path_dim_date = config_dict.get('path_dim_date', './Data_Processing/raw/dim_date.parquet')
    
    logging.info("Successfully for loading metadata path")
    logging.info("{}{}".format("path_sale_store:", path_sale_store))
    logging.info("{}{}".format("path_dim_restaurant:", path_dim_restaurant))
    logging.info("{}{}".format("df_holiday_weight:", path_holiday))
    logging.info("{}{}".format("path_cluster:", path_cluster))
    logging.info("{}{}".format("path_first_month:", path_first_month))
    logging.info("{}{}".format("final_date:", final_date))
    logging.info("{}{}".format("output_dir:", output_dir))
    logging.info("{}{}".format("path_processed_data:", path_processed_data))
    logging.info("{}{}".format("path_feature_data:", path_feature_data))
    logging.info("{}{}".format("path_prediction_data:", path_prediction_data))
    logging.info("{}{}".format("path_cluster_km:", path_cluster_km))
    logging.info("{}{}".format("path_moon_date:", path_moon_date))
    logging.info("{}{}".format("path_dim_date:", path_dim_date))

    logging.info('Please check the metadata path before running the program')

    while True:
        # Get input from user
        input_str = input("Press any key to continue...")
        # Check if input is empty
        if input_str == '':
            continue
        else:
            break

    # Load data from path_dim_restaurant
    logging.info('Loading data from path_dim_restaurant')
    df_dims = pd.read_parquet(
        path_dim_restaurant,
        filesystem=hdfs,
    )

    df_date = pd.read_parquet(
        path_dim_date,
        filesystem=hdfs,
    )
    
    # df_dims = df_dims.loc[df_dims['status'] == 'active']
    # data_list_active = df_dims['restaurant_id'].unique()

    # Load data from path_sale_store
    logging.info('Loading data from path_sale_store')
    df_sale_stores = pd.read_parquet(
        path_sale_store,
        filesystem=hdfs, )
    
    df_sale_stores = check_early_sale_store(df_sale_stores)
    df_sale_stores = df_sale_stores[df_sale_stores['restaurant_id'].notna()]

    # Load data from path_cluster
    logging.info('Loading data from path_cluster')
    df_cluster = pd.read_parquet(
        path_cluster,
        filesystem=hdfs,
    )

    # Load data from path_holiday
    logging.info('Loading data from path_holiday')
    df_holidays = pd.read_parquet(
        path_holiday,
        # filesystem=hdfs,
    )

    df_cluster_km = pd.read_parquet(
        path_cluster_km,
        # filesystem=hdfs,
    )

    df_moon_date = pd.read_parquet(
        path_moon_date,
        filesystem=hdfs,
    )

    df_holidays.rename(columns={'date': 'ds'}, inplace=True)
    df_holidays = df_holidays[['ds', 'holiday']]
    # Delete anomaly from "Sale_Store" by Prophet
    df_anomaly = pd.DataFrame()
    df_total = df_sale_stores.copy()
    logging.info("Detect anomaly by Prophet")
    lst_res_id_except = []
    for restaurant_id in tqdm(df_total.restaurant_id.unique()):
        try:
            df_res_id = df_total.loc[
                (df_total.restaurant_id == restaurant_id) & (df_total.shift_date >= '2022-1-1'), ['shift_date',
                                                                                                  'guest_count']]
            df_res_id.columns = ['ds', 'y']
            model = Prophet(weekly_seasonality=True)
            model.fit(df_res_id)
            forecast = model.predict(df_res_id)
            actual_values = df_res_id['y'].values
            predicted_values = forecast['yhat'].values
            residuals = actual_values - predicted_values
            df_res_id['residuals'] = residuals
            df_res_id['test'] = ((residuals.mean() + residuals.std() * 6) > df_res_id['residuals']) & (
                    df_res_id['residuals'] > (residuals.mean() - residuals.std() * 6))
            df_anomaly_resid = df_res_id[df_res_id['test'] == 0]
            df_anomaly = pd.concat([df_anomaly, pd.DataFrame(
                {
                    'restaurant_id': restaurant_id,
                    'shift_date': df_anomaly_resid['ds'],
                })])
        except Exception as e:
            # logging.error('Restaurant_id: ' + str(restaurant_id) + ' ' + str(e))
            continue

    
    df_anomaly = df_anomaly[~df_anomaly.restaurant_id.isin([383])]

    logging.info("Successfully in detecting anomaly by Prophet")

    # Merged to check anomaly
    logging.info("Merged to check anomaly, exclude holiday")
    df_holidays.rename(columns={'ds': 'shift_date'}, inplace=True)
    df_anomaly = df_anomaly.merge(df_holidays, how='left', on='shift_date')
    df_anomaly = df_anomaly.loc[df_anomaly['holiday'].isnull(), ['shift_date', 'restaurant_id']]

    # Save anomaly data
    path_anomaly = output_dir + "anomaly_data.parquet"
    df_anomaly.to_parquet(
        path_anomaly, index=False)
    logging.info(f'Save successfully the anomaly data to {path_anomaly}')

    # Merge DataFrame theo cột 'shift_date' và 'restaurant_id'
    logging.info('Exclude anomaly data from sale_store data')
    merged_df = pd.merge(df_total, df_anomaly, on=['shift_date', 'restaurant_id'], how='outer', indicator=True)
    # Chọn các hàng chỉ xuất hiện ở DataFrame thứ hai (right_only)
    result_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)
    # logging.info("{:<80}{}".format("Load holiday data", "SUCCESS"))

    # Save xanomaly data
    path_xanomaly = output_dir + "xanomaly_data.parquet"
    result_df.to_parquet(path_xanomaly, index=False)
    logging.info(f"Save successfully the xanomaly data to {path_xanomaly}")

    # Merge with "First_month"
    df_first_andSecond_M = check_first_open_month(df_sale_stores)
    merged_first_data = pd.merge(result_df, df_first_andSecond_M[['restaurant_id', 'first_log_month']],
                                 on="restaurant_id", how="left")
    df_holiday_weight = output_holiday_weight(
    df_sales=merged_first_data,
    df_date=df_date,
    df_res_prep=df_dims,
    start_date=dt.datetime(2023,1,1),
    end_date=dt.datetime(2023,12,31)
)
    df_holiday_weight.to_parquet(path_holiday)

    # Processing to del the first month
    merged_first_data['shift_date'] = pd.to_datetime(merged_first_data['shift_date'])
    merged_first_data['month'] = merged_first_data['shift_date'].dt.strftime('%Y-%m')

    merged_first_data['first_log_month'] = pd.to_datetime(merged_first_data['first_log_month'])
    merged_first_data['month'] = pd.to_datetime(merged_first_data['month'])

    merged_first_data['month_diff'] = (merged_first_data['month'].dt.year - merged_first_data[
        'first_log_month'].dt.year) * 12 + \
                                      (merged_first_data['month'].dt.month - merged_first_data[
                                          'first_log_month'].dt.month)
    merged_first_data_selected = merged_first_data.loc[merged_first_data['month_diff'] > 0]
    logging.info("{}{}".format("Del the first month", "SUCCESS"))

    # Processing dataset
    data_processing = pad_full_date_and_fill_missing(merged_first_data_selected, final_date)
    logging.info("{}{}".format("Process dataset - fill na", "SUCCESS"))

    # Merge "Sale_Store" and "Dim_Restaurant" to choose profile restaurant columns
    logging.info("Merge sale_store and dim_restaurant")
    merged_df_dim = pd.merge(data_processing[
                                 ['shift_date', 'restaurant_id', 'total_sale', 'total_loyalty_sale', 'guest_count',
                                  'total_bill', 'avg_rating']],
                             df_dims[['restaurant_id', 'region_name', 'city', 'brand',
                                      'concept_detail',
                                      'is_mall', 'district_name', 'subdistrict_name',
                                      'sbu']],
                             on=['restaurant_id'], how='left')

    # Merge with "Moon_date"
    logging.info('Merge with "Moon_date"')
    merged_df_dim = pd.merge(merged_df_dim, df_moon_date, on=['shift_date'], how='left')

    # Merge with "Cluster"
    logging.info('Merge with "Cluster"')
    merged_df_dim = pd.merge(merged_df_dim, df_cluster, on=['restaurant_id'], how='left')
    # merged_df_dim = pd.merge(merged_df_dim, df_cluster_km[['restaurant_id','cluster_km']], on=['restaurant_id'], how='left')
    merged_df_dim.fillna(method='ffill', inplace=True)

    # Create folder and save file result
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        logging.info("Successfully created the result folder")
    merged_df_dim.reset_index().to_parquet(path_processed_data, index=False)
    logging.info(f"Save successfully processed data to {path_processed_data}")
