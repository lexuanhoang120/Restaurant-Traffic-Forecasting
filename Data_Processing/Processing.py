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


def GuestCount_ByShiftDate_updated(df_all, final_day):
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
    - df_all: Là dataframe gốc từ bảng "Sale_Store"

    Trả về:
    - merged_data: Là dataframe đã được xử lí không có missing 
                  ở giữa các ngày và xử lí các giá trị âm
    """
    
    # Chuyển cột 'shift_date' sang kiểu dữ liệu ngày tháng
    df_all['shift_date'] = pd.to_datetime(df_all['shift_date'])

    # Giới hạn dữ liệu 
    start_date = pd.to_datetime(df_all['shift_date'].min())
    # end_date = pd.to_datetime('2023-11-30')
    end_date = pd.to_datetime(final_day)
    filtered_data = df_all[(df_all['shift_date'] >= start_date) & (df_all['shift_date'] <= end_date)]

    merged_data_list = []
    for restaurant_id in filtered_data['restaurant_id'].unique():
        # Lọc dữ liệu theo từng restaurant_id
        filtered_subset = filtered_data[filtered_data['restaurant_id'] == restaurant_id]

        start_date_format = pd.to_datetime(filtered_subset['shift_date'].min())
        all_dates = pd.date_range(start=start_date_format, end=end_date, freq='D')
        all_dates_df = pd.DataFrame({'shift_date': all_dates})

        # Thực hiện merge giữa all_dates_df và filtered_subset dựa trên cột 'shift_date'
        merged_subset = pd.merge(all_dates_df, filtered_subset, on='shift_date', how='left')

        # # Fill the nan of restaurant ID after MERGE
        merged_subset['restaurant_id'].fillna(restaurant_id, inplace=True)
        # nan_rows_B = merged_subset[merged_subset['restaurant_id'].isnull()]
        # print(nan_rows_B)

        # Thêm dữ liệu sau khi merge vào danh sách
        merged_data_list.append(merged_subset)

    # Gộp các phần dữ liệu lại thành một DataFrame duy nhất
    merged_data = pd.concat(merged_data_list, ignore_index=True)
    
    merged_data['shift_date'] = pd.to_datetime(merged_data['shift_date'])
    # Tạo cột 'week' và 'year' để lưu số tuần và năm của mỗi ngày
    merged_data['week'] = merged_data['shift_date'].dt.isocalendar().week
    merged_data['month'] = merged_data['shift_date'].dt.month
    merged_data['year'] = merged_data['shift_date'].dt.year

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

    
    # Áp dụng fill_na_with_previous_week_mean cho từng nhóm tuần và năm
    merged_data = merged_data.groupby(['year', 'week', 'restaurant_id'], group_keys=False).apply(fill_na_with_previous_week_mean)
    merged_data = merged_data.groupby(['year', 'week', 'restaurant_id'], group_keys=False).apply(fill_negative_values_with_previous_week_mean)
    

    merged_data['guest_count'] = merged_data.groupby(['year', 'restaurant_id'])['guest_count'].transform(lambda group: group.fillna(group.mean()))

    def replace_negative_with_group_mean(group):
        group['guest_count'] = group['guest_count'].apply(lambda x: x if x >= 0 else group['guest_count'].mean())
        return group
    # Sử dụng transform để áp dụng hàm trên cho mỗi nhóm
    merged_data = merged_data.groupby(['year', 'restaurant_id'], group_keys=False).apply(replace_negative_with_group_mean)
    
    merged_data.drop(columns=['week' ,'month', 'year'], inplace=True)

    # Fill các trị chung bằng giá trị đầu đủ ở dòng cuối cùng
    merged_data.fillna(method='ffill', inplace=True)

    return merged_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_sale_store", default="./Data_Processing/raw/sales_store.parquet", type=str)
    parser.add_argument("--dim_restaurant", default="./Data_Processing/raw/dim_restaurant.parquet", type=str)
    parser.add_argument("--cluster", default="./Data_Processing/result/cluster_km.parquet", type=str)
    parser.add_argument("--first_month", default="./Data_Processing/result/first_log_by_res.parquet", type=str)
    parser.add_argument("--finalDay_predict", default="2023-10-31", type=str)
    parser.add_argument("--output_dir", default="./result", type=str, help="for saving results")
    args = parser.parse_args()

    path_sale_store = args.data_sale_store
    path_dim_restaurant = args.dim_restaurant
    path_cluster = args.cluster
    path_first_month = args.first_month
    final_day = args.finalDay_predict
    
    
    # load data from "Dim Restaurant"
    try:
        df_dims = pd.read_parquet(
            path_dim_restaurant,
            filesystem=hdfs,
        )
        df_dims = df_dims.loc[df_dims['status'] == 'active']
        data_list_active = df_dims['restaurant_id'].unique()
        print("Sucessfully for loading dim restaurant dataset")
    except Exception as e:
                print(f"Failed for loading dim restaurant dataset: {str(e)}")
        
    # load data from "Sale_Store_v5"
    try: 
        df_all = pd.read_parquet(
            path_sale_store,
            filesystem=hdfs,
        )
        df_all = df_all[df_all['restaurant_id'].isin(data_list_active) == True]
        print("Sucessfully for loading Sale Store")
    except:
        print("Fail in loading dataset")


    # Merge "Sale_Store" and "Dim_Restaurant" to choose profile restaurant columns
    try:
        merged_df_dim = pd.merge(df_all[['shift_date', 'restaurant_id', 'total_sale', 'total_loyalty_sale', 'guest_count', 'total_bill', 'avg_rating']], 
                         df_dims[['restaurant_id', 'region_name', 'city', 'brand', 'rnd', 'concept_detail', 'is_mall']], on=['restaurant_id'], how = 'left')
        # Tính giá trị trung bình của 'guest_count' theo từng 'restaurant_id'
        mean_guest_count_by_restaurant = merged_df_dim.groupby('restaurant_id')['total_loyalty_sale'].mean()
        
        # Điền các giá trị bị thiếu trong 'total_loyalty_sale' bằng giá trị trung bình tương ứng của 'restaurant_id'
        merged_df_dim['total_loyalty_sale'].fillna(merged_df_dim['restaurant_id'].map(mean_guest_count_by_restaurant), inplace=True)
        print("Sucessfully in mergeing dataset")
    except:
        print("Fail in merging dataset")

    # Merge with "Cluster"
    try:
        df_cluster = pd.read_parquet(
            path_cluster,
            filesystem=hdfs,
        )
        merged_df_dim = pd.merge(merged_df_dim, df_cluster, on = ['restaurant_id'], how = 'left')
        print("Sucessfully in mergeing with cluster")
    except:
        print("Fail in merging with cluster")

    # Del the first month
    try:
        df_first_andSecond_M = pd.read_parquet(
            path_first_month,
            filesystem=hdfs,
        )
        merged_first_data = pd.merge(merged_df_dim, df_first_andSecond_M[['restaurant_id', 'first_log_month']], on = "restaurant_id", how = "left")
        # Processing to del the first month
        merged_first_data['shift_date'] = pd.to_datetime(merged_first_data['shift_date'])
        merged_first_data['month'] = merged_first_data['shift_date'].dt.strftime('%Y-%m')
        
        merged_first_data['first_log_month'] = pd.to_datetime(merged_first_data['first_log_month'])
        merged_first_data['month'] = pd.to_datetime(merged_first_data['month'])
        
        merged_first_data['month_diff'] = (merged_first_data['month'].dt.year - merged_first_data['first_log_month'].dt.year) * 12 + \
                                          (merged_first_data['month'].dt.month - merged_first_data['first_log_month'].dt.month)
        merged_first_data_selected = merged_first_data.loc[merged_first_data['month_diff'] > 0]
       
        print("Sucessfully in del the first month")
    except:
        print("Fail in del the first month")

        

    # Processing dataset
    try: 
        data_processing = GuestCount_ByShiftDate_updated(merged_first_data_selected, final_day)
        print("Processing dataset is sucessfully")
    except Exception as e:
        print(f"Processing dataset is faild: {str(e)}")

    # Create folder and save file result
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        print("Successfully created the result folder")
    dt = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    # path_to_save_1 = f"./<private_path_removed>"
    # path_to_save = f"./result/Data_Processing_{dt}.parquet"
    path_to_save = f"./result/Data_Processing.parquet"
    try:
        # # Tạo đường dẫn cho tệp mới
        # # saved_file_path = os.path.join(args.output_dir, f'result_{dt}.csv')
        # saved_file_path = os.path.join(args.output_dir, f'result_{dt}.csv')
        # # Tạo và lưu tệp vào đường dẫn đã tạo
        # with open(saved_file_path, 'w') as file:
        #     # file.write("This is the content of the result file.")
        #     # data_processing.to_parquet(
        #     #         file,
        #     #         filesystem=hdfs,  # bắt buộc phải có
        #     #         index=False,  # dùng option này để khi đọc df bằng spark sẽ không xuất hiện thêm column __index_level_0__
        #     # )
        #     # # Save to file CSV
        #     # data_processing.to_csv(file, index=True)
        #     # Save to file Parquet
        #     # data_processing.to_parquet(path_to_save_1, filesystem=hdfs, index=False)
        data_processing.reset_index().to_parquet(path_to_save, index=False)
        print("Saved result successfully")
    except Exception as e:
        print(f"Failed to save result: {str(e)}")

