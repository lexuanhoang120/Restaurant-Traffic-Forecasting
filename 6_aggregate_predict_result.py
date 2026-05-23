# import configparser

# import numpy as np
# import json
# # import torch
# from tqdm import tqdm
# import argparse
# import os
# import pickle
# import glob
# from datetime import datetime

# import os
# import sys
# import subprocess
# import numpy as np
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from pyarrow import fs
# import pyarrow.parquet as pq

# import pandas as pd
# import argparse
# from datetime import datetime
# import shutil
# import warnings

# warnings.filterwarnings('ignore')  # setting ignore as a parameter

# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logging.getLogger('fbprophet').setLevel(logging.WARNING)

# os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf/"
# os.environ["JAVA_HOME"] = "/usr/jdk64/jdk1.8.0_112"
# os.environ["HADOOP_HOME"] = "/usr/hdp/3.1.0.0-78/hadoop"
# os.environ["ARROW_LIBHDFS_DIR"] = "/usr/hdp/3.1.0.0-78/usr/lib/"
# os.environ["CLASSPATH"] = subprocess.check_output(
#     "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True
# ).decode("utf-8")

# hdfs = fs.HadoopFileSystem(
#     host=os.getenv("HDFS_HOST", "localhost"), port=int(os.getenv("HDFS_PORT", "8020"))
# )


# def check_early_sale_store(df: pd.DataFrame) -> pd.DataFrame:
#     df.sort_values(by=['shift_date'], inplace=True)
#     df = df.drop_duplicates(subset=['shift_date', 'restaurant_id'], keep='last')
#     df['shift_date'] = df['shift_date'].astype('datetime64[ns]')
#     return df


# def check_first_open_month(df_sales: pd.DataFrame) -> pd.DataFrame:
#     '''
#     Check first open month to pre-process input data for modeling.
    
#     Parameters:
#     df_sales: sales_store data frame, including restaurant id, date, daily guest count and other information
    
#     Returns:
#     (dataframe): data frame containing first log date and first log month of each restaurant.
#     '''

#     # calculate first log date of each restaurant
#     df_fmonth = df_sales.groupby('restaurant_id')['shift_date'].min() \
#         .reset_index() \
#         .rename(columns={'shift_date': 'first_log_date'})

#     # format date to 'year-month' to get first log month
#     df_fmonth['first_log_month'] = df_fmonth['first_log_date'].dt.strftime('%Y-%m')

#     # create first_month category
#     df_fmonth['is_first_month'] = 1

#     return df_fmonth


# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--local_result", default="./Forecasting_Results/Local_Result.parquet", type=str)
#     # parser.add_argument("--global_result", default="./Forecasting_Results/Global_Result.parquet", type=str)
#     # parser.add_argument("--data_sale_store",
#     #                     default="./<private_path_removed>",
#     #                     type=str)
#     # parser.add_argument("--first_month",
#     #                     default="./<private_path_removed>",
#     #                     type=str)
#     # parser.add_argument("--output_dir", default="./Forecasting_Results", type=str, help="for saving results")
#     # args = parser.parse_args()
#     #
#     # path_local_result = args.local_result
#     # path_global_result = args.global_result
#     # path_sale_store = args.data_sale_store
#     # path_first_month = args.first_month
#     # path_save_hybrid = args.output_dir

#     configFilePath = "./Config/config.ini"
#     print('Loading metadata path from config file')
#     config = configparser.RawConfigParser()
#     config.read(configFilePath)
#     config_dict = dict(config.items('config'))
#     path_sale_store = config_dict['path_sale_store']
#     path_dim_restaurant = config_dict['path_dim_restaurant']
#     path_processed_data = config_dict['path_processed_data']
#     path_feature_data = config_dict['path_feature_data']
#     path_holiday = config_dict['path_holiday']
#     path_cluster = config_dict['path_cluster']
#     path_first_month = config_dict['path_first_month']
#     final_date = config_dict['final_date']
#     output_dir = config_dict['path_output_processed_data']
#     path_prediction_data = config_dict['path_prediction_data']
#     path_local_result = config_dict['path_local_result']
#     path_global_result = config_dict['path_global_result']
#     path_hybrid_result = config_dict['path_hybrid_result']
#     path_output_data = config_dict['path_output_data']

#     print("Successfully for loading metadata path")
#     # print('path_sale_store: ', path_sale_store)
#     # print('path_processed_data: ', path_processed_data)
#     # print('path_feature_data: ', path_feature_data)
#     # print('path_dim_restaurant: ', path_dim_restaurant)
#     # print('df_holiday_weight: ', path_holiday)
#     # print('path_cluster: ', path_cluster)
#     # print('path_first_month: ', path_first_month)
#     # print('final_date: ', final_date)
#     # print('output_dir: ', output_dir)
#     # print('path_global_result: ', path_global_result)
#     # print('path_local_result: ', path_local_result)
#     # print('path_hybrid_result: ', path_hybrid_result)
#     print("{:<30}{}".format("path_sale_store:", path_sale_store))
#     print("{:<30}{}".format("path_global_result:", path_global_result))
#     print("{:<30}{}".format("path_local_result:", path_local_result))
#     print("{:<30}{}".format("path_hybrid_result:", path_hybrid_result))

#     print('Please check the metadata path before running the program')

#     while True:
#         # Get input from user
#         input_str = input("Press any key to continue...")
#         # Check if input is empty
#         if input_str == '':
#             continue
#         else:
#             break

#     # load data from "Local_Result"
#     df_local_result = pd.read_parquet(
#         path_local_result,
#         # filesystem=hdfs,
#     )
#     print("Successfully in loading local result")
#     # load data from "Global_Result"
#     df_global_result = pd.read_parquet(
#         path_global_result,
#         # filesystem=hdfs,
#     )
#     print("Successfully in loading global result")

#     # Hybrid the results
#     df_final_result = pd.concat([df_global_result, df_local_result], axis=0)
#     df_final_result = df_final_result.drop_duplicates(subset=['date', 'restaurant_id'], keep='last')
#     df_final_result = df_final_result.rename(columns={"date": "shift_date"})
#     print("Successfully in combining results")
#     # Del the second month
#     # df_first_andSecond_M = pd.read_parquet(
#     #     path_first_month,
#     #     filesystem=hdfs,
#     # )

#     df_sale_stores = pd.read_parquet(
#         path_sale_store,
#         filesystem=hdfs,
#     )
#     df_sale_stores = check_early_sale_store(df_sale_stores)
#     df_first_andSecond_M = check_first_open_month(df_sale_stores)

#     merged_first_data = pd.merge(df_final_result, df_first_andSecond_M[['restaurant_id', 'first_log_month']],
#                                  on="restaurant_id", how="left")

#     # Processing to del the first month
#     merged_first_data['shift_date'] = pd.to_datetime(merged_first_data['shift_date'])
#     merged_first_data['month'] = merged_first_data['shift_date'].dt.strftime('%Y-%m')

#     merged_first_data['first_log_month'] = pd.to_datetime(merged_first_data['first_log_month'])
#     merged_first_data['month'] = pd.to_datetime(merged_first_data['month'])

#     merged_first_data['month_diff'] = (merged_first_data['month'].dt.year - merged_first_data[
#         'first_log_month'].dt.year) * 12 + \
#                                       (merged_first_data['month'].dt.month - merged_first_data[
#                                           'first_log_month'].dt.month)

#     merged_first_data_selected = merged_first_data.loc[merged_first_data['month_diff'] > 1]
#     print("Successfully in checking first month")

#     # Choose suitable restaurant_id
#     restaurantID_list = [2, 3, 4, 5, 8, 10, 11, 13, 14, 17, 19, 21, 23, 24, 27, 28, 29, 30, 34, 35, 36, 38, 39, 40, 41,
#                          42, 43, 44, 45, 47, 49,
#                          50, 59, 62, 66, 68, 69, 70, 77, 78, 80, 81, 83, 85, 86, 89, 90, 93, 95, 96, 99, 101, 102, 103,
#                          105, 107, 108, 115, 119, 127, 128, 129,
#                          132, 134, 135, 136, 137, 139, 140, 141, 143, 147, 149, 150, 151, 152, 158, 159, 160, 161, 168,
#                          172, 173, 174, 175, 176, 178, 180, 181, 182, 184, 186, 187,
#                          189, 190, 192, 193, 194, 195, 199, 200, 201, 207, 209, 211, 213, 214, 215, 216, 218, 219, 220,
#                          221, 222, 223, 224, 225, 229, 230, 231, 233, 235, 236, 237,
#                          238, 239, 241, 242, 244, 245, 246, 249, 250, 252, 255, 257, 260, 261, 263, 264, 269, 271, 272,
#                          273, 274, 276, 281, 283, 290, 291, 292, 294, 295, 296, 298,
#                          300, 301, 302, 303, 304, 306, 307, 309, 310, 311, 312, 314, 315, 316, 319, 320, 322, 323, 325,
#                          326, 327, 328, 329, 330, 331, 332, 333, 334, 337, 338, 339,
#                          340, 341, 342, 343, 344, 346, 348, 349, 354, 355, 357, 358, 359, 361, 363, 364, 370, 371, 372,
#                          373, 374, 375, 376, 377, 379, 380, 381, 382, 383, 386, 388,
#                          389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 404, 405, 406, 407, 415, 416, 417, 419,
#                          420, 422, 423, 425, 426, 427, 428, 429, 430, 431, 432, 433,
#                          434, 435, 438, 439, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
#                          456, 457, 458, 461, 462, 463, 465, 466, 467, 469, 470, 471,
#                          472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491,
#                          492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 503, 504,
#                          505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 521, 522, 523, 525,
#                          528, 531, 536, 537, 538, 540, 541, 542, 544, 545, 546, 547,
#                          548, 549, 550, 551, 552, 554, 555, 556, 557, 558, 559, 564, 567, 568, 569, 572, 574, 575, 577,
#                          578, 579, 580, 581, 583, 584, 586, 591, 592, 593, 594, 595,
#                          596, 597, 598, 599, 600, 601, 602, 604, 605, 606, 607, 609, 610, 611, 613, 614, 615, 616, 617,
#                          618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629,
#                          632, 634, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 648, 649, 650, 651, 653, 654,
#                          656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667,
#                          668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678]

#     # merged_first_data_selected = merged_first_data_selected[
#     #     merged_first_data_selected['restaurant_id'].isin(restaurantID_list) == True]
#     merged_first_data_selected = merged_first_data_selected[['shift_date', 'restaurant_id', 'number_Prediction']]
#     merged_first_data_selected = merged_first_data_selected.rename(columns={"number_Prediction": "pred"})
#     merged_first_data_selected['pred'] = merged_first_data_selected['pred'].apply(lambda x: round(x, 0))
#     print("Successfully in choosing suitable restaurant_id")

#     # Save result
#     # path_to_save = f"./Forecasting_Results/Hybrid_Result_B.parquet"
#     merged_first_data_selected.to_parquet(path_hybrid_result, index=False)
#     merged_first_data_selected.to_parquet(path_output_data, index=False, filesystem=hdfs)
#     # df_final_result.to_parquet(path_to_save, index=False)
#     print("Successfully in saving result to {}".format(path_hybrid_result))
#     print("Successfully in saving result to {}".format(path_output_data))
























import configparser
import logging
import os
import subprocess
import pandas as pd
from pyarrow import fs
import pyarrow.parquet as pq
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('fbprophet').setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Set up environment variables
os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf/"
os.environ["JAVA_HOME"] = "/usr/jdk64/jdk1.8.0_112"
os.environ["HADOOP_HOME"] = "/usr/hdp/3.1.0.0-78/hadoop"
os.environ["ARROW_LIBHDFS_DIR"] = "/usr/hdp/3.1.0.0-78/usr/lib/"
os.environ["CLASSPATH"] = subprocess.check_output(
    "$HADOOP_HOME/bin/hadoop classpath --glob", shell=True
).decode("utf-8")

# Hadoop FileSystem
hdfs = fs.HadoopFileSystem(
    host=os.getenv("HDFS_HOST", "localhost"), port=int(os.getenv("HDFS_PORT", "8020"))
)

def setup_logging():
    logging.basicConfig(
        filename='process.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_early_sale_store(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Checking early sales store")
    df.sort_values(by=['shift_date'], inplace=True)
    df = df.drop_duplicates(subset=['shift_date', 'restaurant_id'], keep='last')
    df['shift_date'] = df['shift_date'].astype('datetime64[ns]')
    return df

def check_first_open_month(df_sales: pd.DataFrame) -> pd.DataFrame:
    '''
    Check first open month to pre-process input data for modeling.
    '''
    logger.info("Checking first open month")
    df_fmonth = df_sales.groupby('restaurant_id')['shift_date'].min() \
        .reset_index() \
        .rename(columns={'shift_date': 'first_log_date'})
    df_fmonth['first_log_month'] = df_fmonth['first_log_date'].dt.strftime('%Y-%m')
    df_fmonth['is_first_month'] = 1
    return df_fmonth

if __name__ == "__main__":
    setup_logging()
    logger.info("Starting the program")

    configFilePath = "./Config/config.ini"
    logger.info("Loading metadata path from config file")
    config = configparser.RawConfigParser()
    config.read(configFilePath)
    config_dict = dict(config.items('config'))

    path_sale_store = config_dict['path_sale_store']
    path_local_result = config_dict['path_local_result']
    path_global_result = config_dict['path_global_result']
    path_hybrid_result = config_dict['path_hybrid_result']
    path_output_data = config_dict['path_output_data']

    logger.info("Loaded metadata paths")
    logger.info(f"path_sale_store: {path_sale_store}")
    logger.info(f"path_local_result: {path_local_result}")
    logger.info(f"path_global_result: {path_global_result}")
    logger.info(f"path_hybrid_result: {path_hybrid_result}")

    input("Press any key to continue...")

    # Load data
    logger.info("Loading local result")
    df_local_result = pd.read_parquet(path_local_result)
    logger.info("Loading global result")
    df_global_result = pd.read_parquet(path_global_result)

    # Combine results
    logger.info("Combining results")
    df_final_result = pd.concat([df_global_result, df_local_result], axis=0)
    df_final_result = df_final_result.drop_duplicates(subset=['date', 'restaurant_id'], keep='last')
    df_final_result = df_final_result.rename(columns={"date": "shift_date"})

    # Load and process sales store data
    logger.info("Loading sales store data")
    df_sale_stores = pd.read_parquet(path_sale_store, filesystem=hdfs)
    df_sale_stores = check_early_sale_store(df_sale_stores)

    logger.info("Checking first month for each restaurant")
    df_first_andSecond_M = check_first_open_month(df_sale_stores)

    merged_first_data = pd.merge(df_final_result, df_first_andSecond_M[['restaurant_id', 'first_log_month']],
                                 on="restaurant_id", how="left")

    # Processing to remove data for the first month
    merged_first_data['shift_date'] = pd.to_datetime(merged_first_data['shift_date'])
    merged_first_data['month'] = merged_first_data['shift_date'].dt.strftime('%Y-%m')
    merged_first_data['first_log_month'] = pd.to_datetime(merged_first_data['first_log_month'])
    merged_first_data['month'] = pd.to_datetime(merged_first_data['month'])
    merged_first_data['month_diff'] = (merged_first_data['month'].dt.year - merged_first_data['first_log_month'].dt.year) * 12 + \
                                      (merged_first_data['month'].dt.month - merged_first_data['first_log_month'].dt.month)

    merged_first_data_selected = merged_first_data.loc[merged_first_data['month_diff'] > 1]
    merged_first_data_selected = merged_first_data_selected[['shift_date', 'restaurant_id', 'number_Prediction']]
    merged_first_data_selected = merged_first_data_selected.rename(columns={"number_Prediction": "pred"})
    merged_first_data_selected['pred'] = merged_first_data_selected['pred'].apply(lambda x: round(x, 0))

    logger.info("Saving results")
    merged_first_data_selected.to_parquet(path_hybrid_result, index=False)
    merged_first_data_selected.to_parquet(path_output_data, index=False, filesystem=hdfs)
    logger.info("Successfully saved results to {}".format(path_hybrid_result))
    logger.info("Successfully saved results to {}".format(path_output_data))

