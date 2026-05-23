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


def create_features_glob(df):
    df = df.rename(columns={"guest_count": "y", "shift_date": "date"})
    
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    # df['weekofyear'] = df['date'].dt.weekofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    df['y_previous_month'] = df.groupby('restaurant_id')['y'].shift(35)
    
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')
    
    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(35)
    
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')
    
    df['normal_y'] = df['y'] / df['holiday_weight'] 
    
    # Create lag Feature
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)
    df['lag_42s'] = df.groupby('restaurant_id')['normal_y'].shift(42)
    df['lag_49s'] = df.groupby('restaurant_id')['normal_y'].shift(49)
    df['lag_56s'] = df.groupby('restaurant_id')['normal_y'].shift(56)
    
    # Create loyalty feature
    df['loyalty_sale_previous_month'] = df.groupby('restaurant_id')['total_loyalty_sale'].shift(35)
    df['sale_previous_month'] = df.groupby('restaurant_id')['total_sale'].shift(35)
    
    df['monthly_sum_loyalty_sale'] = df.groupby(['year', 'month', 'restaurant_id'])['loyalty_sale_previous_month'].transform('sum')
    df['monthly_sum_sale'] = df.groupby(['year', 'month', 'restaurant_id'])['sale_previous_month'].transform('sum')
    df['weight_sale'] = df['monthly_sum_loyalty_sale'] / df['monthly_sum_sale']

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['new_weekday'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('max')
    
    df['y_previous_week_weekend'] = df[df['new_weekday'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('max')
    
    
    X = df[['y', 'date', 'restaurant_id',
            'brand_label', 'city_label', 'region_label', 
            # 'rnd_label', 'concept_detail_label', 'is_mall_label',
            # 'brand', 'city', 'region_name', 'rnd', 'concept_detail',
            'concept_detail_label', 'is_mall_label',
            'brand', 'city', 'region_name', 'concept_detail',
            'dayofweek','quarter','month','year',
            'monthly_mean_visitors', 'monthly_median_visitors','monthly_min_visitors', 'monthly_max_visitors',
            'weekly_mean_visitors', 'weekly_median_visitors', 'weekly_min_visitors', 'weekly_max_visitors',
            'monthly_mean_visitors_weekeek', 'monthly_median_visitors_weekeek','monthly_min_visitors_weekeek', 'monthly_max_visitors_weekeek',
            'weekly_mean_visitors_weekeek', 'weekly_median_visitors_weekeek', 'weekly_min_visitors_weekeek', 'weekly_max_visitors_weekeek',
            # 'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            'weight_sale', 
            'holiday', 'new_weekday', 'holiday_weight', 'cluster'
           ]]

    return X
    
def create_features(df):
    df = df.rename(columns={"guest_count": "y", "shift_date": "date"})
    
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    # df['weekofyear'] = df['date'].dt.weekofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    df['y_previous_month'] = df.groupby('restaurant_id')['y'].shift(35)
    
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')
    
    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(35)
    
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')
    
    df['normal_y'] = df['y'] / df['holiday_weight'] 
    
    # Create lag Feature
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)
    df['lag_42s'] = df.groupby('restaurant_id')['normal_y'].shift(42)
    df['lag_49s'] = df.groupby('restaurant_id')['normal_y'].shift(49)
    df['lag_56s'] = df.groupby('restaurant_id')['normal_y'].shift(56)
    
    # Create loyalty feature
    df['loyalty_sale_previous_month'] = df.groupby('restaurant_id')['total_loyalty_sale'].shift(35)
    df['sale_previous_month'] = df.groupby('restaurant_id')['total_sale'].shift(35)
    
    df['monthly_sum_loyalty_sale'] = df.groupby(['year', 'month', 'restaurant_id'])['loyalty_sale_previous_month'].transform('sum')
    df['monthly_sum_sale'] = df.groupby(['year', 'month', 'restaurant_id'])['sale_previous_month'].transform('sum')
    df['weight_sale'] = df['monthly_sum_loyalty_sale'] / df['monthly_sum_sale']

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['new_weekday'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('max')
    
    df['y_previous_week_weekend'] = df[df['new_weekday'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('max')
    
    
    X = df[['y', 'date', 'restaurant_id',
            'brand_label', 'city_label', 'region_label', 
            # 'rnd_label', 'concept_detail_label', 'is_mall_label',
            # 'brand', 'city', 'region_name', 'rnd', 'concept_detail',
            'concept_detail_label', 'is_mall_label',
            'brand', 'city', 'region_name', 'concept_detail',
            'dayofweek','quarter','month','year',
            'monthly_mean_visitors', 'monthly_median_visitors','monthly_min_visitors', 'monthly_max_visitors',
            'weekly_mean_visitors', 'weekly_median_visitors', 'weekly_min_visitors', 'weekly_max_visitors',
            'monthly_mean_visitors_weekeek', 'monthly_median_visitors_weekeek','monthly_min_visitors_weekeek', 'monthly_max_visitors_weekeek',
            'weekly_mean_visitors_weekeek', 'weekly_median_visitors_weekeek', 'weekly_min_visitors_weekeek', 'weekly_max_visitors_weekeek',
            'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            'weight_sale', 
            'holiday', 'new_weekday', 'holiday_weight', 'cluster'
           ]]

    return X

def create_features_old(df, label=None):
    
    X = df.drop(columns=['y', 'date'])        
    if label:
        y = df[label]
        return X, y
    return X



def create_features_15(df):
    df = df.rename(columns={"guest_count": "y", "shift_date": "date"})
    
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    # df['weekofyear'] = df['date'].dt.weekofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    df['y_previous_month'] = df.groupby('restaurant_id')['y'].shift(49)
    
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')
    
    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(49)
    
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')
    
    
    df['normal_y'] = df['y'] / df['holiday_weight'] 
     # Tính 'normal_y' bằng cách chia lần lượt 'y' cho 'holiday_weight' theo từng 'restaurant_id'
    # df['normal_y'] = df.groupby('restaurant_id').apply(lambda group: group['y'] / group['holiday_weight']).reset_index(drop=True)
    # Create lag Feature
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(49)
    df['lag_42s'] = df.groupby('restaurant_id')['normal_y'].shift(56)
    df['lag_49s'] = df.groupby('restaurant_id')['normal_y'].shift(63)
    df['lag_56s'] = df.groupby('restaurant_id')['normal_y'].shift(70)
    
    # Create feature loyalty
    df['loyalty_sale_previous_month'] = df['total_loyalty_sale'].shift(49)
    df['sale_previous_month'] = df['total_sale'].shift(49)
    
    df['monthly_sum_loyalty_sale'] = df.groupby(['year', 'month', 'restaurant_id'])['loyalty_sale_previous_month'].transform('sum')
    df['monthly_sum_sale'] = df.groupby(['year', 'month', 'restaurant_id'])['sale_previous_month'].transform('sum')
    df['weight_sale'] = df['monthly_sum_loyalty_sale'] / df['monthly_sum_sale']

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['new_weekday'] == 1].groupby('restaurant_id')['y'].shift(49)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month_weekeek'].transform('max')
    
    df['y_previous_week_weekend'] = df[df['new_weekday'] == 1].groupby('restaurant_id')['y'].shift(49)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week_weekend'].transform('max')
    
    
    X = df[['y', 'date', 'restaurant_id',
            'brand_label', 'city_label', 'region_label', 
            # 'rnd_label', 'concept_detail_label', 'is_mall_label',
            # 'brand', 'city', 'region_name', 'rnd', 'concept_detail',
            'concept_detail_label', 'is_mall_label',
            'brand', 'city', 'region_name', 'concept_detail',
            'dayofweek','quarter','month','year',
            'monthly_mean_visitors', 'monthly_median_visitors','monthly_min_visitors', 'monthly_max_visitors',
            'weekly_mean_visitors', 'weekly_median_visitors', 'weekly_min_visitors', 'weekly_max_visitors',
            'monthly_mean_visitors_weekeek', 'monthly_median_visitors_weekeek','monthly_min_visitors_weekeek', 'monthly_max_visitors_weekeek',
            'weekly_mean_visitors_weekeek', 'weekly_median_visitors_weekeek', 'weekly_min_visitors_weekeek', 'weekly_max_visitors_weekeek',
            'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            'weight_sale', 
            'holiday', 'new_weekday', 'holiday_weight', 'cluster'
           ]]
    return X


def create_features_old_15(df, label=None):
     
    X = df.drop(columns=['y', 'date'])

    if label:
        y = df[label]
        return X, y
    return X

