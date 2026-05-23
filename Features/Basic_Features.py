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
    df['weekofyear'] = df['date'].dt.isocalendar().week

    df['normal_y'] = df['y'] / df['holiday_weight']

    """   
    df['y_previous_month'] = df.groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(7)
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')
    """

    df['y_previous_week'] = df.groupby('restaurant_id')['normal_y'].shift(7)
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')

    df['y_previous_month'] = df.groupby('restaurant_id')['normal_y'].shift(28)
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    # Create lag Feature
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)
    df['lag_42s'] = df.groupby('restaurant_id')['normal_y'].shift(42)
    df['lag_49s'] = df.groupby('restaurant_id')['normal_y'].shift(49)
    df['lag_56s'] = df.groupby('restaurant_id')['normal_y'].shift(56)

    # New Features ###
    df['lag_30s'] = df.groupby(['restaurant_id'])['normal_y'].shift(28)
    df['mean_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'mean')
    df['median_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'median')
    df['min_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'min')
    df['max_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'max')

    df['lag_7s'] = df.groupby(['restaurant_id'])['normal_y'].shift(7)
    df['mean_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'mean')
    df['median_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_7s'].transform('median')
    df['min_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'min')
    df['max_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'max')

    df['lag_14s'] = df.groupby(['restaurant_id'])['normal_y'].shift(14)
    df['mean_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'mean')
    df['median_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_14s'].transform('median')
    df['min_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'min')
    df['max_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'max')

    df['lag_21s'] = df.groupby(['restaurant_id'])['normal_y'].shift(21)
    df['mean_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'mean')
    df['median_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_21s'].transform('median')
    df['min_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'min')
    df['max_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'max')
    df['lag_30s_y'] = df.groupby(['restaurant_id'])['y'].shift(35)
    df['mean_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_30s_y'].transform(
        'mean')
    df['median_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_30s_y'].transform(
        'median')
    df['min_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s_y'].transform(
        'min')
    df['max_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s_y'].transform(
        'max')

    df['lag_1s'] = df.groupby('restaurant_id')['normal_y'].shift(1)
    df['lag_2s'] = df.groupby('restaurant_id')['normal_y'].shift(2)
    df['lag_3s'] = df.groupby('restaurant_id')['normal_y'].shift(3)
    df['lag_4s'] = df.groupby('restaurant_id')['normal_y'].shift(4)
    df['lag_5s'] = df.groupby('restaurant_id')['normal_y'].shift(5)
    df['lag_6s'] = df.groupby('restaurant_id')['normal_y'].shift(6)
    df['lag_7s'] = df.groupby('restaurant_id')['normal_y'].shift(7)
    df['lag_8s'] = df.groupby('restaurant_id')['normal_y'].shift(8)
    df['lag_9s'] = df.groupby('restaurant_id')['normal_y'].shift(9)
    df['lag_10s'] = df.groupby('restaurant_id')['normal_y'].shift(10)
    df['lag_11s'] = df.groupby('restaurant_id')['normal_y'].shift(11)
    df['lag_12s'] = df.groupby('restaurant_id')['normal_y'].shift(12)
    df['lag_13s'] = df.groupby('restaurant_id')['normal_y'].shift(13)

    df['lag_14s'] = df.groupby('restaurant_id')['normal_y'].shift(14)
    df['lag_15s'] = df.groupby('restaurant_id')['normal_y'].shift(15)
    df['lag_16s'] = df.groupby('restaurant_id')['normal_y'].shift(16)
    df['lag_64s'] = df.groupby('restaurant_id')['normal_y'].shift(64)

    df['lag_21s'] = df.groupby('restaurant_id')['normal_y'].shift(21)
    df['lag_28s'] = df.groupby('restaurant_id')['normal_y'].shift(28)
    df['lag_29s'] = df.groupby('restaurant_id')['normal_y'].shift(29)
    df['lag_30s'] = df.groupby('restaurant_id')['normal_y'].shift(30)
    df['lag_31s'] = df.groupby('restaurant_id')['normal_y'].shift(31)
    df['lag_32s'] = df.groupby('restaurant_id')['normal_y'].shift(32)
    df['lag_33s'] = df.groupby('restaurant_id')['normal_y'].shift(33)
    df['lag_34s'] = df.groupby('restaurant_id')['normal_y'].shift(34)
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)

    df['lag_1s_y'] = df.groupby('restaurant_id')['y'].shift(1)
    df['lag_2s_y'] = df.groupby('restaurant_id')['y'].shift(2)
    df['lag_3s_y'] = df.groupby('restaurant_id')['y'].shift(3)
    df['lag_4s_y'] = df.groupby('restaurant_id')['y'].shift(4)
    df['lag_5s_y'] = df.groupby('restaurant_id')['y'].shift(5)
    df['lag_6s_y'] = df.groupby('restaurant_id')['y'].shift(6)
    df['lag_7s_y'] = df.groupby('restaurant_id')['y'].shift(7)
    df['lag_14s_y'] = df.groupby('restaurant_id')['y'].shift(14)
    df['lag_21s_y'] = df.groupby('restaurant_id')['y'].shift(21)
    df['lag_28s_y'] = df.groupby('restaurant_id')['y'].shift(28)
    df['lag_35s_y'] = df.groupby('restaurant_id')['y'].shift(35)
    df['lag_42s_y'] = df.groupby('restaurant_id')['y'].shift(42)
    df['lag_49s_y'] = df.groupby('restaurant_id')['y'].shift(49)
    df['lag_56s_y'] = df.groupby('restaurant_id')['y'].shift(56)

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('max')

    df['y_previous_week_weekend'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(7)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('max')

    # df['dining'] = df['dining'].astype(bool)
    df['is_mall'] = df['is_mall'].astype(bool)
    # df['month_day'] = df['month'].astype('category')
    # df['dayofweek'] = df['dayofweek'].astype('category')
    # df['moon_day'] = df['moon_day'].astype('category')
    # df['dayofmonth'] = df['dayofmonth'].astype('category')

    X = df[['y', 'date', 'restaurant_id',

            # 'brand',
            # 'city',
            # 'concept_detail',
            # 'sbu',
            # 'is_mall',
            # 'dining',
            # 'moon_day',
            # 'moon_month',
            'is_first_moon_day',
            'is_half_moon_day',

            # 'dayofweek',
            # 'quarter',
            # 'month',
            # 'year',

            # 'monthly_mean_visitors',
            # 'monthly_median_visitors',
            # 'monthly_min_visitors',
            # 'monthly_max_visitors',

            # 'mean_dow_previous_7days',
            # 'median_dow_previous_7days',
            # 'min_dow_previous_7days',
            # 'max_dow_previous_7days',

            'weekly_mean_visitors',
            'weekly_median_visitors',
            'weekly_min_visitors',
            'weekly_max_visitors',

            'pre_holiday',
            'holiday',
            'holiday_weight',
            'weekend',
            'dayofmonth',
            'cluster_km',

            # 'monthly_mean_visitors_weekeek',
            # 'monthly_median_visitors_weekeek',
            # 'monthly_min_visitors_weekeek',
            # 'monthly_max_visitors_weekeek',
            # 'weekly_mean_visitors_weekeek',
            # 'weekly_median_visitors_weekeek',
            # 'weekly_min_visitors_weekeek',
            # 'weekly_max_visitors_weekeek',
            # 'is_Sunday','is_Saturday',

            # 'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            # 'lag_7s', 'lag_14s', 'lag_21s', 'lag_28s',
            # 'lag_7s_y', 'lag_14s_y', 'lag_21s_y', 'lag_28s_y', 'lag_35s_y', 'lag_42s_y', 'lag_49s_y', 'lag_56s_y',

            'mean_dow_previous_month',
            'median_dow_previous_month',
            'min_dow_previous_month',
            'max_dow_previous_month',

            # 'mean_dow_previous_month_y',
            # 'median_dow_previous_month_y',
            # 'min_dow_previous_month_y',
            # 'max_dow_previous_month_y',

            'lag_1s',
            'lag_2s', 'lag_3s', 'lag_4s', 'lag_5s', 'lag_6s',
            'lag_7s', 'lag_8s', 'lag_9s',
            'lag_10s', 'lag_11s', 'lag_12s', 'lag_13s',
            'lag_14s', 'lag_21s', 'lag_28s', 'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s', 'lag_64s',

            # 'lag_29s', 'lag_30s', 'lag_31s', 'lag_32s', 'lag_33s', 'lag_34s',

            # 'mean_dow_previous_14days', 'median_dow_previous_14days', 'min_dow_previous_14days',
            # 'max_dow_previous_14days',

            # 'mean_dow_previous_21days', 'median_dow_previous_21days', 'min_dow_previous_21days',
            # 'max_dow_previous_21days',

            # 'lag_1s_y', 'lag_2s_y', 'lag_3s_y', 'lag_4s_y', 'lag_5s_y', 'lag_6s_y', 'lag_7s_y', 'lag_14s_y',
            # 'lag_21s_y', 'lag_28s_y', 'lag_35s_y', 'lag_42s_y',
            # 'lag_49s_y', 'lag_56s_y'

            ]]
    return X



def create_features_global_2(df):
    df = df.rename(columns={"guest_count": "y", "shift_date": "date"})

    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week

    df['normal_y'] = df['y'] / df['holiday_weight']

    """   
    df['y_previous_month'] = df.groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(7)
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')
    """

    df['y_previous_week'] = df.groupby('restaurant_id')['normal_y'].shift(7)
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')

    df['y_previous_month'] = df.groupby('restaurant_id')['normal_y'].shift(28)
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    # Create lag Feature
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)
    df['lag_42s'] = df.groupby('restaurant_id')['normal_y'].shift(42)
    df['lag_49s'] = df.groupby('restaurant_id')['normal_y'].shift(49)
    df['lag_56s'] = df.groupby('restaurant_id')['normal_y'].shift(56)

    # New Features ###
    df['lag_30s'] = df.groupby(['restaurant_id'])['normal_y'].shift(28)
    df['mean_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'mean')
    df['median_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'median')
    df['min_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'min')
    df['max_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'max')

    df['lag_7s'] = df.groupby(['restaurant_id'])['normal_y'].shift(7)
    df['mean_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'mean')
    df['median_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_7s'].transform('median')
    df['min_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'min')
    df['max_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'max')

    df['lag_14s'] = df.groupby(['restaurant_id'])['normal_y'].shift(14)
    df['mean_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'mean')
    df['median_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_14s'].transform('median')
    df['min_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'min')
    df['max_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'max')

    df['lag_21s'] = df.groupby(['restaurant_id'])['normal_y'].shift(21)
    df['mean_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'mean')
    df['median_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_21s'].transform('median')
    df['min_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'min')
    df['max_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'max')

    df['lag_30s_y'] = df.groupby(['restaurant_id'])['y'].shift(35)
    df['mean_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_30s_y'].transform(
        'mean')
    df['median_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_30s_y'].transform(
        'median')
    df['min_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s_y'].transform(
        'min')
    df['max_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s_y'].transform(
        'max')

    df['lag_1s'] = df.groupby('restaurant_id')['normal_y'].shift(1)
    df['lag_2s'] = df.groupby('restaurant_id')['normal_y'].shift(2)
    df['lag_3s'] = df.groupby('restaurant_id')['normal_y'].shift(3)
    df['lag_4s'] = df.groupby('restaurant_id')['normal_y'].shift(4)
    df['lag_5s'] = df.groupby('restaurant_id')['normal_y'].shift(5)
    df['lag_6s'] = df.groupby('restaurant_id')['normal_y'].shift(6)
    df['lag_7s'] = df.groupby('restaurant_id')['normal_y'].shift(7)
    df['lag_8s'] = df.groupby('restaurant_id')['normal_y'].shift(8)
    df['lag_9s'] = df.groupby('restaurant_id')['normal_y'].shift(9)
    df['lag_10s'] = df.groupby('restaurant_id')['normal_y'].shift(10)
    df['lag_11s'] = df.groupby('restaurant_id')['normal_y'].shift(11)
    df['lag_12s'] = df.groupby('restaurant_id')['normal_y'].shift(12)
    df['lag_13s'] = df.groupby('restaurant_id')['normal_y'].shift(13)

    df['lag_14s'] = df.groupby('restaurant_id')['normal_y'].shift(14)
    df['lag_15s'] = df.groupby('restaurant_id')['normal_y'].shift(15)
    df['lag_16s'] = df.groupby('restaurant_id')['normal_y'].shift(16)
    df['lag_64s'] = df.groupby('restaurant_id')['normal_y'].shift(64)

    df['lag_21s'] = df.groupby('restaurant_id')['normal_y'].shift(21)
    df['lag_28s'] = df.groupby('restaurant_id')['normal_y'].shift(28)
    df['lag_29s'] = df.groupby('restaurant_id')['normal_y'].shift(29)
    df['lag_30s'] = df.groupby('restaurant_id')['normal_y'].shift(30)
    df['lag_31s'] = df.groupby('restaurant_id')['normal_y'].shift(31)
    df['lag_32s'] = df.groupby('restaurant_id')['normal_y'].shift(32)
    df['lag_33s'] = df.groupby('restaurant_id')['normal_y'].shift(33)
    df['lag_34s'] = df.groupby('restaurant_id')['normal_y'].shift(34)
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)

    df['lag_1s_y'] = df.groupby('restaurant_id')['y'].shift(1)
    df['lag_2s_y'] = df.groupby('restaurant_id')['y'].shift(2)
    df['lag_3s_y'] = df.groupby('restaurant_id')['y'].shift(3)
    df['lag_4s_y'] = df.groupby('restaurant_id')['y'].shift(4)
    df['lag_5s_y'] = df.groupby('restaurant_id')['y'].shift(5)
    df['lag_6s_y'] = df.groupby('restaurant_id')['y'].shift(6)
    df['lag_7s_y'] = df.groupby('restaurant_id')['y'].shift(7)
    df['lag_14s_y'] = df.groupby('restaurant_id')['y'].shift(14)
    df['lag_21s_y'] = df.groupby('restaurant_id')['y'].shift(21)
    df['lag_28s_y'] = df.groupby('restaurant_id')['y'].shift(28)
    df['lag_35s_y'] = df.groupby('restaurant_id')['y'].shift(35)
    df['lag_42s_y'] = df.groupby('restaurant_id')['y'].shift(42)
    df['lag_49s_y'] = df.groupby('restaurant_id')['y'].shift(49)
    df['lag_56s_y'] = df.groupby('restaurant_id')['y'].shift(56)

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('max')

    df['y_previous_week_weekend'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(7)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('max')

    # df['dining'] = df['dining'].astype(bool)
    df['is_mall'] = df['is_mall'].astype(bool)

    X = df[['y', 'date', 'restaurant_id',

            # 'brand',
            # 'city',
            # 'concept_detail',
            # 'sbu',
            #
            # 'is_mall',
            # 'dining',
            # 'moon_day',
            'is_first_moon_day',
            'is_half_moon_day',


            'dayofweek',
            # 'quarter',
            # 'month',
            # 'year',

            # 'monthly_mean_visitors',
            # 'monthly_median_visitors',
            # 'monthly_min_visitors',
            # 'monthly_max_visitors',

            # 'mean_dow_previous_7days',
            # 'median_dow_previous_7days',
            # 'min_dow_previous_7days',
            # 'max_dow_previous_7days',

            'weekly_mean_visitors',
            'weekly_median_visitors',
            'weekly_min_visitors',
            'weekly_max_visitors',

            'pre_holiday',
            'holiday',
            'holiday_weight',
            'weekend',
            'dayofmonth',
            'cluster_km',

            # 'monthly_mean_visitors_weekeek',
            # 'monthly_median_visitors_weekeek',
            # 'monthly_min_visitors_weekeek',
            # 'monthly_max_visitors_weekeek',
            # 'weekly_mean_visitors_weekeek',
            # 'weekly_median_visitors_weekeek',
            # 'weekly_min_visitors_weekeek',
            # 'weekly_max_visitors_weekeek',
            # 'is_Sunday','is_Saturday',

            # 'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            # 'lag_7s', 'lag_14s', 'lag_21s', 'lag_28s',
            # 'lag_7s_y', 'lag_14s_y', 'lag_21s_y', 'lag_28s_y', 'lag_35s_y', 'lag_42s_y', 'lag_49s_y', 'lag_56s_y',

            'mean_dow_previous_month',
            'median_dow_previous_month',
            'min_dow_previous_month',
            'max_dow_previous_month',

            # 'mean_dow_previous_month_y',
            # 'median_dow_previous_month_y',
            # 'min_dow_previous_month_y',
            # 'max_dow_previous_month_y',

            # 'lag_1s',
            # 'lag_2s', 'lag_3s', 'lag_4s', 'lag_5s', 'lag_6s',
            # 'lag_7s',
            # 'lag_8s', 'lag_9s',
            # 'lag_10s', 'lag_11s', 'lag_12s', 'lag_13s',
            # 'lag_14s', 'lag_21s', 'lag_28s', 'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s', 'lag_64s',


            # 'lag_29s', 'lag_30s', 'lag_31s', 'lag_32s', 'lag_33s', 'lag_34s',

            # 'mean_dow_previous_14days', 'median_dow_previous_14days', 'min_dow_previous_14days',
            # 'max_dow_previous_14days',

            # 'mean_dow_previous_21days', 'median_dow_previous_21days', 'min_dow_previous_21days',
            # 'max_dow_previous_21days',

            # 'lag_1s_y', 'lag_2s_y', 'lag_3s_y', 'lag_4s_y', 'lag_5s_y', 'lag_6s_y', 'lag_7s_y', 'lag_14s_y',
            # 'lag_21s_y', 'lag_28s_y', 'lag_35s_y', 'lag_42s_y',
            # 'lag_49s_y', 'lag_56s_y'
            ]]
    return X



def create_features_local(df):
    df = df.rename(columns={"guest_count": "y", "shift_date": "date"})

    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week

    df['normal_y'] = df['y'] / df['holiday_weight']

    """   
    df['y_previous_month'] = df.groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(7)
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')
    """

    df['y_previous_week'] = df.groupby('restaurant_id')['normal_y'].shift(7)
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')

    df['y_previous_month'] = df.groupby('restaurant_id')['normal_y'].shift(30)
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    df['y_previous_year'] = df.groupby('restaurant_id')['normal_y'].shift(365)
    df['yearly_mean_visitors'] = df.groupby(['year', 'restaurant_id'])['y_previous_year'].transform('mean')
    df['yearly_median_visitors'] = df.groupby(['year', 'restaurant_id'])['y_previous_year'].transform(
        'median')
    df['yearly_min_visitors'] = df.groupby(['year', 'restaurant_id'])['y_previous_year'].transform('min')
    df['yearly_max_visitors'] = df.groupby(['year', 'restaurant_id'])['y_previous_year'].transform('max')

    # Create lag Feature
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)
    df['lag_42s'] = df.groupby('restaurant_id')['normal_y'].shift(42)
    df['lag_49s'] = df.groupby('restaurant_id')['normal_y'].shift(49)
    df['lag_56s'] = df.groupby('restaurant_id')['normal_y'].shift(56)

    # New Features ###
    df['lag_30s'] = df.groupby(['restaurant_id'])['normal_y'].shift(28)
    df['mean_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'mean')
    df['median_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'median')
    df['min_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'min')
    df['max_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'max')

    df['lag_7s'] = df.groupby(['restaurant_id'])['normal_y'].shift(7)
    df['mean_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'mean')
    df['median_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_7s'].transform('median')
    df['min_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'min')
    df['max_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'max')

    df['lag_14s'] = df.groupby(['restaurant_id'])['normal_y'].shift(14)
    df['mean_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'mean')
    df['median_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_14s'].transform('median')
    df['min_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'min')
    df['max_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'max')

    df['lag_21s'] = df.groupby(['restaurant_id'])['normal_y'].shift(21)
    df['mean_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'mean')
    df['median_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_21s'].transform('median')
    df['min_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'min')
    df['max_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'max')

    df['lag_30s_y'] = df.groupby(['restaurant_id'])['y'].shift(32)
    df['mean_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_30s_y'].transform(
        'mean')
    df['median_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_30s_y'].transform(
        'median')
    df['min_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s_y'].transform(
        'min')
    df['max_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s_y'].transform(
        'max')

    df['lag_1s'] = df.groupby('restaurant_id')['normal_y'].shift(1)
    df['lag_2s'] = df.groupby('restaurant_id')['normal_y'].shift(2)
    df['lag_3s'] = df.groupby('restaurant_id')['normal_y'].shift(3)
    df['lag_4s'] = df.groupby('restaurant_id')['normal_y'].shift(4)
    df['lag_5s'] = df.groupby('restaurant_id')['normal_y'].shift(5)
    df['lag_6s'] = df.groupby('restaurant_id')['normal_y'].shift(6)
    df['lag_7s'] = df.groupby('restaurant_id')['normal_y'].shift(7)
    df['lag_14s'] = df.groupby('restaurant_id')['normal_y'].shift(14)
    df['lag_21s'] = df.groupby('restaurant_id')['normal_y'].shift(21)
    df['lag_28s'] = df.groupby('restaurant_id')['normal_y'].shift(28)
    df['lag_29s'] = df.groupby('restaurant_id')['normal_y'].shift(29)
    df['lag_30s'] = df.groupby('restaurant_id')['normal_y'].shift(30)
    df['lag_31s'] = df.groupby('restaurant_id')['normal_y'].shift(31)
    df['lag_32s'] = df.groupby('restaurant_id')['normal_y'].shift(32)
    df['lag_33s'] = df.groupby('restaurant_id')['normal_y'].shift(33)
    df['lag_34s'] = df.groupby('restaurant_id')['normal_y'].shift(34)
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)

    df['lag_1s_y'] = df.groupby('restaurant_id')['y'].shift(1)
    df['lag_2s_y'] = df.groupby('restaurant_id')['y'].shift(2)
    df['lag_3s_y'] = df.groupby('restaurant_id')['y'].shift(3)
    df['lag_4s_y'] = df.groupby('restaurant_id')['y'].shift(4)
    df['lag_5s_y'] = df.groupby('restaurant_id')['y'].shift(5)
    df['lag_6s_y'] = df.groupby('restaurant_id')['y'].shift(6)
    df['lag_7s_y'] = df.groupby('restaurant_id')['y'].shift(7)
    df['lag_14s_y'] = df.groupby('restaurant_id')['y'].shift(14)
    df['lag_21s_y'] = df.groupby('restaurant_id')['y'].shift(21)
    df['lag_28s_y'] = df.groupby('restaurant_id')['y'].shift(28)
    df['lag_35s_y'] = df.groupby('restaurant_id')['y'].shift(35)
    df['lag_42s_y'] = df.groupby('restaurant_id')['y'].shift(42)
    df['lag_49s_y'] = df.groupby('restaurant_id')['y'].shift(49)
    df['lag_56s_y'] = df.groupby('restaurant_id')['y'].shift(56)

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('max')

    df['y_previous_week_weekend'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(7)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('max')

    df['dining'] = df['dining'].astype(bool)
    df['is_mall'] = df['is_mall'].astype(bool)

    X = df[['y', 'date', 'restaurant_id',

            # 'brand',
            # 'city',
            # 'concept_detail',
            # 'sbu',

            # 'is_mall',
            # 'dining',

            'dayofweek',
            # 'quarter',
            # 'month',
            # 'year',

            # 'yearly_mean_visitors',
            # 'yearly_median_visitors',
            # 'yearly_min_visitors',
            # 'yearly_max_visitors',

            'monthly_mean_visitors',
            'monthly_median_visitors',
            'monthly_min_visitors',
            'monthly_max_visitors',

            # 'weekly_mean_visitors',
            # 'weekly_median_visitors',
            # 'weekly_min_visitors',
            # 'weekly_max_visitors',

            'pre_holiday',
            'holiday',
            'holiday_weight',
            'weekend',
            'dayofmonth',
            # 'cluster',
            # 'cluster_km',

            # 'monthly_mean_visitors_weekeek',
            # 'monthly_median_visitors_weekeek',
            # 'monthly_min_visitors_weekeek',
            # 'monthly_max_visitors_weekeek',
            # 'weekly_mean_visitors_weekeek',
            # 'weekly_median_visitors_weekeek',
            # 'weekly_min_visitors_weekeek',
            # 'weekly_max_visitors_weekeek',

            # 'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            # 'lag_7s', 'lag_14s', 'lag_21s', 'lag_28s',
            # 'lag_7s_y', 'lag_14s_y', 'lag_21s_y', 'lag_28s_y', 'lag_35s_y', 'lag_42s_y', 'lag_49s_y', 'lag_56s_y',

            # 'mean_dow_previous_month',
            # 'median_dow_previous_month',
            # 'min_dow_previous_month',
            # 'max_dow_previous_month',

            # 'mean_dow_previous_month_y',
            # 'median_dow_previous_month_y',
            # 'min_dow_previous_month_y',
            # 'max_dow_previous_month_y',

            'lag_1s', 'lag_2s', 'lag_3s', 'lag_4s', 'lag_5s', 'lag_6s', 'lag_7s', 'lag_14s', 'lag_21s', 'lag_28s',
            'lag_35s',
            'lag_42s', 'lag_49s', 'lag_56s',

            # 'lag_29s', 'lag_30s', 'lag_31s', 'lag_32s', 'lag_33s', 'lag_34s',

            # 'mean_dow_previous_7days',
            # 'median_dow_previous_7days',
            # 'min_dow_previous_7days',
            # 'max_dow_previous_7days',

            # 'mean_dow_previous_14days', 'median_dow_previous_14days', 'min_dow_previous_14days',
            # 'max_dow_previous_14days',
            #
            # 'mean_dow_previous_21days', 'median_dow_previous_21days', 'min_dow_previous_21days',
            # 'max_dow_previous_21days',

            # 'lag_1s_y', 'lag_2s_y', 'lag_3s_y', 'lag_4s_y', 'lag_5s_y', 'lag_6s_y', 'lag_7s_y', 'lag_14s_y',
            # 'lag_21s_y', 'lag_28s_y', 'lag_35s_y', 'lag_42s_y',
            # 'lag_49s_y', 'lag_56s_y'

            ]]
    return X


def create_features_local_old(df, label=None):
    X = df.drop(columns=['y', 'date', 'restaurant_id'])
    if label:
        y = df[label]
        return X, y
    return X


def create_features_local_15(df):
    df = df.rename(columns={"guest_count": "y", "shift_date": "date"})

    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week

    df['normal_y'] = df['y'] / df['holiday_weight']

    """   
    df['y_previous_month'] = df.groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(7)
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')
    """

    df['y_previous_week'] = df.groupby('restaurant_id')['normal_y'].shift(28)
    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')

    df['y_previous_month'] = df.groupby('restaurant_id')['normal_y'].shift(63)
    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    # Create lag Feature
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(63)
    df['lag_42s'] = df.groupby('restaurant_id')['normal_y'].shift(70)
    df['lag_49s'] = df.groupby('restaurant_id')['normal_y'].shift(77)
    df['lag_56s'] = df.groupby('restaurant_id')['normal_y'].shift(84)

    # New Features ###
    df['lag_30s'] = df.groupby(['restaurant_id'])['normal_y'].shift(28)
    df['mean_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'mean')
    df['median_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'median')
    df['min_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'min')
    df['max_dow_previous_month'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s'].transform(
        'max')

    df['lag_7s'] = df.groupby(['restaurant_id'])['normal_y'].shift(7)
    df['mean_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'mean')
    df['median_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_7s'].transform('median')
    df['min_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'min')
    df['max_dow_previous_7days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_7s'].transform(
        'max')

    df['lag_14s'] = df.groupby(['restaurant_id'])['normal_y'].shift(14)
    df['mean_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'mean')
    df['median_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_14s'].transform('median')
    df['min_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'min')
    df['max_dow_previous_14days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_14s'].transform(
        'max')

    df['lag_21s'] = df.groupby(['restaurant_id'])['normal_y'].shift(21)
    df['mean_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'mean')
    df['median_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_21s'].transform('median')
    df['min_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'min')
    df['max_dow_previous_21days'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_21s'].transform(
        'max')

    df['lag_30s_y'] = df.groupby(['restaurant_id'])['y'].shift(32)
    df['mean_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_30s_y'].transform(
        'mean')
    df['median_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])[
        'lag_30s_y'].transform(
        'median')
    df['min_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s_y'].transform(
        'min')
    df['max_dow_previous_month_y'] = df.groupby(['restaurant_id', 'year', 'month', 'dayofweek'])['lag_30s_y'].transform(
        'max')

    df['lag_1s'] = df.groupby('restaurant_id')['normal_y'].shift(29)
    df['lag_2s'] = df.groupby('restaurant_id')['normal_y'].shift(30)
    df['lag_3s'] = df.groupby('restaurant_id')['normal_y'].shift(31)
    df['lag_4s'] = df.groupby('restaurant_id')['normal_y'].shift(31)
    df['lag_5s'] = df.groupby('restaurant_id')['normal_y'].shift(33)
    df['lag_6s'] = df.groupby('restaurant_id')['normal_y'].shift(34)
    df['lag_7s'] = df.groupby('restaurant_id')['normal_y'].shift(35)
    df['lag_14s'] = df.groupby('restaurant_id')['normal_y'].shift(42)
    df['lag_21s'] = df.groupby('restaurant_id')['normal_y'].shift(49)
    df['lag_28s'] = df.groupby('restaurant_id')['normal_y'].shift(56)
    df['lag_29s'] = df.groupby('restaurant_id')['normal_y'].shift(29)
    df['lag_30s'] = df.groupby('restaurant_id')['normal_y'].shift(30)
    df['lag_31s'] = df.groupby('restaurant_id')['normal_y'].shift(31)
    df['lag_32s'] = df.groupby('restaurant_id')['normal_y'].shift(32)
    df['lag_33s'] = df.groupby('restaurant_id')['normal_y'].shift(33)
    df['lag_34s'] = df.groupby('restaurant_id')['normal_y'].shift(34)
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)

    df['lag_1s_y'] = df.groupby('restaurant_id')['y'].shift(1)
    df['lag_2s_y'] = df.groupby('restaurant_id')['y'].shift(2)
    df['lag_3s_y'] = df.groupby('restaurant_id')['y'].shift(3)
    df['lag_4s_y'] = df.groupby('restaurant_id')['y'].shift(4)
    df['lag_5s_y'] = df.groupby('restaurant_id')['y'].shift(5)
    df['lag_6s_y'] = df.groupby('restaurant_id')['y'].shift(6)
    df['lag_7s_y'] = df.groupby('restaurant_id')['y'].shift(7)
    df['lag_14s_y'] = df.groupby('restaurant_id')['y'].shift(14)
    df['lag_21s_y'] = df.groupby('restaurant_id')['y'].shift(21)
    df['lag_28s_y'] = df.groupby('restaurant_id')['y'].shift(28)
    df['lag_35s_y'] = df.groupby('restaurant_id')['y'].shift(35)
    df['lag_42s_y'] = df.groupby('restaurant_id')['y'].shift(42)
    df['lag_49s_y'] = df.groupby('restaurant_id')['y'].shift(49)
    df['lag_56s_y'] = df.groupby('restaurant_id')['y'].shift(56)

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('max')

    df['y_previous_week_weekend'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(7)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('max')

    df['dining'] = df['dining'].astype(bool)
    df['is_mall'] = df['is_mall'].astype(bool)

    X = df[['y', 'date', 'restaurant_id',

            # 'brand',
            # 'city',
            # 'concept_detail',
            # 'sbu',

            # 'is_mall',
            # 'dining',

            'dayofweek',
            # 'quarter',
            # 'month',
            # 'year',

            'monthly_mean_visitors',
            'monthly_median_visitors',
            'monthly_min_visitors',
            'monthly_max_visitors',

            # 'weekly_mean_visitors',
            # 'weekly_median_visitors',
            # 'weekly_min_visitors',
            # 'weekly_max_visitors',

            'pre_holiday',
            'holiday',
            'holiday_weight',
            'weekend',
            'dayofmonth',
            # 'cluster',
            # 'cluster_km',

            # 'monthly_mean_visitors_weekeek',
            # 'monthly_median_visitors_weekeek',
            # 'monthly_min_visitors_weekeek',
            # 'monthly_max_visitors_weekeek',
            # 'weekly_mean_visitors_weekeek',
            # 'weekly_median_visitors_weekeek',
            # 'weekly_min_visitors_weekeek',
            # 'weekly_max_visitors_weekeek',

            # 'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            # 'lag_7s', 'lag_14s', 'lag_21s', 'lag_28s',
            # 'lag_7s_y', 'lag_14s_y', 'lag_21s_y', 'lag_28s_y', 'lag_35s_y', 'lag_42s_y', 'lag_49s_y', 'lag_56s_y',

            # 'mean_dow_previous_month',
            # 'median_dow_previous_month',
            # 'min_dow_previous_month',
            # 'max_dow_previous_month',

            # 'mean_dow_previous_month_y',
            # 'median_dow_previous_month_y',
            # 'min_dow_previous_month_y',
            # 'max_dow_previous_month_y',

            'lag_1s', 'lag_2s', 'lag_3s', 'lag_4s', 'lag_5s', 'lag_6s', 'lag_7s', 'lag_14s', 'lag_21s', 'lag_28s',
            'lag_35s',
            'lag_42s', 'lag_49s', 'lag_56s',

            # 'lag_29s', 'lag_30s', 'lag_31s', 'lag_32s', 'lag_33s', 'lag_34s',

            # 'mean_dow_previous_7days',
            # 'median_dow_previous_7days',
            # 'min_dow_previous_7days',
            # 'max_dow_previous_7days',

            # 'mean_dow_previous_14days', 'median_dow_previous_14days', 'min_dow_previous_14days',
            # 'max_dow_previous_14days',
            #
            # 'mean_dow_previous_21days', 'median_dow_previous_21days', 'min_dow_previous_21days',
            # 'max_dow_previous_21days',

            # 'lag_1s_y', 'lag_2s_y', 'lag_3s_y', 'lag_4s_y', 'lag_5s_y', 'lag_6s_y', 'lag_7s_y', 'lag_14s_y',
            # 'lag_21s_y', 'lag_28s_y', 'lag_35s_y', 'lag_42s_y',
            # 'lag_49s_y', 'lag_56s_y'

            ]]
    return X


def create_features_local_old_15(df, label=None):
    X = df.drop(columns=['y', 'date', 'restaurant_id'])
    if label:
        y = df[label]
        return X, y
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

    df['y_previous_month'] = df.groupby('restaurant_id')['y'].shift(30)

    df['monthly_mean_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('mean')
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')

    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(7)

    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
    df['weekly_min_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('min')
    df['weekly_max_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform('max')

    df['normal_y'] = df['y'] / df['holiday_weight']

    # Create lag Feature
    df['lag_35s'] = df.groupby('restaurant_id')['normal_y'].shift(35)
    df['lag_42s'] = df.groupby('restaurant_id')['normal_y'].shift(42)
    df['lag_49s'] = df.groupby('restaurant_id')['normal_y'].shift(49)
    df['lag_56s'] = df.groupby('restaurant_id')['normal_y'].shift(56)

    df['lag_35s_y'] = df.groupby('restaurant_id')['y'].shift(35)
    df['lag_42s_y'] = df.groupby('restaurant_id')['y'].shift(42)
    df['lag_49s_y'] = df.groupby('restaurant_id')['y'].shift(49)
    df['lag_56s_y'] = df.groupby('restaurant_id')['y'].shift(56)

    # Create loyalty feature
    # df['loyalty_sale_previous_month'] = df.groupby('restaurant_id')['total_loyalty_sale'].shift(35)
    # df['sale_previous_month'] = df.groupby('restaurant_id')['total_sale'].shift(35)
    #
    # df['monthly_sum_loyalty_sale'] = df.groupby(['year', 'month', 'restaurant_id'])[
    #     'loyalty_sale_previous_month'].transform('sum')
    # df['monthly_sum_sale'] = df.groupby(['year', 'month', 'restaurant_id'])['sale_previous_month'].transform('sum')
    # df['weight_sale'] = df['monthly_sum_loyalty_sale'] / df['monthly_sum_sale']

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('max')

    df['y_previous_week_weekend'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(35)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('max')

    X = df[['y', 'date', 'restaurant_id',

            # 'brand_label', 'city_label', 'region_label',
            # 'rnd_label', 'concept_detail_label', 'is_mall_label',
            # 'brand', 'city', 'region_name', 'rnd', 'concept_detail',
            # 'concept_detail_label', 'is_mall_label',
            # 'brand', 'city', 'region_name', 'concept_detail',

            'dayofweek',
            'quarter',
            'month',
            'year',
            'monthly_mean_visitors', 'monthly_median_visitors', 'monthly_min_visitors', 'monthly_max_visitors',
            'weekly_mean_visitors', 'weekly_median_visitors', 'weekly_min_visitors', 'weekly_max_visitors',
            'monthly_mean_visitors_weekeek', 'monthly_median_visitors_weekeek', 'monthly_min_visitors_weekeek',
            'monthly_max_visitors_weekeek',
            'weekly_mean_visitors_weekeek', 'weekly_median_visitors_weekeek', 'weekly_min_visitors_weekeek',
            'weekly_max_visitors_weekeek',
            'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            'lag_35s_y', 'lag_42s_y', 'lag_49s_y', 'lag_56s_y',
            # 'weight_sale',
            'holiday',
            'weekend',
            'holiday_weight',
            'cluster',
            # 'close_date'
            ]]

    return X


def create_features_old(df, label=None):
    X = df.drop(columns=['y', 'date', 'restaurant_id'])
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
    df['monthly_median_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform(
        'median')
    df['monthly_min_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('min')
    df['monthly_max_visitors'] = df.groupby(['year', 'month', 'restaurant_id'])['y_previous_month'].transform('max')

    df['y_previous_week'] = df.groupby('restaurant_id')['y'].shift(49)

    df['weekly_mean_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'mean')
    df['weekly_median_visitors'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])['y_previous_week'].transform(
        'median')
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

    df['lag_35s_y'] = df.groupby('restaurant_id')['y'].shift(49)
    df['lag_42s_y'] = df.groupby('restaurant_id')['y'].shift(56)
    df['lag_49s_y'] = df.groupby('restaurant_id')['y'].shift(63)
    df['lag_56s_y'] = df.groupby('restaurant_id')['y'].shift(70)

    # Create feature loyalty
    # df['loyalty_sale_previous_month'] = df['total_loyalty_sale'].shift(49)
    # df['sale_previous_month'] = df['total_sale'].shift(49)
    #
    # df['monthly_sum_loyalty_sale'] = df.groupby(['year', 'month', 'restaurant_id'])[
    #     'loyalty_sale_previous_month'].transform('sum')
    # df['monthly_sum_sale'] = df.groupby(['year', 'month', 'restaurant_id'])['sale_previous_month'].transform('sum')
    # df['weight_sale'] = df['monthly_sum_loyalty_sale'] / df['monthly_sum_sale']

    ### Statistic Features for weekend ###
    df['y_previous_month_weekeek'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(49)
    df['monthly_mean_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('mean')
    df['monthly_median_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('median')
    df['monthly_min_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('min')
    df['monthly_max_visitors_weekeek'] = df.groupby(['year', 'month', 'restaurant_id'])[
        'y_previous_month_weekeek'].transform('max')

    df['y_previous_week_weekend'] = df[df['weekend'] == 1].groupby('restaurant_id')['y'].shift(49)
    df['weekly_mean_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('mean')
    df['weekly_median_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('median')
    df['weekly_min_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('min')
    df['weekly_max_visitors_weekeek'] = df.groupby(['year', 'weekofyear', 'restaurant_id'])[
        'y_previous_week_weekend'].transform('max')

    X = df[['y', 'date', 'restaurant_id',
            'brand_label', 'city_label', 'region_label',
            # 'rnd_label', 'concept_detail_label', 'is_mall_label',
            # 'brand', 'city', 'region_name', 'rnd', 'concept_detail',
            'concept_detail_label', 'is_mall_label',
            'brand', 'city', 'region_name', 'concept_detail',
            'dayofweek', 'quarter', 'month', 'year',
            'monthly_mean_visitors', 'monthly_median_visitors', 'monthly_min_visitors', 'monthly_max_visitors',
            'weekly_mean_visitors', 'weekly_median_visitors', 'weekly_min_visitors', 'weekly_max_visitors',
            'monthly_mean_visitors_weekeek', 'monthly_median_visitors_weekeek', 'monthly_min_visitors_weekeek',
            'monthly_max_visitors_weekeek',
            'weekly_mean_visitors_weekeek', 'weekly_median_visitors_weekeek', 'weekly_min_visitors_weekeek',
            'weekly_max_visitors_weekeek',
            'lag_35s', 'lag_42s', 'lag_49s', 'lag_56s',
            'lag_35s_y', 'lag_42s_y', 'lag_49s_y', 'lag_56s_y',
            # 'weight_sale',
            'holiday', 'weekend', 'holiday_weight'
        , 'cluster',
            ]]
    return X


def create_features_old_15(df, label=None):
    X = df.drop(columns=['y', 'date', 'restaurant_id'])

    if label:
        y = df[label]
        return X, y
    return X
