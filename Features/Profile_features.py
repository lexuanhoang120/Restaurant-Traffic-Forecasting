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

"""
Xử lí các feature về dạng "Category"

Tham số:
- filtered_data: Là dataframe dạng đã được tiền xử lí các
                điểm bất thường

Trả về:
- filtered_data(after): Là dataframe mà các feature đã được 
                      xử lí về dạng category
"""


def label_brand(filtered_data):
    unique_brand = filtered_data['brand'].dropna().unique()
    brand_labels = {brand: label + 1 for label, brand in enumerate(unique_brand)}
    filtered_data['brand_label'] = filtered_data['brand'].map(brand_labels)
    filtered_data['brand_label'].fillna(0, inplace=True)
    return filtered_data


def label_city(filtered_data):
    unique_brand = filtered_data['city'].dropna().unique()
    brand_labels = {city: sub + 1 for sub, city in enumerate(unique_brand)}
    filtered_data['city_label'] = filtered_data['city'].map(brand_labels)
    filtered_data['city_label'].fillna(0, inplace=True)
    return filtered_data


try:
    def label_region(filtered_data):
        unique_brand = filtered_data['sbu'].dropna().unique()
        brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
        filtered_data['region_label'] = filtered_data['sbu'].map(brand_labels)
        filtered_data['region_label'].fillna(0, inplace=True)
        return filtered_data
except Exception as e:
    print(f"Failed for loading dim restaurant dataset: {str(e)}")


def label_rnd(filtered_data):
    unique_brand = filtered_data['rnd'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['rnd_label'] = filtered_data['rnd'].map(brand_labels)
    filtered_data['rnd_label'].fillna(0, inplace=True)
    return filtered_data


def label_concept_detail(filtered_data):
    unique_brand = filtered_data['concept_detail'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['concept_detail_label'] = filtered_data['concept_detail'].map(brand_labels)
    filtered_data['concept_detail_label'].fillna(0, inplace=True)
    return filtered_data


def label_is_mall(filtered_data):
    unique_brand = filtered_data['is_mall'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['is_mall_label'] = filtered_data['is_mall'].map(brand_labels)
    filtered_data['is_mall_label'].fillna(0, inplace=True)
    return filtered_data


def label_dayofweek(filtered_data):
    unique_brand = filtered_data['dayofweek'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['dayofweek_label'] = filtered_data['dayofweek'].map(brand_labels)
    filtered_data['dayofweek_label'].fillna(0, inplace=True)
    return filtered_data


def label_restaurant_id(filtered_data):
    unique_brand = filtered_data['restaurant_id'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['restaurant_id_label'] = filtered_data['restaurant_id'].map(brand_labels)
    filtered_data['restaurant_id_label'].fillna(0, inplace=True)
    return filtered_data


def label_holiday(filtered_data):
    unique_brand = filtered_data['holiday'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['holiday_label'] = filtered_data['holiday'].map(brand_labels)
    filtered_data['holiday_label'].fillna(0, inplace=True)
    return filtered_data


def label_quarter(filtered_data):
    unique_brand = filtered_data['quarter'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['quarter_label'] = filtered_data['quarter'].map(brand_labels)
    filtered_data['quarter_label'].fillna(0, inplace=True)
    return filtered_data


def label_month(filtered_data):
    unique_brand = filtered_data['month'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['month_label'] = filtered_data['month'].map(brand_labels)
    filtered_data['month_label'].fillna(0, inplace=True)
    return filtered_data


def label_cluster(filtered_data):
    unique_brand = filtered_data['cluster'].dropna().unique()
    brand_labels = {region: sub + 1 for sub, region in enumerate(unique_brand)}
    filtered_data['cluster_label'] = filtered_data['cluster'].map(brand_labels)
    filtered_data['cluster_label'].fillna(0, inplace=True)
    return filtered_data
