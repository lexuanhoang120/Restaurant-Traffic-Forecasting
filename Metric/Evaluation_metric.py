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




def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Hàm tính NAPE
def calculate_pape(y_true, y_pred):
    nape_values = []
    for true, pred in zip(y_true, y_pred):
        nape = (true - pred) / true * 100
        if nape > 0:
            nape_values.append(nape)
    return nape_values

# Hàm tính PAPE
def calculate_nape(y_true, y_pred):
    pape_values = []
    for true, pred in zip(y_true, y_pred):
        pape = ((true - pred) / true) * 100
        if pape < 0:
            pape_values.append(pape)
    return pape_values


def weighted_absolute_percentage_error(actual, prediction):
    """
    Tính Weighted Absolute Percentage Error (WAPE) cho từng giá trị trong actual và prediction.

    Tham số:
    - actual: Mảng chứa giá trị thực tế.
    - prediction: Mảng chứa giá trị dự đoán.

    Trả về:
    - Mảng chứa giá trị WAPE cho từng giá trị.
    """
    wape_values = []
    for i in range(len(actual)):
        abs_percentage_error = np.abs((actual.iloc[i] - prediction.iloc[i]) / actual.iloc[i])
        weights = actual.iloc[i] / sum(actual)
        wape = weights * abs_percentage_error
        wape_values.append(wape * 100)
    ave_wape = sum(wape_values) / len(wape_values)
    
    return wape_values, ave_wape

def weighted_absolute_percentage_error_v2(actual, prediction):
    """
    Tính Weighted Absolute Percentage Error (WAPE) cho từng giá trị trong actual và prediction.

    Tham số:
    - actual: Mảng chứa giá trị thực tế.
    - prediction: Mảng chứa giá trị dự đoán.

    Trả về:
    - Mảng chứa giá trị WAPE cho từng giá trị.
    """
    sum_abs_percentage_error = sum(np.abs(actual - prediction))
    sum_actual = sum(actual)
    wape = sum_abs_percentage_error / sum_actual * 100
    
    return wape

def m_absolute_percentage_error(actual, prediction):
    """
    Tính Weighted Absolute Percentage Error (WAPE) cho từng giá trị trong actual và prediction.

    Tham số:
    - actual: Mảng chứa giá trị thực tế.
    - prediction: Mảng chứa giá trị dự đoán.

    Trả về:
    - Mảng chứa giá trị WAPE cho từng giá trị.
    """
    mape_values = []
    for i in range(len(actual)):
        abs_percentage_error = np.abs((actual.iloc[i] - prediction.iloc[i]) / actual.iloc[i])
        weights = 1
        mape = weights * abs_percentage_error
        mape_values.append(mape * 100)
    ave_mape = sum(mape_values) / len(mape_values)
    
    return mape_values, ave_mape


def m_absolute_percentage_error_positive_negative(actual, prediction):
    """
    Tính Weighted Absolute Percentage Error (WAPE) cho từng giá trị trong actual và prediction.

    Tham số:
    - actual: Mảng chứa giá trị thực tế.
    - prediction: Mảng chứa giá trị dự đoán.

    Trả về:
    - Mảng chứa giá trị WAPE cho từng giá trị.
    - Giá trị MAPE cho giá trị dương.
    - Giá trị MAPE cho giá trị âm.
    """
    positive_mape_values = []
    negative_mape_values = []
    for i in range(len(actual)):
        abs_percentage_error = np.abs((actual.iloc[i] - prediction.iloc[i]) / actual.iloc[i])
        if (actual.iloc[i] - prediction.iloc[i]) >= 0:
            positive_mape_values.append(abs_percentage_error * 100)
        else:
            negative_mape_values.append(-abs_percentage_error * 100)
    
    try:
        positive_mape = sum(positive_mape_values) / len(positive_mape_values)
    except:
        positive_mape = 0
        
    try:
        negative_mape = sum(negative_mape_values) / len(negative_mape_values)
    except:
        negative_mape = 0
    
    return positive_mape_values, negative_mape_values, positive_mape, negative_mape 


def calculate_mae(actual, prediction):
    """
    Tính Mean Absolute Error (MAE) giữa actual và prediction.

    Tham số:
    - actual: Mảng chứa giá trị thực tế.
    - prediction: Mảng chứa giá trị dự đoán.

    Trả về:
    - Giá trị MAE.
    """
    mae = np.mean(np.abs(np.array(actual) - np.array(prediction)))
    return mae



