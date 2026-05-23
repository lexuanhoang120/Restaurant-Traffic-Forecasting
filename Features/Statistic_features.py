from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import matplotlib.pyplot as plt
import pandas as pd

from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures
from feature_engine.timeseries.forecasting import (
    LagFeatures,
    WindowFeatures,
)
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
# --- The transformers from earlier in the course. --- #
# Lag and window features
from sktime.transformations.series.summarize import WindowSummarizer
# Time features for trend 
from sktime.transformations.series.time_since import TimeSince
from sklearn.preprocessing import PolynomialFeatures
# Rescaling transformer for linear models with regularisation
from sklearn.preprocessing import MinMaxScaler
# Pipelines to create feature engineering pipeline
from sklearn.pipeline import make_pipeline, make_union
# Used to reset sklearn estimators
from sklearn.base import clone
# Let's ensure all sklearn transformers output pandas dataframes
from sklearn import set_config

set_config(transform_output="pandas")  # Upgrade to scikit-learn >= 0.12
# for this feature


"""
Xử lí nhóm feature ("total_sale", "guest_count", "total_bill")
thành các nhóm feature Statistic mang đặc điểm của 1-2 tháng trước đó

Tham số:
- window_pairs_local: Là khoảng thống kê trong khoảng thời gian ngắn (Từ ngày 35 đến 56 về trước)
- window_pairs_global: Là khoảng thống kê trong khoảng thời dai ngắn (Từ ngày 35 đến 56 về trước)

Trả về:
- max, min, median, mean, std, sum và lag theo các nhóm window_pairs_local và window_pairs_global
"""


def create_features_lag_window(X):
    features = [
        "total_sale",
        "total_bill",
        "guest_count"]

    window_pairs_local = [(35, 37), (37, 42), (42, 49), (49, 56)]
    window_pairs_global = [(35, 42), (35, 49), (35, 56)]
    lag_f = LagFeatures(variables=features,
                        periods=[i for i in range(35, 57)],
                        drop_original=True,
                        missing_values="ignore"
                        )
    X_lag = lag_f.fit_transform(X)
    X_window = windown_feature(window_pairs_local, window_pairs_global, X_lag, features)
    return X_window


def generate_sequence(start, end):
    if start > end:
        return []
    return list(range(start, end + 1))


def windown_feature(window_pairs_local, window_pairs_global, X, features):
    for feature in features:
        for window_pair in window_pairs_local:
            window_lags = generate_sequence(window_pair[0], window_pair[1])
            windows = [feature + "_lag_" + str(i) for i in window_lags]

            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "local_max"] = X[windows].max(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "local_mean"] = X[windows].mean(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "local_sum"] = X[windows].sum(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "local_median"] = X[windows].median(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "local_std"] = X[windows].std(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "local_min"] = X[windows].min(axis=1)

        for window_pair in window_pairs_global:
            window_lags = generate_sequence(window_pair[0], window_pair[1])
            windows = [feature + "_lag_" + str(i) for i in window_lags]

            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "global_max"] = X[windows].max(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "global_min"] = X[windows].min(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "global_mean"] = X[windows].mean(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "global_sum"] = X[windows].sum(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "global_median"] = X[windows].median(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "global_std"] = X[windows].std(axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "global_quantile90"] = X[windows].quantile(
                .90, axis=1)
            X[feature + "_{}_{}_".format(window_pair[0], window_pair[1]) + "global_quantile10"] = X[windows].quantile(
                .10, axis=1)
    return X
