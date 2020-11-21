import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from common.utils import *
from common.classifiers import *

"""
Find good hyper-parameters of gradient boosting.

Try to train also using multiple classes (with all thresholds).
"""

data_path = r"C:\DATA2\BITCOIN\GENERATED"
data_file = r"BTCUSDT-1m-features.csv"


features_kline = [
    'close_1','close_5','close_15','close_60','close_180','close_720',
    'close_std_5','close_std_15','close_std_60','close_std_180','close_std_720',  # Removed "std_1" which is constant
    'volume_1','volume_5','volume_15','volume_60','volume_180','volume_720',
    'span_1', 'span_5', 'span_15', 'span_60', 'span_180', 'span_720',
    'trades_1','trades_5','trades_15','trades_60','trades_180','trades_720',
    'tb_base_1','tb_base_5','tb_base_15','tb_base_60','tb_base_180','tb_base_720',
    'tb_quote_1','tb_quote_5','tb_quote_15','tb_quote_60','tb_quote_180','tb_quote_720',
    'close_area_60','close_area_120','close_area_180','close_area_300','close_area_720',
    ]  # 46 features

features_futur = [
    "f_close_1", "f_close_2", "f_close_5", "f_close_20", "f_close_60", "f_close_180",
    "f_close_std_2", "f_close_std_5", "f_close_std_20", "f_close_std_60", "f_close_std_180",  # Removed "std_1" which is constant
    "f_volume_1", "f_volume_2", "f_volume_5", "f_volume_20", "f_volume_60", "f_volume_180",
    "f_span_1", "f_span_2", "f_span_5", "f_span_20", "f_span_60", "f_span_180",
    "f_trades_1", "f_trades_2", "f_trades_5", "f_trades_20", "f_trades_60", "f_trades_180",
    'f_close_area_20', 'f_close_area_60', 'f_close_area_120', 'f_close_area_180',
]  # 33 features


labels = [
    'high_10', 'high_15', 'high_20',
    'low_10', 'low_15', 'low_20',
]
labels_regr = [
    'high_max_60', 'high_max_120', 'high_max_180',  # Maximum high (relative)
    'low_min_60', 'low_min_120', 'low_min_180',  # Minimum low (relative)
    'high_to_low_60', 'high_to_low_120', 'high_to_low_180',
    'close_area_future_60', 'close_area_future_120', 'close_area_future_180', 'close_area_future_300',
]

#
# Parameters of rolling predict
#

nrows = 10_000_000  # For debug
# Columns
train_features = features_futur  # features_futur features_kline
predict_label = "low_20"
# Rows
prediction_start_str = "2020-02-01 00:00:00"  # Use it when rolling prediction will work
#prediction_start_str = "2020-06-01 00:00:00"
train_length = int(4 * 43_800)  # 1.5 * 525_600 for long/spot, 4 * 43_800 for short/futur
stride = 1*7*1440  # Length of one rolling prediction step: mid: 1 month 43_800=4*7*1440, long: 1,5 months 6*7*1440
steps = 1  # How many rolling prediction steps. ~40 weeks in [1.2-1.11]

algorithm = "lc"  # gb nn lc

#
# Parameters for algorithms
#

params_grid_gb = {  # First parameter is the slowest
    # binary (logloss - logistic regression) cross_entropy cross_entropy_lambda
    "objective": ["cross_entropy"],  # "cross_entropy", "cross_entropy_lambda", "binary"
    "max_depth": [1],
    "learning_rate": [0.01],
    "num_boost_round": [1_500],

    "lambda_l1": [1.0], # (reg_alpha) 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100] 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05], 1.5
    "lambda_l2": [1.0],  # (reg_lambda), [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
}
def params_to_line_gb(params):
    line = [
        params.get("objective"),
        params.get("max_depth"),
        params.get("learning_rate"),
        params.get("num_boost_round"),
        params.get("lambda_l1"),
        params.get("lambda_l2"),
    ]
    return line

# TODO: Implement and run GB with scaling (by default it uses NO scaling)
# 1-0.01-1500
# Futur
#high_10, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.625, 0.167, 0.602, 0.097
#high_15, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.641, 0.121, 0.609, 0.067
#high_20, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.652, 0.066, 0.524, 0.035
#low_10, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.632, 0.131, 0.684, 0.073
#low_15, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.649, 0.075, 0.634, 0.040
#low_20, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.653, 0.074, 0.661, 0.039
# klines
#high_10, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.680, 0.171, 0.695, 0.098
#high_15, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.720, 0.075, 0.622, 0.040
#high_20, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.745, 0.001, 0.577, 0.001
#low_10, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.688, 0.200, 0.642, 0.118
#low_15, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.718, 0.072, 0.620, 0.038
#low_20, cross_entropy, 1, 0.01, 1500, 1.0, 1.0, 0.734, 0.011, 0.506, 0.006

params_grid_nn = {  # First parameter is the slowest
    "layers": [[33]],  # Number of layers depends on no of features: 33 future, 46 for spot
    "learning_rate": [0.001],
    "n_epochs": [20],
    "bs": [64],
}
def params_to_line_nn(params):
    line = [
        params.get("layers"),
        params.get("learning_rate"),
        params.get("n_epochs"),
        params.get("bs"),
    ]
    return line

# TODO: Implement and run NN without scaling (by default it usese scaling)
# Futur
#high_10, [33], 0.001, 20, 64, 0.626, 0.171, 0.606, 0.099
#high_15, [33], 0.001, 20, 64, 0.623, 0.118, 0.621, 0.065
#high_20, [33], 0.001, 20, 64, 0.646, 0.061, 0.507, 0.033
#low_10, [33], 0.001, 20, 64, 0.634, 0.179, 0.625, 0.104
#low_15, [33], 0.001, 20, 64, 0.623, 0.106, 0.665, 0.058
#low_20, [33], 0.001, 20, 64, 0.632, 0.058, 0.588, 0.031
# Spot/klines
#high_10, [46], 0.001, 20, 64, 0.675, 0.269, 0.653, 0.169
#high_15, [46], 0.001, 20, 64, 0.707, 0.207, 0.674, 0.122
#high_20, [46], 0.001, 20, 64, 0.717, 0.123, 0.610, 0.068
#low_10, [46], 0.001, 20, 64, 0.696, 0.283, 0.646, 0.181
#low_15, [46], 0.001, 20, 64, 0.713, 0.190, 0.628, 0.112
#low_20, [46], 0.001, 20, 64, 0.729, 0.150, 0.617, 0.085

# Best liblinear: is_scale=False, balance=False, penalty=l2, max_iter=100
params_grid_lc = {  # First parameter is the slowest
    # Best is False, but True can give almost same result (under different conditions)
    "is_scale": [False],
    "penalty": ["l2"],  # "l2" "l1" "elasticnet" "none"
    "C": [1.0],  # small values stronger regularization
    # 1) None balance is always better
    "class_weight": [None],  # "balanced"
    # 1) liblinear - fast convergence (100 is enough), lbfgs (l2 or none) - good convergence, "newton-cg" "sag" "saga"
    "solver": ["liblinear"],
    "max_iter": [200],
}
def params_to_line_lc(params):
    line = [
        params.get("is_scale"),
        params.get("penalty"),
        params.get("C"),
        params.get("class_weight"),
        params.get("solver"),
        params.get("max_iter"),
    ]
    return line

# Results for futur:
#high_10, False, l2, 1.0, None, liblinear, 100, 0.458, 0.032, 0.553, 0.016
#high_15, False, l2, 1.0, None, liblinear, 100, 0.470, 0.029, 0.652, 0.015
#high_20, False, l2, 1.0, None, liblinear, 100, 0.472, 0.030, 0.623, 0.015
#low_10, False, l2, 1.0, None, liblinear, 100, 0.471, 0.017, 0.367, 0.009
#low_15, False, l2, 1.0, None, liblinear, 100, 0.468, 0.005, 0.399, 0.002
#low_20, False, l2, 1.0, None, liblinear, 100, 0.473, 0.010, 0.572, 0.005
# Results for klines (spot):
#high_10, False, l2, 1.0, None, liblinear, 200, 0.551, 0.050, 0.522, 0.026
#high_15, False, l2, 1.0, None, liblinear, 200, 0.558, 0.023, 0.484, 0.012
#high_20, False, l2, 1.0, None, liblinear, 200, 0.564, 0.017, 0.440, 0.009
#low_10, False, l2, 1.0, None, liblinear, 200, 0.565, 0.044, 0.538, 0.023
#low_15, False, l2, 1.0, None, liblinear, 200, 0.580, 0.015, 0.556, 0.008
#low_20, False, l2, 1.0, None, liblinear, 200, 0.584, 0.019, 0.784, 0.010

#
# Grid search
#

def driver():
    #
    # Load and prepare all data
    #

    # Load all data
    df_all = pd.read_csv(data_path + "\\" + data_file, parse_dates=['timestamp'], nrows=nrows)
    for label in labels:
        df_all[label] = df_all[label].astype(int)  # "category" NN does not work without this

    # Select necessary features and label
    df_all = df_all[train_features + labels + ["timestamp"]]

    pd.set_option('use_inf_as_na', True)
    df_all = df_all.dropna()  # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    df_all = df_all.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    prediction_start = find_index(df_all, prediction_start_str)

    del df_all["timestamp"]

    #
    # Prepare params and train by getting precision
    #
    if algorithm == "gb":
        params_grid = params_grid_gb
    elif algorithm == "nn":
        params_grid = params_grid_nn
    elif algorithm == "lc":
        params_grid = params_grid_lc
    else:
        raise ValueError(f"Unknown algorithm value {algorithm}.")

    records = []  # One record with metrics for one param object
    grid = ParameterGrid(params_grid)
    params_list = list(grid)  # List of hyper-param dicts
    for i, params in enumerate(params_list):

        print("\n{}/{} rolling train start...".format(i+1, len(params_list)))

        #
        # Loop over all rolling steps
        #
        y_true = pd.Series(dtype=float)
        y_predicted = pd.Series(dtype=float)
        print("Steps ({}): ".format(steps), end="")
        for step in range(steps):

            # Train step data
            end = prediction_start + (step * stride)
            start = end - train_length
            df_train = df_all.iloc[start:end]

            df_X = df_train[train_features]
            df_y = df_train[predict_label]

            # Test step data
            df_test = df_all.iloc[end:end+stride]

            df_X_test = df_test[train_features]
            df_y_test = df_test[predict_label]

            # ---
            if algorithm == "gb":
                y_test_hat = train_predict_gb(df_X, df_y, df_X_test, params)
            elif algorithm == "nn":
                y_test_hat = train_predict_nn(df_X, df_y, df_X_test, params)
            elif algorithm == "lc":
                y_test_hat = train_predict_lc(df_X, df_y, df_X_test, params)
            # ---

            y_test_hat = pd.Series(index=df_y_test.index, data=y_test_hat)

            # Append true and predicted array
            y_true = y_true.append(df_y_test)
            y_predicted = y_predicted.append(y_test_hat)

            print(".", end="")

        print("")
        print("Finished {} steps of train with {} true and {} predicted results.".format(steps, len(y_true), len(y_predicted)))

        # Computing metrics
        y_predicted_class = np.where(y_predicted > 0.5, 1, 0)
        auc = metrics.roc_auc_score(y_true, y_predicted)

        f1 = metrics.f1_score(y_true, y_predicted_class)
        precision = precision_score(y_true, y_predicted_class)
        recall = recall_score(y_true, y_predicted_class)
        score = {"auc": auc, "f1": f1, "precision": precision, "recall": recall}

        records.append(score)

    #
    # Process all collected results and save
    #
    def get_line_gb(params, record):
        line = [
            predict_label,
            params.get("objective"),
            params.get("max_depth"),
            params.get("learning_rate"),
            params.get("num_boost_round"),
            params.get("lambda_l1"),
            params.get("lambda_l2"),
            "{:.3f}".format(record["auc"]),
            "{:.3f}".format(record["f1"]),
            "{:.3f}".format(record["precision"]),
            "{:.3f}".format(record["recall"]),
        ]
        return line

    def get_line_nn(params, record):
        line = [
            predict_label,
            params.get("layers"),
            params.get("learning_rate"),
            params.get("n_epochs"),
            params.get("bs"),
            "{:.3f}".format(record["auc"]),
            "{:.3f}".format(record["f1"]),
            "{:.3f}".format(record["precision"]),
            "{:.3f}".format(record["recall"]),
        ]
        return line

    def get_line_lc(params, record):
        line = [
            predict_label,
            params.get("is_scale"),
            params.get("penalty"),
            params.get("C"),
            params.get("class_weight"),
            params.get("solver"),
            params.get("max_iter"),
            "{:.3f}".format(record["auc"]),
            "{:.3f}".format(record["f1"]),
            "{:.3f}".format(record["precision"]),
            "{:.3f}".format(record["recall"]),
        ]

        return line

    lines = []
    for i, params in enumerate(params_list):
        line = [predict_label]
        # Add parameters
        if algorithm == "gb":
            line += params_to_line_gb(params)
        elif algorithm == "nn":
            line += params_to_line_nn(params)
        elif algorithm == "lc":
            line += params_to_line_lc(params)
        # Add scores
        rec = records[i]
        score_str = [
            "{:.3f}".format(rec["auc"]),
            "{:.3f}".format(rec["f1"]),
            "{:.3f}".format(rec["precision"]),
            "{:.3f}".format(rec["recall"]),
        ]
        line += score_str

        lines.append(", ".join([str(x) for x in line]))

    with open('metrics.txt', 'a+') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    driver()
