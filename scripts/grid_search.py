import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from service.App import App
from common.utils import *
from common.classifiers import *

"""
Find good hyper-parameters of algorithms applied to generated featues.
"""

data_path = r"C:\DATA2\BITCOIN\GENERATED"
data_file = r"BTCUSDT-1m-features.csv"

labels = App.config["labels"]
features_kline = App.config["features_kline"]
features_futur = App.config["features_futur"]

# features_horizon = 720  # Features are generated using this past window length (max feature window)
labels_horizon = 180  # Labels are generated using this number of steps ahead (max label window)

#
# Parameters of rolling predict
#

nrows = 10_000_000  # For debug
# Columns
train_features = features_kline  # features_futur features_kline
predict_label = "high_15"
# Rows
prediction_start_str = "2020-09-01 00:00:00"  # Use it when rolling prediction will work (2020-02-01 00:00:00 - for futur)
#prediction_start_str = "2020-06-01 00:00:00"
train_length = int(1.5 * 525_600)  # 1.5 * 525_600 for long/spot, 4 * 43_800 for short/futur
stride = 4*7*1440  # Length of one rolling prediction step: mid: 1 month 43_800=4*7*1440, long: 1,5 months 6*7*1440
steps = 2  # How many rolling prediction steps. ~40 weeks in [1.2-1.11]

algorithm = "nn"  # gb nn lc

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
    "layers": [[33]],  # Number of layers depends on the number of input features
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

def main():
    #
    # Load and prepare all data
    #

    # Load all data
    df_all = pd.read_csv(data_path + "\\" + data_file, parse_dates=['timestamp'], nrows=nrows)

    print(f"Feature matrix loaded. Length: {len(df_all)}. Width: {len(df_all.columns)}")

    for label in labels:
        df_all[label] = df_all[label].astype(int)  # "category" NN does not work without this

    # Select necessary features and label
    df_all = df_all[["timestamp"] + features_kline + features_futur + labels]

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #df_all = df_all.dropna(subset=labels)
    df_all = df_all.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    prediction_start = find_index(df_all, prediction_start_str)
    print(f"Start index: {prediction_start}")

    if len(df_all) - prediction_start < steps * stride:
        raise ValueError(f"Number of steps {steps} is too high (not enough data after start). Data available for prediction: {len(df_all) - prediction_start}. Data to be predicted: {steps * stride} ")

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

    metrics = []  # One record with metrics for one param object
    grid = ParameterGrid(params_grid)
    params_list = list(grid)  # List of hyper-param dicts
    for i, params in enumerate(params_list):

        print("\n{}/{} rolling train start...".format(i+1, len(params_list)))

        # Here we will collect true and predicted values for one label
        # These series must have the same indexes and these indexes should correspond to main input indexes even if some rows are dropped (for them we store Null)
        y_true = pd.Series(dtype=float)
        y_predicted = pd.Series(dtype=float)

        for step in range(steps):

            print(f"\nStart step {step}/{steps}")

            # Predict data

            predict_start = prediction_start + (step * stride)
            predict_end = predict_start + stride

            df_test = df_all.iloc[predict_start:predict_end]
            #df_test = df_test.dropna(subset=train_features)  # Nans will be droped by the algorithms themselves

            df_X_test = df_test[train_features]
            df_y_test = df_test[predict_label]

            # Train data

            # We exclude recent objects from training, because they do not have labels yet - the labels are in future
            # In real (stream) data, we will have null labels for recent objects. During simulation, labels are available and hence we need to ignore/exclude them manually
            train_end = predict_start - labels_horizon - 1
            train_start = train_end - train_length
            train_start = 0 if train_start < 0 else train_start

            df_train = df_all.iloc[int(train_start):int(train_end)]
            df_train = df_train.dropna(subset=train_features)

            df_X = df_train[train_features]
            df_y = df_train[predict_label]

            print(f"Train range: [{train_start}, {train_end}]={train_end-train_start}. Prediction range: [{predict_start}, {predict_end}]={predict_end-predict_start}. ")

            # ---
            if algorithm == "gb":
                y_test_hat = train_predict_gb(df_X, df_y, df_X_test, params)
            elif algorithm == "nn":
                y_test_hat = train_predict_nn(df_X, df_y, df_X_test, params)
            elif algorithm == "lc":
                y_test_hat = train_predict_lc(df_X, df_y, df_X_test, params)
            # ---

            # Append true and predicted array
            y_true = y_true.append(df_y_test)
            y_predicted = y_predicted.append(y_test_hat)

            print(f"End step {step}/{steps}. ")

        print("")
        print("Finished {} steps of train with {} true and {} predicted results.".format(steps, len(y_true), len(y_predicted)))

        # y_true and y_predicted might have nans which can confuse some scoring functions
        df_scores = pd.DataFrame({"y_true": y_true, "y_predicted": y_predicted})
        num_scores = len(df_scores)
        df_scores = df_scores.dropna()
        print(f"Total number of collected predictions: {num_scores}. After dropping NaNs: {len(df_scores)}")
        print(f"Number of non-NaN predictions used for scoring: {len(df_scores)}")
        num_scores = len(df_scores)
        y_true = df_scores["y_true"]
        y_predicted = df_scores["y_predicted"]

        score = compute_scores(y_true, y_predicted)

        metrics.append(score)

    #
    # Process all collected results and save
    #
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
        rec = metrics[i]
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
    main()
