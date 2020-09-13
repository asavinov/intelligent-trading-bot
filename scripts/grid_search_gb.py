import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import lightgbm as lgbm

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from trade.utils import *

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
    ]
features_kline_small = [
    'close_1','close_5','close_15','close_60','close_180','close_720',
    'close_std_5','close_std_15','close_std_60','close_std_180','close_std_720',  # Removed "std_1" which is constant
    'volume_1','volume_5','volume_15','volume_60','volume_180','volume_720',
    ]

labels = [
    'high_10', 'high_15', 'high_20',
    'low_10', 'low_15', 'low_20',
]

params_grid = {  # First parameter is the slowest
    # binary (logloss - logistic regression) cross_entropy cross_entropy_lambda
    "objective": ["cross_entropy"],  # "cross_entropy", "cross_entropy_lambda", "binary"
    "max_depth": [1, 2, 3],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_boost_round": [500, 1_000, 1_500],
    # BEST:
    # auc: depth=2: 0.1-500, 0.05-1000, 0.01-1000, 0.005-2000
    # f1: depth 4,5: lr=0.1, rounds: 1000-2000
    # precision: any depth: 0.001 - 500-1500 (any)

    "lambda_l1": [0.1, 0.5, 1.0], # (reg_alpha) 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100] 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05], 1.5
    "lambda_l2": [0.1, 0.5, 1.0],  # (reg_lambda), [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
}

#
# Parameters of rolling predict
#
nrows = 10_000_000  # For debug
# Columns
train_features = features_kline  # Assuming that same features for the whole grid
predict_label = "low_10"
# Rows
prediction_start_str = "2020-02-01 00:00:00"  # Use it when rolling prediction will work
#prediction_start_str = "2020-06-01 00:00:00"
train_length = int(1.0 * 525_600)  # 525_600
stride = 6*7*1440  # Length of one rolling prediction step: 4 weeks, 1 month = 43800
steps = 5  # How many rolling prediction steps

def train(df_X, df_y, df_X_test, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions.
    """
    is_scale = False

    #
    # Prepare data
    #
    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        X_train = df_X.values

    y_train = df_y.values

    #
    # Create model
    #

    objective = params.get("objective")

    max_depth = params.get("max_depth")
    learning_rate = params.get("learning_rate")
    num_boost_round = params.get("num_boost_round")

    lambda_l1 = params.get("lambda_l1")
    lambda_l2 = params.get("lambda_l2")

    lgbm_params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,  # Can be -1
        #"n_estimators": 10000,

        #"min_split_gain": params['min_split_gain'],
        "min_data_in_leaf": int(0.01*len(df_X)),  # 10_000 best (~0.02 * len())
        #'subsample': 0.8,
        #'colsample_bytree': 0.8,
        'num_leaves': 32,  # or (2 * 2**max_depth)
        #"bagging_freq": 5,
        #"bagging_fraction": 0.4,
        #"feature_fraction": 0.05,

        # gamma=0.1 ???
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,

        'is_unbalance': 'true',
        # 'scale_pos_weight': scale_pos_weight,  # is_unbalance must be false

        'boosting_type': 'gbdt',  # dart (slow but best, worse than gbdt), goss, gbdt

        'objective': objective, # binary cross_entropy cross_entropy_lambda

        'metric': {'cross_entropy'},  # auc auc_mu map (mean_average_precision) cross_entropy binary_logloss cross_entropy_lambda binary_error

        'verbose': -1,
    }

    model = lgbm.train(
        lgbm_params,
        train_set=lgbm.Dataset(X_train, y_train),
        num_boost_round=num_boost_round,
        #valid_sets=[lgbm.Dataset(X_validate, y_validate)],
        #early_stopping_rounds=int(num_boost_round / 5),
        verbose_eval=100,
    )

    #
    # Test score
    #
    # TODO: If we used scaling during training, then we must scale the test data
    y_test_hat = model.predict(df_X_test.values)

    return y_test_hat


def driver():
    #
    # Load and prepare all data
    #

    # Load all data
    df_all = pd.read_csv(data_path + "\\" + data_file, parse_dates=['timestamp'], nrows=nrows)
    for label in labels:
        df_all[label] = df_all[label].astype(int)  # "category"

    # Select necessary features and label
    df_all = df_all[features_kline + labels + ["timestamp"]]

    df_all = df_all.dropna()  # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    df_all = df_all.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    prediction_start = find_index(df_all, prediction_start_str)

    del df_all["timestamp"]

    #
    # Prepare params and train by getting precision
    #
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
            y_test_hat = train(df_X, df_y, df_X_test, params)
            # ---
            y_test_hat = pd.Series(index=df_y_test.index, data=y_test_hat)

            # Append true and predicted array
            y_true = y_true.append(df_y_test)
            y_predicted = y_predicted.append(y_test_hat)

            print(".", end="")

        print("")
        print("Finished {} steps of train with {} true and {} predicted results.".format(steps, len(y_true), len(y_predicted)))

        # Computing metrics
        y_predicted_bool = np.where(y_predicted < 0.5, 0, 1)
        auc = metrics.roc_auc_score(y_true, y_predicted)

        f1 = metrics.f1_score(y_true, y_predicted_bool)
        precision = precision_score(y_true, y_predicted_bool)
        recall = recall_score(y_true, y_predicted_bool)
        score = {"auc": auc, "f1": f1, "precision": precision, "recall": recall}

        records.append(score)

    #
    # Process all collected results and save
    #
    lines = []
    for i, params in enumerate(params_list):
        line = [
            predict_label,
            params.get("objective"),
            params.get("max_depth"),
            params.get("learning_rate"),
            params.get("num_boost_round"),
            params.get("lambda_l1"),
            params.get("lambda_l2"),
            "{:.3f}".format(records[i]["auc"]),
            "{:.3f}".format(records[i]["f1"]),
            "{:.3f}".format(records[i]["precision"]),
            "{:.3f}".format(records[i]["recall"]),
        ]
        lines.append(", ".join([str(x) for x in line]))

    with open('metrics.txt', 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    driver()
    pass
