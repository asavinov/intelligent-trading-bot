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

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *
from keras.regularizers import *
from keras.callbacks import *

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
    ]  # 41 features
features_kline_small = [
    'close_1','close_5','close_15','close_60','close_180','close_720',
    'close_std_5','close_std_15','close_std_60','close_std_180','close_std_720',  # Removed "std_1" which is constant
    'volume_1','volume_5','volume_15','volume_60','volume_180','volume_720',
    ]

features_futur = [
    "f_close_1", "f_close_2", "f_close_5", "f_close_20", "f_close_60", "f_close_180",
    "f_close_std_2", "f_close_std_5", "f_close_std_20", "f_close_std_60", "f_close_std_180",
    # Removed "std_1" which is constant
    "f_volume_1", "f_volume_2", "f_volume_5", "f_volume_20", "f_volume_60", "f_volume_180",
    "f_span_1", "f_span_2", "f_span_5", "f_span_20", "f_span_60", "f_span_180",
    "f_trades_1", "f_trades_2", "f_trades_5", "f_trades_20", "f_trades_60", "f_trades_180",
]  # 29 features

labels = [
    'high_10', 'high_15', 'high_20',
    'low_10', 'low_15', 'low_20',
]

params_grid_gb = {  # First parameter is the slowest
    # binary (logloss - logistic regression) cross_entropy cross_entropy_lambda
    "objective": ["cross_entropy"],  # "cross_entropy", "cross_entropy_lambda", "binary"
    "max_depth": [1, 2, 3],
    "learning_rate": [0.001, 0.005, 0.01],
    "num_boost_round": [500, 1_000, 1_500, 2_000],

    "lambda_l1": [1.0], # (reg_alpha) 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100] 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05], 1.5
    "lambda_l2": [1.0],  # (reg_lambda), [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
}

params_grid_nn = {  # First parameter is the slowest
    "layers": [[29]],
    "learning_rate": [0.001],
    "n_epochs": [20],
    "bs": [64],
}

#
# Parameters of data and features
#

params_grid = params_grid_nn

#
# Parameters of rolling predict
#
nrows = 10_000_000  # For debug
# Columns
train_features = features_futur
predict_label = "low_20"
# Rows
prediction_start_str = "2020-02-01 00:00:00"  # Use it when rolling prediction will work
#prediction_start_str = "2020-06-01 00:00:00"
train_length = int(4.0 * 43_800)  # 525_600 * 1.5 for long, 43_800 * 4 for short (futur)
stride = 6*7*1440  # Length of one rolling prediction step: 4 weeks, 1 month = 43800
steps = 5  # 5 How many rolling prediction steps


#
# NN
#


def train_predict_nn(df_X, df_y, df_X_test, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    models = train_nn(df_X, df_y, params)
    y_test_hat = predict_nn(models, df_X_test)
    return y_test_hat

def train_nn(df_X, df_y, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    is_scale = True

    #
    # Prepare data
    #
    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    #
    # Create model
    #
    n_features = X_train.shape[1]
    layers = params.get("layers")  # List of ints
    learning_rate = params.get("learning_rate")
    n_epochs = params.get("n_epochs")
    batch_size = params.get("bs")

    # Topology
    model = Sequential()
    # sigmoid, relu, tanh, selu, elu, exponential
    # kernel_regularizer=l2(0.001)

    reg_l2 = 0.001

    model.add(Dense(n_features, activation='sigmoid', input_dim=n_features, kernel_regularizer=l2(reg_l2)))

    #model.add(Dense(layers[0], activation='sigmoid', input_dim=n_features, kernel_regularizer=l2(reg_l2)))
    #if len(layers) > 1:
    #    model.add(Dense(layers[1], activation='sigmoid', kernel_regularizer=l2(reg_l2)))
    #if len(layers) > 2:
    #    model.add(Dense(layers[2], activation='sigmoid', kernel_regularizer=l2(reg_l2)))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    optimizer = Adam(lr=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    es = EarlyStopping(
        monitor="loss",  # val_loss loss
        min_delta=0.0001,  # Minimum change qualified as improvement
        patience=0,  # Number of epochs with no improvements
        verbose=0,
        mode='auto',
    )

    #
    # Train
    #
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        #validation_data=(X_validate, y_validate),
        #class_weight={0: 1, 1: 20},
        callbacks=[es],
    )

    return (model, scaler)

def predict_nn(models: tuple, df_X_test):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    scaler = models[1]
    is_scale = scaler is not None

    if is_scale:
        df_X_test = scaler.transform(df_X_test)
    else:
        df_X_test = df_X_test.values

    y_test_hat = models[0].predict(df_X_test)
    # NN model returns several columns as predictions

    return y_test_hat[:, 0]  # Or y_test_hat.flatten()


#
# GB
#


def train_predict_gb(df_X, df_y, df_X_test, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    models = train_gb(df_X, df_y, params)
    y_test_hat = predict_gb(models, df_X_test)
    return y_test_hat

def train_gb(df_X, df_y, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for new data.
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
        scaler = None
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
        "min_data_in_leaf": int(0.01*len(df_X)),  # Best: ~0.02 * len() - 2% of size
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

        'verbose': 0,
    }

    model = lgbm.train(
        lgbm_params,
        train_set=lgbm.Dataset(X_train, y_train),
        num_boost_round=num_boost_round,
        #valid_sets=[lgbm.Dataset(X_validate, y_validate)],
        #early_stopping_rounds=int(num_boost_round / 5),
        verbose_eval=100,
    )

    return (model, scaler)

def predict_gb(models: tuple, df_X_test):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    scaler = models[1]
    is_scale = scaler is not None

    if is_scale:
        df_X_test = scaler.transform(df_X_test)
    else:
        df_X_test = df_X_test.values

    y_test_hat = models[0].predict(df_X_test)

    return y_test_hat

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
            #y_test_hat = train_gb(df_X, df_y, df_X_test, params)
            y_test_hat = train_predict_nn(df_X, df_y, df_X_test, params)
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

    lines = []
    for i, params in enumerate(params_list):
        #line = get_line_gb(params, records[i])
        line = get_line_nn(params, records[i])
        lines.append(", ".join([str(x) for x in line]))

    with open('metrics.txt', 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    driver()
    pass
