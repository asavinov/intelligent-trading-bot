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

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *
from keras.regularizers import *

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from trade.utils import *

"""
Find good hyper-parameters of the neural network.

We want to train with different network parameters like number of hidden layers and their size, learning rate etc.

We also want to train on different input data (to exclude situation where we get good or bad results on thsi specific subset.
Therefore, for each set of hyper-parameters, we train several times with different input data.
The test data set always follows the input data set using some future horizon.
The final quality of the hyper-parameters is computed as average for different datasets.
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

n = len(features_kline)
params_grid = {
    # Hyper-parameters of the algorithm
    "layers": [[20, 10]],  # , [15], [20], [25], [30], [35], [40], [45], [50]

    # Which targets to predict
    "label": labels[:1],

    # Rolling predict
    "ends": [4],  # How many steps

    # Constant values
    "df": [None],
    "df_validate": [None],
    "features": [features_kline],
    "bs": [64],
}

# Algorithm training parameters
n_epochs = 5
learning_rate = 0.001

# Parameters of rolling predict
train_length = 525_600  # 525_600
stride = 2 * 7 * 1_440  # 4 weeks, 1 month = 43800

def train(params: dict):
    """
    Train model with the specified hyper-parameters and return its best metric.
    """
    is_scale = True

    #
    # Prepare data
    #
    df_train = params.get("df")
    X_train = df_train[params.get("features")]
    y_train = df_train[params.get("label")].values

    if is_scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

    df_validate = params.get("df_validate")
    X_validate = df_validate[params.get("features")]
    y_validate = df_validate[params.get("label")].values

    if is_scale:
        X_validate = scaler.transform(X_validate)

    #
    # Create model
    #
    n_features = len(params.get("features"))
    layers = params.get("layers")  # List of ints
    optimizer = Adam(lr=learning_rate)
    model = Sequential()
    model.add(Dense(layers[0], activation='sigmoid', kernel_regularizer=l2(0.001), input_dim=n_features))
    model.add(Dense(layers[1], activation='sigmoid', kernel_regularizer=l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), 'accuracy'],
    )

    #
    # Train
    #
    model.fit(
        X_train,
        y_train,
        batch_size=params.get("bs"),
        epochs=n_epochs,
        validation_data=(X_validate, y_validate),
        class_weight={0: 1, 1: 20}
    )

    #
    # Test score
    #
    #y_pred = model.predict_classes(X_test)
    scores = model.evaluate(X_validate, y_validate)
    for i in range(len(scores)):
        print("{}={:.2f}".format(model.metrics_names[i], scores[i]))

    #return learn.final_record

    return scores


def driver():
    #
    # Load and prepare all data
    #

    # Load all data
    df_all = pd.read_csv(data_path + "\\" + data_file, parse_dates=['timestamp'], nrows=100_000_000)
    for label in labels:
        df_all[label] = df_all[label].astype(int)  # "category"

    # Select necessary features and label
    df_all = df_all[features_kline + labels + ["timestamp"]]

    df_all = df_all.dropna()  # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    df_all = df_all.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    prediction_start_str = "2020-02-01 00:00:00"
    prediction_start = find_index(df_all, prediction_start_str)

    del df_all["timestamp"]

    #
    # Prepare params and train by getting precision
    #

    records = []
    grid = ParameterGrid(params_grid)
    params_list = list(grid)  # List of hyper-param dicts
    for i, params in enumerate(params_list):

        # Given end value, extract the corresponding dataset number (range)
        end = prediction_start + (stride * params.get("ends"))
        start = end - train_length

        df = df_all.iloc[start:end]
        params["df"] = df

        df_validate = df_all.iloc[end:end+stride]
        params["df_validate"] = df_validate

        print("{}/{} train".format(i, len(params_list)))
        # ---
        record = train(params)
        # ---
        records.append(record)

    #
    # Process all collected results and save
    #
    lines = []
    for i, params in enumerate(params_list):
        line = [
            params.get("label"),
            params.get("layers"),
            params.get("ends"),
            "{:.3f}".format(records[i][1]),
            "{:.3f}".format(records[i][2]),
            "{:.3f}".format(records[i][3]),
        ]
        lines.append(", ".join([str(x) for x in line]))

    with open('metrics.txt', 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    driver()
    pass
