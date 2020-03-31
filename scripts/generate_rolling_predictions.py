import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

import lightgbm as lgbm

from trade.utils import *
from trade.feature_generation import *
from trade.feature_prediction import *

"""
Generate label predictions for the whole input feature matrix by iteratively training models using historic data and predicting labels for some future horizon.
The main parameter is the step of iteration, that is, the future horizon for prediction.
As usual, we can specify past history length used to train a model.
The output file will store predicted labels in addition to all input columns (generated features and true labels).
This file is intended for training signal models (by simulating trade process and computing overall performance for some long period).
The output predicted labels will cover shorter period of time because we need some relatively long history to train the very first model.
"""

# TODO: Grid search for best hyper-parameters. Either here as a function, or as a driver script calling rolling predictions.
#   We need to write a driver which will execute rolling predictions for a grid of parameters by storing the resulting performance
grid_predictions = [
    {
        'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'num_boost_round': [500],
        'n_neighbors': [5, 10, 20], 'weights': ['uniform', 'distance'],
    },
    # Debug
    #{
    #    'max_depth': [3], 'learning_rate': [0.05], 'num_boost_round': [500],
    #    'n_neighbors': [1000, 2000], 'weights': ['uniform', 'distance'],
    #},
]


#
# Parameters
#
class P:
    in_path_name = r"_TEMP_FEATURES"
    in_file_name = r"_BTCUSDT-1m-features.csv"

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"_BTCUSDT-1m-rolling-predictions"

    features_horizon = 300  # Features are generated using this past window length
    labels_horizon = 60  # Labels are generated using this number of steps ahead
    label_histories = {"12": 525_600, "06": 262_800, "03": 131_400, "01": 43_920}
    #label_histories = {"03": 131_400, "01": 43_800}

    labels = ['high_60_10', 'high_60_15', 'high_60_20']  # Target columns with true values

    # Offsets
    # 2017-08-18 00:00:00 = 1200
    # 2017-08-21 00:00:00 = 5520
    # 2017-10-02 00:00:00 = 65580
    # 2017-12-04 06:00:20.799 = 156663 - start of anomaly with 0.0 in maker/taker and time alignment
    # 2017-12-04 06:47:20.799 = 156710 - end of anomaly (last record) with zeros
    # 2017-12-18 10:00:20.799 = 177063 - end of anomaly (last record) with time alignment
    # 2018-01-01 00:00:00 = 196_546
    # 2019-01-01 00:00:00 = 718_170
    # 2019-05-01 00:00:00 = 890_610

    # ---
    # Debug:
    #in_nrows = 500_000
    #prediction_start_str = "2018-02-01 00:00:00"  # First row for predictions
    #prediction_length = 60  # 1 hour: 60, 1 day: 1_440 = 60 * 24, one week: 10_080
    #prediction_count = 2  # How many prediction steps. If None or 0, then from prediction start till the data end

    # ---
    # Production:
    in_nrows = 10_000_000
    prediction_start_str = "2019-01-01 00:00:00"  # First row for starting predictions
    prediction_length = 1_440  # 1 day: 1_440 = 60 * 24, one week: 10_080
    prediction_count = None  # How many prediction steps. If None or 0, then from prediction start till the data end

    class_labels_all = [
        'high_60_10', 'high_60_15', 'high_60_20', 'high_60_25',
        'high_60_01', 'high_60_02', 'high_60_03', 'high_60_04',
        'low_60_01', 'low_60_02', 'low_60_03', 'low_60_04',
        'low_60_10', 'low_60_15', 'low_60_20', 'low_60_25',
        ]

    features_0 = [
        'close_1','close_2','close_5','close_20','close_60','close_180',
        'close_std_1','close_std_2','close_std_5','close_std_20','close_std_60','close_std_180',
        'volume_1','volume_2','volume_5','volume_20','volume_60','volume_180',
        ]
    features_1 = [
        'close_1','close_2','close_5','close_20','close_60','close_180',
        'close_std_1','close_std_2','close_std_5','close_std_20','close_std_60','close_std_180',
        'volume_1','volume_2','volume_5','volume_20','volume_60','volume_180',
        'trades_1','trades_2','trades_5','trades_20','trades_60','trades_180',
        'tb_base_1','tb_base_2','tb_base_5','tb_base_20','tb_base_60','tb_base_180',
        'tb_quote_1','tb_quote_2','tb_quote_5','tb_quote_20','tb_quote_60','tb_quote_180',
        ]

    features_gb = features_1
    features_knn = features_0

def main(args=None):
    pd.set_option('use_inf_as_na', True)
    in_df = None

    start_dt = datetime.now()

    #
    # Load feature matrix
    #
    print(f"Loading feature matrix from input file...")

    in_path = Path(P.in_path_name).joinpath(P.in_file_name)
    if not in_path.exists():
        print(f"ERROR: Input file does not exist: {in_path}")
        return

    if P.in_file_name.endswith(".csv"):
        in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)
    elif P.in_file_name.endswith(".parq"):
        in_df = pd.read_parquet(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    pd.set_option('use_inf_as_na', True)
    in_df = in_df.dropna()

    #
    # Algorithm parameters
    #
    max_depth = os.getenv("max_depth", None)
    learning_rate = os.getenv("learning_rate", None)
    num_boost_round = os.getenv("num_boost_round", None)
    params_gb = {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_boost_round": num_boost_round,
    }

    #
    # Rolling train-predict loop
    #
    print(f"Starting train-predict loop with {P.prediction_count} prediction steps. Each step with {P.prediction_length} horizon...")

    prediction_start = find_index(in_df, P.prediction_start_str)
    if P.prediction_count is None or P.prediction_count == 0:
        # Use all available rest data (from the prediction start to the dataset end)
        P.prediction_count = (len(in_df) - prediction_start) // P.prediction_length

    # Result rows. Here store only row for which we make predictions
    labels_hat_df = pd.DataFrame()

    for i in range(0, P.prediction_count):

        print(f"---> Iteration {i} / {P.prediction_count}")

        # We cannot use recent data becuase its labels use information form the predicted segment(add 1 to ensure that there is no future leak in the model)
        train_end = prediction_start - P.labels_horizon - 1
        # Train start will be set depending on the parameter in the loop

        models_gb = {}
        for history_name, history_length in P.label_histories.items():  # Example: {"12": 525_600, "06": 262_800, "03": 131_400}
            train_start = train_end - history_length
            train_start = 0 if train_start < 0 else train_start

            #
            # Prepare train data of the necessary history length
            #
            train_df = in_df.iloc[train_start:train_end]
            train_length = len(train_df)

            #
            # Train gb models for all labels
            #
            X = train_df[P.features_gb].values
            for label in P.labels:
                print(f"Train gb model: label '{label}', history {history_name}, rows {len(train_df)}, features {len(P.features_gb)}...")
                y = train_df[label].values
                y = y.reshape(-1)
                model = train_model_gb_classifier(X, y, params=params_gb)
                models_gb[label+"_gb_"+history_name] = model

        #
        # Use just trained models to predict future values within the horizon
        #
        prediction_end = prediction_start + P.prediction_length
        predict_df = in_df.iloc[prediction_start:prediction_end]

        predict_labels_df = pd.DataFrame(index=predict_df.index)

        # Predict labels using gb models
        X = predict_df[P.features_gb].values
        for label, model in models_gb.items():
            y_hat = model.predict(X)
            predict_labels_df[label] = y_hat

        # Predictions for all labels and histories (and algorithms) have been generated for the iteration
        # Append predicted rows to the end of previous predicted rows
        labels_hat_df = labels_hat_df.append(predict_labels_df)

        #
        # Iterate
        #
        prediction_start += P.prediction_length

    #
    # Prepare output
    #

    # Append all features including true labels to the predicted labels
    out_df = labels_hat_df.join(in_df)

    #
    # Compute accuracy for the whole data set (all segments)
    #

    # For gb
    aucs_gb = {}
    for history_name, history_length in P.label_histories.items():
        for label in P.labels:
            try:
                auc = metrics.roc_auc_score(out_df[label].astype(int), out_df[label+"_gb_"+history_name])
            except ValueError:
                label_auc = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging)
            aucs_gb[label+"_gb_"+history_name] = f"{auc:.2f}"

    auc_gb_mean = np.mean([float(x) for x in aucs_gb.values()])
    out_str = f"Mean AUC {auc_gb_mean:.2f}: {aucs_gb}"
    print(out_str)

    #
    # Store hyper-parameters and scores
    #
    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
    out_path = out_path.joinpath(P.out_file_name)

    out_str = f"{max_depth}, {learning_rate}, {num_boost_round}, " + out_str
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(out_str + "\n")

    #
    # Store data
    #
    print(f"Storing output file...")

    out_df.to_csv(out_path.with_suffix('.csv'), index=False, float_format="%.4f")

    #out_df.to_parquet(out_path.with_suffix('.parq'), engine='auto', compression=None, index=None, partition_cols=None)

    elapsed = datetime.now() - start_dt

    print(f"Finished feature prediction in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main(sys.argv[1:])
