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
Generate predicted features by training a model and using its parameters to generate values (for some next intervals).
Note that in contrast to simple feature generation where we use fixed (constant) parameters,
the parameters of this approach change as we move in time because previous values are used to tune the model and then this model is used to generate features.
One general parameter of this approach is how frequently we update (re-train) the model used to generate a feature, that is, which data is used for that purpose.
For example, we could re-train a model for each next moment or do it less frequently. Also, we could use only some limited past window to re-train a model.

Store the results of prediction in the output file along with original features and labels.
This file can be then used for simulation and optimizing signal parameters.
Note that since we need some significant past data to train the very first model, the output will be normally shorter than the source input data (or at least only the last generated values will be meaningful).
"""

#
# Parameters
#
class P:
    in_path_name = r"_TEMP_FEATURES"
    in_file_name = r"_BTCUSDT-1m-data-features.csv"
    in_nrows = 10_000_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"_BTCUSDT-1m-data-predictions"

    features_horizon = 300  # Features are generated using this past window length
    labels_horizon = 60  # Labels are generated using this number of steps ahead

    # Debug:
    # 2017-08-18 00:00:00 = 1200
    # 2017-08-21 00:00:00 = 5520
    # 2017-10-02 00:00:00 = 65580
    #train_start_str = "2018-01-01 00:00:00"  # First row to include in train data sets
    #prediction_start_str = "2018-02-01 00:00:00"  # First row for predictions

    # Production:
    # 2018-01-01 00:00:00 = 196_546
    # 2019-01-01 00:00:00 = 718_170
    # 2019-05-01 00:00:00 = 890_610
    train_start_str = "2018-01-01 00:00:00"  # First row to include in train data sets
    prediction_start_str = "2019-01-01 00:00:00"  # First row for starting predictions

    train_step = 1_440  # 1_440 = 60 * 24 Re-train after each day, 10_080 - one week
    train_count = None  # How many prediction steps. If None or 0, then all data from prediction start to the data end

    train_max_length = 10_000_000  # 1 year: 525600, 6 months: 262800, 3 months: 131400, 3 months: 43800

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
    regression_labels_all = [
        'high_60_max', 
        'low_60_min',
        ]
    class_labels_all = [
        'high_60_10', 'high_60_15', 'high_60_20', 'high_60_25', 
        'high_60_01', 'high_60_02', 'high_60_03', 'high_60_04', 
        'low_60_01', 'low_60_02', 'low_60_03', 'low_60_04', 
        'low_60_10', 'low_60_15', 'low_60_20', 'low_60_25',
        ]

    features_gb = features_1
    features_knn = features_0

    labels = ['high_60_10', 'high_60_20']

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

    #
    # Algorithm parameters
    #
    max_depth = os.getenv("max_depth", "DEFAULT")
    learning_rate = os.getenv("learning_rate", "DEFAULT")
    num_boost_round = os.getenv("num_boost_round", "DEFAULT")
    params_gb = {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_boost_round": num_boost_round,
    }

    n_neighbors = int(os.getenv("n_neighbors", 20))
    weights = os.getenv("weights", "distance")  # ['uniform', 'distance']
    params_knn = {
        "n_neighbors": n_neighbors,
        "weights": weights,
    }

    #
    # Initialize parameters of the loop
    #

    train_start = find_index(in_df, P.train_start_str)
    prediction_start = find_index(in_df, P.prediction_start_str)

    if P.train_count is None or P.train_count == 0:
        # Use all available data from the prediction start to the dataset end
        P.train_count = (len(in_df) - prediction_start) // P.train_step

    if os.getenv("train_max_length"):
        P.train_max_length = int(os.getenv("train_max_length"))

    # Result rows. Here store only row for which we make predictions
    labels_hat_df = pd.DataFrame()  # All label columns with predicted values

    #
    # Predict loop
    #
    print(f"Starting train-predict loop with {P.train_count} predictions...")

    i_start = 0
    prediction_start += (i_start * P.train_step)

    for i in range(i_start, P.train_count):
        print(f"---> Iteration {i} / {P.train_count} with start {prediction_start}")
        #
        # Prepare train data with past values
        #
        s = train_start + P.features_horizon + 1  # Ignore some past data which is None because of not enough history
        e = prediction_start - P.labels_horizon - 1  # Ignore some recent data for which there is no future data to generate true labels (1 is to ensure that there is no future leak in the model)

        if e - s > P.train_max_length:
            s = e - P.train_max_length  # Shorten the training set by moving the start forward so that it has the allowed max length

        train_df = in_df.iloc[s:e]

        # Only knn is sensitive to nan/inf, therefore we do it only for knn features
        train_df_na_count = train_df[P.features_knn].isnull().sum().sum()
        if train_df_na_count > 0:
            full_length = len(train_df)
            train_df = train_df.dropna()
            new_length = len(train_df)
            print(f"WARNING: {train_df_na_count} NaN found on step {i}. Drop {full_length-new_length} rows.")
        #train_df_inf_has = np.isinf(train_df).any()
        #if train_df_inf_has:
        #    # train_df.replace([np.inf, -np.inf], np.nan)
        #    # pd.set_option('use_inf_as_na', True)

        #
        # Train gb models for all labels
        #
        X = train_df[P.features_gb].values
        models_gb = {}
        for label in P.labels:
            print(f"Train gb model for label '{label}' {len(train_df)} records...")
            y = train_df[label].values
            y = y.reshape(-1)
            model = train_model_gb_classifier(X, y, params=params_gb)
            models_gb[label] = model

        X = train_df[P.features_knn].values
        models_knn = {}
        for label in P.labels:
            print(f"Train knn model for label '{label}' {len(train_df)} records...")
            y = train_df[label].values
            y = y.reshape(-1)
            model = train_model_knn_classifier(X, y, params=params_knn)
            models_knn[label] = model

        #
        # Use the models to predict future values within some horizon
        #
        s = prediction_start
        e = s + P.train_step

        predict_df = in_df.iloc[s:e]

        predict_labels_df = pd.DataFrame(index=predict_df.index)

        # Predict labels using gb models
        X = predict_df[P.features_gb].values
        for label, model in models_gb.items():
            y_hat = model.predict(X)
            predict_labels_df[label+"_gb"] = y_hat

        # Predict labels using knn models
        X = predict_df[P.features_knn].values
        for label, model in models_knn.items():
            y_hat = predict_model_knn(X, model)
            predict_labels_df[label+"_knn"] = y_hat

        # Append predicted rows to the end of previous predicted rows
        labels_hat_df = labels_hat_df.append(predict_labels_df)

        #
        # Iterate
        #
        prediction_start += P.train_step

    #
    # Prepare output
    #

    # Append all features including true labels to the predicted labels
    out_df = labels_hat_df.join(in_df)

    #
    # Compute accuracy
    #

    # For gb
    aucs_gb = []
    for label in P.labels:
        label_auc = metrics.roc_auc_score(out_df[label].astype(int), out_df[label+"_gb"])
        aucs_gb.append(label_auc)

    aucs_gb_str = [f"{x:.2f}" for x in aucs_gb]
    auc_gb_mean = np.mean(aucs_gb)
    print(f"Average gb AUC: {auc_gb_mean:.2f}: {aucs_gb_str}")

    # For knn
    aucs_knn = []
    for label in P.labels:
        label_auc = metrics.roc_auc_score(out_df[label].astype(int), out_df[label+"_knn"])
        aucs_knn.append(label_auc)

    aucs_knn_str = [f"{x:.2f}" for x in aucs_knn]
    auc_knn_mean = np.mean(aucs_knn)
    print(f"Average knn AUC: {auc_knn_mean:.2f}: {aucs_knn_str}")

    #
    # Store hyper-parameters and scores
    #
    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
    out_path = out_path.joinpath(P.out_file_name)

    elapsed = datetime.now() - start_dt

    out_str = ""
    out_str += f"{P.train_max_length}, "
    out_str += f"{max_depth}, {learning_rate}, {num_boost_round}, "
    out_str += f"{n_neighbors}, {weights}, "
    out_str += f"{auc_gb_mean:.2f}, {str(aucs_gb_str)}, "
    out_str += f"{auc_knn_mean:.2f}, {str(aucs_knn_str)}, "

    header_str = \
        f"train_max_length," \
        f"max_depth,learning_rate,num_boost_round," \
        f"n_neighbors,weights," \
        f"auc_gb,aucs_gb," \
        f"auc_knn,aucs_knn"

    if out_path.with_suffix('.txt').is_file():
        add_header = False
    else:
        add_header = True

    with open(out_path.with_suffix('.txt'), "a+") as f:
        if add_header:
            f.write(header_str + "\n")
        f.write(out_str + "\n")

    #
    # Store data
    #
    print(f"Storing output file...")

    out_df.to_csv(out_path.with_suffix('.csv'), index=False, float_format="%.4f")

    #out_df.to_parquet(out_path.with_suffix('.parq'), engine='auto', compression=None, index=None, partition_cols=None)

    print(f"Finished feature prediction in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main(sys.argv[1:])
