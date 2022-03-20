import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle
import click

import numpy as np
import pandas as pd

from service.App import *
from common.utils import *
from common.classifiers import *
from common.feature_generation import *

"""
Use input feature matrix to train *one* label predict model for each label using all specified historic data.
The output is a number of prediction models - one for each label and prediction algorithm (like gb or svm)
The script will train models by using only one specified data range. For other (shorter or longer) data ranges, 
several runs with other parameters are needed. Accuracy (for the train data set) will be reported.
No grid search or hyper-parameter optimization is done - use generate_rolling_predictions for that purpose 
(from some grid search driver). This script is used to train models which will be used in the service.

Parameters:
- the data range always ends with last loaded (non-null) record
- the algorithm will select and use for training features listed in config
- models (for each algorithm) will be trained separately for the labels specified in the list "labels"
- The hyper-parameters are specified in this script. They are supposed to be the best hyper-parameters
obtained elsewhere (e.g., using grid search).   
"""


#
# Parameters
#
class P:
    feature_sets = ["kline"]  # futur

    in_nrows = 10_000_000  # For debugging
    in_nrows_tail = None  # How many last rows to select (for testing)

    # How much data we want to use for training
    kline_train_length = int(1.5 * 525_600)  # 1.5 * 525_600
    futur_train_length = int(4 * 43_800)

    # Whether to store file with predictions
    store_predictions = False

#
# (Best) train parameters (found by and copied from grid search scripts)
#

params_gb = {
    "objective": "cross_entropy",
    "max_depth": 1,
    "learning_rate": 0.01,
    "num_boost_round": 1_500,

    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
}

params_nn = {
    "layers": [29],  # It is equal to the number of input features (different for spot and futur)
    "learning_rate": 0.001,
    "n_epochs": 60,
    "bs": 64,
}

params_lc = {
    "is_scale": False,
    "penalty": "l2",
    "C": 1.0,
    "class_weight": None,
    "solver": "liblinear",
    "max_iter": 200,
}


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    label_horizon = App.config["label_horizon"]  # Labels are generated using this number of steps ahead
    labels = App.config["labels"]

    #features_horizon = 720  # Features are generated using this past window length
    features_kline = App.config["features_kline"]
    features_futur = App.config["features_futur"]
    features_depth = App.config["features_depth"]

    freq = "1m"
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"])
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return
    out_path = Path(App.config["model_folder"])
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

    #
    # Load feature matrix
    #
    print(f"Loading feature matrix from input file...")
    start_dt = datetime.now()

    in_file_name = f"{symbol}-{freq}-features.csv"
    in_path = data_path / in_file_name
    if not in_path.exists():
        print(f"ERROR: Input file does not exist: {in_path}")
        return

    in_df = None
    if in_file_name.endswith(".csv"):
        in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)
    elif in_file_name.endswith(".parquet"):
        in_df = pd.read_parquet(in_path)
    elif in_file_name.endswith(".pickle"):
        in_df = pd.read_pickle(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    print(f"Feature matrix loaded. Length: {len(in_df)}. Width: {len(in_df.columns)}")

    if P.in_nrows_tail:
        in_df = in_df.tail(P.in_nrows_tail)

    for label in labels:
        in_df[label] = in_df[label].astype(int)  # "category" NN does not work without this

    # Select necessary features and label
    out_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    all_features = []
    if "kline" in P.feature_sets:
        all_features += features_kline
    if "futur" in P.feature_sets:
        all_features += features_futur
    all_features += labels
    in_df = in_df[all_features + out_columns]

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna(subset=labels)
    in_df = in_df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    # Remove the tail data for which no labels are available (since their labels are computed from future which is not available)
    in_df = in_df.head(-label_horizon)

    models = dict()
    scores = dict()

    df_out = pd.DataFrame()  # Collect predictions

    # ===
    # kline feature set
    # ===
    if "kline" in P.feature_sets:
        features = features_kline
        name_tag = "_k_"

        print(f"Start training 'kline' package with {len(features)} features, name tag {name_tag}', and train length {P.kline_train_length}")

        train_df = in_df.tail(P.kline_train_length)
        train_df = train_df.dropna(subset=features)

        df_X = train_df[features]

        for label in labels:  # Train-predict different labels (and algorithms) using same X
            df_y = train_df[label]

            # --- GB
            score_column_name = label + name_tag + "gb"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_gb(df_X, df_y, params=params_gb)
            models[score_column_name] = model_pair
            df_y_hat = predict_gb(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            df_out[score_column_name] = df_y_hat

            # --- NN
            score_column_name = label + name_tag + "nn"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_nn(df_X, df_y, params=params_nn)
            models[score_column_name] = model_pair
            df_y_hat = predict_nn(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            df_out[score_column_name] = df_y_hat

            # --- LC
            score_column_name = label + name_tag + "lc"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_lc(df_X, df_y, params=params_lc)
            models[score_column_name] = model_pair
            df_y_hat = predict_lc(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            df_out[score_column_name] = df_y_hat

    # ===
    # futur feature set
    # ===
    if "futur" in P.feature_sets:
        features = features_futur
        name_tag = "_f_"

        print(f"Start training 'futur' package with {len(features)} features, name tag {name_tag}', and train length {P.futur_train_length}")

        train_df = in_df.tail(P.futur_train_length)
        train_df = train_df.dropna(subset=features)

        df_X = train_df[features]

        for label in labels:  # Train-predict different labels (and algorithms) using same X
            df_y = train_df[label]

            # --- GB
            score_column_name = label + name_tag + "gb"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_gb(df_X, df_y, params=params_gb)
            models[score_column_name] = model_pair
            df_y_hat = predict_gb(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            df_out[score_column_name] = df_y_hat

            # --- NN
            score_column_name = label + name_tag + "nn"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_nn(df_X, df_y, params=params_nn)
            models[score_column_name] = model_pair
            df_y_hat = predict_nn(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            df_out[score_column_name] = df_y_hat

            # --- LC
            score_column_name = label + name_tag + "lc"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_lc(df_X, df_y, params=params_lc)
            models[score_column_name] = model_pair
            df_y_hat = predict_lc(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            df_out[score_column_name] = df_y_hat

    #
    # Store all collected models in files
    #
    for score_column_name, model_pair in models.items():
        save_model_pair(out_path, score_column_name, model_pair)

    print(f"Models stored in path: {out_path.absolute()}")

    #
    # Store scores
    #
    lines = list()
    for score_column_name, score in scores.items():
        line = score_column_name + ", " + str(score)
        lines.append(line)

    metrics_file_name = out_path.joinpath("metrics").with_suffix(".txt")
    with open(metrics_file_name, 'a+') as f:
        f.write("\n".join(lines) + "\n")

    print(f"Metrics stored in path: {metrics_file_name.absolute()}")

    #
    # Store predictions if necessary
    #
    if P.store_predictions:
        out_file_name = f"{symbol}-{freq}-predictions.csv"
        out_path = data_path / out_file_name

        # We do not store features. Only selected original data, labels, and their predictions
        df_out = df_out.join(in_df[out_columns + labels])

        df_out.to_csv(out_path, index=False)
        print(f"Predictions stored in file: {out_path}. Length: {len(df_out)}. Columns: {len(df_out.columns)}")

    #
    # End
    #
    elapsed = datetime.now() - start_dt
    print(f"Finished in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main()
