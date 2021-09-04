import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle

import numpy as np
import pandas as pd

from service.App import App
from common.utils import *
from common.classifiers import *
from common.feature_generation import *

"""
Use input feature matrix to train *one* label predict model for each label using all specified historic data.
The output is a number of prediction models - one for each label and prediction algorithm (like gb or svm)
The script will train models by using only one specified data range. For other (shorter or longer) data ranges, several runs with other parameters are needed.
Accuracy (for the train data set) will be reported.
No grid search or hyper-parameter optimization is done - use generate_rolling_predictions for that purpose (from some grid search driver).

Parameters:
- the data range always ends with last loaded (non-null) record
- the algorithm will select and use for training features listed in "features_gb" (for GB) and "features_knn" (for KNN)
  (same features must be used later for prediction so ensure that these lists are identical)
- models (for each algorithm) will be trained separately for the labels specified in the list "labels"
  (the file model files will include these labels in the name so ensure this name convention in the predictions)
- !!! train hyper-parameters will be read from env and passed to the training algorithm.
  If no env parameters are set, then the hard-coded defaults will be used - CHECK defaults because there are debug and production values    
"""

#
# Parameters
#
class P:
    feature_sets = ["kline"]  # futur

    labels = App.config["labels"]
    features_kline = App.config["features_kline"]
    features_futur = App.config["features_futur"]
    features_depth = App.config["features_depth"]

    #features_horizon = 720  # Features are generated using this past window length
    labels_horizon = 180  # Labels are generated using this number of steps ahead

    in_path_name = r"C:\DATA2\BITCOIN\GENERATED"  # File with all necessary derived features
    in_file_name = r"BTCUSDT-1m-features.csv"
    in_nrows = 10_000_000  # <-- PARAMETER
    in_nrows_tail = None  # How many last rows to select (for testing)

    out_path_name = r"_TEMP_MODELS"
    out_file_name = r""

#
# (Best) algorithm parameters (found by and copied from grid search scripts)
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


def main(args=None):
    in_df = None

    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

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
    elif P.in_file_name.endswith(".parquet"):
        in_df = pd.read_parquet(in_path)
    elif P.in_file_name.endswith(".pickle"):
        in_df = pd.read_pickle(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    print(f"Feature matrix loaded. Length: {len(in_df)}. Width: {len(in_df.columns)}")

    if P.in_nrows_tail:
        in_df = in_df.tail(P.in_nrows_tail)

    for label in P.labels:
        in_df[label] = in_df[label].astype(int)  # "category" NN does not work without this

    # Select necessary features and label
    all_features = []
    if "kline" in P.feature_sets:
        all_features += P.features_kline
    if "futur" in P.feature_sets:
        all_features += P.features_futur
    all_features += P.labels
    in_df = in_df[all_features]

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna(subset=P.labels)
    in_df = in_df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    # Remove the tail data for which no labels are available (since their labels are computed from future which is not available)
    in_df = in_df.head(-P.labels_horizon)

    models = dict()
    scores = dict()

    # ===
    # kline feature set
    # ===
    if "kline" in P.feature_sets:
        features = P.features_kline
        name_tag = "_k_"
        train_length = int(1.5 * 525_600)  # 1.5 * 525_600

        print(f"Start training 'kline' package with {len(features)} features, name tag {name_tag}', and train length {train_length}")

        train_df = in_df.tail(train_length)
        train_df = train_df.dropna(subset=features)

        df_X = train_df[features]

        for label in P.labels:  # Train-predict different labels (and algorithms) using same X
            df_y = train_df[label]

            # --- GB
            score_column_name = label + name_tag + "gb"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_gb(df_X, df_y, params=params_gb)
            models[score_column_name] = model_pair
            df_y_hat = predict_gb(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)

            # --- NN
            score_column_name = label + name_tag + "nn"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_nn(df_X, df_y, params=params_nn)
            models[score_column_name] = model_pair
            df_y_hat = predict_nn(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)

            # --- LC
            score_column_name = label + name_tag + "lc"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_lc(df_X, df_y, params=params_lc)
            models[score_column_name] = model_pair
            df_y_hat = predict_lc(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)

    # ===
    # futur feature set
    # ===
    if "futur" in P.feature_sets:
        features = P.features_futur
        name_tag = "_f_"
        train_length = int(4 * 43_800)

        print(f"Start training 'futur' package with {len(features)} features, name tag {name_tag}', and train length {train_length}")

        train_df = in_df.tail(train_length)
        train_df = train_df.dropna(subset=features)

        df_X = train_df[features]

        for label in P.labels:  # Train-predict different labels (and algorithms) using same X
            df_y = train_df[label]

            # --- GB
            score_column_name = label + name_tag + "gb"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_gb(df_X, df_y, params=params_gb)
            models[score_column_name] = model_pair
            df_y_hat = predict_gb(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)

            # --- NN
            score_column_name = label + name_tag + "nn"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_nn(df_X, df_y, params=params_nn)
            models[score_column_name] = model_pair
            df_y_hat = predict_nn(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)

            # --- LC
            score_column_name = label + name_tag + "lc"
            print(f"Train '{score_column_name}'... ")
            model_pair = train_lc(df_X, df_y, params=params_lc)
            models[score_column_name] = model_pair
            df_y_hat = predict_lc(model_pair, df_X)
            scores[score_column_name] = compute_scores(df_y, df_y_hat)

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
    # Store accuracies
    #

    #
    # End
    #
    elapsed = datetime.now() - start_dt

    print(f"Finished in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main(sys.argv[1:])
