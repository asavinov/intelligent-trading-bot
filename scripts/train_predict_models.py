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
Use input feature matrix to train prediction models for its labels.
The output is a number of prediction models - one for each label and predictoin algorithm (like gb or svm)
The script will train models by using only one specified data range. For other (shorter or longer) data ranges, several runs with other parameters are needed.

Parameters:
- the data range always ends with last loaded (non-null) record
- !!! the size of the training data is specified by the "train_max_length" parameter - it is used to find the start record
  If specified, the value from env will overwrite this parameter
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
    in_path_name = r"_TEMP_FEATURES"
    in_file_name = r"_BTCUSDT-1m-data-features.csv"
    in_nrows = 10_000_000  # <-- PARAMETER

    out_path_name = r"_TEMP_MODELS"
    out_file_name = r""

    features_horizon = 300  # Features are generated using this past window length
    labels_horizon = 60  # Labels are generated using this number of steps ahead

    train_max_length = 10_000_000  # <-- PARAMETER: 1 year: 10_000_000, 525_600, 6 months: 262_800, 3 months: 131_400, 3 months: 43_800

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
    elif P.in_file_name.endswith(".parq"):
        in_df = pd.read_parquet(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    #
    # Algorithm parameters
    #
    max_depth = os.getenv("max_depth", 1)  # 10 (long, production)
    learning_rate = os.getenv("learning_rate", 0.1)
    num_boost_round = os.getenv("num_boost_round", 1_000)  # 20_000
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

    if os.getenv("train_max_length"):
        P.train_max_length = int(os.getenv("train_max_length"))

    #
    # Prepare train data with past values
    #
    print(f"Prepare training data set. Loaded size: {len(in_df)}...")

    full_length = len(in_df)
    train_df = in_df.dropna()
    new_length = len(train_df)
    print(f"Number of NaN records dropped from the loaded data: {full_length - new_length}. New length: {new_length}")

    pd.set_option('use_inf_as_na', True)
    if new_length > P.train_max_length:
        train_df = train_df.tail(P.train_max_length)
        new_length = len(train_df)
        print(f"Select last {P.train_max_length} rows. New length: {new_length}")

    #
    # Train gb models for all labels
    #
    X = train_df[P.features_gb].values
    models_gb = {}
    accuracies_gb = {}
    for label in P.labels:
        print(f"Train gb model for label '{label}' {len(train_df)} records and {len(P.features_gb)} features...")
        y = train_df[label].values
        y = y.reshape(-1)
        model = train_model_gb_classifier(X, y, params=params_gb)
        models_gb[label] = model

        # Accuracy for training set
        y_hat = model.predict(X)
        auc = metrics.roc_auc_score(y, y_hat)  # maybe y.astype(int)
        accuracies_gb[label] = auc
        print(f"Model training finished. AUC: {auc:.2f}")

        model_file = out_path.joinpath("gb_"+label).with_suffix('.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print(f"Model stored in file: {model_file}")

    #
    # Train knn models for all labels
    #
    #X = train_df[P.features_knn].values
    #models_knn = {}
    #accuracies_knn = {}
    #for label in P.labels:
    #    print(f"Train knn model for label '{label}' {len(train_df)} records and {len(P.features_gb)} features...")
    #    y = train_df[label].values
    #    y = y.reshape(-1)
    #    model = train_model_knn_classifier(X, y, params=params_knn)
    #    models_knn[label] = model

    #
    # End
    #
    elapsed = datetime.now() - start_dt

    print(f"Finished feature prediction in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main(sys.argv[1:])
