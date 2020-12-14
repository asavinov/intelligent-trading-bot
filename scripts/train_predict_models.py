import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle

import numpy as np
import pandas as pd

from trade.App import App
from common.utils import *
from common.feature_generation import *
from common.feature_prediction import *

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
    source_type = "futur"  # Selector: klines (our main approach), futur (only futur), depth (only depth), merged

    in_path_name = r"C:\DATA2\BITCOIN\GENERATED"  # File with all necessary derived features
    in_file_name = r"BTCUSDT-1m-features.csv"
    in_nrows = 10_000_000  # <-- PARAMETER
    in_nrows_tail = None  # How many last rows to select (for testing)

    out_path_name = r"_TEMP_MODELS"
    out_file_name = r""

    in_features_kline = [
        "timestamp",
        "open","high","low","close","volume",
        #"close_time",
        #"quote_av","trades","tb_base_av","tb_quote_av","ignore"
    ]

    labels = App.config["labels"]
    features_kline = App.config["features_kline"]
    features_futur = App.config["features_futur"]
    features_depth = App.config["features_depth"]

    #
    # Selector: here we choose what input features to use, what algorithm to use and what histories etc.
    #
    if source_type == "klines":
        features_gb = features_kline
        label_histories = {"12": 525_600}  # Example: {"12": 525_600, "06": 262_800, "03": 131_400}
    elif source_type == "futur":
        features_gb = features_futur
        label_histories = {"03": 131_400}  # Example: {"12": 525_600, "06": 262_800, "03": 131_400}
    elif source_type == "depth":
        features_gb = features_depth
        label_histories = {"03": 131_400}  # Example: {"12": 525_600, "06": 262_800, "03": 131_400}
    elif source_type == "merged":
        print(f"NOT IMPLEMENTED")
        exit()

    features_horizon = 300  # Features are generated using this past window length
    labels_horizon = 60  # Labels are generated using this number of steps ahead


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
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    if P.in_nrows_tail:
        in_df = in_df.tail(P.in_nrows_tail)

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

    n_neighbors = int(os.getenv("n_neighbors", 0))
    weights = os.getenv("weights", None)
    params_knn = {
        "n_neighbors": n_neighbors,
        "weights": weights,
    }

    #
    # Prepare train data with past values
    #
    print(f"Prepare training data set. Loaded size: {len(in_df)}...")

    full_length = len(in_df)
    pd.set_option('use_inf_as_na', True)
    in_df = in_df.dropna()
    full_length = len(in_df)

    models_gb = {}
    accuracies_gb = {}
    for history_name, history_length in P.label_histories.items():  # Example: {"12": 525_600, "06": 262_800, "03": 131_400}

        train_df = in_df.tail(history_length)  # Latest history
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

            # Accuracy for training set
            y_hat = model.predict(X)
            try:
                auc = metrics.roc_auc_score(y, y_hat)  # maybe y.astype(int)
            except ValueError:
                auc = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging)
            accuracies_gb[label+"_"+history_name] = auc
            print(f"Model training finished. AUC: {auc:.2f}")

            model_file = out_path.joinpath(label+"_gb_"+history_name).with_suffix('.pickle')
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

    print(f"Finished in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main(sys.argv[1:])
