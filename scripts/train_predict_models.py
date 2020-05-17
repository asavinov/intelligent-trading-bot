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
    in_file_name = r"_BTCUSDT-1m-features-merged.csv"
    in_nrows = 10_000_000  # <-- PARAMETER

    out_path_name = r"_TEMP_MODELS"
    out_file_name = r""

    labels = [  # Target columns with true values which will be predicted
        'high_60_10', 'high_60_15', 'high_60_20',
        'low_60_10', 'low_60_15', 'low_60_20',
    ]
    class_labels_all = [  # All existing target labels generated from feature generation procedure
        'high_60_10', 'high_60_15', 'high_60_20', 'high_60_25',  # At least one time above
        'high_60_01', 'high_60_02', 'high_60_03', 'high_60_04',  # Always below
        'low_60_01', 'low_60_02', 'low_60_03', 'low_60_04',  # Always above
        'low_60_10', 'low_60_15', 'low_60_20', 'low_60_25',  # At least one time below
    ]

    in_features_klines = [
        "timestamp",
        "open","high","low","close","volume",
        "close_time",
        "quote_av","trades","tb_base_av","tb_quote_av","ignore"
    ]

    features_klines_small = [
        'close_1','close_2','close_5','close_20','close_60','close_180',
        'close_std_1','close_std_2','close_std_5','close_std_20','close_std_60','close_std_180',
        'volume_1','volume_2','volume_5','volume_20','volume_60','volume_180',
    ]
    features_klines = [
        'close_1','close_2','close_5','close_20','close_60','close_180',
        'close_std_1','close_std_2','close_std_5','close_std_20','close_std_60','close_std_180',
        'volume_1','volume_2','volume_5','volume_20','volume_60','volume_180',
        'trades_1','trades_2','trades_5','trades_20','trades_60','trades_180',
        'tb_base_1','tb_base_2','tb_base_5','tb_base_20','tb_base_60','tb_base_180',
        'tb_quote_1','tb_quote_2','tb_quote_5','tb_quote_20','tb_quote_60','tb_quote_180',
    ]

    features_futur = [
        "f_close_1", "f_close_2", "f_close_5", "f_close_10", "f_close_30", "f_close_60",
        "f_close_std_1", "f_close_std_2", "f_close_std_5", "f_close_std_10", "f_close_std_30", "f_close_std_60",
        "f_volume_1", "f_volume_2", "f_volume_5", "f_volume_10", "f_volume_30", "f_volume_60",
        "f_span_1", "f_span_2", "f_span_5", "f_span_10", "f_span_30", "f_span_60",
        "f_trades_1", "f_trades_2", "f_trades_5", "f_trades_10", "f_trades_30", "f_trades_60",
    ]

    features_depth = [
        "gap_2","gap_5","gap_10",
        "bids_1_2","bids_1_5","bids_1_10", "asks_1_2","asks_1_5","asks_1_10",
        "bids_2_2","bids_2_5","bids_2_10", "asks_2_2","asks_2_5","asks_2_10",
        "bids_5_2","bids_5_5","bids_5_10", "asks_5_2","asks_5_5","asks_5_10",
        "bids_10_2","bids_10_5","bids_10_10", "asks_10_2","asks_10_5","asks_10_10",
        "bids_20_2","bids_20_5","bids_20_10", "asks_20_2","asks_20_5","asks_20_10",
    ]

    #
    # Selector: here we choose what input features to use, what algorithm to use and what histories etc.
    #
    if source_type == "klines":
        features_gb = features_klines
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
    elif P.in_file_name.endswith(".parq"):
        in_df = pd.read_parquet(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

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

            model_file = out_path.joinpath(label+"_gb_"+history_name).with_suffix('.pkl')
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
