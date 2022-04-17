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
from common.model_store import *

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
    in_nrows = 100_000_000  # For debugging
    in_nrows_tail = None  # How many last rows to select (for testing)

    # How much data we want to use for training
    kline_train_length = int(2.0 * 525_600)  # 1.5 * 525_600
    futur_train_length = int(4 * 43_800)

    # Whether to store file with predictions
    store_predictions = True


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    label_horizon = App.config["label_horizon"]  # Labels are generated using this number of steps ahead
    labels = App.config["labels"]
    feature_sets = App.config.get("feature_sets")
    algorithms = App.config.get("algorithms")

    #features_horizon = 720  # Features are generated using this past window length
    features_kline = App.config.get("features_kline")
    features_futur = App.config.get("features_futur")
    features_depth = App.config.get("features_depth")

    freq = "1m"
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return

    out_path = data_path / "MODELS"
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

    config_file_modifier = App.config.get("config_file_modifier")
    config_file_modifier = ("-" + config_file_modifier) if config_file_modifier else ""

    start_dt = datetime.now()

    #
    # Load feature matrix
    #
    in_file_suffix = App.config.get("matrix_file_modifier")

    in_file_name = f"{in_file_suffix}{config_file_modifier}.csv"
    in_path = data_path / in_file_name
    if not in_path.exists():
        print(f"ERROR: Input file does not exist: {in_path}")
        return

    print(f"Loading feature matrix from input file: {in_path}")
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
    if "kline" in feature_sets:
        all_features += features_kline
    if "futur" in feature_sets:
        all_features += features_futur
    all_features += labels
    in_df = in_df[all_features + out_columns]

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna(subset=labels)
    in_df = in_df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    # Remove the tail data for which no labels are available
    # The reason is that these labels are computed from future which is not available
    if label_horizon:
        in_df = in_df.head(-label_horizon)

    models = dict()
    scores = dict()

    out_df = pd.DataFrame()  # Collect predictions

    for feature_set in feature_sets:
        if feature_set == "kline":
            features = features_kline
            fs_tag = "_k_"
            feature_set_train_length = P.kline_train_length
        elif feature_set == "futur":
            features = features_futur
            fs_tag = "_f_"
            feature_set_train_length = P.futur_train_length
        else:
            print(f"ERROR: Unknown feature set {feature_set}. Check feature set list in config.")
            return

        print(f"Start training {feature_set} feature set with {len(features)} features, name tag {fs_tag}', and train length {feature_set_train_length}")

        # Limit maximum length
        train_df = in_df.tail(feature_set_train_length)
        train_df = train_df.dropna(subset=features)

        for label in labels:

            for algo_name in algorithms:
                model_config = get_model(algo_name)
                algo_type = model_config.get("algo")
                train_length = model_config.get("train", {}).get("length")
                score_column_name = label + fs_tag + algo_name

                # Limit length according to algorith parameters
                if train_length and train_length < feature_set_train_length:
                    train_df_2 = train_df.iloc[-train_length:]
                else:
                    train_df_2 = train_df
                df_X = train_df_2[features]
                df_y = train_df_2[label]

                print(f"Train '{score_column_name}'. Train length {len(df_X)}. Algorithm {algo_name}")
                if algo_type == "gb":
                    model_pair = train_gb(df_X, df_y, model_config)
                    models[score_column_name] = model_pair
                    df_y_hat = predict_gb(model_pair, df_X, model_config)
                elif algo_type == "nn":
                    model_pair = train_nn(df_X, df_y, model_config)
                    models[score_column_name] = model_pair
                    df_y_hat = predict_nn(model_pair, df_X, model_config)
                elif algo_type == "lc":
                    model_pair = train_lc(df_X, df_y, model_config)
                    models[score_column_name] = model_pair
                    df_y_hat = predict_lc(model_pair, df_X, model_config)
                else:
                    print(f"ERROR: Unknown algorithm type {algo_type}. Check algorithm list.")
                    return

                scores[score_column_name] = compute_scores(df_y, df_y_hat)
                out_df[score_column_name] = df_y_hat

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

    metrics_file_name = f"metrics.txt"
    metrics_path = (out_path / metrics_file_name).resolve()
    with open(metrics_path, 'a+') as f:
        f.write("\n".join(lines) + "\n")

    print(f"Metrics stored in path: {metrics_path.absolute()}")

    #
    # Store predictions if necessary
    #
    if P.store_predictions:
        out_file_suffix = App.config.get("predict_file_modifier")

        out_file_name = f"{out_file_suffix}{config_file_modifier}.csv"
        out_path = data_path / out_file_name

        # We do not store features. Only selected original data, labels, and their predictions
        out_df = out_df.join(in_df[out_columns + labels])

        print(f"Storing output file...")
        out_df.to_csv(out_path, index=False)
        print(f"Predictions stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    #
    # End
    #
    elapsed = datetime.now() - start_dt
    print(f"Finished in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main()
