import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
import click

import numpy as np
import pandas as pd

from service.App import *
from common.utils import *
from common.classifiers import *
from common.feature_generation import *
from common.model_store import *

"""
Generate label predictions for the whole input feature matrix by iteratively training models using historic data and predicting labels for some future horizon.
The main parameter is the step of iteration, that is, the future horizon for prediction.
As usual, we can specify past history length used to train a model.
The output file will store predicted labels in addition to all input columns (generated features and true labels).
This file is intended for training signal models (by simulating trade process and computing overall performance for some long period).
The output predicted labels will cover shorter period of time because we need some relatively long history to train the very first model.
"""

#
# Parameters
#
class P:
    start_index = 0
    end_index = None

    # How much data we want to use for training
    kline_train_length = int(1.5 * 525_600)  # 1.5 * 525_600
    futur_train_length = int(4 * 43_800)

    # First row for starting predictions: "2020-02-01 00:00:00" - minimum start for futures
    prediction_start_str = "2020-02-01 00:00:00"
    # How frequently re-train models: 1 day: 1_440 = 60 * 24, one week: 10_080
    prediction_length = 2*7*1440
    prediction_count = 56  # How many prediction steps. If None or 0, then from prediction start till the data end. Use: https://www.timeanddate.com/date/duration.html

    use_multiprocessing = True
    max_workers = 8  # None means number of processors


#
# Main
#

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    label_horizon = App.config["label_horizon"]  # Labels are generated using this number of steps ahead
    labels = App.config["labels"]
    train_features = App.config.get("train_features")
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
    df = None
    if in_file_name.endswith(".csv"):
        df = pd.read_csv(in_path, parse_dates=['timestamp'])
    elif in_file_name.endswith(".parquet"):
        df = pd.read_parquet(in_path)
    elif in_file_name.endswith(".pickle"):
        df = pd.read_pickle(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    print(f"Feature matrix loaded. Length: {len(df)}. Width: {len(df.columns)}")

    #
    # Limit length according to parameters start_index end_index
    #
    df = df.iloc[P.start_index:P.end_index]
    df = df.reset_index()

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna(subset=labels)
    df = df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    prediction_start = find_index(df, P.prediction_start_str)
    print(f"Start index: {prediction_start}")

    #
    # Limit columns
    #
    for label in labels:
        df[label] = df[label].astype(int)  # "category" NN does not work without this

    # Select necessary features and label
    out_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    all_features = []
    if "kline" in train_features:
        all_features += features_kline
    if "futur" in train_features:
        all_features += features_futur
    all_features += labels
    df = df[all_features + out_columns]

    #
    # Rolling train-predict loop
    #
    stride = P.prediction_length
    steps = P.prediction_count
    if not steps:
        # Use all available rest data (from the prediction start to the dataset end)
        steps = (len(df) - prediction_start) // stride
    if len(df) - prediction_start < steps * stride:
        raise ValueError(f"Number of steps {steps} is too high (not enough data after start). Data available for prediction: {len(df) - prediction_start}. Data to be predicted: {steps * stride} ")

    print(f"Starting rolling predict loop with {steps} steps. Each step with {stride} horizon...")

    # Result rows. Here store only rows for which we make predictions
    labels_hat_df = pd.DataFrame()

    for step in range(steps):

        print(f"\n===>>> Start step {step}/{steps}")

        # Predict data

        predict_start = prediction_start + (step * stride)
        predict_end = predict_start + stride

        predict_df = df.iloc[predict_start:predict_end]  # We assume that iloc is equal to index
        # predict_df = predict_df.dropna(subset=features)  # Nans will be droped by the algorithms themselves

        # Here we will collect predicted columns
        predict_labels_df = pd.DataFrame(index=predict_df.index)

        for tf in train_features:
            if tf == "kline":
                features = features_kline
                fs_tag = "_k_"
                features_train_length = P.kline_train_length
            elif tf == "futur":
                features = features_futur
                fs_tag = "_f_"
                features_train_length = P.futur_train_length
            else:
                print(f"ERROR: Unknown feature set {tf}. Check feature set list in config.")
                return

            print(f"Start training {tf} feature set with {len(features)} features, name tag {fs_tag}', and train length {features_train_length}")

            # Predict data

            df_X_test = predict_df[features]
            #df_y_test = predict_df[predict_label]  # It will be set in the loop over labels

            # Train data

            # We exclude recent objects from training, because they do not have labels yet - the labels are in future
            # In real (stream) data, we will have null labels for recent objects. During simulation, labels are available and hence we need to ignore/exclude them manually
            train_end = predict_start - label_horizon - 1
            train_start = train_end - features_train_length
            train_start = 0 if train_start < 0 else train_start

            train_df = df.iloc[int(train_start):int(train_end)]  # We assume that iloc is equal to index
            train_df = train_df.dropna(subset=features)

            print(f"Train range: [{train_start}, {train_end}]={train_end-train_start}. Prediction range: [{predict_start}, {predict_end}]={predict_end-predict_start}. ")

            for label in labels:  # Train-predict different labels (and algorithms) using same X

                if P.use_multiprocessing:
                    # Submit train-predict algorithms to the pool
                    execution_results = dict()
                    with ProcessPoolExecutor(max_workers=P.max_workers) as executor:
                        for algo_name in algorithms:
                            model_config = get_model(algo_name)
                            algo_type = model_config.get("algo")
                            train_length = model_config.get("train", {}).get("length")
                            score_column_name = label + fs_tag + algo_name

                            # Limit length according to algorith parameters
                            if train_length and train_length < features_train_length:
                                train_df_2 = train_df.iloc[-train_length:]
                            else:
                                train_df_2 = train_df
                            df_X = train_df_2[features]
                            df_y = train_df_2[label]
                            df_y_test = predict_df[label]

                            if algo_type == "gb":
                                execution_results[score_column_name] = executor.submit(train_predict_gb, df_X, df_y, df_X_test, model_config)
                            elif algo_type == "nn":
                                execution_results[score_column_name] = executor.submit(train_predict_nn, df_X, df_y, df_X_test, model_config)
                            elif algo_type == "lc":
                                execution_results[score_column_name] = executor.submit(train_predict_lc, df_X, df_y, df_X_test, model_config)
                            else:
                                print(f"ERROR: Unknown algorithm type {algo_type}. Check algorithm list.")
                                return

                    # Process the results as the tasks are finished
                    for score_column_name, future in execution_results.items():
                        predict_labels_df[score_column_name] = future.result()
                        if future.exception():
                            print(f"Exception while train-predict {score_column_name}.")
                            return
                else:  # No multiprocessing - sequential execution
                    for algo_name in algorithms:
                        model_config = get_model(algo_name)
                        algo_type = model_config.get("algo")
                        train_length = model_config.get("train", {}).get("length")
                        score_column_name = label + fs_tag + algo_name

                        # Limit length according to algorith parameters
                        if train_length and train_length < features_train_length:
                            train_df_2 = train_df.iloc[-train_length:]
                        else:
                            train_df_2 = train_df
                        df_X = train_df_2[features]
                        df_y = train_df_2[label]
                        df_y_test = predict_df[label]

                        if algo_type == "gb":
                            predict_labels_df[score_column_name] = train_predict_gb(df_X, df_y, df_X_test, model_config)
                        elif algo_type == "nn":
                            predict_labels_df[score_column_name] = train_predict_nn(df_X, df_y, df_X_test, model_config)
                        elif algo_type == "lc":
                            predict_labels_df[score_column_name] = train_predict_lc(df_X, df_y, df_X_test, model_config)
                        else:
                            print(f"ERROR: Unknown algorithm type {algo_type}. Check algorithm list.")
                            return

        #
        # Append predicted *rows* to the end of previous predicted rows
        #
        # Predictions for all labels and histories (and algorithms) have been generated for the iteration
        labels_hat_df = pd.concat([labels_hat_df, predict_labels_df])

        print(f"End step {step}/{steps}.")
        print(f"Predicted {len(predict_labels_df.columns)} labels.")


    # End of loop over prediction steps
    print("")
    print(f"Finished all {steps} prediction steps each with {stride} predicted rows (stride). ")
    print(f"Size of predicted dataframe {len(labels_hat_df)}. Number of rows in all steps {steps*stride} (steps * stride). ")
    print(f"Number of predicted columns {len(labels_hat_df.columns)}")

    #
    # Store data
    #
    # We do not store features. Only selected original data, labels, and their predictions
    out_df = labels_hat_df.join(df[out_columns + labels])

    out_file_suffix = App.config.get("predict_file_modifier")

    out_file_name = f"{out_file_suffix}{config_file_modifier}.csv"
    out_path = data_path / out_file_name

    print(f"Storing output file...")
    out_df.to_csv(out_path, index=False)
    #out_df.to_parquet(out_path.with_suffix('.parquet'), engine='auto', compression=None, index=None, partition_cols=None)
    print(f"Predictions stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    #
    # Compute accuracy for the whole data set (all segments)
    #

    # Alternatively, generate all score column names: score_column_name = label + "_k_" + history_name
    score_lines = []
    for score_column_name in labels_hat_df.columns:
        label_column = score_column_name[0:-5]

        # Drop nans from scores
        df_scores = pd.DataFrame({"y_true": out_df[label_column], "y_predicted": out_df[score_column_name]})
        df_scores = df_scores.dropna()

        y_true = df_scores["y_true"].astype(int)
        y_predicted = df_scores["y_predicted"]
        y_predicted_class = np.where(y_predicted.values > 0.5, 1, 0)

        print(f"Using {len(df_scores)} non-nan rows for scoring.")

        score = compute_scores(y_true, y_predicted)

        score_lines.append(f"{score_column_name}, {score.get('auc'):.3f}, {score.get('ap'):.3f}, {score.get('f1'):.3f}, {score.get('precision'):.3f}, {score.get('recall'):.3f}")

    #
    # Store hyper-parameters and scores
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write("\n".join([str(x) for x in score_lines]) + "\n")

    elapsed = datetime.now() - start_dt
    print(f"Finished feature prediction in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main()
