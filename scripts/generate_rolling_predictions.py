import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle

import numpy as np
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from trade.App import App
from common.utils import *
from common.classifiers import *
from common.feature_generation import *

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
    # Each one is a separate procedure with its algorithm and (expected) input features
    # Leave only what we want to generate (say, only klines for debug purposes)
    feature_sets = ["kline", "futur"]

    # These source columns will be added to the output file
    in_features_kline = [
        "timestamp",
        "open","high","low","close","volume",
        #"close_time",
        #"quote_av","trades","tb_base_av","tb_quote_av","ignore",
    ]

    labels = App.config["labels"]
    features_kline = App.config["features_kline"]
    features_futur = App.config["features_futur"]
    features_depth = App.config["features_depth"]

    #features_horizon = 720  # Features are generated using this past window length (max feature window)
    labels_horizon = 180  # Labels are generated using this number of steps ahead (max label window)

    in_path_name = r"C:\DATA2\BITCOIN\GENERATED"  # File with all necessary derived features
    in_file_name = r"BTCUSDT-1m-features.csv"
    in_nrows = 100_000_000
    in_nrows_tail = None  # How many last rows to select (for testing)
    skiprows = 500_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"BTCUSDT-1m-features-rolling.csv"

    # First row for starting predictions: "2020-02-01 00:00:00" - minimum start for futures
    prediction_start_str = "2020-02-01 00:00:00"
    # How frequently re-train models: 1 day: 1_440 = 60 * 24, one week: 10_080
    prediction_length = 4*7*1440
    prediction_count = 1  # How many prediction steps. If None or 0, then from prediction start till the data end


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
    "n_epochs": 20,
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

#
# Main
#

def main(args=None):
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
        in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)  # , skiprows=range(1,P.skiprows)
        #in_df.to_pickle('aaa.pickle')
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
    in_df = in_df[P.in_features_kline + P.features_kline + P.features_futur + P.labels]

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna(subset=P.labels)
    in_df = in_df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    prediction_start = find_index(in_df, P.prediction_start_str)
    print(f"Start index: {prediction_start}")

    #
    # Rolling train-predict loop
    #
    stride = P.prediction_length
    steps = P.prediction_count
    if not steps:
        # Use all available rest data (from the prediction start to the dataset end)
        steps = (len(in_df) - prediction_start) // stride
    if len(in_df) - prediction_start < steps * stride:
        raise ValueError(f"Number of steps {steps} is too high (not enough data after start). Data available for prediction: {len(in_df) - prediction_start}. Data to be predicted: {steps * stride} ")

    print(f"Starting rolling predict loop with {steps} steps. Each step with {stride} horizon...")

    # Result rows. Here store only rows for which we make predictions
    labels_hat_df = pd.DataFrame()

    for step in range(steps):

        print(f"Start step {step}/{steps}")

        # Predict data

        predict_start = prediction_start + (step * stride)
        predict_end = predict_start + stride

        predict_df = in_df.iloc[predict_start:predict_end]  # We assume that iloc is equal to index
        # predict_df = predict_df.dropna(subset=features)  # Nans will be droped by the algorithms themselves

        # Here we will collect predicted columns
        predict_labels_df = pd.DataFrame(index=predict_df.index)

        # ===
        # kline feature set
        # ===
        if "kline" in P.feature_sets:
            features = P.features_kline
            name_tag = "_k_"
            train_length = int(1.5 * 525_600)  # 1.5 * 525_600

            print(f"Start training 'kline' package with {len(features)} features, name tag {name_tag}', and train length {train_length}")

            # Predict data

            df_X_test = predict_df[features]
            #df_y_test = predict_df[predict_label]  # It will be set in the loop over labels

            # Train data

            # We exclude recent objects from training, because they do not have labels yet - the labels are in future
            # In real (stream) data, we will have null labels for recent objects. During simulation, labels are available and hence we need to ignore/exclude them manually
            train_end = predict_start - P.labels_horizon - 1
            train_start = train_end - train_length
            train_start = 0 if train_start < 0 else train_start

            train_df = in_df.iloc[int(train_start):int(train_end)]  # We assume that iloc is equal to index
            train_df = train_df.dropna(subset=features)

            df_X = train_df[features]
            #df_y = train_df[predict_label]  # It will be set in the loop over labels

            print(f"Train range: [{train_start}, {train_end}]={train_end-train_start}. Prediction range: [{predict_start}, {predict_end}]={predict_end-predict_start}. ")

            for label in P.labels:  # Train-predict different labels (and algorithms) using same X
                df_y = train_df[label]
                df_y_test = predict_df[label]

                # --- GB
                score_column_name = label + name_tag + "gb"
                print(f"Train-predict '{score_column_name}'... ")
                y_hat = train_predict_gb(df_X, df_y, df_X_test, params=params_gb)
                predict_labels_df[score_column_name] = y_hat

                # --- NN
                score_column_name = label + name_tag + "nn"
                print(f"Train-predict '{score_column_name}'... ")
                y_hat = train_predict_nn(df_X, df_y, df_X_test, params=params_nn)
                predict_labels_df[score_column_name] = y_hat

                # --- LC
                score_column_name = label + name_tag + "lc"
                print(f"Train-predict '{score_column_name}'... ")
                y_hat = train_predict_lc(df_X, df_y, df_X_test, params=params_lc)
                predict_labels_df[score_column_name] = y_hat

        # ===
        # futur feature set
        # ===
        if "futur" in P.feature_sets:
            features = P.features_futur
            name_tag = "_f_"
            train_length = int(4 * 43_800)

            print(f"Start training 'futur' package with {len(features)} features, name tag {name_tag}', and train length {train_length}")

            # Predict data

            df_X_test = predict_df[features]
            #df_y_test = predict_df[predict_label]  # It will be set in the loop over labels

            # Train data

            # We exclude recent objects from training, because they do not have labels yet - the labels are in future
            # In real (stream) data, we will have null labels for recent objects. During simulation, labels are available and hence we need to ignore/exclude them manually
            train_end = predict_start - P.labels_horizon - 1
            train_start = train_end - train_length
            train_start = 0 if train_start < 0 else train_start

            train_df = in_df.iloc[int(train_start):int(train_end)]  # We assume that iloc is equal to index
            train_df = train_df.dropna(subset=features)

            df_X = train_df[features]
            #df_y = train_df[predict_label]  # It will be set in the loop over labels

            print(f"Train range: [{train_start}, {train_end}]={train_end-train_start}. Prediction range: [{predict_start}, {predict_end}]={predict_end-predict_start}. ")

            for label in P.labels:  # Train-predict different labels (and algorithms) using same X
                df_y = train_df[label]
                df_y_test = predict_df[label]

                # --- GB
                score_column_name = label + name_tag + "gb"
                print(f"Train-predict '{score_column_name}'... ")
                y_hat = train_predict_gb(df_X, df_y, df_X_test, params=params_gb)
                predict_labels_df[score_column_name] = y_hat

                # --- NN
                score_column_name = label + name_tag + "nn"
                print(f"Train-predict '{score_column_name}'... ")
                y_hat = train_predict_nn(df_X, df_y, df_X_test, params=params_nn)
                predict_labels_df[score_column_name] = y_hat

                # --- LC
                score_column_name = label + name_tag + "lc"
                print(f"Train-predict '{score_column_name}'... ")
                y_hat = train_predict_lc(df_X, df_y, df_X_test, params=params_lc)
                predict_labels_df[score_column_name] = y_hat

        #
        # Append predicted *rows* to the end of previous predicted rows
        #
        # Predictions for all labels and histories (and algorithms) have been generated for the iteration
        labels_hat_df = labels_hat_df.append(predict_labels_df)

        print(f"End step {step}/{steps}.")
        print(f"Predicted {len(predict_labels_df.columns)} labels.")


    # End of loop over prediction steps
    print("")
    print(f"Finished all {steps} prediction steps each with {stride} predicted rows (stride). ")
    print(f"Size of predicted dataframe {len(labels_hat_df)}. Number of rows in all steps {steps*stride} (steps * stride). ")
    print(f"Number of predicted columns {len(labels_hat_df.columns)}")

    #
    # Prepare output
    #

    # Attach all original columns including true labels to the predicted labels
    # Here we assume that df with predictions has the same index as the original df
    out_columns = P.in_features_kline + P.labels
    out_df = labels_hat_df.join(in_df[out_columns])

    #
    # Store data
    #
    print(f"Storing output file...")

    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
    out_path = out_path.joinpath(P.out_file_name)

    out_df.to_csv(out_path.with_suffix('.csv'), index=False, float_format="%.4f")

    #out_df.to_parquet(out_path.with_suffix('.parquet'), engine='auto', compression=None, index=None, partition_cols=None)

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

        score_lines.append(f"{score_column_name}, {score.auc:.3f}, {score.ap:.3f}, {score.f1:.3f}, {score.precision:.3f}, {score.recall:.3f}")

    #
    # Store hyper-parameters and scores
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write("\n".join([str(x) for x in score_lines]) + "\n")

    elapsed = datetime.now() - start_dt
    print(f"Finished feature prediction in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main(sys.argv[1:])
