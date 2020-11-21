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

from common.classifiers import *
from common.utils import *
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

#
# Parameters
#
class P:
    # Each one is a separate procedure with its algorithm and (expected) input features
    # Leave only what we want to generate (say, only klines for debug purposes)
    prediction_types = ["kline", "futur"]

    # Target columns with true values which will be predicted
    # Leave only what we want to be generated (e.g., only one label for debug purposes)
    labels = [
        'high_10', 'high_15', 'high_20',
        'low_10', 'low_15', 'low_20',
    ]
    labels_regr = [
        'high_max_60','high_max_120','high_max_180',  # Maximum high (relative)
        'low_min_60','low_min_120','low_min_180',  # Minimum low (relative)
        'high_to_low_60', 'high_to_low_120', 'high_to_low_180',
        'close_area_future_60', 'close_area_future_120', 'close_area_future_180', 'close_area_future_300',
    ]
    class_labels_all = [  # All existing target labels generated from feature generation procedure
        'high_max_60','high_max_120','high_max_180',  # Maximum high (relative)
        'high_10', 'high_15', 'high_20', 'high_25',  # At least one time above
        'high_01', 'high_02', 'high_03', 'high_04',  # Always below

        'low_min_60','low_min_120','low_min_180',  # Minimum low (relative)
        'low_01', 'low_02', 'low_03', 'low_04',  # Always above
        'low_10', 'low_15', 'low_20', 'low_25',  # At least one time below

        'high_to_low_60','high_to_low_120','high_to_low_180',

        'close_area_future_60','close_area_future_120','close_area_future_180','close_area_future_300',
        ]

    # These source columns will be added to the output file
    in_features_kline = [
        "timestamp",
        "open","high","low","close","volume",
        "close_time",
        "quote_av","trades","tb_base_av","tb_quote_av","ignore"
    ]

    features_kline = [
        'close_1','close_5','close_15','close_60','close_180','close_720',
        'close_std_5','close_std_15','close_std_60','close_std_180','close_std_720',  # Removed "std_1" which is constant
        'volume_1','volume_5','volume_15','volume_60','volume_180','volume_720',
        'span_1', 'span_5', 'span_15', 'span_60', 'span_180', 'span_720',
        'trades_1','trades_5','trades_15','trades_60','trades_180','trades_720',
        'tb_base_1','tb_base_5','tb_base_15','tb_base_60','tb_base_180','tb_base_720',
        'tb_quote_1','tb_quote_5','tb_quote_15','tb_quote_60','tb_quote_180','tb_quote_720',
        'close_area_60', 'close_area_120', 'close_area_180', 'close_area_300', 'close_area_720',
    ]
    features_kline_small = [
        'close_1','close_5','close_15','close_60','close_180','close_720',
        'close_std_5','close_std_15','close_std_60','close_std_180','close_std_720',  # Removed "std_1" which is constant
        'volume_1','volume_5','volume_15','volume_60','volume_180','volume_720',
        ]

    features_futur = [
        "f_close_1", "f_close_2", "f_close_5", "f_close_20", "f_close_60", "f_close_180",
        "f_close_std_2", "f_close_std_5", "f_close_std_20", "f_close_std_60", "f_close_std_180",  # Removed "std_1" which is constant
        "f_volume_1", "f_volume_2", "f_volume_5", "f_volume_20", "f_volume_60", "f_volume_180",
        "f_span_1", "f_span_2", "f_span_5", "f_span_20", "f_span_60", "f_span_180",
        "f_trades_1", "f_trades_2", "f_trades_5", "f_trades_20", "f_trades_60", "f_trades_180",
        'f_close_area_20', 'f_close_area_60', 'f_close_area_120', 'f_close_area_180',
    ]

    features_depth = [
        "gap_2","gap_5","gap_10",
        "bids_1_2","bids_1_5","bids_1_10", "asks_1_2","asks_1_5","asks_1_10",
        "bids_2_2","bids_2_5","bids_2_10", "asks_2_2","asks_2_5","asks_2_10",
        "bids_5_2","bids_5_5","bids_5_10", "asks_5_2","asks_5_5","asks_5_10",
        "bids_10_2","bids_10_5","bids_10_10", "asks_10_2","asks_10_5","asks_10_10",
        "bids_20_2","bids_20_5","bids_20_10", "asks_20_2","asks_20_5","asks_20_10",
    ]


    in_path_name = r"C:\DATA2\BITCOIN\GENERATED"  # File with all necessary derived features
    in_file_name = r"BTCUSDT-1m-features.pkl"
    in_nrows = 10_000_000
    in_nrows_tail = None  # How many last rows to select (for testing)
    skiprows = 500_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"BTCUSDT-1m-features-rolling.csv"

    # First row for starting predictions: "2020-02-01 00:00:00" - minimum start for futures
    prediction_start_str = "2020-02-01 00:00:00"
    # How frequently re-train models: 1 day: 1_440 = 60 * 24, one week: 10_080
    prediction_length = 4*7*1440
    prediction_count = 9  # How many prediction steps. If None or 0, then from prediction start till the data end

    # We can define several history lengths. A separate model and prediction scores will be generated for each of them.
    # Example: {"18": 788_400, "12": 525_600, "06": 262_800, "04": 175_200, "03": 131_400, "02": 87_600}
    label_histories_kline = {"18": 788_400}  # 1.5 years - 788_400=1.5*525_600
    label_histories_futur = {"04": 175_200}  # 4 months - 175_200=4*43_800

    features_horizon = 720  # Features are generated using this past window length (max feature window)
    labels_horizon = 180  # Labels are generated using this number of steps ahead (max label window)


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

params_grid_lc = {
    "is_scale": False,
    "penalty": "l2",
    "C": 1.0,
    "class_weight": None,
    "solver": "liblinear",
    "max_iter": 200,
}


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
        in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)  # , skiprows=range(1,P.skiprows)
        #in_df.to_pickle('aaa.pkl')
    elif P.in_file_name.endswith(".parq"):
        in_df = pd.read_parquet(in_path)
    elif P.in_file_name.endswith(".pkl"):
        in_df = pd.read_pickle(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    if P.in_nrows_tail:
        in_df = in_df.tail(P.in_nrows_tail)

    for label in P.labels:
        in_df[label] = in_df[label].astype(int)  # "category" NN does not work without this

    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna()  # We drop nulls after selecting features (not here) because futures have more nans than klines
    in_df = in_df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    prediction_start = find_index(in_df, P.prediction_start_str)

    #
    # Rolling train-predict loop
    #
    stride = P.prediction_length
    steps = P.prediction_count
    if not steps:
        # Use all available rest data (from the prediction start to the dataset end)
        steps = (len(in_df) - prediction_start) // stride

    print(f"Starting rolling predict loop with {steps} steps. Each step with {stride} horizon...")

    # Result rows. Here store only rows for which we make predictions
    labels_hat_df = pd.DataFrame()

    for step in range(steps):

        # New start for this step
        start = prediction_start + (step * stride)

        predict_df = in_df.iloc[start:start+stride]  # We assume that iloc is equal to index
        predict_labels_df = pd.DataFrame(index=predict_df.index)

        # We exclude recent data from training, because we do not have labels (to generate labels, we need future which is not available yet)
        # In real data, we will have Nulls in labels, but in simulation labels are available, and we need to exclude them manually
        train_end = start - P.labels_horizon - 1

        # ===
        # kline package
        # ===
        if "kline" in P.prediction_types:

            label_histories = P.label_histories_kline
            features = P.features_kline
            name_tag = "_k_"

            for history_name, history_length in label_histories.items():
                # Prepare train data of the necessary history length
                train_start = start - history_length
                train_start = 0 if train_start < 0 else train_start

                train_df = in_df.iloc[train_start:train_end]  # We assume that iloc is equal to index
                train_df = train_df.dropna(subset=features + P.labels)

                df_X = train_df[features]
                df_X_test = predict_df[features]

                for label in P.labels:  # Train-predict different labels (and algorithms) using same X
                    df_y = train_df[label]

                    # --- GB
                    score_column_name = label + name_tag + "gb"
                    print(f"Train model: label '{label}', history {history_name}, rows {len(train_df)}, features {len(features)}, score column {score_column_name}...")
                    y_hat = train_predict_gb(df_X, df_y, df_X_test, params=params_gb)
                    predict_labels_df[score_column_name] = y_hat

                    # --- NN
                    score_column_name = label + name_tag + "nn"
                    print(f"Train model: label '{label}', history {history_name}, rows {len(train_df)}, features {len(features)}, score column {score_column_name}...")
                    y_hat = train_predict_nn(df_X, df_y, df_X_test, params=params_nn)
                    predict_labels_df[score_column_name] = y_hat

                    # --- LC
                    score_column_name = label + name_tag + "lc"
                    print(f"Train model: label '{label}', history {history_name}, rows {len(train_df)}, features {len(features)}, score column {score_column_name}...")
                    y_hat = train_predict_lc(df_X, df_y, df_X_test, params=params_lc)
                    predict_labels_df[score_column_name] = y_hat

        # ===
        # futur package
        # ===
        if "futur" in P.prediction_types:

            label_histories = P.label_histories_futur
            features = P.features_futur
            name_tag = "_f_"

            for history_name, history_length in label_histories.items():
                # Prepare train data of the necessary history length
                train_start = start - history_length
                train_start = 0 if train_start < 0 else train_start

                train_df = in_df.iloc[train_start:train_end]  # We assume that iloc is equal to index
                train_df = train_df.dropna(subset=features + P.labels)

                df_X = train_df[features]
                df_X_test = predict_df[features]

                for label in P.labels:  # Train-predict different labels (and algorithms) using same X
                    df_y = train_df[label]

                    # --- GB
                    score_column_name = label + name_tag + "gb"
                    print(f"Train model: label '{label}', history {history_name}, rows {len(train_df)}, features {len(features)}, score column {score_column_name}...")
                    y_hat = train_predict_gb(df_X, df_y, df_X_test, params=params_gb)
                    predict_labels_df[score_column_name] = y_hat

                    # --- NN
                    score_column_name = label + name_tag + "nn"
                    print(f"Train model: label '{label}', history {history_name}, rows {len(train_df)}, features {len(features)}, score column {score_column_name}...")
                    y_hat = train_predict_nn(df_X, df_y, df_X_test, params=params_nn)
                    predict_labels_df[score_column_name] = y_hat

                    # --- LC
                    score_column_name = label + name_tag + "lc"
                    print(f"Train model: label '{label}', history {history_name}, rows {len(train_df)}, features {len(features)}, score column {score_column_name}...")
                    y_hat = train_predict_lc(df_X, df_y, df_X_test, params=params_lc)
                    predict_labels_df[score_column_name] = y_hat

        #
        # Append predicted *rows* to the end of previous predicted rows
        #
        # Predictions for all labels and histories (and algorithms) have been generated for the iteration
        labels_hat_df = labels_hat_df.append(predict_labels_df)

    #
    # Prepare output
    #

    # Append all features including true labels to the predicted labels
    out_columns = ["timestamp", "open", "high", "low", "close", "volume"] + P.labels
    out_df = labels_hat_df.join(in_df[out_columns])

    #
    # Store data
    #
    print(f"Storing output file...")

    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
    out_path = out_path.joinpath(P.out_file_name)

    out_df.to_csv(out_path.with_suffix('.csv'), index=False, float_format="%.4f")

    #out_df.to_parquet(out_path.with_suffix('.parq'), engine='auto', compression=None, index=None, partition_cols=None)

    #
    # Compute accuracy for the whole data set (all segments)
    #

    # Alternatively, generate all score column names: score_column_name = label + "_k_" + history_name
    score_lines = []
    for score_column_name in labels_hat_df.columns:

        y_predicted = out_df[score_column_name]
        y_predicted_class = np.where(out_df[score_column_name].values > 0.5, 1, 0)

        label_column = score_column_name[0:-5]
        y_true = out_df[label_column].astype(int)

        try:
            auc = metrics.roc_auc_score(y_true, y_predicted.fillna(value=0))
        except ValueError:
            auc = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

        try:
            ap = metrics.average_precision_score(y_true, y_predicted.fillna(value=0))
        except ValueError:
            ap = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

        f1 = metrics.f1_score(y_true, y_predicted_class)
        precision = metrics.precision_score(y_true, y_predicted_class)
        recall = metrics.recall_score(y_true, y_predicted_class)

        score_lines.append(f"{score_column_name}, {auc:.3f}, {ap:.3f}, {f1:.3f}, {precision:.3f}, {recall:.3f}")

    #
    # Store hyper-parameters and scores
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write("\n".join([str(x) for x in score_lines]) + "\n")

    elapsed = datetime.now() - start_dt
    print(f"Finished feature prediction in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main(sys.argv[1:])
