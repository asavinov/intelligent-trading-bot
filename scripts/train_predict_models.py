from pathlib import Path
from datetime import datetime, timezone, timedelta
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from service.App import *
from common.classifiers import *
from common.feature_generation import *
from common.model_store import *

"""
Train models for all target labels and all algorithms declared in the configuration using the specified features.
"""


#
# Parameters
#
class P:
    in_nrows = 100_000_000  # For debugging
    tail_rows = 0  # How many last rows to select (for debugging)

    # Whether to store file with predictions
    store_predictions = True


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    #
    # Load feature matrix
    #
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol

    file_path = (data_path / App.config.get("matrix_file_name")).with_suffix(".csv")
    if not file_path.is_file():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading data from source data file {file_path}...")
    df = pd.read_csv(file_path, parse_dates=[time_column], nrows=P.in_nrows)
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:]
    df = df.reset_index(drop=True)

    #
    # Prepare data by selecting columns and rows
    #
    label_horizon = App.config["label_horizon"]  # Labels are generated from future data and hence we might want to explicitly remove some tail rows
    train_length = App.config.get("train_length")
    train_features = App.config.get("train_features")
    labels = App.config["labels"]
    algorithms = App.config.get("algorithms")

    # Select necessary features and label
    out_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [x for x in out_columns if x in df.columns]
    all_features = train_features + labels
    df = df[out_columns + all_features]

    for label in labels:
        # "category" NN does not work without this (note that we assume a classification task here)
        df[label] = df[label].astype(int)

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna(subset=labels)
    df = df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    # Remove the tail data for which no labels are available
    # The reason is that these labels are computed from future which is not available
    if label_horizon:
        df = df.head(-label_horizon)

    # Limit maximum length
    train_df = df.tail(train_length)

    train_df = train_df.dropna(subset=train_features)

    if len(train_df) == 0:
        print(f"ERROR: Empty data set after removing NULLs in feature columns. Some features might have all NULL values.")
        #print(train_df.isnull().sum().sort_values(ascending=False))
        return

    models = dict()
    scores = dict()
    out_df = pd.DataFrame()  # Collect predictions

    for label in tqdm(labels, desc="LABELS", colour='red', position=0):

        for algo_name in tqdm(algorithms, desc="ALGORITHMS", colour='red', leave=False, position=1):
            model_config = get_model(algo_name)  # Get algorithm description from the algo store
            algo_type = model_config.get("algo")
            algo_train_length = model_config.get("train", {}).get("length")
            score_column_name = label + label_algo_separator + algo_name

            # Limit length according to the algorith parameters
            if algo_train_length and algo_train_length < train_length:
                train_df_2 = train_df.iloc[-algo_train_length:]
            else:
                train_df_2 = train_df
            df_X = train_df_2[train_features]
            df_y = train_df_2[label]

            print(f"Train '{score_column_name}'. Train length {len(df_X)}. Train columns {len(df_X.columns)}. Algorithm {algo_name}")
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
            elif algo_type == "svc":
                model_pair = train_svc(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_svc(model_pair, df_X, model_config)
            else:
                print(f"ERROR: Unknown algorithm type {algo_type}. Check algorithm list.")
                return

            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            out_df[score_column_name] = df_y_hat

    #
    # Store all collected models in files
    #
    model_path = data_path / "MODELS"
    model_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

    for score_column_name, model_pair in models.items():
        save_model_pair(model_path, score_column_name, model_pair)

    print(f"Models stored in path: {model_path.absolute()}")

    #
    # Store scores
    #
    lines = list()
    for score_column_name, score in scores.items():
        line = score_column_name + ", " + str(score)
        lines.append(line)

    metrics_file_name = f"prediction-metrics.txt"
    metrics_path = (data_path / metrics_file_name).resolve()
    with open(metrics_path, 'a+') as f:
        f.write("\n".join(lines) + "\n\n")

    print(f"Metrics stored in path: {metrics_path.absolute()}")

    #
    # Store predictions if necessary
    #
    if P.store_predictions:
        # Store only selected original data, labels, and their predictions
        out_df = out_df.join(df[out_columns + labels])

        out_path = data_path / App.config.get("predict_file_name")

        print(f"Storing output file...")
        out_df.to_csv(out_path.with_suffix(".csv"), index=False, float_format='%.4f')
        print(f"Predictions stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    #
    # End
    #
    elapsed = datetime.now() - now
    print(f"Finished training models in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
