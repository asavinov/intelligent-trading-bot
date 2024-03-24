from pathlib import Path
from datetime import datetime, timezone, timedelta
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from service.App import *
from common.gen_features import *
from common.classifiers import *
from common.model_store import *
from common.generators import train_feature_set

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

    file_path = data_path / App.config.get("matrix_file_name")
    if not file_path.is_file():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading data from source data file {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unknown extension of the 'matrix_file_name' file '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    #
    # Prepare data by selecting columns and rows
    #

    # Default (common) values for all trained features
    label_horizon = App.config["label_horizon"]  # Labels are generated from future data and hence we might want to explicitly remove some tail rows
    train_length = App.config.get("train_length")
    train_features = App.config.get("train_features")
    labels = App.config["labels"]
    algorithms = App.config.get("algorithms")

    # Select necessary features and labels
    out_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [x for x in out_columns if x in df.columns]
    all_features = train_features + labels
    df = df[out_columns + [x for x in all_features if x not in out_columns]]

    for label in labels:
        # "category" NN does not work without this (note that we assume a classification task here)
        df[label] = df[label].astype(int)

    # Remove the tail data for which no (correct) labels are available
    # The reason is that these labels are computed from future values which are not available and hence labels might be wrong
    if label_horizon:
        df = df.head(-label_horizon)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df = df.dropna(subset=labels)
    df = df.dropna(subset=train_features)
    if len(df) == 0:
        print(f"ERROR: Empty data set after removing NULLs in feature columns. Some features might have all NULL values.")
        #print(df.isnull().sum().sort_values(ascending=False))
        return

    # Limit maximum length for all algorithms (algorithms can further limit their train size)
    if train_length:
        df = df.tail(train_length)

    df = df.reset_index(drop=True)  # To remove gaps in index before use

    #
    # Train feature models
    #
    train_feature_sets = App.config.get("train_feature_sets", [])
    if not train_feature_sets:
        print(f"ERROR: no train feature sets defined. Nothing to process.")
        return

    print(f"Start training models for {len(df)} input records.")

    out_df = pd.DataFrame()  # Collect predictions
    models = dict()
    scores = dict()

    for i, fs in enumerate(train_feature_sets):
        fs_now = datetime.now()
        print(f"Start train feature set {i}/{len(train_feature_sets)}. Generator {fs.get('generator')}...")

        fs_out_df, fs_models, fs_scores = train_feature_set(df, fs, App.config)

        out_df = pd.concat([out_df, fs_out_df], axis=1)
        models.update(fs_models)
        scores.update(fs_scores)

        fs_elapsed = datetime.now() - fs_now
        print(f"Finished train feature set {i}/{len(train_feature_sets)}. Generator {fs.get('generator')}. Time: {str(fs_elapsed).split('.')[0]}")

    print(f"Finished training models.")

    #
    # Store all collected models in files
    #
    model_path = Path(App.config["model_folder"])
    if not model_path.is_absolute():
        model_path = data_path / model_path
    model_path = model_path.resolve()

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

        print(f"Storing predictions with {len(out_df)} records and {len(out_df.columns)} columns in output file {out_path}...")
        if out_path.suffix == ".parquet":
            out_df.to_parquet(out_path, index=False)
        elif out_path.suffix == ".csv":
            out_df.to_csv(out_path, index=False, float_format='%.6f')
        else:
            print(f"ERROR: Unknown extension of the 'predict_file_name' file '{out_path.suffix}'. Only 'csv' and 'parquet' are supported")
            return

        print(f"Predictions stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    #
    # End
    #
    elapsed = datetime.now() - now
    print(f"Finished training models in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
