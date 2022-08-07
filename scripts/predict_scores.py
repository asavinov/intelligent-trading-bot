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
Apply models to (previously generated) features and compute prediction scores.
"""


#
# Parameters
#
class P:
    in_nrows = 100_000_000  # For debugging
    tail_rows = 0  # How many last rows to select (for debugging)


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

    file_path = (data_path / App.config.get("feature_file_name")).with_suffix(".csv")
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
    train_features = App.config.get("train_features")
    labels = App.config["labels"]
    algorithms = App.config.get("algorithms")

    # Select necessary features and label
    out_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [x for x in out_columns if x in df.columns]
    labels_present = set(labels).issubset(df.columns)
    if labels_present:
        all_features = train_features + labels
    else:
        all_features = train_features
    df = df[out_columns + all_features]

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna(subset=labels)
    df = df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    train_df = df[train_features].dropna(subset=train_features)

    if len(train_df) == 0:
        print(f"ERROR: Empty data set after removing NULLs in feature columns. Some features might have all NULL values.")
        #print(train_df.isnull().sum().sort_values(ascending=False))
        return

    #
    # Load models for all score columns
    #
    model_path = data_path / "MODELS"
    if not model_path.is_absolute():
        model_path = PACKAGE_ROOT / model_path
    model_path = model_path.resolve()

    buy_labels = App.config["buy_labels"]
    sell_labels = App.config["sell_labels"]
    models = {label: load_model_pair(model_path, label) for label in buy_labels + sell_labels}

    #
    # Loop over score columns with models and apply them to features
    #
    scores = dict()
    out_df = pd.DataFrame(index=train_df.index)  # Collect predictions
    for score_column_name, model_pair in tqdm(models.items(), desc="PREDICTIONS"):

        label, algo_name = score_to_label_algo_pair(score_column_name)
        model_config = get_model(algo_name)  # Get algorithm description from the algo store
        algo_type = model_config.get("algo")

        if algo_type == "gb":
            df_y_hat = predict_gb(model_pair, train_df, model_config)
        elif algo_type == "nn":
            df_y_hat = predict_nn(model_pair, train_df, model_config)
        elif algo_type == "lc":
            df_y_hat = predict_lc(model_pair, train_df, model_config)
        elif algo_type == "svc":
            df_y_hat = predict_svc(model_pair, train_df, model_config)
        else:
            raise ValueError(f"Unknown algorithm type '{algo_type}'")

        if labels_present:
            scores[score_column_name] = compute_scores(df[label], df_y_hat)
        out_df[score_column_name] = df_y_hat

    #
    # Store scores
    #
    if labels_present:
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
    # Store predictions
    #
    # Store only selected original data, labels, and their predictions
    out_df = out_df.join(df[out_columns + (labels if labels_present else [])])

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
