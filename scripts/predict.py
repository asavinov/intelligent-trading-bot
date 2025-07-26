from pathlib import Path
from datetime import datetime, timezone, timedelta
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from service.App import *
from common.model_store import *
from common.generators import predict_feature_set

"""
Apply models to (previously generated) features and compute prediction scores.
"""

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)
    config = App.config

    App.model_store = ModelStore(config)
    App.model_store.load_models()

    time_column = config["time_column"]

    now = datetime.now()

    symbol = config["symbol"]
    data_path = Path(config["data_folder"]) / symbol

    # Determine desired data length depending on train/predict mode
    is_train = config.get("train")
    if is_train:
        window_size = config.get("train_length")
        print(f"WARNING: Train mode is specified although this script is intended for prediction and will not train models.")
    else:
        window_size = config.get("predict_length")
    features_horizon = config.get("features_horizon")
    if window_size:
        window_size += features_horizon

    #
    # Load data
    #
    file_path = data_path / config.get("matrix_file_name")
    if not file_path.is_file():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading data from source data file {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
    else:
        print(f"ERROR: Unknown extension of the input file '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Finished loading {len(df)} records with {len(df.columns)} columns from the source file {file_path}")

    # Select only the data necessary for analysis
    if window_size:
        df = df.tail(window_size)
        df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    #
    # Apply ML algorithm predictors
    #
    train_features = config.get("train_features")
    labels = config["labels"]
    algorithms = config.get("algorithms")

    # Select necessary features and label
    out_columns = [time_column, 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [x for x in out_columns if x in df.columns]
    labels_present = set(labels).issubset(df.columns)
    if labels_present:
        all_features = train_features + labels
    else:
        all_features = train_features
    df = df[out_columns + [x for x in all_features if x not in out_columns]]

    # Handle NULLs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    na_df = df[ df[train_features].isna().any(axis=1) ]
    if len(na_df) > 0:
        print(f"WARNING: There exist {len(na_df)} rows with NULLs in some feature columns. These rows will be removed.")
        df = df.dropna(subset=train_features)
        df = df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    #
    # Generate/predict train features
    #
    train_feature_sets = config.get("train_feature_sets", [])
    if not train_feature_sets:
        print(f"ERROR: no train feature sets defined. Nothing to process.")
        return

    print(f"Start generating trained features for {len(df)} input records.")

    out_df = pd.DataFrame()  # Collect predictions
    features = []
    scores = dict()

    for i, fs in enumerate(train_feature_sets):
        fs_now = datetime.now()
        print(f"Start train feature set {i}/{len(train_feature_sets)}. Generator {fs.get('generator')}...")

        fs_out_df, fs_features, fs_scores = predict_feature_set(df, fs, config, App.model_store)

        out_df = pd.concat([out_df, fs_out_df], axis=1)
        features.extend(fs_features)
        scores.update(fs_scores)

        fs_elapsed = datetime.now() - fs_now
        print(f"Finished train feature set {i}/{len(train_feature_sets)}. Generator {fs.get('generator')}. Time: {str(fs_elapsed).split('.')[0]}")

    print(f"Finished generating trained features.")

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
    out_df = df[out_columns + (labels if labels_present else [])].join(out_df)

    out_path = data_path / config.get("predict_file_name")

    print(f"Storing predictions with {len(out_df)} records and {len(out_df.columns)} columns in output file {out_path}...")
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format='%.6f')
    else:
        print(f"ERROR: Unknown extension of the output file '{out_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Predictions stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    #
    # End
    #
    elapsed = datetime.now() - now
    print(f"Finished predicting in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
