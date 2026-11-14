from pathlib import Path

import numpy as np
import pandas as pd

import click

from service.App import *
from common.model_store import *
from scripts.features import generate_feature_set

"""
This script will load a feature file (or any file with close price), and add
top-bot columns according to the label parameter, by finally storing both input
data and the labels in the output file (can be the same file as input).
"""

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Load a file with close price (typically feature matrix),
    compute top-bottom labels, add them to the data, and store to output file.
    """
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
    else:
        window_size = config.get("predict_length")
    features_horizon = config.get("features_horizon")
    if window_size:
        window_size += features_horizon

    #
    # Load merged data with regular time series
    #
    file_path = data_path / config.get("feature_file_name")
    if not file_path.is_file():
        print(f"Data file does not exist: {file_path}")
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
    # Generate derived features
    #
    label_sets = config.get("label_sets", [])
    if not label_sets:
        print(f"ERROR: no label sets defined. Nothing to process.")
        return

    # Apply all feature generators to the data frame which get accordingly new derived columns
    # The feature parameters will be taken from config (depending on generator)
    print(f"Start generating labels for {len(df)} input records.")

    all_features = []
    for i, fs in enumerate(label_sets):
        fs_now = datetime.now()
        print(f"Start label set {i}/{len(label_sets)}. Generator {fs.get('generator')}...")

        df, new_features = generate_feature_set(df, fs, config, App.model_store, last_rows=0)

        all_features.extend(new_features)
        fs_elapsed = datetime.now() - fs_now
        print(f"Finished label set {i}/{len(label_sets)}. Generator {fs.get('generator')}. Labels: {len(new_features)}. Time: {str(fs_elapsed).split('.')[0]}")

    print(f"Finished generating labels.")

    # Handle NULLs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    na_df = df[ df[all_features].isna().any(axis=1) ]
    if len(na_df) > 0:
        print(f"WARNING: There exist {len(na_df)} rows with NULLs in some feature columns")

    print(f"Number of NULL values:")
    print(df[all_features].isnull().sum().sort_values(ascending=False))

    #
    # Store feature matrix in output file
    #
    out_file_name = config.get("matrix_file_name")
    out_path = (data_path / out_file_name).resolve()

    print(f"Storing file with labels. {len(df)} records and {len(df.columns)} columns in output file {out_path}...")
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        print(f"ERROR: Unknown extension of the output file '{out_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Stored output file {out_path} with {len(df)} records")

    #
    # Store labels
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f'"{f}"' for f in all_features] ) + "\n\n")

    print(f"Stored {len(all_features)} labels in output file {out_path.with_suffix('.txt')}")

    elapsed = datetime.now() - now
    print(f"Finished generating {len(all_features)} labels in {str(elapsed).split('.')[0]}. Time per label: {str(elapsed/len(all_features)).split('.')[0]}")


if __name__ == '__main__':
    main()
