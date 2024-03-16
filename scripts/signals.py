from pathlib import Path
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from common.generators import generate_feature_set
from service.App import *

"""
Generate new derived columns according to the signal definitions.
The transformations are applied to the results of ML predictions.
"""

#
# Parameters
#
class P:
    in_nrows = 100_000_000

    start_index = 0  # 200_000 for 1m btc
    end_index = None


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    """
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return
    out_path = Path(App.config["data_folder"]) / symbol
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

    #
    # Load data with (rolling) label point-wise predictions
    #
    file_path = data_path / App.config.get("predict_file_name")
    if not file_path.exists():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading predictions from input file: {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unknown extension of the 'predict_file_name' file '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return
    print(f"Predictions loaded. Length: {len(df)}. Width: {len(df.columns)}")

    # Limit size according to parameters start_index end_index
    df = df.iloc[P.start_index:P.end_index]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    #
    # Signals
    #
    feature_sets = App.config.get("signal_sets", [])
    if not feature_sets:
        print(f"ERROR: no signal sets defined. Nothing to process.")
        return

    print(f"Start generating features for {len(df)} input records.")

    all_features = []
    for i, fs in enumerate(feature_sets):
        fs_now = datetime.now()
        print(f"Start feature set {i}/{len(feature_sets)}. Generator {fs.get('generator')}...")
        df, new_features = generate_feature_set(df, fs, last_rows=0)
        all_features.extend(new_features)
        fs_elapsed = datetime.now() - fs_now
        print(f"Finished feature set {i}/{len(feature_sets)}. Generator {fs.get('generator')}. Features: {len(new_features)}. Time: {str(fs_elapsed).split('.')[0]}")

    print(f"Finished generating features.")

    print(f"Number of NULL values:")
    print(df[all_features].isnull().sum().sort_values(ascending=False))

    #
    # Choose columns to stored
    #
    out_columns = ["timestamp", "open", "high", "low", "close"]  # Source data
    out_columns.extend(App.config.get('labels'))  # True labels
    out_columns.extend(all_features)

    out_df = df[out_columns]

    #
    # Store data
    #
    out_path = data_path / App.config.get("signal_file_name")

    print(f"Storing signals with {len(out_df)} records and {len(out_df.columns)} columns in output file {out_path}...")
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format='%.6f')
    else:
        print(f"ERROR: Unknown extension of the 'signal_file_name' file '{out_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Signals stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    elapsed = datetime.now() - now
    print(f"Finished signal generation in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
