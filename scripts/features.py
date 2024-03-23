from typing import Tuple
from pathlib import Path
import click

import numpy as np
import pandas as pd

from service.App import *
from common.generators import generate_feature_set


#
# Parameters
#
class P:
    in_nrows = 50_000_000  # Load only this number of records
    tail_rows = int(10.0 * 525_600)  # Process only this number of last rows


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    #
    # Load merged data with regular time series
    #
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol

    file_path = data_path / App.config.get("merge_file_name")
    if not file_path.is_file():
        print(f"Data file does not exist: {file_path}")
        return

    print(f"Loading data from source data file {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unknown extension of the 'merge_file_name' file '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    #
    # Generate derived features
    #
    feature_sets = App.config.get("feature_sets", [])
    if not feature_sets:
        print(f"ERROR: no feature sets defined. Nothing to process.")
        return

    # Apply all feature generators to the data frame which get accordingly new derived columns
    # The feature parameters will be taken from App.config (depending on generator)
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
    # Store feature matrix in output file
    #
    out_file_name = App.config.get("feature_file_name")
    out_path = (data_path / out_file_name).resolve()

    print(f"Storing features with {len(df)} records and {len(df.columns)} columns in output file {out_path}...")
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        print(f"ERROR: Unknown extension of the 'feature_file_name' file '{out_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Stored output file {out_path} with {len(df)} records")

    #
    # Store feature list
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f'"{f}"' for f in all_features] ) + "\n\n")

    print(f"Stored {len(all_features)} features in output file {out_path.with_suffix('.txt')}")

    elapsed = datetime.now() - now
    print(f"Finished generating {len(all_features)} features in {str(elapsed).split('.')[0]}. Time per feature: {str(elapsed/len(all_features)).split('.')[0]}")


if __name__ == '__main__':
    main()
