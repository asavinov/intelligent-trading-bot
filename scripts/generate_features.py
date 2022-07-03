import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union, Tuple
import json
import pickle
import click

import numpy as np
import pandas as pd

from service.App import *
from common.feature_generation import *

#
# Parameters
#
class P:
    in_nrows = 50_000_000  # Load only this number of records
    tail_rows = int(2.0 * 525_600)  # Process only this number of last rows


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

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
    # Load merged data with regular time series
    #
    in_file_suffix = App.config.get("merge_file_modifier")

    in_file_name = f"{in_file_suffix}.csv"
    in_path = data_path / in_file_name

    print(f"Loading data from source file {str(in_path)}...")
    df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:]
    df = df.reset_index(drop=True)

    #
    # Generate derived features
    #
    feature_sets = App.config.get("feature_sets", [])
    if not feature_sets:
        # By default, we generate standard kline features
        feature_sets = [{"column_prefix": "", "generator": "klines", "feature_prefix": ""}]

    # Apply all feature generators to the data frame which get accordingly new derived columns
    # The feature parameters will be taken from App.config (depending on generator)
    print(f"Start generating features for {len(df)} input records.")
    df, all_features = generate_feature_sets(df, feature_sets, last_rows=0)
    print(f"Finished generating features.")

    #
    # Store feature matrix in output file
    #
    out_file_suffix = App.config.get("feature_file_modifier")

    out_file_name = f"{out_file_suffix}{config_file_modifier}.csv"
    out_path = (data_path / out_file_name).resolve()

    print(f"Storing feature matrix with {len(df)} records and {len(df.columns)} columns in output file...")
    df.to_csv(out_path, index=False, float_format="%.4f")
    #df.to_parquet(out_path.with_suffix('.parquet'), engine='auto', compression=None, index=None, partition_cols=None)

    #
    # Store features
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f"'{f}'" for f in all_features] ) + "\n")

    print(f"Stored {len(all_features)} features in output file {out_path}")

    elapsed = datetime.now() - start_dt
    print(f"Finished feature generation in {int(elapsed.total_seconds())} seconds")
    print(f"Output file location: {out_path}")


def generate_feature_sets(df: pd.DataFrame, feature_sets: list, last_rows: int) -> Tuple[pd.DataFrame, list]:
    """Apply different feature generators to the input data set according to descriptors."""

    all_features = []
    for fs in feature_sets:
        # Select columns from the data set to be processed by the feature generator
        cp = fs.get("column_prefix")
        if cp:
            cp = cp + "_"
            f_cols = [col for col in df if col.startswith(cp)]
            f_df = df[f_cols]  # Alternatively: f_df = df.loc[:, df.columns.str.startswith(cf)]
            # Remove prefix because feature generators are generic (a prefix will be then added to derived features before adding them back to the main frame)
            f_df = f_df.rename(columns=lambda x: x[len(cp):] if x.startswith(cp) else x)  # Alternatively: f_df.columns = f_df.columns.str.replace(cp, "")
        else:
            f_df = df[df.columns.to_list()]  # We want to have a different data frame object to add derived featuers and then join them back to the main frame with prefix

        generator = fs.get("generator")
        if generator == "klines":
            features = generate_features(
                f_df, use_differences=False,
                base_window=App.config["base_window_kline"], windows=App.config["windows_kline"],
                area_windows=App.config["area_windows_kline"], last_rows=last_rows
            )
        elif generator == "futures":
            features = generate_features_futures(f_df)
        elif generator == "depth":
            features = generate_features_depth(f_df)
        else:
            print(f"Unknown feature generator {generator}")
            return

        f_df = f_df[features]
        # Add feature columns from feature frame to main input frame and add prefix
        fp = fs.get("feature_prefix")
        if fp:
            f_df = f_df.add_prefix(fp + "_")

        all_features += f_df.columns.to_list()

        df = df.join(f_df)  # Attach all derived features to the main frame

    return df, all_features


if __name__ == '__main__':
    main()
