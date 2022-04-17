import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
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
    feature_sets = ["kline", ]  # "futur"

    in_nrows = 10_000_000


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
    in_path = (data_path / f"data.csv").resolve()

    print(f"Loading data from source file {str(in_path)}...")
    in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)
    print(f"Finished loading {len(in_df)} records with {len(in_df.columns)} columns.")

    #
    # Generate derived features
    #

    if "kline" in P.feature_sets:
        print(f"Generating klines features...")
        k_features = generate_features(
            in_df, use_differences=False,
            base_window=App.config["base_window_kline"], windows=App.config["windows_kline"],
            area_windows=App.config["area_windows_kline"], last_rows=0
        )
        print(f"Finished generating {len(k_features)} kline features")
    else:
        k_features = []

    if "futur" in P.feature_sets:
        print(f"Generating futur features...")
        f_features = generate_features_futur(in_df)
        print(f"Finished generating {len(f_features)} futur features")
    else:
        f_features = []

    if "depth" in P.feature_sets:
        print(f"Generating depth features...")
        d_features = generate_features_depth(in_df)
        print(f"Finished generating {len(f_features)} depth features")
    else:
        d_features = []

    all_features = k_features + f_features + d_features

    #
    # Store feature matrix in output file
    #
    out_file_suffix = App.config.get("feature_file_modifier")

    out_file_name = f"{out_file_suffix}{config_file_modifier}.csv"
    out_path = (data_path / out_file_name).resolve()

    print(f"Storing feature matrix with {len(in_df)} records and {len(in_df.columns)} columns in output file...")
    in_df.to_csv(out_path, index=False, float_format="%.4f")
    #in_df.to_parquet(out_path.with_suffix('.parquet'), engine='auto', compression=None, index=None, partition_cols=None)

    #
    # Store features
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f"'{f}'" for f in all_features] ) + "\n")

    print(f"Stored {len(all_features)} features in output file {out_path}")

    elapsed = datetime.now() - start_dt
    print(f"Finished feature generation in {int(elapsed.total_seconds())} seconds")
    print(f"Output file location: {out_path}")


if __name__ == '__main__':
    main()
