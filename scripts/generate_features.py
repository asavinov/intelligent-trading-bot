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
from common.label_generation import *

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
    data_path = Path(App.config["data_folder"])
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return

    start_dt = datetime.now()

    #
    # Load historic data
    #
    in_path = (data_path / f"{symbol}-{freq}.csv").resolve()

    print(f"Loading data from source file {str(in_path)}...")

    in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)

    print(f"Finished loading {len(in_df)} records with {len(in_df.columns)} columns.")

    #
    # Generate derived features
    #

    if "kline" in P.feature_sets:
        print(f"Generating klines features...")
        k_features = generate_features(in_df)
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

    #
    # Generate labels (always the same, currently based on kline data which must be therefore present)
    #
    print(f"Generating labels...")
    labels = []

    # Binary labels whether max has exceeded a threshold or not
    labels += generate_labels_thresholds(in_df, horizon=180)

    # Numeric label which is ration between areas over and under the latest price
    labels += add_area_ratio(in_df, is_future=True, column_name="close", windows=[60, 120, 180, 300], suffix = "_area_future")

    print(f"Finished generating {len(labels)} labels")

    #
    # Store feature matrix in output file
    #
    out_file_name = f"{symbol}-{freq}-features.csv"
    out_file = (data_path / out_file_name).resolve()

    print(f"Storing feature matrix with {len(in_df)} records and {len(in_df.columns)} columns in output file...")

    in_df.to_csv(out_file, index=False, float_format="%.4f")

    #in_df.to_parquet(out_path.with_suffix('.parquet'), engine='auto', compression=None, index=None, partition_cols=None)

    elapsed = datetime.now() - start_dt
    print(f"Finished feature generation in {int(elapsed.total_seconds())} seconds")
    print(f"Output file location: {out_file}")


if __name__ == '__main__':
    main()
