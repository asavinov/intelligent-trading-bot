from datetime import timedelta, datetime, time
from pathlib import Path

import pandas as pd
import numpy as np

import click

from common.utils import merge_data_sources
from service.App import *

"""
Create one output file from multiple input data files. 
"""

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)
    config = App.config

    time_column = config["time_column"]

    now = datetime.now()

    symbol = config["symbol"]
    data_path = Path(config["data_folder"])

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
    # Load data from multiple sources and merge
    #
    data_sources = config.get("data_sources", [])
    if not data_sources:
        print(f"ERROR: Data sources are not defined. Nothing to merge.")
        #data_sources = [{"folder": symbol, "file": "klines", "column_prefix": ""}]

    # Read data from input files
    for ds in data_sources:
        # What is want is for each source, load file into df, determine its properties (columns, start, end etc.), and then merge all these dfs

        quote = ds.get("folder")
        if not quote:
            print(f"ERROR. Folder is not specified.")
            continue

        # If file name is not specified then use symbol name as file name
        file = ds.get("file", quote)
        if not file:
            file = quote

        file_path = (data_path / quote / file)
        if not file_path.suffix:
            file_path = file_path.with_suffix(".csv")  # CSV by default

        print(f"Reading data file: {file_path}")
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
        else:
            print(f"ERROR: Unknown extension of the input file '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
            return
        print(f"Loaded file with {len(df)} records.")

        # Select only the data necessary for analysis
        if window_size:
            df = df.tail(window_size)
            df = df.reset_index(drop=True)

        ds["df"] = df

    # Merge in one df with prefixes and common regular time index
    freq = App.config["freq"]
    merge_interpolate = App.config.get("merge_interpolate", False)
    df_out = merge_data_sources(data_sources, time_column, freq, merge_interpolate)

    #
    # Store file with features
    #
    out_path = data_path / symbol / config.get("merge_file_name")

    print(f"Storing output file...")
    df_out = df_out.reset_index(drop=(df_out.index.name in df_out.columns))
    if out_path.suffix == ".parquet":
        df_out.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df_out.to_csv(out_path, index=False)  # float_format="%.6f"
    else:
        print(f"ERROR: Unknown extension of the output file '{out_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    range_start = df_out.index[0]
    range_end = df_out.index[-1]
    print(f"Stored output file {out_path} with {len(df_out)} records. Range: ({range_start}, {range_end})")

    elapsed = datetime.now() - now
    print(f"Finished merging data in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
