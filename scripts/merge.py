from datetime import timedelta, datetime, time
from pathlib import Path

import pandas as pd
import numpy as np

import click

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
    df_out = merge_data_sources(data_sources)

    #
    # Store file with features
    #
    out_path = data_path / symbol / config.get("merge_file_name")

    print(f"Storing output file...")
    df_out = df_out.reset_index()
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


def merge_data_sources(data_sources: list):

    time_column = App.config["time_column"]
    freq = App.config["freq"]

    for ds in data_sources:
        df = ds.get("df")

        if time_column in df.columns:
            df = df.set_index(time_column)
        elif df.index.name == time_column:
            pass
        else:
            print(f"ERROR: Timestamp column is absent.")
            return

        # Add prefix if not already there
        if ds['column_prefix']:
            #df = df.add_prefix(ds['column_prefix']+"_")
            df.columns = [
                ds['column_prefix']+"_"+col if not col.startswith(ds['column_prefix']+"_") else col
                for col in df.columns
            ]

        ds["start"] = df.first_valid_index()  # df.index[0]
        ds["end"] = df.last_valid_index()  # df.index[-1]

        ds["df"] = df

    #
    # Create common (main) index and empty data frame
    #
    range_start = min([ds["start"] for ds in data_sources])
    range_end = min([ds["end"] for ds in data_sources])

    # Generate a discrete time raster according to the (pandas) frequency parameter
    index = pd.date_range(range_start, range_end, freq=freq)

    df_out = pd.DataFrame(index=index)
    df_out.index.name = time_column

    for ds in data_sources:
        # Note that timestamps must have the same semantics, for example, start of kline (and not end of kline)
        # If different data sets have different semantics for timestamps, then data must be shifted accordingly
        df_out = df_out.join(ds["df"])

    # Interpolate numeric columns
    merge_interpolate = App.config.get("merge_interpolate", False)
    if merge_interpolate:
        num_columns = df_out.select_dtypes((float, int)).columns.tolist()
        for col in num_columns:
            df_out[col] = df_out[col].interpolate()

    return df_out


if __name__ == '__main__':
    main()
