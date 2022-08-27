from datetime import timedelta, datetime, time
from pathlib import Path

import pandas as pd
import numpy as np

import click

from service.App import *

"""
This script is intended for creating one output file from multiple input data files. 
It is needed when we want to use additional data source in order to predict the main parameter.
For example, in order to predict BTC price, we might want to add ETH prices. 
This script solves the following problems:
- Input files might have the same column names (e.g., open, high, low, close) and therefore it adds prefixes to the columns of the output file
- Input data may have gaps and therefore the script generates a regular time raster for the output file. The granularity of the time raster is determined by the parameter
"""


depth_file_names = [  # Leave empty to skip
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch1.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch2.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch3.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch4.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch5.csv",
]


#
# Readers from inputs files (DEPRECATED)
#

def load_futur_files(futur_file_path):
    """Return a data frame with future features."""

    df = pd.read_csv(futur_file_path, parse_dates=['timestamp'])
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]

    df = df.set_index("timestamp")

    print(f"Loaded futur file with {len(df)} records in total. Range: ({start}, {end})")

    return df, start, end


def load_kline_files(kline_file_path):
    """Return a data frame with kline features."""

    df = pd.read_csv(kline_file_path, parse_dates=['timestamp'])
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]

    df = df.set_index("timestamp")

    print(f"Loaded kline file with {len(df)} records in total. Range: ({start}, {end})")

    return df, start, end


def load_depth_files():
    """Return a list of data frames with depth features."""

    dfs = []
    start = None
    end = None
    for depth_file_name in depth_file_names:
        df = pd.read_csv(depth_file_name, parse_dates=['timestamp'])
        # Start
        if start is None:
            start = df["timestamp"].iloc[0]
        elif df["timestamp"].iloc[0] < start:
            start = df["timestamp"].iloc[0]
        # End
        if end is None:
            end = df["timestamp"].iloc[-1]
        elif df["timestamp"].iloc[-1] > end:
            end = df["timestamp"].iloc[-1]

        df = df.set_index("timestamp")

        dfs.append(df)

    length = np.sum([len(df) for df in dfs])
    print(f"Loaded {len(depth_file_names)} depth files with {length} records in total. Range: ({start}, {end})")

    return dfs, start, end

#
# Merger
#


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"]

    data_sources = App.config.get("data_sources", [])
    if not data_sources:
        print(f"ERROR: Data sources are not defined. Nothing to merge.")
        #data_sources = [{"folder": symbol, "file": "klines", "column_prefix": ""}]

    now = datetime.now()

    # Read data from input files
    data_path = Path(App.config["data_folder"])
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

        file_path = (data_path / quote / file).with_suffix(".csv")
        if not file_path.is_file():
            print(f"Data file does not exist: {file_path}")
            return

        print(f"Reading data file: {file_path}")
        df = pd.read_csv(file_path, parse_dates=[time_column])
        print(f"Loaded file with {len(df)} records.")

        ds["df"] = df

    # Merge in one df with prefixes and common regular time index
    df_out = merge_data_sources(data_sources)

    #
    # Store file with features
    #
    out_path = data_path / App.config["symbol"] / App.config.get("merge_file_name")

    print(f"Storing output file...")
    df_out.to_csv(out_path.with_suffix(".csv"), index=True)  # float_format="%.6f"
    range_start = df_out.index[0]
    range_end = df_out.index[-1]
    print(f"Stored output merged file with {len(df_out)} records. Range: ({range_start}, {range_end})")

    elapsed = datetime.now() - now
    print(f"Finished merging data in {str(elapsed).split('.')[0]}")
    print(f"Output file location: {out_path.with_suffix('.csv')}")


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
    # Create 1m common (main) index and empty data frame
    #
    range_start = min([ds["start"] for ds in data_sources])
    range_end = min([ds["end"] for ds in data_sources])

    # Regular time raster according to the parameter
    if freq == "1m":
        index = pd.date_range(range_start, range_end, freq="T")
    elif freq == "1d":
        index = pd.date_range(range_start, range_end, freq="B")  # D - daily, B - business days (no weekends)
        #index = pd.bdate_range(start=range_start, end=range_end)  # tz='UTC'
    else:
        print(f"ERROR: Frequency parameter 'freq' is unknown or not specified: {freq}")
        return

    df_out = pd.DataFrame(index=index)
    df_out.index.name = time_column

    for ds in data_sources:
        # Note that timestamps must have the same semantics, for example, start of kline (and not end of kline)
        # If different data sets have different semantics for timestamps, then data must be shifted accordingly
        df_out = df_out.join(ds["df"])

    return df_out


if __name__ == '__main__':
    main()
