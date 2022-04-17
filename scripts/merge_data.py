import pandas as pd
import math
#import os.path
from pathlib import Path
import json
import time
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)
import click

import numpy as np

from common.utils import *
from service.App import *
from common.feature_generation import *

"""
Create one output file from many input files:
- symbols like BTC or ETH
- different data types like klines, futures, depth (depth can be in several files)

Align the timestamps and create a uniform time axis without gaps.

Prefix column names with the corresponding origin modifiers so that each column name stores its origin.

IMPLEMENTATION.
What we need to know:
- data folder with all files, symbol list to get subfolders, data types for each symbol to find its individual file, prefix to be used for this file
- once files are loaded, create raster index, and merge all loaded data to this index by also providing column prefixes.
- what to do with empty (initial or trailing)? Automatically detect maximum common timestamp.

list of:
<symbol>-<data source> - prefix

"""


depth_file_names = [  # Leave empty to skip
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch1.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch2.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch3.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch4.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch5.csv",
]

futur_column_prefix = "f_"
range_type = "kline"  # Selector: kline, futur, depth, merged (common range)


#
# Historic data
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


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    freq = "1m"
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"])
    data_sources = App.config.get("data_sources", [])
    if not data_sources:
        data_sources = [{"folder": symbol, "file": "klines", "column_prefix": ""}]

    config_file_modifier = App.config.get("config_file_modifier")
    config_file_modifier = ("-" + config_file_modifier) if config_file_modifier else ""

    start_dt = datetime.now()

    for ds in data_sources:
        # What is want is for each source, load file into df, determine its properties (columns, start, end etc.), and then merge all these dfs
        file_path = (data_path / ds.get("folder") / ds.get("file")).with_suffix(".csv")
        if not file_path.is_file():
            print(f"Data file does not exist: {file_path}")
            return

        print(f"Reading data file: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = df.set_index("timestamp")  # We only assume that there is timestamp attribute

        if ds['column_prefix']:
            df = df.add_prefix(ds['column_prefix']+"_")
        #df.columns = [f"{ds['column_prefix']}_{col}" for col in df.columns]

        ds["df"] = df
        ds["start"] = df.index[0]
        ds["end"] = df.index[-1]

        print(f"Loaded file with {len(df)} records. Range: ({ds['start']}, {ds['end']})")

    #
    # Create 1m common (main) index and empty data frame
    #
    range_start = min([ds["start"] for ds in data_sources])
    range_end = max([ds["end"] for ds in data_sources])

    index = pd.date_range(range_start, range_end, freq="T")
    df_out = pd.DataFrame(index=index)
    df_out.index.name = "timestamp"

    print(f"Start merging...")

    for ds in data_sources:
        # Note that timestamps must have the same semantics, for example, start of kline (and not end of kline)
        # If different data sets have different semantics for timestamps, then data must be shifted accordingly
        df_out = df_out.join(ds["df"])

    #
    # Store file with features
    #
    out_file_suffix = App.config.get("merge_file_modifier")

    out_file_name = f"{out_file_suffix}{config_file_modifier}.csv"
    out_path = data_path / symbol / out_file_name

    print(f"Storing output file...")
    df_out.to_csv(out_path, index=True)  # float_format="%.6f"
    print(f"Stored output merged file with {len(df_out)} records. Range: ({range_start}, {range_end})")

    elapsed = datetime.now() - start_dt
    print(f"Finished processing in {int(elapsed.total_seconds())} seconds.")
    print(f"Output file location: {out_path}")


if __name__ == '__main__':
    main()
