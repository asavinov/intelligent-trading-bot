import pandas as pd
import math
#import os.path
from pathlib import Path
import json
import time
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)

import numpy as np

from common.utils import *
from common.feature_generation import *

"""
Create one output file from input files of different types: klines, futures, depth.
Depth data can be provided in several files. 
Also, depth timestamps correspond to end of 1m interval and hence they will be changed to kline convention.
Futures and klines are in single files and their timestamp is start of 1m interval.
Future column names are same as in klines, and hence they will be prefixed in the output.
Output file has continuous index by removing possible gaps in input files.
"""


symbol = "BTCUSDT"  # BTCUSDT ETHBTC IOTAUSDT

kline_file_name = r"C:\DATA2\BITCOIN\DOWNLOADED\BTCUSDT-1m-klines.csv"

futur_file_name = r"C:\DATA2\BITCOIN\DOWNLOADED\BTCUSDT-1m-futurs.csv"

depth_file_names = [  # Leave empty to skip
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch1.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch2.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch3.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch4.csv",
    #r"C:\DATA2\BITCOIN\GENERATED\depth-BTCUSDT-batch5.csv",
]

out_file_name = r"BTCUSDT-1m.csv"

futur_column_prefix = "f_"
range_type = "kline"  # Selector: kline, futur, depth, merged (common range)

#
# Historic data
#

def get_symbol_files(symbol):
    """
    Get a list of file names with data for this symbol and frequency.
    We find all files with this symbol in name in the directly recursively.
    """
    file_pattern = f"*{symbol}*.txt"
    paths = Path(in_path_name).rglob(file_pattern)
    return list(paths)

def load_futur_files():
    """Return a data frame with future features."""

    df = pd.read_csv(futur_file_name, parse_dates=['timestamp'])
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]

    df = df.set_index("timestamp")

    print(f"Loaded futur file with {len(df)} records in total. Range: ({start}, {end})")

    return df, start, end

def load_kline_files():
    """Return a data frame with kline features."""

    df = pd.read_csv(kline_file_name, parse_dates=['timestamp'])
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

def main(args=None):

    start_dt = datetime.now()
    print(f"Start processing...")

    k_df, k_start, k_end = load_kline_files()
    f_df, f_start, f_end = load_futur_files()
    if depth_file_names:
        d_dfs, d_start, d_end = load_depth_files()
    else:
        d_dfs = []
        d_start = 0
        d_end = 0

    #
    # Determine range
    #
    if range_type.startswith("kline"):
        start = k_start
        end = k_end
    elif range_type.startswith("futur"):
        start = f_start
        end = f_end
    elif range_type.startswith("depth"):
        start = d_start
        end = d_end
    elif range_type.startswith("merge"):
        start = np.max([d_start, f_start, k_start])
        end = np.min([d_end, f_end, k_end])
    else:
        print(f"Unknown parameter value. Exit.")
        exit()

    #
    # Create 1m common (main) index and empty data frame
    #
    index = pd.date_range(start, end, freq="T")
    df_out = pd.DataFrame(index=index)
    df_out.index.name = "timestamp"

    #
    # Attach all necessary columns to the common data frame
    #

    # Attach kline data frame
    df_out = df_out.join(k_df)

    # Attach futur data frame by also renaming columns
    f_df = f_df.rename(lambda x: futur_column_prefix + x if x != "timestamp" else x, axis='columns')
    df_out = df_out.join(f_df)

    # Attach several depth data frames using the same columns
    for i, df in enumerate(d_dfs):
        if i == 0:
            df_out = df_out.join(df)
        else:
            df_out.update(df)

    #
    # Store file with features
    #
    out_path = Path(out_file_name).absolute()

    df_out.to_csv(out_path, index=True)  # float_format="%.6f"
    print(f"Stored output merged file with {len(df_out)} records. Range: ({start}, {end})")

    elapsed = datetime.now() - start_dt
    print(f"Finished processing in {int(elapsed.total_seconds())} seconds.")
    print(f"Output file location: {out_path}")


if __name__ == '__main__':

    main()

    pass
