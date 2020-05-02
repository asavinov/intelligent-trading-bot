import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

from trade.utils import *

"""
Feature/label generation.
These features are computed using explict transformations.
(True) labels are features computed from future data but stored as properties of the current row (in contrast to normal features which are computed from past data).
(Currently) feature/label generation is not based on (explicit) models - all parameters are hard-coded.
Also, no parameter training is performed.
"""

def generate_features(df, use_differences=False):
    """
    Generate derived features by adding them as new columns to the data frame.
    This (same) function must be used for both training and prediction.
    If we use it for training to produce models then this same function has to be used before applying these models.
    """
    # Parameters of moving averages
    windows = [1, 2, 5, 20, 60, 180]
    base_window = 300

    features = []
    to_drop = []

    if use_differences:
        df['close'] = to_diff(df['close'])
        df['volume'] = to_diff(df['volume'])
        df['trades'] = to_diff(df['trades'])

    # close mean
    weight_column_name = 'volume'  # Set None to use no weighting or 'volume' for volume average
    to_drop += add_past_weighted_aggregations(df, 'close', weight_column_name, np.mean, base_window, suffix='')  # Base column
    features += add_past_weighted_aggregations(df, 'close', weight_column_name, np.mean, windows, '', to_drop[-1], 100.0)
    # ['close_1', 'close_2', 'close_5', 'close_20', 'close_60', 'close_180']

    # close std
    to_drop += add_past_aggregations(df, 'close', np.std, base_window)  # Base column
    features += add_past_aggregations(df, 'close', np.std, windows, '_std', to_drop[-1], 100.0)
    # ['close_std_1', 'close_std_2', 'close_std_5', 'close_std_20', 'close_std_60', 'close_std_180']

    # volume mean
    to_drop += add_past_aggregations(df, 'volume', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'volume', np.mean, windows, '', to_drop[-1], 100.0)
    # ['volume_1', 'volume_2', 'volume_5', 'volume_20', 'volume_60', 'volume_180']

    # Span: high-low difference
    df['span'] = df['high'] - df['low']
    to_drop += add_past_aggregations(df, 'span', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'span', np.mean, windows, '', to_drop[-1], 100.0)
    # ['span_1', 'span_2', 'span_5', 'span_20', 'span_60', 'span_180']

    # Number of trades
    to_drop += add_past_aggregations(df, 'trades', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'trades', np.mean, windows, '', to_drop[-1], 100.0)
    # ['trades_1', 'trades_2', 'trades_5', 'trades_20', 'trades_60', 'trades_180']

    # tb_base_av / volume varies around 0.5 in base currency
    df['tb_base'] = df['tb_base_av'] / df['volume']
    to_drop.append('tb_base')
    to_drop += add_past_aggregations(df, 'tb_base', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'tb_base', np.mean, windows, '', to_drop[-1], 100.0)
    # ['tb_base_1', 'tb_base_2', 'tb_base_5', 'tb_base_20', 'tb_base_60', 'tb_base_180']

    # tb_quote_av / quote_av varies around 0.5 in quote currency
    df['tb_quote'] = df['tb_quote_av'] / df['quote_av']
    to_drop.append('tb_quote')
    to_drop += add_past_aggregations(df, 'tb_quote', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'tb_quote', np.mean, windows, '', to_drop[-1], 100.0)
    # ['tb_quote_1', 'tb_quote_2', 'tb_quote_5', 'tb_quote_20', 'tb_quote_60', 'tb_quote_180']

    df.drop(columns=to_drop, inplace=True)

    return features

def depth_to_df(depth: list):
    """
    Input is a list of json objects each representing current market depth with a list bids and asks
    The method computes features from the market depth and returns a data frame with the corresponding columns.

    NOTE:
    - Important: "timestamp" is real time of the depth data which corresponds to "close_time" in klines
      "timestamp" in klines is 1m before current time
      It has to be taken into account when matching/joining records, e.g., by shifting columns (if we match "timestamp" then the reslt will be wrong)
    - data frame index is continuous and may contain gaps. its start is first line and end is last line

    # TODO Questions:
    # !!! - what is zone for our timestamps - ensure that it is the same as Binance server
    # - is it possible to create a data frame with a column containing json object or string?
    # - how to match json/string values with data frame index?
    """
    bin_size = 1.0  # In USDT
    windows = [1, 2, 5, 10, 20]  # No of price bins for aggregate/smoothing

    #
    # Generate a table with feature records
    #
    table = []
    for entry in depth:
        record = depth_to_features(entry, windows, bin_size)
        table.append(record)

    #
    # Convert json table to data frame
    #
    df = pd.DataFrame.from_dict(table)
    # Alternatively, from_records() or json_normalize()

    # Timestamp is an index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df = df.set_index("timestamp")
    df = df.sort_index()

    #
    # Find start and end dates
    #
    # NOTE: timestamp is request time (in our implementation) and hence it is end of 1m interval while kline id is start of 1m inteval
    #  It is important for matching, so maybe align this difference here by shifting data
    start_line = depth[0]
    end_line = depth[-1]
    start_ts = start_line.get("timestamp")
    #start_ts -= 60_000  # To ensure that we do not lose any data
    end_ts = end_line.get("timestamp")
    #end_ts += 60_000  # To ensure that we do not lose any data

    #
    # Create index for this interval of timestamps
    #
    # NOTE: Add utc=True to get tz-aware object (with tz="UTC" instead of tz-unaware object with tz=None), so it seems that no tz means UTC
    start = pd.to_datetime(start_ts, unit='ms')
    end = pd.to_datetime(end_ts, unit='ms')

    # Alternatively:
    # If tz is not specified then 1 hour difference will be added so it seems that no tz means locale tz
    #datetime.fromtimestamp(float(start_ts) / 1e3, tz=pytz.UTC)

    # Create DatetimeIndex
    # NOTE: if tz is not specified then the index is tz-naive
    #   closed can be specified (which side to include/exclude: left, right or both). it influences if we want ot include/exclude start or end of the interval
    index = pd.date_range(start, end, freq="T")
    df_out = pd.DataFrame(index=index)

    #
    # Join data with this empty index (to ensure continuous range of timestamps)
    #
    df_out = df_out.join(df)

    return df_out

def depth_to_features(entry: list, windows: list, bin_size: float):
    """Convert one record of market depth to a dict of features"""

    bids = entry.get("bids")
    asks = entry.get("asks")

    timestamp = entry.get("timestamp")

    # Gap feature
    gap = asks[0][0] - bids[0][0]

    if gap < 0: gap = 0

    # Price feature
    price = bids[0][0] + (gap / 2)

    # Densities for bids and asks (volume per price unit)
    densities = mean_volumes(depth=entry, windows=windows, bin_size=bin_size)

    record = {"timestamp": timestamp, "gap": gap, "price": price}
    record.update(densities)

    return record


if __name__ == "__main__":
    pass
