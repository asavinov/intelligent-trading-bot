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

from trade.utils import *

"""
Given a file with depth data, produce a time-series file with feature extracted from the depth data
Input depth data:
- [json] One line is a json object with timestamp and two time-series "bids" and "asks"
- [gaps] It is not necessarily a regular time series and may have gaps (for whatever reason)
- [higher frequency] The depth records may have higher frequency so the data needs to be filtered
- [varying depth] the depth (length of ask/bid time series) can theoretically vary so we need at least check it (for output consistency)
Output file:
- [csv] output has a fixed header and column values
- [base features] these are feature directly derived from the depth data
- [aggregated features] these features are derived from many rows (but cannot be derived from base features)
- [time range] one file with full time range but (optionally) removed null records (that is, with gaps) 
Features:
- [ask_area, bid_area] area under the asks and bids curves - smoothing for some intervals
- [ba_ratio] ratio between two areas - smoothing for some intervals - can be derived
- [ask_slope, bid_slope] - angle of the line computed either via regression or using min and max 
- do we need and is it meaningful to process price and volume information separately or together?

Tasks:
- specify input parameters: symbol(s), file(s), frequency (currently fixed), whether combine or separately, delete gaps
- process individual files:
  - load files
  - for each file, create a data frame with necessary index from start to end
  - compute features for each row. each feature has a function which takes json object as input and returns a value
- combine all file data frames (if necessary to store in one combined file:
  - create one index with start-end for all files
  - assign data from individual files to this common data frame using time index for matching
- store one output or individual files

Features:
Once we defined a box with absolute values, we can plot data (data might be more than the box or not enough for the box).
We can compute the following characteristics:

- gap

- bid/ask_density_xx bid/ask_volume_xx
  mean volume per price unit/bin
  param: number of price bins

- bid/ask_density_std_xx
  standard deviation of volume

- price_span_0, bid_span_20, ask_span_20: 
price span for this fixed number of items (say, 10, 20, 50). It is easy: price N-th element minus price 0-th element)
0 means gap. Note that the number if fixed price units

- bid_slope_20: either regression or between min and max. Note that in order to compare the slope for different timestamps or averaging,
  we need to have the same box or otherwise comparable values, that is,
  the units of slope have to be the same, say, Volume increase per Price unit (average for the whole price interval).

- bid_slope_std_20 deviation from linear growth
  It is deviation from linear line. We can measure linear rate (average growth per price unit).
  And then find average deviation from this average value. It will reflect the deviation from linear growth.

- bid_area_20 which characterizes second derivative. Note that the areas for different timestamps have to be measured in the same units.
  For example, sum of volumes for fixed price interval. Yet, it will be essentially growth rate.
  What we really want to measure is second derivative when it first grows quickly and the slowly.
  However, we would like to measure how convex the growth is: first fast growth and then slow or vice versa - it is essentially area.
  We can measure area with respect to the whole box. For that purpose, we need to find max volume which volum of the last price.
  Essentially, this ratio is average ration between volume and (max - volume) for each price bin (probably)

Example data:

bids - buy (lower prices), --> prices drop
"bids": [["7185.49000000", "0.01663600"], ["7185.34000000", "0.00163600"], ["7185.25000000", "0.17722200"],
[7166.66000000, 7185.49000000] = 18,83 - price span

asks - sell (higher prices), --> prices grow
"asks": [["7186.47000000", "0.11475300"], ["7186.49000000", "0.04812900"], ["7186.52000000", "2.00000000"],
[7186.47000000, 7198.31000000] = 11,84 price span

Meaningful values of price bins: 0.5 (wo we will get >20 bins) or 1 USDT
Meaningful values of price bin numbers for averaging: 1, 2, 5, 10

"""

symbol = "BTCUSDT"  # BTCUSDT ETHBTC IOTAUSDT
in_path_name = r"C:\DATA2\BITCOIN\COLLECTED\DEPTH"
bin_size = 1.0  # In USDT
windows = [1, 2, 5, 10, 20]  # No of price bins for aggregate/smoothing

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

#
# Feature definitions
#

def mean_volume(depth: list, windows: list):
    """
    Density. Mean volume per price unit (bin) computed using the specified number of price bins.
    First, we discreteize and then find average value for the first element (all if length is not specified).
    Return a list of values each value being a mean volume for one aggregation window (number of bins)
    """
    volumes = discretize(depth=depth, bin_size=bin_size, start=None)
    if not windows:
        windows = [len(volumes)]

    ret = []
    for length in windows:
        density = np.mean(volumes[0:min(length, len(volumes))])

    return ret

#
# Data processing
#

def depth_to_df(depth: list):
    """
    Process input list of depth records and return a data frame with computed features.
    The list is supposed to loaded from a file so records are ordered but gaps are not excluded.

    # TODO Questions:
    # !!! - what is depth timestamp: previous 1m interval start or end? - is it like kline identifier?
    # !!! - what is zone for our timestamps - ensure that it is the same as Binance server
    # - is it possible to create a data frame with a column containing json object or string?
    # - how to match json/string values with data frame index?
    """

    #
    # Find start and end dates
    #
    # NOTE: timestamp is request time (in our implementation) and hence it is end of 1m interval while kline id is start of 1m inteval
    #  It is important for matching, so maybe align this difference here by shifting data
    start_line = depth[0]
    end_line = depth[-1]
    start_ts = start_line.get("timestamp")
    end_ts = end_line.get("timestamp")

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

    #
    # For each depth record from the list, compute features from its depth data
    #
    for entry in depth:
        ts = entry.get("timestamp")
        bids = entry.get("bids")
        asks = entry.get("asks")

        gap = asks[0][0] - bids[0][0]
        if gap < 0: gap = 0
        price = bids[0][0] + (gap/2)

        # Asks densities for all windows
        densities = mean_volume(depth=bids, windows=windows)

    pass

def main(args=None):

    start_dt = datetime.now()
    print(f"Start processing...")

    pass  # Do processing

    elapsed = datetime.now() - start_dt
    print(f"Finished processing in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':

    start = pd.to_datetime(1576324740000, unit='ms')

    datetime.fromtimestamp(float(1576324740000) / 1e3, tz=pytz.UTC)

    paths = get_symbol_files(symbol)

    main()

    pass
