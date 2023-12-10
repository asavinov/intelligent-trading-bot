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
from common.gen_features import *
from common.depth_processing import *

"""
Produce features from market depth (json) data for a set of files by writing the result in several output (csv) files.
Given a file with depth data, produce a time-series file with features extracted from the depth data.
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

Parameters defined in feature_generation::depth_to_df():
- bin_size=1.0
- windows=[1, 2, 5, 10, 20]
Output columns:
-timestamp, end of latest 1m interval (time of request)
-gap, - difference between ask and bid
-price, - middle price between ask and bid
-bids_1,asks_1, - abs volume in BTC per X price bins (defined by bin_size)
-bids_2,asks_2,
-bids_5,asks_5,
-bids_10,asks_10,
-bids_20,asks_20


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

Possible features:
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
"""

"""
Statistics retrieved using find_depth_statistics():
Bid spans: min=3.68, max=339.85, mean=18.37
Ask spans: min=2.19, max=378.55, mean=17.23
Bid lens: min=100.00, max=100.00, mean=100.00
Ask lens: min=100.00, max=100.00, mean=100.00
Bid vols: min=8.46, max=1141.25, mean=68.83
Ask vols: min=5.92, max=915.25, mean=66.85

Meaningful bin_size: 1 USDT or 2 USDT
Meaningful windows: for 1 size: 1, 2, 5, 10, for 2 size: 1, 2, 5
"""

symbol = "BTCUSDT"  # BTCUSDT ETHBTC IOTAUSDT

in_path_name = r"C:\DATA2\BITCOIN\COLLECTED\DEPTH\batch6-partial-till-0307"
#in_path_name = r"C:\DATA2\BITCOIN\COLLECTED\DEPTH\_test_"


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


def find_depth_statistics():
    """Utility to research the depth data by computing: price span (min, max, mean)"""

    paths = get_symbol_files(symbol)
    bad_lines = 0
    bid_spans = []
    ask_spans = []
    bid_lens = []
    ask_lens = []
    bid_vols = []
    ask_vols = []
    for path in paths:

        with open(path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except:
                    bad_lines += 1
                    continue
                # File can contain error lines which we skip
                if not entry.get("bids") or not entry.get("asks"):
                    bad_lines += 1
                    continue
                # Replace all price-volume strings by floats
                bids = [float(x[0]) for x in entry.get("bids")]
                asks = [float(x[0]) for x in entry.get("asks")]
                bid_spans.append(np.max(bids) - np.min(bids))
                ask_spans.append(np.max(asks) - np.min(asks))
                bid_lens.append(len(bids))
                ask_lens.append(len(asks))
                bid_vols.append(np.sum([float(x[1]) for x in entry.get("bids")]))
                ask_vols.append(np.sum([float(x[1]) for x in entry.get("asks")]))

    print(f"Bid spans: min={np.min(bid_spans):.2f}, max={np.max(bid_spans):.2f}, mean={np.mean(bid_spans):.2f}")
    print(f"Ask spans: min={np.min(ask_spans):.2f}, max={np.max(ask_spans):.2f}, mean={np.mean(ask_spans):.2f}")
    print(f"Bid lens: min={np.min(bid_lens):.2f}, max={np.max(bid_lens):.2f}, mean={np.mean(bid_lens):.2f}")
    print(f"Ask lens: min={np.min(ask_lens):.2f}, max={np.max(ask_lens):.2f}, mean={np.mean(ask_lens):.2f}")
    print(f"Bid vols: min={np.min(bid_vols):.2f}, max={np.max(bid_vols):.2f}, mean={np.mean(bid_vols):.2f}")
    print(f"Ask vols: min={np.min(ask_vols):.2f}, max={np.max(ask_vols):.2f}, mean={np.mean(ask_vols):.2f}")
    print(f"Bad lines: {bad_lines}")


def main(args=None):

    start_dt = datetime.now()
    print(f"Start processing...")

    paths = get_symbol_files(symbol)
    for path in paths:

        # Load file as a list of dict records
        bad_lines = 0
        table = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except:
                    bad_lines += 1
                    continue
                # File can contain error lines which we skip
                if not entry.get("bids") or not entry.get("asks"):
                    bad_lines += 1
                    continue
                # If it is not 1m data then skip
                timestamp = entry.get("timestamp")
                if timestamp % 60_000 != 0:
                    continue
                # Replace all price-volume strings by floats
                bids = [[float(x[0]), float(x[1])] for x in entry.get("bids")]
                asks = [[float(x[0]), float(x[1])] for x in entry.get("asks")]
                entry["bids"] = bids
                entry["asks"] = asks
                table.append(entry)

        # Transform json table to data frame with features
        # ---
        df = depth_to_df(table)
        # ---
        df = df.reset_index().rename(columns={"index": "timestamp"})

        # Make timestamp conform to klines: it has to be start of 1m interval (and not end as it is in collected depth data)
        df["timestamp"] = df["timestamp"].shift(periods=1)  # Move forward (down) - use previous timestamp

        # Store file with features
        df.to_csv(path.with_suffix('.csv').name, index=False, float_format="%.4f")

        print(f"Finished processing file: {path}")

        print(f"Bad lines: {bad_lines}")

        pass

    elapsed = datetime.now() - start_dt
    print(f"Finished processing in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':

    #start = pd.to_datetime(1576324740000, unit='ms')
    #datetime.fromtimestamp(float(1576324740000) / 1e3, tz=pytz.UTC)

    #find_depth_statistics()

    main()

    pass
