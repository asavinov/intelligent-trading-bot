import dateparser
import pytz
from datetime import datetime, timezone, timedelta
from typing import Union
import json
from decimal import *

import numpy as np
import pandas as pd


#
# Feature generation
#

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

#
# Utils
#

def price_to_volume(side, depth, price_limit):
    """
    Given limit, compute the available volume from the depth data on the specified side.
    The limit is inclusive.
    Bids (buyers) are on the left of X and asks (sellers) are on the right of X.

    :return: volume if limit is in the book and None otherwise
    :rtype: float
    """
    if side == "buy":
        orders = depth.get("asks", [])  # Sellers. Prices increase
        orders = [o for o in orders if o[0] <= price_limit]  # Select low prices
    elif side == "sell":
        orders = depth.get("bids", [])  # Buyers. Prices decrease
        orders = [o for o in orders if o[0] >= price_limit]  # Select high prices
    else:
        return None

    return orders[-1][1]  # Last element contains cumulative volume


def volume_to_price(side, depth, volume_limit):
    """
    Given volume, compute the corresponding limit from the depth data on the specified side.

    :return: limit if volume is available in book and None otherwise
    :rtype: float
    """
    if side == "buy":
        orders = depth.get("asks", [])  # Sellers. Prices increase
    elif side == "sell":
        orders = depth.get("bids", [])  # Buyers. Prices decrease
    else:
        return None

    orders = [o for o in orders if o[1] <= volume_limit]
    return orders[-1][0]  # Last element contains cumulative volume


def depth_accumulate(depth: list, start, end):
    """
    Convert a list of bid/ask volumes into an accumulated (monotonically increasing) volume curve.
    The result is the same list but each volume value in the pair is the sum of all previous volumes.
    For the very first bid/ask, the volume is that same.
    """
    prev_value = 0.0
    for point in depth:
        point[1] += prev_value
        prev_value = point[1]

    return depth


def discretize(side: str, depth: list, bin_size: float, start: float):
    """
    Main problem: current point can contribute to this bin (till bin end) and next bin (from bin end till next point)
    Iterate over bins. For each iteration, initial function value must be provided which works till first point or end
      With each bin iteration, iterate over points (global pointer).
      If point within this bin, the set current volume instead of initial, and compute contribution of the previous value
      If point in next bin, then still use current volume for the next bin, compute contribution till end only. Do not iterate point (it is needed when starting next bin)
        When we start next bin, compute contribution

    :param side:
    :param depth:
    :param bin_size:
    :param start:
    :return:
    """
    if side.startswith("ask") or side.startswith("sell"):
        price_increase = True
    elif side in ["bid", "buy"]:
        price_increase = False
    else:
        print("Wrong use. Side is either bid or ask.")

    # Start is either explict or first point
    if start is None:
        start = depth[0][0]  # First point

    # End covers the last point
    bin_count = int(abs(depth[-1][0] - start) // bin_size) + 1
    all_bins_length = bin_count * bin_size
    end = start + all_bins_length if price_increase else start - all_bins_length

    bin_volumes = []
    for b in range(bin_count):
        bin_start = start + b*bin_size if price_increase else start - b*bin_size
        bin_end = bin_start + bin_size if price_increase else bin_start - bin_size

        # Find point ids within this bin
        if price_increase:
            bin_point_ids = [i for i, x in enumerate(depth) if bin_start <= x[0] < bin_end]
        else:
            bin_point_ids = [i for i, x in enumerate(depth) if bin_end < x[0] <= bin_start]

        if bin_point_ids:
            first_point_id = min(bin_point_ids)
            last_point_id = max(bin_point_ids)
            prev_point = depth[first_point_id-1] if first_point_id >= 1 else None
        else:
            first_point_id = None
            last_point_id = None

        #
        # Iterate over points in this bin by collecting their contribution using previous interval
        #
        prev_price = bin_start
        prev_volume = prev_point[1] if prev_point else 0.0
        bin_volume = 0.0

        if first_point_id is None:  # Bin is empty
            # Update current bin volume
            price = bin_end
            price_delta = abs(price - prev_price)
            price_coeff = price_delta / bin_size  # Portion of this interval in bin
            bin_volume += prev_volume * price_coeff  # Each point in the bin contributes to this bin final value

            # Store current bin as finished
            bin_volumes.append(bin_volume)

            continue

        # Bin is not empty
        for point_id in range(first_point_id, last_point_id+1):
            point = depth[point_id]

            # Update current bin volume
            price = point[0]
            price_delta = abs(price - prev_price)
            price_coeff = price_delta / bin_size  # Portion of this interval in bin
            bin_volume += prev_volume * price_coeff  # Each point in the bin contributes to this bin final value

            # Iterate
            prev_price = point[0]
            prev_volume = point[1]
            prev_point = point
        #
        # Last point contributes till the end of this bin
        #
        # Update current bin volume
        price = bin_end
        price_delta = abs(price - prev_price)
        price_coeff = price_delta / bin_size  # Portion of this interval in bin
        bin_volume += prev_volume * price_coeff  # Each point in the bin contributes to this bin final value

        # Store current bin as finished
        bin_volumes.append(bin_volume)

    return bin_volumes


# OBSOLETE: Because works only for increasing prices (ask). Use general version instead.
def discretize_ask(depth: list, bin_size: float, start: float):
    """
    Find (volume) area between the specified interval (of prices) given the step function volume(price).

    The step-function is represented as list of points (price,volume) ordered by price.
    Volume is the function value for the next step (next price delta - not previous one). A point specifies volume till the next point.

    One bin has coefficient 1 and then all sub-intervals within one bin are coefficients to volume

    Criterion: whole volume area computed for the input data and output data (for the same price interval) must be the same

    side: "ask" (prices in depth list increase) or "bid" (prices in depth list decrease)

    TODO: It works only for increasing prices (asks). It is necessary to make it work also for decreasing prices.
    TODO: it does not work if start is after first point (only if before or equal/none)
    """
    if start is None:
        start = depth[0][0]  # First point

    prev_point = [start, 0.0]

    bin_start = start
    bin_end = bin_start + bin_size
    bin_volume = 0.0

    bin_volumes = []
    for i, point in enumerate(depth):
        if point[0] <= bin_start:  # Point belongs to previous bin (when start is in the middle of series)
            prev_point = point
            continue

        if point[0] >= bin_end:  # Point in the next bin
            price = bin_end
        else:  # Point within bin
            price = point[0]

        # Update current bin volume
        price_delta = abs(price - prev_point[0])
        price_coeff = price_delta / bin_size  # Portion of this interval in bin
        bin_volume += prev_point[1] * price_coeff  # Each point in the bin contributes to this bin final value

        # Iterate bin (if current is finished)
        if point[0] >= bin_end:  # Point in the next bin
            # Store current bin as finished
            bin_volumes.append(bin_volume)
            # Iterate to next bin
            bin_start = bin_end
            bin_end = bin_start + bin_size
            bin_volume = 0.0

            price = point[0]

            # Initialize bin volume with the rest of current point
            price_delta = abs(price - bin_start)
            price_coeff = price_delta / bin_size  # Portion of this interval in bin
            bin_volume += prev_point[1] * price_coeff  # Each point in the bin contributes to this bin final value

        # Iterate point
        prev_point = point

    #
    # Finalize by closing last bin which does not have enough points
    #
    price = bin_end

    # Update current bin volume
    price_delta = abs(price - prev_point[0])
    price_coeff = price_delta / bin_size  # Portion of this interval in bin
    bin_volume += prev_point[1] * price_coeff  # Each point in the bin contributes to this bin final value

    # Store current bin as finished
    bin_volumes.append(bin_volume)

    return bin_volumes


def mean_volumes(depth: list, windows: list, bin_size: 1.0):
    """
    Density. Mean volume per price unit (bin) computed using the specified number of price bins.
    First, we discreteize and then find average value for the first element (all if length is not specified).
    Return a list of values each value being a mean volume for one aggregation window (number of bins)
    """

    bid_volumes = discretize(side="bid", depth=depth.get("bids"), bin_size=bin_size, start=None)
    ask_volumes = discretize(side="ask", depth=depth.get("asks"), bin_size=bin_size, start=None)

    ret = {}
    for length in windows:
        density = np.nanmean(bid_volumes[0:min(length, len(bid_volumes))])
        feature_name = f"bids_{length}"
        ret[feature_name] = density

        density = np.nanmean(ask_volumes[0:min(length, len(ask_volumes))])
        feature_name = f"asks_{length}"
        ret[feature_name] = density

    return ret
