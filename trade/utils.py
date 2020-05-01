from __future__ import annotations  # Eliminates problem with type annotations like list[int] and error "'type' object is not subscriptable"
import dateparser
import pytz
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import pandas as pd

from binance.helpers import date_to_milliseconds, interval_to_milliseconds

def get_interval(freq: str, timestamp: int=None):
    """
    Return a triple of interval start (including), end (excluding) in milliseconds for the specified timestamp or now

    INFO:
    https://github.com/sammchardy/python-binance/blob/master/binance/helpers.py
        interval_to_milliseconds(interval) - binance freq string (like 1m) to millis

    :return: tuple of start (inclusive) and end (exclusive) of the interval in millis
    :rtype: (int, int)
    """
    if not timestamp:
        timestamp = datetime.utcnow()  # datetime.now(timezone.utc)
    elif isinstance(timestamp, int):
        timestamp = pd.to_datetime(timestamp, unit='ms').to_pydatetime()

    # Although in 3.6 (at least), datetime.timestamp() assumes a timezone naive (tzinfo=None) datetime is in UTC
    timestamp = timestamp.replace(microsecond=0, tzinfo=timezone.utc)

    if freq == "1s":
        start = timestamp.timestamp()
        end = timestamp + timedelta(seconds=1)
        end = end.timestamp()
    elif freq == "5s":
        reference_timestamp = timestamp.replace(second=0)
        now_duration = timestamp - reference_timestamp

        freq_duration = timedelta(seconds=5)

        full_intervals_no = now_duration.total_seconds() // freq_duration.total_seconds()

        start = reference_timestamp + freq_duration * full_intervals_no
        end = start + freq_duration

        start = start.timestamp()
        end = end.timestamp()
    elif freq == "1m":
        timestamp = timestamp.replace(second=0)
        start = timestamp.timestamp()
        end = timestamp + timedelta(minutes=1)
        end = end.timestamp()
    elif freq == "5m":
        # Here we need to find 1 h border (or 1 day border) by removing minutes
        # Then divide (now-1hourstart) by 5 min interval length by finding 5 min border for now
        print(f"Frequency 5m not implemented.")
    elif freq == "1h":
        timestamp = timestamp.replace(minute=0, second=0)
        start = timestamp.timestamp()
        end = timestamp + timedelta(hours=1)
        end = end.timestamp()
    else:
        print(f"Unknown frequency.")

    return int(start * 1000), int(end * 1000)

def now_timestamp():
    """
    INFO:
    https://github.com/sammchardy/python-binance/blob/master/binance/helpers.py
    date_to_milliseconds(date_str) - UTC date string to millis

    :return: timestamp in millis
    :rtype: int
    """
    return int(datetime.utcnow().replace(tzinfo=timezone.utc).timestamp() * 1000)

def find_index(df: pd.DataFrame, date_str: str, column_name: str= "timestamp"):
    """
    Return index of the record with the specified datetime string.

    :return: row id in the input data frame which can be then used in iloc function
    :rtype: int
    """
    d = dateparser.parse(date_str)
    try:
        res = df[df[column_name] == d]
    except TypeError:  # "Cannot compare tz-naive and tz-aware datetime-like objects"
        # Change timezone (set UTC timezone or reset timezone)
        if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
            d = d.replace(tzinfo=pytz.utc)
        else:
            d = d.replace(tzinfo=None)

        # Repeat
        res = df[df[column_name] == d]

    id = res.index[0]

    return id

#
# Depth processing
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

def discretize(depth: list, bin_size: float):
    """
    Find (volume) area between the specified interval (of prices) given the step function volume(price).

    The step-function is represented as list of points (price,volume) ordered by price.
    Volume is the function value for the next step (next price delta - not previous one). A point specifies volume till the next point.

    One bin has coefficient 1 and then all subintervals within one bin are coeffocients to volume

    Criterion: whole volume area computed for the input data and output data (for the same price interval) must be the same
    """
    # Add bin points on the bin borders with volume equal to the previous point
    prev_point = depth[0]

    bin_start = depth[0][0]
    bin_end = bin_start + bin_size
    bin_volume = 0.0

    bins = []
    for i, point in enumerate(depth):
        if i == 0:
            continue

        if point[0] >= bin_end:  # Point in the next bin
            price = bin_end
        else:  # Point within bin
            price = point[0]

        # Update current bin volume
        price_delta = price - prev_point[0]
        price_coeff = price_delta / bin_size  # Portion of this interval in bin
        bin_volume += prev_point[1] * price_coeff  # Each point in the bin contributes to this bin final value

        # Iterate bin (if current is finished)
        if point[0] >= bin_end:  # Point in the next bin
            # Store current bin as finished
            bins.append([bin_start, bin_volume])
            # Iterate to next bin
            bin_start = bin_end
            bin_end = bin_start + bin_size
            bin_volume = 0.0

            price = point[0]

            # Initialize bin volume with the rest of current point
            price_delta = price - bin_start
            price_coeff = price_delta / bin_size  # Portion of this interval in bin
            bin_volume += prev_point[1] * price_coeff  # Each point in the bin contributes to this bin final value

        # Iterate point
        prev_point = point

    #
    # Finalize by closing last bin which does not have enough points
    #
    price = bin_end

    # Update current bin volume
    price_delta = price - prev_point[0]
    price_coeff = price_delta / bin_size  # Portion of this interval in bin
    bin_volume += prev_point[1] * price_coeff  # Each point in the bin contributes to this bin final value

    # Store current bin as finished
    bins.append([bin_start, bin_volume])

    return bins

#
# Klnes processing
#

def klines_to_df(klines: list):
    """
    Convert a list of klines to a data frame.
    """
    columns = [
        'timestamp',
        'open', 'high', 'low', 'close', 'volume',
        'close_time',
        'quote_av', 'trades', 'tb_base_av', 'tb_quote_av',
        'ignore'
    ]

    df = pd.DataFrame(klines, columns=columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    df["open"] = pd.to_numeric(df["open"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])

    df["quote_av"] = pd.to_numeric(df["quote_av"])
    df["trades"] = pd.to_numeric(df["trades"])
    df["tb_base_av"] = pd.to_numeric(df["tb_base_av"])
    df["tb_quote_av"] = pd.to_numeric(df["tb_quote_av"])

    df.set_index('timestamp', inplace=True)

    return df

#
# Feature/label generation utilities
#

def to_diff_NEW(sr):
    # TODO: Use an existing library function to compute difference
    #   We used it in fast hub for computing datetime difference - maybe we can use it for numeric diffs
    pass

def to_diff(sr):
    """
    Convert the specified input column to differences.
    Each value of the output series is equal to the difference between current and previous values divided by the current value.
    """

    def diff_fn(x):  # ndarray. last element is current row and first element is most old historic value
        return 100 * (x[1] - x[0]) / x[0]

    diff = sr.rolling(window=2, min_periods=2).apply(diff_fn, raw=True)
    return diff

def add_past_weighted_aggregations(df, column_name: str, weight_column_name: str, fn, windows: Union[int, list[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0):
    return _add_weighted_aggregations(df, False, column_name, weight_column_name, fn, windows, suffix, rel_column_name, rel_factor)

def add_past_aggregations(df, column_name: str, fn, windows: Union[int, list[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0):
    return _add_aggregations(df, False, column_name, fn, windows, suffix, rel_column_name, rel_factor)

def add_future_aggregations(df, column_name: str, fn, windows: Union[int, list[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0):
    return _add_aggregations(df, True, column_name, fn, windows, suffix, rel_column_name, rel_factor)
    #return _add_weighted_aggregations(df, True, column_name, None, fn, windows, suffix, rel_column_name, rel_factor)

def _add_aggregations(df, is_future: bool, column_name: str, fn, windows: Union[int, list[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0):
    """
    Compute moving aggregations over past or future values of the specified base column using the specified windows.

    Windowing. Window size is the number of elements to be aggregated.
    For past aggregations, the current value is always included in the window.
    For future aggregations, the current value is not included in the window.

    Naming. The result columns will start from the base column name then suffix is used and then window size is appended (separated by underscore).
    If suffix is not provided then it is function name.
    The produced names will be returned as a list.

    Relative values. If the base column is provided then the result is computed as a relative change.
    If the coefficient is provided then the result is multiplied by it.

    The result columns are added to the data frame (and their names are returned).
    The length of the data frame is not changed even if some result values are None.
    """

    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]

    if rel_column_name:
        rel_column = df[rel_column_name]

    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:
        # Aggregate
        ro = column.rolling(window=w, min_periods=max(1, w // 10))
        feature = ro.apply(fn, raw=True)

        # Convert past aggregation to future aggregation
        if is_future:
            feature = feature.shift(periods=-w)

        # Normalize
        feature_name = column_name + suffix + '_' + str(w)
        features.append(feature_name)
        if rel_column_name:
            df[feature_name] = rel_factor * (feature - rel_column) / rel_column
        else:
            df[feature_name] = rel_factor * feature

    return features

def _add_weighted_aggregations(df, is_future: bool, column_name: str, weight_column_name: str, fn, windows: Union[int, list[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0):
    """
    Weighted aggregation. Normally using np.sum function.
    """

    column = df[column_name]

    if weight_column_name:
        weight_column = df[weight_column_name]
    else:
        # If weight column is not specified then it is equal to constant 1.0
        weight_column = pd.Series(data=1.0, index=column.index)

    products_column = column * weight_column

    if isinstance(windows, int):
        windows = [windows]

    if rel_column_name:
        rel_column = df[rel_column_name]

    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:

        # Sum of products
        ro = products_column.rolling(window=w, min_periods=max(1, w // 10))
        feature = ro.apply(fn, raw=True)

        # Sum of weights
        w_ro = weight_column.rolling(window=w, min_periods=max(1, w // 10))
        weights = w_ro.apply(fn, raw=True)

        # Weighted feature
        feature = feature / weights

        # Convert past aggregation to future aggregation
        if is_future:
            feature = feature.shift(periods=-w)

        # Normalize
        feature_name = column_name + suffix + '_' + str(w)
        features.append(feature_name)
        if rel_column_name:
            df[feature_name] = rel_factor * (feature - rel_column) / rel_column
        else:
            df[feature_name] = rel_factor * feature

    return features

def add_threshold_feature(df, column_name: str, thresholds: list, out_names: list):
    """

    :param df:
    :param column_name: Column with values to compare with the thresholds
    :param thresholds: List of thresholds. For each of them an output column will be generated
    :param out_names: List of output column names (same length as thresholds)
    :return: List of output column names
    """

    for i, threshold in enumerate(thresholds):
        out_name = out_names[i]
        if threshold > 0.0:  # Max high
            if abs(threshold) >= 0.75:  # Large threshold
                df[out_name] = df[column_name] >= threshold  # At least one high is greater than the threshold
            else:  # Small threshold
                df[out_name] = df[column_name] <= threshold  # All highs are less than the threshold
        else:  # Min low
            if abs(threshold) >= 0.75:  # Large negative threshold
                df[out_name] = df[column_name] <= threshold  # At least one low is less than the (negative) threshold
            else:  # Small threshold
                df[out_name] = df[column_name] >= threshold  # All lows are greater than the (negative) threshold

    return out_names

# TODO: DEPRCATED: check that it is not used. Or refactor by using no apply (direct computation of relative) and remove row filter
def ___add_label_column(df, window, threshold, max_column_name='<HIGH>', ref_column_name='<CLOSE>',
                        out_column_name='label'):
    """
    Add a goal column to the dataframe which stores a label computed from future data.
    We take the column with maximum values and find its maximum for the specified window.
    Then we find relative deviation of this maximum from the value in the reference column.
    Finally, we compare this relative deviation with the specified threashold and write either 1 or 0 into the output.
    The resulted column with 1s and 0s is attached to the dataframe.
    """

    ro = df[max_column_name].rolling(window=window, min_periods=window)

    max = ro.max()  # Aggregate

    df['max'] = max.shift(periods=-window)  # Make it future max value (will be None if not enough history)

    # count = df.count()
    # labelnacount = df['label'].isna().sum()
    # nacount =  df.isna().sum()
    df.dropna(subset=['max', ref_column_name], inplace=True)  # Number of nans (at the end) is equal to the window size

    # Compute relative max value
    def relative_max_fn(row):
        # if np.isnan(row['max']) or np.isnan(row['<CLOSE>']):
        #    return None
        return 100 * (row['max'] - row[ref_column_name]) / row[ref_column_name]  # Percentage

    df['max'] = df.apply(relative_max_fn, axis=1)

    # Whether it exceeded the threshold
    df[out_column_name] = df.apply(
        lambda row: 1 if row['max'] > threshold else (0 if row['max'] <= threshold else None), axis=1)

    # Uncomment to use relative max as a numeric label
    # df['label'] = df['max']

    # df.drop(columns=['max'], inplace=True)  # Not needed anymore

    return df


if __name__ == "__main__":

    print(now_timestamp())

    # INFO:
    # timedelta(seconds=1)
    # timedelta(minutes=1))
    # timedelta(hours=1))
    # datetime.now()
    # datetime.utcnow()
    # datetime.today()
    # pd.to_datetime(df['Millisecond'], unit='ms')
    # datetime.datetime.fromtimestamp(milliseconds/1000.0)


    #raster = pd.date_range(start, end, tz=timezone.utc, normalize=True, closed="left", freq="1T")
    #raster = pd.interval_range(start, end, tz=timezone.utc, normalize=True, closed="left", freq="1S")
    #raster = pd.period_range(start, end, tz=timezone.utc, normalize=True, closed="left", freq="1S")


    ts = 1502942460000

    ts_now = datetime.now()
    ts_today =  datetime.today()
    ts_utcnow = datetime.utcnow()
    ts_dt = datetime.fromtimestamp(ts/1000.0)

    ts_pd = pd.to_datetime(ts, unit='ms')
    ts_pd = pd.to_datetime(ts, unit='ms', utc=True)

    pass
