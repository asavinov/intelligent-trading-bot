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
