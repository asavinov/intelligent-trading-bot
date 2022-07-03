import dateparser
import pytz
from datetime import datetime, timezone, timedelta
from typing import Union, List
import json
from decimal import *

import numpy as np
import pandas as pd

from binance.helpers import date_to_milliseconds, interval_to_milliseconds

from common.feature_generation import *

#
# Decimals
#

def to_decimal(value):
    """Convert to a decimal with the required precision. The value can be string, float or decimal."""
    # Possible cases: string, 4.1-e7, float like 0.1999999999999 (=0.2), Decimal('4.1E-7')

    # App.config["trade"]["symbol_info"]["baseAssetPrecision"]

    n = 8
    rr = Decimal(1) / (Decimal(10) ** n)  # Result: 0.00000001
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_DOWN)
    return ret


def round_str(value, digits):
    rr = Decimal(1) / (Decimal(10) ** digits)  # Result for 8 digits: 0.00000001
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_HALF_UP)
    return f"{ret:.{digits}f}"


def round_down_str(value, digits):
    rr = Decimal(1) / (Decimal(10) ** digits)  # Result for 8 digits: 0.00000001
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_DOWN)
    return f"{ret:.{digits}f}"


#
# Date and time
#

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
