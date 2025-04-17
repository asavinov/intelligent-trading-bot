import logging
import re
import dateparser
import pytz
from datetime import datetime, timezone, timedelta
from typing import Union, List
import json
from decimal import *

import numpy as np
import pandas as pd

import MetaTrader5 as mt5

from apscheduler.triggers.cron import CronTrigger

from binance.helpers import date_to_milliseconds, interval_to_milliseconds

from common.gen_features import *


logger = logging.getLogger(__name__)

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
# Binance specific
#

def klines_to_df(klines, df):

    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    dtypes = {
        'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64',
        'close_time': 'int64',
        'quote_av': 'float64',
        'trades': 'int64',
        'tb_base_av': 'float64',
        'tb_quote_av': 'float64',
        'ignore': 'float64',
    }
    data = data.astype(dtypes)

    if df is None or len(df) == 0:
        df = data
    else:
        df = pd.concat([df, data])

    # Drop duplicates
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    #df = df[~df.index.duplicated(keep='last')]  # alternatively, drop duplicates in index

    df.set_index('timestamp', inplace=True)

    return df


def binance_klines_to_df(klines: list):
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

    if "timestamp" in df.columns:
        df.set_index('timestamp', inplace=True)

    return df


def mt5_freq_from_pandas(freq: str) -> int:
    """
    Dynamically map pandas frequency strings to MetaTrader5 API timeframe constants.

    Handles inputs like '1min', '15min', '1h', '4h', '1D', 'D', '1W', 'W', '1MS', 'MS'.

    :param freq: pandas frequency string (e.g., '5min', '1h', '1D').
                 See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    :return: Corresponding MetaTrader5 TIMEFRAME_* constant (integer).
             See https://www.mql5.com/en/docs/integration/python_metatrader5/
    :raises ValueError: If the frequency string is not recognized or the corresponding
                        MT5 constant cannot be found.
    """
    # Map Pandas units (lowercase) to MT5 prefixes and whether they always imply '1'
    unit_map = {
        'min': ('M', False),
        'h':   ('H', False),
        'd':   ('D', True),
        'w':   ('W', True),
        'ms':  ('MN', True), # Month Start maps to MN1
    }

    # Try to match pattern: optional number + unit letters
    match = re.fullmatch(r"(\d+)?([A-Za-z]+)", str(freq))

    if not match:
        raise ValueError(f"Input frequency '{freq}' does not match expected format (e.g., '1min', '4h', '1D').")

    num_str, unit_pandas_raw = match.groups()
    unit_pandas = unit_pandas_raw.lower() # Normalize unit to lower case for map lookup

    # Find the corresponding MT5 unit info
    mt5_prefix, is_always_one = None, False
    found_unit = False
    if unit_pandas == 'min':
         mt5_prefix, is_always_one = unit_map['min']
         found_unit = True
    elif unit_pandas == 'h':
         mt5_prefix, is_always_one = unit_map['h']
         found_unit = True
    # Use original case for D, W, MS check as they are distinct in Pandas
    elif unit_pandas_raw == 'D':
         mt5_prefix, is_always_one = unit_map['d'] # map key is lowercase
         found_unit = True
    elif unit_pandas_raw == 'W':
         mt5_prefix, is_always_one = unit_map['w'] # map key is lowercase
         found_unit = True
    elif unit_pandas_raw == 'MS':
         mt5_prefix, is_always_one = unit_map['ms'] # map key is lowercase
         found_unit = True

    if not found_unit:
         raise ValueError(f"Unsupported Pandas frequency unit '{unit_pandas_raw}' in '{freq}'.")

    # Determine the number part
    if is_always_one:
        number = 1
    elif num_str:
        number = int(num_str)
    else:
        # If number is missing for min/h (e.g., 'h'), assume 1
        number = 1

    # Construct the MT5 constant name (e.g., "TIMEFRAME_M15", "TIMEFRAME_H4", "TIMEFRAME_D1")
    mt5_constant_name = f"TIMEFRAME_{mt5_prefix}{number}"

    # Retrieve the constant value from the mt5 module
    try:
        return getattr(mt5, mt5_constant_name)
    except AttributeError:
        # Provide a more informative error if the constant doesn't exist
        supported_timeframes = [tf for tf_name, tf in mt5.__dict__.items() if tf_name.startswith('TIMEFRAME_')]
        raise ValueError(
            f"Could not find or map MetaTrader5 constant '{mt5_constant_name}' for frequency '{freq}'. "
            f"Check if this timeframe is supported by the MetaTrader5 library/API. "
            f"Available TIMEFRAME constants might include: {sorted(list(set(supported_timeframes)))}"
        )


def get_timedelta_for_mt5_timeframe(mt5_timeframe: int, count: int) -> timedelta:
    """
    Calculate the total duration corresponding to 'count' bars
    of the specified MT5 timeframe constant.

    Internally maintains a cache of parsed timeframe details
    and a compiled regex for parsing attribute names.

    :param mt5_timeframe: MT5 constant (e.g., mt5.TIMEFRAME_M15)
    :param count: Number of bars
    :return: timedelta representing the aggregated duration
    :raises ValueError: If the timeframe is unknown or unsupported
    """
    # Initialize static attributes on the function for cache and pattern
    if not hasattr(get_timedelta_for_timeframe, "_pattern"):
        # Compile regex once
        get_timedelta_for_timeframe._pattern = re.compile(r"TIMEFRAME_([A-Z]+)(\d+)$")
        # Build cache mapping MT5 timeframe constants to (name, unit, number)
        cache: dict[int, tuple[str, str, int]] = {}
        for attr_name, attr_value in mt5.__dict__.items():
            if not attr_name.startswith("TIMEFRAME_") or not isinstance(attr_value, int):
                continue
            match = get_timedelta_for_timeframe._pattern.match(attr_name)
            if match:
                unit_prefix, number_str = match.groups()
                cache[attr_value] = (attr_name, unit_prefix, int(number_str))
            elif attr_name == "TIMEFRAME_MN1":
                # Special case for monthly timeframe without explicit number
                cache[attr_value] = (attr_name, "MN", 1)
        get_timedelta_for_timeframe._cache = cache
        logger.debug("Initialized MT5 timeframe pattern and cache")

    # Retrieve static attributes
    pattern = get_timedelta_for_timeframe._pattern
    cache = get_timedelta_for_timeframe._cache

    details = cache.get(mt5_timeframe)
    if details is None:
        raise ValueError(f"Unknown MetaTrader5 timeframe constant: {mt5_timeframe}")

    name, unit_prefix, number = details

    # Mapping of unit prefix to a factory function returning a timedelta
    unit_to_timedelta = {
        'M': lambda n, c: timedelta(minutes=n * c),
        'H': lambda n, c: timedelta(hours=n * c),
        'D': lambda n, c: timedelta(days=n * c),
        'W': lambda n, c: timedelta(weeks=n * c),
        'MN': lambda n, c: timedelta(days=n * c * 30.5),  # approximate month
    }

    factory = unit_to_timedelta.get(unit_prefix)
    if factory is None:
        raise ValueError(f"Unsupported timeframe unit '{unit_prefix}' derived from {name}")

    if unit_prefix == 'MN':
        logger.warning("Using approximate duration of 30.5 days for monthly timeframes.")

    return factory(number, count)


def binance_freq_from_pandas(freq: str) -> str:
    """
    Map pandas frequency to binance API frequency

    :param freq: pandas frequency https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    :return: binance frequency https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/Kline-Candlestick-Data
    """
    if freq.endswith("min"):  # Binance: 1m, 3m, 5m, 15m, 30m
        freq = freq.replace("min", "m")
    elif freq.endswith("D"):
        freq = freq.replace("D", "d")  # Binance: 1d, 3d
    elif freq.endswith("W"):
        freq = freq.replace("W", "w")
    elif freq == "BMS":
        freq = freq.replace("BMS", "M")

    if len(freq) == 1:
        freq = "1" + freq

    if not (2 <= len(freq) <= 3) or not freq[:-1].isdigit() or freq[-1] not in ["m", "h", "d", "w", "M"]:
        raise ValueError(f"Not supported Binance frequency {freq}. It should be one or two digits followed by a character.")

    return freq


def binance_get_interval(freq: str, timestamp: int=None):
    """
    Return a triple of interval start (including), end (excluding) in milliseconds for the specified timestamp or now

    INFO:
    https://github.com/sammchardy/python-binance/blob/master/binance/helpers.py
        interval_to_milliseconds(interval) - binance freq string (like 1m) to millis

    :param freq: binance frequency https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/Kline-Candlestick-Data
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


def pandas_get_interval(freq: str, timestamp: int=None):
    """
    Find a discrete interval for the given timestamp and return its start and end.

    :param freq: pandas frequency
    :param timestamp: milliseconds (13 digits)
    :return: triple (start, end)
    """
    if not timestamp:
        timestamp = int(datetime.now(pytz.utc).timestamp())  # seconds (10 digits)
        # Alternatively: timestamp = int(datetime.utcnow().replace(tzinfo=pytz.utc).timestamp())
    elif isinstance(timestamp, datetime):
        timestamp = int(timestamp.replace(tzinfo=pytz.utc).timestamp())
    elif isinstance(timestamp, int):
        pass
    else:
        ValueError(f"Error converting timestamp {timestamp} to millis. Unknown type {type(timestamp)} ")

    # Interval length for the given frequency
    interval_length_sec = pandas_interval_length_ms(freq) / 1000

    start = (timestamp // interval_length_sec) * interval_length_sec
    end = start + interval_length_sec

    return int(start*1000), int(end*1000)


def pandas_interval_length_ms(freq: str):
    return int(pd.Timedelta(freq).to_pytimedelta().total_seconds() * 1000)

#
# Date and time
#

def freq_to_CronTrigger(freq: str):
    # Add small time after interval end to get a complete interval from the server
    if freq.endswith("min"):
        if freq[:-3] == "1":
            trigger = CronTrigger(minute="*", second="1", timezone="UTC")
        else:
            trigger = CronTrigger(minute="*/" + freq[:-3], second="1", timezone="UTC")
    elif freq.endswith("h"):
        if freq[:-1] == "1":
            trigger = CronTrigger(hour="*", minute="0", second="2", timezone="UTC")
        else:
            trigger = CronTrigger(hour="*/" + freq[:-1], minute="0", second="2", timezone="UTC")
    elif freq.endswith("D"):
        if freq[:-1] == "1":
            trigger = CronTrigger(day="*", second="5", timezone="UTC")
        else:
            trigger = CronTrigger(day="*/" + freq[:-1], second="5", timezone="UTC")
    elif freq.endswith("W"):
        if freq[:-1] == "1":
            trigger = CronTrigger(week="*", second="10", timezone="UTC")
        else:
            trigger = CronTrigger(day="*/" + freq[:-1], second="10", timezone="UTC")
    elif freq.endswith("MS"):
        if freq[:-2] == "1":
            trigger = CronTrigger(month="*", second="30", timezone="UTC")
        else:
            trigger = CronTrigger(month="*/" + freq[:-1], second="30", timezone="UTC")
    else:
        raise ValueError(f"Cannot convert frequency '{freq}' to cron.")

    return trigger


def now_timestamp():
    """
    INFO:
    https://github.com/sammchardy/python-binance/blob/master/binance/helpers.py
    date_to_milliseconds(date_str) - UTC date string to millis

    :return: timestamp in millis
    :rtype: int
    """
    return int(datetime.utcnow().replace(tzinfo=timezone.utc).timestamp() * 1000)


def find_index(df: pd.DataFrame, date_str: str, column_name: str = "timestamp"):
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

    if res is None or len(res) == 0:
        raise ValueError(f"Cannot find date '{date_str}' in the column '{column_name}'. Either it does not exist or wrong format")

    id = res.index[0]

    return id


def notnull_tail_rows(df):
    """Maximum number of tail rows without nulls."""

    nan_df = df.isnull()
    nan_cols = nan_df.any()  # Series with columns having at least one NaN
    nan_cols = nan_cols[nan_cols].index.to_list()
    if len(nan_cols) == 0:
        return len(df)

    # Indexes of last NaN for all columns and then their minimum
    tail_rows = nan_df[nan_cols].values[::-1].argmax(axis=0).min()

    return tail_rows

#
# System etc.
#

def resolve_generator_name(gen_name: str):
    """
    Resolve the specified name to a function reference.
    Fully qualified name consists of module name and function name separated by a colon,
    for example:  'mod1.mod2.mod3:my_func'.

    Example: fn = resolve_generator_name("common.gen_features_topbot:generate_labels_topbot3")
    """

    mod_and_func = gen_name.split(':', 1)
    mod_name = mod_and_func[0] if len(mod_and_func) > 1 else None
    func_name = mod_and_func[-1]

    if not mod_name:
        return None

    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return None
    if mod is None:
        return None

    try:
        func = getattr(mod, func_name)
    except AttributeError as e:
        return None

    return func


def double_columns(df, shifts: List[int]):
    """
    Use previous rows as features appended to this row. This allows us to move history to the current time.
    One limitation is that this function will duplicate *all* features and only using the explicitly specified list of offsets.
    """
    if not shifts:
        return df
    df_list = [df.shift(shift) for shift in shifts]
    df_list.insert(0, df)
    max_shift = max(shifts)

    # Shift and add same columns
    df_out = pd.concat(df_list, axis=1)  # keys=('A', 'B')

    return df_out
