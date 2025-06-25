import re
from datetime import datetime, timezone, timedelta
import logging

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


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
    if not hasattr(get_timedelta_for_mt5_timeframe, "_pattern"):
        # Compile regex once
        get_timedelta_for_mt5_timeframe._pattern = re.compile(r"TIMEFRAME_([A-Z]+)(\d+)$")
        # Build cache mapping MT5 timeframe constants to (name, unit, number)
        cache: dict[int, tuple[str, str, int]] = {}
        for attr_name, attr_value in mt5.__dict__.items():
            if not attr_name.startswith("TIMEFRAME_") or not isinstance(attr_value, int):
                continue
            match = get_timedelta_for_mt5_timeframe._pattern.match(attr_name)
            if match:
                unit_prefix, number_str = match.groups()
                cache[attr_value] = (attr_name, unit_prefix, int(number_str))
            elif attr_name == "TIMEFRAME_MN1":
                # Special case for monthly timeframe without explicit number
                cache[attr_value] = (attr_name, "MN", 1)
        get_timedelta_for_mt5_timeframe._cache = cache
        logger.debug("Initialized MT5 timeframe pattern and cache")

    # Retrieve static attributes
    pattern = get_timedelta_for_mt5_timeframe._pattern
    cache = get_timedelta_for_mt5_timeframe._cache

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
