from datetime import datetime, timezone, timedelta
import pandas as pd
from binance.helpers import date_to_milliseconds, interval_to_milliseconds

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
        timestamp = datetime.now(timezone.utc)
    elif isinstance(timestamp, int):
        timestamp = pd.to_datetime(timestamp, unit='ms', utc=True).to_pydatetime()

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
