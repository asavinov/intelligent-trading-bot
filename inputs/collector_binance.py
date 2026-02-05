import os
import sys
import argparse
import math, time
from datetime import datetime
from decimal import *
from typing import Any, Coroutine

import pandas as pd
import asyncio

from binance import Client
from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from binance.enums import *

from service.App import *
from common.utils import *
from inputs.utils_binance import *

import logging
log = logging.getLogger('binance.base_client')

client = None

#
# Parameters
#
append_overlap_records = 5  # How many records to request in addition to the missing data (overlap length)

# Binance-specific columns name corresponding to the values returned from API
column_names = [
    'timestamp',
    'open', 'high', 'low', 'close', 'volume',
    'close_time',
    'quote_av', 'trades', 'tb_base_av', 'tb_quote_av',
    'ignore'
]
column_types = {
    'timestamp': 'datetime64[ns, UTC]',  # datetime64[ns, UTC] datetime64[ns]
    'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64',
    'close_time': 'int64',
    'quote_av': 'float64', 'trades': 'int64', 'tb_base_av': 'float64', 'tb_quote_av': 'float64',
    'ignore': 'float64',
}
time_column = 'timestamp'

async def sync_data_collector_task(config: dict, start_from_dt) -> dict[str, pd.DataFrame] | None:
    """
    Retrieve and return latest data from binance client.

    Limitation: maximum 999 latest klines can be retrieved. If more is needed then some other function has to be used

    :return: For each symbol (key of the dict), data frame with data and binance-specific columns
    """

    data_sources = config.get("data_sources", [])
    symbols = [x.get("folder") for x in data_sources]
    freq = config["freq"]
    binance_freq = binance_freq_from_pandas(freq)

    if not symbols:
        symbols = [config["symbol"]]

    # Compute how many records need to be fetched from the specified start timestamp
    interval_length_td = pd.Timedelta(freq).to_pytimedelta()
    now = datetime.now(timezone.utc)
    intervals_count = (now - start_from_dt) // interval_length_td  # How many whole intervals
    request_count = intervals_count + append_overlap_records

    # Create a list of tasks for retrieving data
    missing_klines_counts = [request_count for sym in symbols]
    #coros = [request_klines(sym, "1m", 5) for sym in symbols]
    tasks = [asyncio.create_task(request_klines(s, freq, c)) for c, s in zip(missing_klines_counts, symbols)]

    results = {}
    timeout = 10  # Seconds to wait for the result

    # Process responses in the order of arrival
    for fut in asyncio.as_completed(tasks, timeout=timeout):
        # Get the results
        res = None
        try:
            res = await fut  # res is dict for symbol, which is a list of record lists of 12 fields
        except TimeoutError as te:
            log.warning(f"Timeout {timeout} seconds when requesting kline data.")
            return None
        except Exception as e:
            log.warning(f"Exception when requesting kline data.")
            return None

        # Add to the database (will overwrite existing klines if any)
        if res and res.keys():
            results.update(res)
        else:
            log.error("Received empty or wrong result from klines request.")
            return None

    for symbol, klines in results.items():
        df = klines_to_df(klines)
        df.name = symbol
        results[symbol] = df

    return results

async def request_klines(symbol, freq, limit: int):
    """
    Request klines data from the service for one symbol.
    Maximum the specified number of klines will be returned.

    :param symbol:
    :param freq: pandas frequency like '1min' which is supported by Binance API
    :param limit: desired and maximum number of klines
    :return: Dict with the symbol as a key and a list of klines as a value. One kline is also a list.
    """
    global client

    klines_per_request = 400  # Limitation of API

    now_ts = now_timestamp()
    start_ts, end_ts = pandas_get_interval(freq)

    binance_freq = binance_freq_from_pandas(freq)
    interval_length_ms = pandas_interval_length_ms(freq)

    try:
        if limit <= klines_per_request:  # Server will return these number of klines in one request
            # INFO:
            # - startTime: include all intervals (ids) with same or greater id: if within interval then excluding this interval; if is equal to open time then include this interval
            # - endTime: include all intervals (ids) with same or smaller id: if equal to left border then return this interval, if within interval then return this interval
            # - It will return also incomplete current interval (in particular, we could collect approximate klines for higher frequencies by requesting incomplete intervals)
            klines = client.get_klines(symbol=symbol, interval=binance_freq, limit=limit, endTime=now_ts)
            # Return: list of lists, that is, one kline is a list (not dict) with items ordered: timestamp, open, high, low, close etc.
        else:
            # https://sammchardy.github.io/binance/2018/01/08/historical-data-download-binance.html
            # get_historical_klines(symbol, interval, start_str, end_str=None, limit=500)
            # Find start from the number of records and frequency (interval length in milliseconds)
            request_start_ts = now_ts - interval_length_ms * (limit+1)
            klines = client.get_historical_klines(symbol=symbol, interval=binance_freq, start_str=request_start_ts, end_str=now_ts)
    except BinanceRequestException as bre:
        # {"code": 1103, "msg": "An unknown parameter was sent"}
        log.error(f"BinanceRequestException while requesting klines: {bre}")
        return {}
    except BinanceAPIException as bae:
        # {"code": 1002, "msg": "Invalid API call"}
        log.error(f"BinanceAPIException while requesting klines: {bae}")
        return {}
    except Exception as e:
        log.error(f"Exception while requesting klines: {e}")
        return {}

    #
    # Post-process
    #

    # Find last complete interval in the result list
    # The problem is that the result also contains the current (still running) interval which we want to exclude
    # Exclude last kline if it corresponds to the current interval
    klines_full = [kl for kl in klines if kl[0] < start_ts]
    last_full_kline_ts = klines_full[-1][0]

    if last_full_kline_ts != start_ts - interval_length_ms:
        log.error(f"UNEXPECTED RESULT: Last full kline timestamp {last_full_kline_ts} is not equal to previous full interval start {start_ts - interval_length_ms}. Maybe some results are missing and there are gaps.")

    # Return all received klines with the symbol as a key
    return {symbol: klines_full}

async def data_provider_health_check():
    """
    Request information about the data provider server state.
    """
    global client

    # Get server state (ping) and trade status (e.g., trade can be suspended on some symbol)
    system_status = client.get_system_status()
    #{
    #    "status": 0,  # 0: normal，1：system maintenance
    #    "msg": "normal"  # normal or System maintenance.
    #}
    if not system_status or system_status.get("status") != 0:
        App.server_status = 1
        return 1
    App.server_status = 0

    # Ping the server

    # Check time synchronization
    #server_time = client.get_server_time()
    #time_diff = int(time.time() * 1000) - server_time['serverTime']
    # TODO: Log large time differences (or better trigger time synchronization procedure)

    return 0

def klines_to_df(klines: list):
    """
    Convert a list of klines (for one symbol) to a data frame by using the binance-specific convention for (a sequence of) column names and their types.
    """
    df = pd.DataFrame(klines, columns=column_names)
    df[time_column] = pd.to_datetime(df[time_column], unit='ms', utc=True)
    df = df.astype(column_types)

    # Explicitly assign or convert time zone not needed because we convert millis directly to UTC
    #if df[time_column].dt.tz is None:
    #    df[time_column] = df[time_column].dt.tz_localize('UTC')
    #else:
    #    df[time_column] = df[time_column].dt.tz_convert('UTC')

    #df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

    #df["open"] = pd.to_numeric(df["open"])
    #df["high"] = pd.to_numeric(df["high"])
    #df["low"] = pd.to_numeric(df["low"])
    #df["close"] = pd.to_numeric(df["close"])
    #df["volume"] = pd.to_numeric(df["volume"])

    #df["quote_av"] = pd.to_numeric(df["quote_av"])
    #df["trades"] = pd.to_numeric(df["trades"])
    #df["tb_base_av"] = pd.to_numeric(df["tb_base_av"])
    #df["tb_quote_av"] = pd.to_numeric(df["tb_quote_av"])

    # Set index by retaining the time column
    df.set_index(time_column, inplace=True, drop=False)

    # Validate
    if df.isnull().any().any():
        null_columns = {k: v for k, v in df.isnull().any().to_dict().items() if v}
        print(f"WARNING: Null in raw data found during conversion. Columns with Nulls: {null_columns}")
    # TODO: We might receive empty strings or 0s in numeric data - how can we detect them?
    # TODO: Check that timestamps in 'close_time' are strictly consecutive. It is warning - not error

    return df
