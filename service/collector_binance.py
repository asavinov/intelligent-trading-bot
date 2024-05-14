import os
import sys
import argparse
import math, time
from datetime import datetime
from decimal import *

import pandas as pd
import asyncio

from binance.client import Client
from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from binance.enums import *

from service.App import *
from common.utils import *
from service.analyzer import *

import logging
log = logging.getLogger('collector')
logging.basicConfig(
    filename="collector.log",  # parameter in App
    level=logging.DEBUG,
    #format = "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    format = "%(asctime)s %(levelname)s %(message)s",
    #datefmt = '%Y-%m-%d %H:%M:%S',
)


async def main_collector_task():
    """
    It is a highest level task which is added to the event loop and executed normally every 1 minute and then it calls other tasks.
    """
    symbol = App.config["symbol"]
    freq = App.config["freq"]
    start_ts, end_ts = pandas_get_interval(freq)
    now_ts = now_timestamp()

    log.info(f"===> Start collector task. Timestamp {now_ts}. Interval [{start_ts},{end_ts}].")

    #
    # 0. Check server state (if necessary)
    #
    if data_provider_problems_exist():
        await data_provider_health_check()
        if data_provider_problems_exist():
            log.error(f"Problems with the data provider server found. No signaling, no trade. Will try next time.")
            return 1

    #
    # 1. Ensure that we are up-to-date with klines
    #
    res = await sync_data_collector_task()

    if res > 0:
        log.error(f"Problem getting data from the server. No signaling, no trade. Will try next time.")
        return 1

    log.info(f"<=== End collector task.")
    return 0


#
# Request/update market data
#

async def sync_data_collector_task():
    """
    Collect latest data.
    After executing this task our local (in-memory) data state is up-to-date.
    Hence, we can do something useful like data analysis and trading.

    Limitations and notes:
    - Currently, we can work only with one symbol
    - We update only local state by loading latest data. If it is necessary to initialize the db then another function should be used.
    """

    data_sources = App.config.get("data_sources", [])
    symbols = [x.get("folder") for x in data_sources]
    freq = App.config["freq"]
    binance_freq = binance_freq_from_pandas(freq)

    if not symbols:
        symbols = [App.config["symbol"]]

    # How many records are missing (and to be requested) for each symbol
    missing_klines_counts = [App.analyzer.get_missing_klines_count(sym) for sym in symbols]

    # Create a list of tasks for retrieving data
    #coros = [request_klines(sym, "1m", 5) for sym in symbols]
    tasks = [asyncio.create_task(request_klines(s, freq, c)) for c, s in zip(missing_klines_counts, symbols)]

    results = {}
    timeout = 10  # Seconds to wait for the result

    # Process responses in the order of arrival
    for fut in asyncio.as_completed(tasks, timeout=timeout):
        # Get the results
        res = None
        try:
            res = await fut
        except TimeoutError as te:
            log.warning(f"Timeout {timeout} seconds when requesting kline data.")
            return 1
        except Exception as e:
            log.warning(f"Exception when requesting kline data.")
            return 1

        # Add to the database (will overwrite existing klines if any)
        if res and res.keys():
            # res is dict for symbol, which is a list of record lists of 12 fields
            # ==============================
            # TODO: We need to check these fields for validity (presence, non-null)
            # TODO: We can load maximum 999 latest klines, so if more 1600, then some other method
            # TODO: Print somewhere diagnostics about how many lines are in history buffer of db, and if nans are found
            results.update(res)
            try:
                added_count = App.analyzer.store_klines(res)
            except Exception as e:
                log.error(f"Error storing kline result in the database. Exception: {e}")
                return 1
        else:
            log.error("Received empty or wrong result from klines request.")
            return 1

    return 0


async def request_klines(symbol, freq, limit):
    """
    Request klines data from the service for one symbol.
    Maximum the specified number of klines will be returned.

    :param symbol:
    :param freq: pandas frequency like '1min' which is supported by Binance API
    :param limit: desired and maximum number of klines
    :return: Dict with the symbol as a key and a list of klines as a value. One kline is also a list.
    """
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
            klines = App.client.get_klines(symbol=symbol, interval=binance_freq, limit=limit, endTime=now_ts)
            # Return: list of lists, that is, one kline is a list (not dict) with items ordered: timestamp, open, high, low, close etc.
        else:
            # https://sammchardy.github.io/binance/2018/01/08/historical-data-download-binance.html
            # get_historical_klines(symbol, interval, start_str, end_str=None, limit=500)
            # Find start from the number of records and frequency (interval length in milliseconds)
            request_start_ts = now_ts - interval_length_ms * (limit+1)
            klines = App.client.get_historical_klines(symbol=symbol, interval=binance_freq, start_str=request_start_ts, end_str=now_ts)
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

#
# Server and account info
#


async def data_provider_health_check():
    """
    Request information about the data provider server state.
    """
    symbol = App.config["symbol"]

    # Get server state (ping) and trade status (e.g., trade can be suspended on some symbol)
    system_status = App.client.get_system_status()
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
    #server_time = App.client.get_server_time()
    #time_diff = int(time.time() * 1000) - server_time['serverTime']
    # TODO: Log large time differences (or better trigger time synchronization procedure)

    return 0
