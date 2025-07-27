"""
This module handles the collection of market data from MetaTrader 5 (MT5).
It defines tasks for connecting to MT5, requesting historical kline data,
and storing it in the database. It also includes health checks for the MT5 connection.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import asyncio
import MetaTrader5 as mt5
import pytz

from common.utils_mt5 import mt5_freq_from_pandas, get_timedelta_for_mt5_timeframe
from service.App import *
from common.utils import *
from service.analyzer import *
from service.mt5 import connect_mt5

import logging

log = logging.getLogger('collector')


async def main_collector_task() -> int:
    """
    The main collector task that is executed periodically.
    It checks the MT5 connection, synchronizes data, and handles any errors.

    Returns:
        int: 0 if the task completed successfully, 1 otherwise.
    """

    symbol = App.config["symbol"]
    pandas_freq = App.config["freq"]
    start_ts, end_ts = pandas_get_interval(pandas_freq)
    now_ts = now_timestamp()

    log.info(f"===> Start collector task. Timestamp {now_ts}. Interval [{start_ts},{end_ts}].")

    #
    # 0. Check MT5 connection
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

async def sync_data_collector_task() -> int:
    """
    Synchronizes the local data state with the latest data from MT5.
    This task retrieves the most recent kline data for specified symbols and
    stores it in the database.

    Returns:
        int: 0 if the data was synchronized successfully, 1 otherwise.

    Raises:
        TimeoutError: If the data request times out.
        Exception: If any other error occurs during data retrieval.
    """

    CHUNK_SIZE = 10000  # How many bars worth of duration to request in each chunk
    RATE_LIMIT_DELAY = 0.1  # Small delay between requests (seconds)

    data_sources = App.config.get("data_sources", [])
    symbols = [x.get("folder") for x in data_sources]
    pandas_freq = App.config["freq"]
    mt5_timeframe = mt5_freq_from_pandas(pandas_freq)

    if not symbols:
        symbols = [App.config["symbol"]]

    # Connect to trading account (same as before)
    mt5_account_id = App.config.get("mt5_account_id")
    mt5_password = App.config.get("mt5_password")
    mt5_server = App.config.get("mt5_server")
    if mt5_account_id and mt5_password and mt5_server:
        authorized = connect_mt5(int(mt5_account_id), password=str(mt5_password), server=str(mt5_server))
        if not authorized:
            log.error(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            return 1


    # How many records are missing (and to be requested) for each symbol (not used here)
    # missing_klines_counts = [App.analyzer.get_missing_klines_count(sym) for sym in symbols]

    # Create a list of tasks for retrieving data
    tasks = [asyncio.create_task(request_klines(s, pandas_freq, mt5_timeframe, CHUNK_SIZE, RATE_LIMIT_DELAY)) for s in symbols]

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
                added_count = App.analyzer.append_klines(res)
            except Exception as e:
                log.error(f"Error storing kline result in the database. Exception: {e}")
                return 1
        else:
            log.error("Received empty or wrong result from klines request.")
            return 1

    # --- Shutdown MT5 (same as before) ---
    log.info("\nShutting down MetaTrader 5 connection...")
    mt5.shutdown()

    return 0


async def request_klines(symbol: str, pandas_freq: str, mt5_timeframe: int, CHUNK_SIZE: int, RATE_LIMIT_DELAY: float) -> dict:
    """
    Requests kline data from MT5 for a given symbol.
    It fetches data in chunks to avoid overloading the server and handles
    potential errors during the data retrieval process.

    Args:
        symbol (str): The trading symbol (e.g., "EURUSD").
        pandas_freq (str): The pandas frequency string (e.g., "1min", "5min").
        mt5_timeframe (int): The MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1).
        CHUNK_SIZE (int): The number of bars to request in each chunk.
        RATE_LIMIT_DELAY (float): The delay in seconds between requests.

    Returns:
        dict: A dictionary with the symbol as the key and a list of klines as the value.
              Each kline is a list of data points.

    Raises:
        ValueError: If an invalid MT5 timeframe is provided.
        Exception: If any other error occurs during data retrieval.
    """

    time_column = App.config["time_column"]
    timezone = pytz.timezone("Etc/UTC")

    # Define end point for download (now)
    end_dt = datetime.now(timezone)

    # Get the last kline from the database
    last_kline_dt = App.analyzer.get_last_kline_time(symbol)
    if last_kline_dt:
        current_start_dt = last_kline_dt
        log.info(f"Existing data found. Will download data starting from {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        # Define a default historical start if no file exists | 2017 | 2024
        current_start_dt = datetime(2014, 1, 1, tzinfo=timezone)  # Or get from config if needed
        log.info(f"No existing data found. Starting download from {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}.")

    all_klines_list = []

    try:
        while current_start_dt < end_dt:
            try:
                # Calculate the duration for CHUNK_SIZE bars
                chunk_duration = get_timedelta_for_mt5_timeframe(mt5_timeframe, CHUNK_SIZE)
            except ValueError as e:
                log.error(f"Error calculating duration: {e}. Stopping download for {symbol}.")
                break

            # Calculate the temporary end date for this chunk request
            temp_end_dt = current_start_dt + chunk_duration

            # Ensure the temporary end date doesn't go beyond the overall end date
            temp_end_dt = min(temp_end_dt, end_dt)

            # Add a small buffer (e.g., 1 second) to start_dt for the request
            # to definitively exclude the current_start_dt bar itself in the range request.
            request_start_dt = current_start_dt + timedelta(seconds=1)

            # Avoid making a request if the adjusted start is already >= temp_end
            if request_start_dt >= temp_end_dt:
                log.info(f"  Skipping request: Calculated range is empty or invalid ({request_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {temp_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}).")
                break  # Likely means we are caught up

            log.info(f"  Fetching range from {request_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {temp_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

            # Use copy_rates_range, since copy_rates_from seems to fetch data using backward lookback(present to past)
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, request_start_dt, temp_end_dt)

            if rates is None:
                log.error(f"  mt5.copy_rates_range returned None. Error: {mt5.last_error()}. Stopping download for {symbol}.")
                break
            if len(rates) == 0:
                log.info("  No data returned in this range. Download may be complete or data gap.")
                # If no data, advance start time past this chunk's end to avoid getting stuck
                current_start_dt = temp_end_dt
                if current_start_dt >= end_dt:
                    log.info("  Reached end date after empty range.")
                    break
                else:
                    log.info(f"  Advancing start time to {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} and continuing.")
                    time.sleep(RATE_LIMIT_DELAY)  # Still pause slightly
                    continue  # Try the next chunk

            chunk_df = pd.DataFrame(rates)
            # Convert 'time' (Unix seconds) to datetime objects (UTC)
            chunk_df[time_column] = pd.to_datetime(chunk_df['time'], unit='s', utc=True)
            all_klines_list.append(chunk_df)
            last_bar_time_in_chunk = chunk_df[time_column].iloc[-1]
            log.info(f"  Fetched {len(chunk_df)} bars. Last timestamp in chunk: {last_bar_time_in_chunk.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            # Update the start time for the next chunk request to be the end time of THIS chunk's last bar
            current_start_dt = last_bar_time_in_chunk
            # Small delay before next request
            time.sleep(RATE_LIMIT_DELAY)

    except Exception as e:
        log.error(f"Exception while requesting klines: {e}")
        return {}

    # Return all received klines with the symbol as a key
    return {symbol: all_klines_list}

#
# Server and account info
#


async def data_provider_health_check() -> int:
    """
    Performs a health check on the MT5 connection.
    It verifies if the MT5 terminal is accessible and returns the status.

    Returns:
        int: 0 if the MT5 connection is healthy, 1 otherwise.

    """

    # Check MT5 connection
    if not mt5.terminal_info():
        log.error(f"MT5 terminal not found.")
        return 1
    return 0
