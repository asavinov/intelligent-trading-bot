"""
This module handles the collection of market data from MetaTrader 5 (MT5).
It defines tasks for connecting to MT5, requesting historical kline data,
and storing it in the database. It also includes health checks for the MT5 connection.
"""

import time
from typing import Optional

import pytz

import asyncio

import MetaTrader5 as mt5

from service.App import *
from inputs.utils_mt5 import mt5_freq_from_pandas, get_timedelta_for_mt5_timeframe

import logging
log = logging.getLogger('mt5')

client = None

#
# Parameters
#
CHUNK_SIZE = 10000  # (int): The number of bars to request in each chunk. How many bars worth of duration to request in each chunk
RATE_LIMIT_DELAY = 0.1  # (float): The delay in seconds between requests. Small delay between requests (seconds)
default_start_dt = datetime(2014, 1, 1, tzinfo=timezone)  # Or get from config if needed

time_column = 'timestamp'
timezone = pytz.timezone("Etc/UTC")

async def fetch_klines(config: dict, start_from_dt) -> dict[str, pd.DataFrame] | None:
    """
    Synchronizes the local data state with the latest data from MT5.
    This task retrieves the most recent kline data for specified symbols and
    stores it in the database.

    Returns:
        dict: For each symbol (key of the dict), a data frame with the new data

    Raises:
        TimeoutError: If the data request times out.
        Exception: If any other error occurs during data retrieval.
    """
    data_sources = config.get("data_sources", [])
    symbols = [x.get("folder") for x in data_sources]
    pandas_freq = config["freq"]
    mt5_timeframe = mt5_freq_from_pandas(pandas_freq)

    if not symbols:
        symbols = [config["symbol"]]

    # Connect to trading account (same as before)
    mt5_account_id = config.get("mt5_account_id")
    mt5_password = config.get("mt5_password")
    mt5_server = config.get("mt5_server")
    if mt5_account_id and mt5_password and mt5_server:
        authorized = connect_mt5(int(mt5_account_id), password=str(mt5_password), server=str(mt5_server))
        if not authorized:
            log.error(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            return None

    # How many records are missing (and to be requested) for each symbol (not used here)
    # missing_klines_counts = [App.analyzer.get_missing_klines_count(sym) for sym in symbols]

    # Create a list of tasks for retrieving data
    tasks = [asyncio.create_task(request_symbol_klines(s, mt5_timeframe, start_from_dt)) for s in symbols]

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

    # --- Shutdown MT5 (same as before) ---
    log.info("\nShutting down MetaTrader 5 connection...")
    mt5.shutdown()

    return results

async def request_symbol_klines(symbol: str, mt5_timeframe: int, start_from_dt) -> dict:
    """
    Requests kline data from MT5 for a given symbol.
    It fetches data in chunks to avoid overloading the server and handles
    potential errors during the data retrieval process.

    Args:
        symbol (str): The trading symbol (e.g., "EURUSD").
        mt5_timeframe (int): The MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1).

    Returns:
        dict: A dictionary with the symbol as the key and a data frame with klines

    Raises:
        ValueError: If an invalid MT5 timeframe is provided.
        Exception: If any other error occurs during data retrieval.
    """
    # Define end point for download (now)
    end_dt = datetime.now(timezone)

    if start_from_dt:
        current_start_dt = start_from_dt
        log.info(f"Existing data found. Will download data starting from {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        # Define a default historical start if no file exists | 2017 | 2024
        current_start_dt = default_start_dt
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

    df = pd.concat(all_klines_list, axis=0, ignore_index=False)
    df = df.drop_duplicates(subset=time_column)
    df = df.set_index(time_column, inplace=False, drop=False)
    df.name = symbol

    # Return all received klines with the symbol as a key
    return {symbol: df}

async def health_check() -> int:
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

def connect_mt5(mt5_account_id: Optional[int] = None, mt5_password: Optional[str] = None, mt5_server: Optional[str] = None, **kwargs):
    """
    Initializes the MetaTrader 5 connection and attempts to log in with the provided credentials.
    """
    # Initialize MetaTrader 5 connection
    if not mt5.initialize():
        log.error(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    log.info(f"MT5 Initialized. Version: {mt5.version()}")

    if mt5_account_id and mt5_password and mt5_server:
        authorized = mt5.login(int(mt5_account_id), password=str(mt5_password), server=str(mt5_server), **kwargs)
        if not authorized:
            log.error(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
    return True
