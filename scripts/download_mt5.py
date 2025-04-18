import time
from datetime import datetime, timedelta
from pathlib import Path
import os

import pandas as pd
import click
import MetaTrader5 as mt5
import pytz

from common.utils import mt5_freq_from_pandas, get_timedelta_for_mt5_timeframe
from service.App import App, load_config
from service.mt5 import connect_mt5


print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

# --- Configuration ---
DEFAULT_BAR_CHUNK_SIZE = 10000  # How many bars worth of duration to request in each chunk
DEFAULT_TICK_CHUNK_SIZE = 5 # How many ticks worth of duration to request in each chunk
RATE_LIMIT_DELAY = 0.1 # Small delay between requests (seconds)
# ---------------------


# -------------------------------------------------

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Retrieving historic klines from MetaTrader5 server incrementally using copy_rates_range
    with calculated chunk durations. Downloads data from the last record in the existing file
    (or a historical start date) up to the current time, fetching in duration-based chunks.
    """
    load_config(config_file)

    data_sources = App.config["data_sources"]
    time_column = App.config["time_column"]
    data_path = Path(App.config["data_folder"])
    download_max_rows = App.config.get("download_max_rows", 0)
    mt5_account_id = App.config.get("mt5_account_id")
    mt5_password = App.config.get("mt5_password")
    mt5_server = App.config.get("mt5_server")

    script_start_time = datetime.now()

    pandas_freq = App.config["freq"]
    print(f"Pandas frequency: {pandas_freq}")

    mt5_timeframe = mt5_freq_from_pandas(pandas_freq)
    # Use timeframe_description for clearer output if available
    try:
        tf_description = mt5.timeframe_description(mt5_timeframe)
        print(f"MetaTrader5 frequency: {tf_description} ({mt5_timeframe})")
    except AttributeError: # Handle older MT5 versions potentially lacking this func
         print(f"MetaTrader5 frequency: {mt5_timeframe}")


    # Define the timezone for MT5 (usually UTC)
    timezone = pytz.timezone("Etc/UTC")
    # Define a default historical start if no file exists => 2014 | 2017 | 2024
    historical_start_date = datetime(2017, 1, 1, tzinfo=timezone) # Or get from config if needed

    # Connect to trading account 
    if mt5_account_id and mt5_password and mt5_server:
        authorized = connect_mt5(mt5_account_id, password=str(mt5_password), server=str(mt5_server))
        if authorized:
            print("MT5 Login successful.")
            account_info = mt5.account_info()
            if account_info:
                print(f"Account Info: Login={account_info.login}, Server={account_info.server}, Balance={account_info.balance}")
            else:
                 print(f"Could not retrieve account info. Error: {mt5.last_error()}")
        else:
            print(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            mt5.shutdown()
            return
    else:
        print("MT5 credentials not fully provided in config. Proceeding without login (might affect available symbols/data).")

    print(f"Terminal Info: {mt5.terminal_info()}")


    # --- Loop through data sources ---

    processed_symbols = []

    for ds in data_sources:
        quote = str(ds.get("folder")).upper()
        file_type = str(ds.get("file")).lower()

        if not quote:
            print("ERROR: Folder (symbol) is not specified in data_sources.")
            continue


        print(f"\n--- Processing symbol: {quote} ---")

        file_path = data_path / quote
        file_path.mkdir(parents=True, exist_ok=True)
        file_name = (file_path / "klines").with_suffix(".csv")
        chunk_size = int(ds.get("chunk_size", DEFAULT_BAR_CHUNK_SIZE))


        if file_type == "ticks":
            file_name = (file_path / "ticks").with_suffix(".csv")
            chunk_size = int(ds.get("chunk_size", DEFAULT_TICK_CHUNK_SIZE))


        existing_df = pd.DataFrame()
        start_dt = historical_start_date


        # Check if file exists and load data
        if file_name.is_file():
            try:
                print(f"Loading existing data from: {file_name}")
                # Specify date format for potentially faster parsing if consistent
                existing_df = pd.read_csv(file_name, parse_dates=[time_column], date_format='ISO8601')
                # Ensure timezone is set correctly after parsing
                if pd.api.types.is_datetime64_any_dtype(existing_df[time_column]) and existing_df[time_column].dt.tz is None:
                     existing_df[time_column] = existing_df[time_column].dt.tz_localize('UTC')
                elif pd.api.types.is_datetime64_any_dtype(existing_df[time_column]):
                     existing_df[time_column] = existing_df[time_column].dt.tz_convert('UTC')
                else: # Fallback if parsing failed or column is not datetime
                    print(f"Warning: Column '{time_column}' not parsed as datetime. Attempting conversion.")
                    existing_df[time_column] = pd.to_datetime(existing_df[time_column], errors='coerce', utc=True)

                existing_df = existing_df.dropna(subset=[time_column]) # Drop rows where conversion failed

                if not existing_df.empty:
                    # Sort just in case file wasn't sorted
                    existing_df = existing_df.sort_values(by=time_column)
                    # Start downloading from the timestamp of the last record
                    start_dt = existing_df[time_column].iloc[-1]
                    print(f"Existing file found. Will download data starting from {start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                else:
                     print("Existing file was empty or had invalid dates after loading. Starting from historical date.")
                     existing_df = pd.DataFrame() # Reset to empty
            except Exception as e:
                print(f"Error loading existing file {file_name}: {e}. Starting from historical date.")
                existing_df = pd.DataFrame() # Reset to empty
        else:
            print(f"File not found. Starting download from {historical_start_date.strftime('%Y-%m-%d %H:%M:%S %Z')}.")
            start_dt = historical_start_date # Ensure start_dt is set


        # Define end point for download (now)
        end_dt = datetime.now(timezone)

        # Check if symbol is available
        symbol_info = mt5.symbol_info(quote)
        if not symbol_info:
            print(f"Symbol {quote} not found or not available in MT5 terminal. Skipping. Error: {mt5.last_error()}")
            continue
        if file_type == "ticks" and not symbol_info.trade_tick_size:
            print(f"Ticks data is not available for {quote}. Skipping. Error: {mt5.last_error()}")
            os.remove(file_name)
            continue
        print(f"Symbol {quote} found in MT5.")


        all_klines_list = []
        current_start_dt = start_dt

        print(f"Starting download loop for {quote} from {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

        # --- Download Loop using copy_rates_range or copy_ticks_range with calculated duration ---
        while current_start_dt < end_dt:
            try:
                # Calculate the duration for chunk_size bars or ticks


                chunk_duration = get_timedelta_for_mt5_timeframe(mt5_timeframe, chunk_size)
            except ValueError as e:
                 print(f"Error calculating duration: {e}. Stopping download for {quote}.")
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
                 print(f"  Skipping request: Calculated range is empty or invalid ({request_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {temp_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}).")
                 break # Likely means we are caught up

            print(f"  Fetching range from {request_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {temp_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

            # Use copy_rates_range or copy_ticks_range
            if file_type == "ticks":
                rates = mt5.copy_ticks_range(quote, request_start_dt, temp_end_dt, mt5.COPY_TICKS_ALL)
                if rates is None:
                    print(f"  mt5.copy_ticks_range returned None. Error: {mt5.last_error()}. Stopping download for {quote}.")
                    break
            else:
                rates = mt5.copy_rates_range(quote, mt5_timeframe, request_start_dt, temp_end_dt)
                if rates is None:
                    print(f"  mt5.copy_rates_range returned None. Error: {mt5.last_error()}. Stopping download for {quote}.")
                    break

            if len(rates) == 0:
                print("  No data returned in this range. Download may be complete or data gap.")
                # If no data, advance start time past this chunk's end to avoid getting stuck
                current_start_dt = temp_end_dt
                if current_start_dt >= end_dt:
                    print("  Reached end date after empty range.")
                    break
                else:
                    print(f"  Advancing start time to {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} and continuing.")
                    time.sleep(RATE_LIMIT_DELAY) # Still pause slightly
                    continue # Try the next chunk

            chunk_df = pd.DataFrame(rates)
            if file_type == "ticks":
                # Convert 'time_msc' (Unix milliseconds) to datetime objects (UTC)
                chunk_df[time_column] = pd.to_datetime(chunk_df['time_msc'], unit='ms', utc=True)
            else:
                # Convert 'time' (Unix seconds) to datetime objects (UTC)
                chunk_df[time_column] = pd.to_datetime(chunk_df['time'], unit='s', utc=True)

            # --- IMPORTANT: Filtering is no longer needed here ---
            # Since we requested data *starting after* current_start_dt using request_start_dt,
            # the check `chunk_df = chunk_df[chunk_df[time_column] > current_start_dt]`
            # is redundant and can be removed.

            # if chunk_df.empty: # This check is effectively handled by len(rates) == 0 now
            #      print("  No new bars found in the fetched chunk (after filtering). Stopping.")
            #      break

            all_klines_list.append(chunk_df)
            last_bar_time_in_chunk = chunk_df[time_column].iloc[-1]
            print(f"  Fetched {len(chunk_df)} bars. Last timestamp in chunk: {last_bar_time_in_chunk.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            # Update the start time for the next chunk request to be the end time of THIS chunk's last bar
            current_start_dt = last_bar_time_in_chunk

            # Check if we've downloaded past the intended end time (redundant check, loop condition handles it)
            # if current_start_dt >= end_dt:
            #     print("  Reached or passed target end time. Download complete.")
            #     break

            # Small delay before next request
            time.sleep(RATE_LIMIT_DELAY)

        # --- Combine and Process Data ---
        if not all_klines_list:
            print(f"No new data downloaded for {quote}.")
            if existing_df.empty:
                print(f"No existing or new data for {quote}. Skipping save.")
                continue
            else:
                print("Saving existing data only (no updates).")
                final_df = existing_df # Use existing data if no new data was fetched
        else:
            print("Combining downloaded data...")
            new_df = pd.concat(all_klines_list, ignore_index=True)

            # Combine existing and new data
            if not existing_df.empty:
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df

            print("Processing combined data (duplicates, sorting, columns)...")
            if file_type == "ticks":
                # Standardize columns (assuming MT5 names)
                final_df.rename(columns={
                    'time_msc': 'time',
                    'flags': 'flags',
                    'bid': 'bid',
                    'ask': 'ask',
                    'last': 'last',
                    'volume': 'volume',
                }, inplace=True, errors='ignore') # Added errors='ignore'
            else:
                final_df.rename(columns={
                    'tick_volume': 'volume', # Use tick_volume as 'volume'
                }, inplace=True, errors='ignore') # Added errors='ignore'


            # Ensure time column is the primary datetime column and drop time column if exist
            if 'time' in final_df.columns and time_column != 'time':
                 final_df = final_df.drop('time', axis=1)

            # Select desired columns (ensure time_column is first)
            required_columns = [time_column, 'open', 'high', 'low', 'close', 'volume']
            # Keep only columns that actually exist in the dataframe
            final_df = final_df[[col for col in required_columns if col in final_df.columns]]

            # Remove duplicates based on timestamp, keeping the latest entry
            initial_rows = len(final_df)
            final_df = final_df.drop_duplicates(subset=[time_column], keep='last')
            if initial_rows > len(final_df):
                print(f"  Removed {initial_rows - len(final_df)} duplicate rows based on '{time_column}'.")

            # Sort by timestamp
            final_df = final_df.sort_values(by=time_column)

            # Remove the last row *if* it represents the current, incomplete bar.
            if not final_df.empty:
                 # Check if the last bar's time is too close to the script end time
                 # A simple heuristic: if the last bar's time is after the loop's end_dt minus one interval, it might be incomplete.
                 # Or just always drop the last row after bulk download.
                 print("Removing potentially incomplete last bar.")
                 final_df = final_df.iloc[:-1]


        # Apply max rows limit if specified (same as before)
        if download_max_rows and len(final_df) > download_max_rows:
            print(f"Applying download_max_rows limit: {download_max_rows}")
            final_df = final_df.tail(download_max_rows)

        # Final check if DataFrame is valid before saving (same as before)
        if final_df.empty:
             print(f"Final dataframe for {quote} is empty after processing. Skipping save.")
             continue

        # Reset index before saving (same as before)
        final_df = final_df.reset_index(drop=True)

        # --- Save Data (same as before) ---

        try:
            if file_type == "ticks":
                final_df = final_df.drop(['time'], axis=1)

            print(f"Saving {len(final_df)} rows to {file_name}...")
            final_df.to_csv(file_name, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
            print(f"Finished saving '{quote}'.")
            processed_symbols.append(quote)
        except Exception as e:
            print(f"Error saving file {file_name}: {e}")


    # --- Shutdown MT5 (same as before) ---
    print("\nShutting down MetaTrader 5 connection...")
    mt5.shutdown()

    elapsed = datetime.now() - script_start_time
    print(f"\nFinished downloading data for symbols: {', '.join(processed_symbols) if processed_symbols else 'None'}")
    print(f"Total time: {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
