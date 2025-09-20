from datetime import datetime, date, timedelta

import click
import pandas as pd
from pathlib import Path

import yfinance as yf
from curl_cffi import requests  # Without its Session object, yahoo will reject requests with YFRateLimitError

from service.App import *

"""
Download quotes from Yahoo
"""

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    """
    load_config(config_file)

    time_column = App.config["time_column"]
    data_path = Path(App.config["data_folder"])

    download_max_rows = App.config.get("download_max_rows", 0)

    # normalize time column name for robust handling (CSV headers may have different case)
    time_col = time_column.lower()

    now = datetime.now()

    # This session will be used in all requests to avoid YFRateLimitError
    session = requests.Session(impersonate="chrome")

    data_sources = App.config["data_sources"]
    for ds in data_sources:
        # Assumption: folder name is equal to the symbol name we want to download
        quote = ds.get("folder")
        if not quote:
            print(f"ERROR. Folder is not specified.")
            continue

        # If file name is not specified then use symbol name as file name
        file = ds.get("file", quote)
        if not file:
            file = quote

        print(f"Start downloading '{quote}' ...")

        file_path = data_path / quote
        file_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

        file_name = (file_path / file).with_suffix(".csv")

        if file_name.is_file():
            # read file and normalize column names to lowercase so we can find the time column reliably
            df = pd.read_csv(file_name)
            df.columns = df.columns.str.lower()
            # ensure time column exists and is datetime
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce').dt.date
                if not df.empty and pd.api.types.is_datetime64_any_dtype(pd.Series(pd.to_datetime(df[time_col], errors='coerce'))):
                    # use last available date
                    last_date = df.iloc[-1][time_col]

                    overlap = 2  # The overlap can be longer because the difference in days includes also weekends which are not trade days
                    days = (pd.Timestamp(now) - pd.Timestamp(last_date)).days + overlap

                    # === Download from the remote server
                    # Download more data than we need and then overwrite the older data
                    new_df = yf.download(quote, period=f"{days}d", auto_adjust=True, multi_level_index=False, session=session)

                    new_df = new_df.reset_index()
                    new_df['Date'] = pd.to_datetime(new_df['Date'], format="ISO8601").dt.date
                    new_df.rename({'Date': time_column}, axis=1, inplace=True)
                    new_df.columns = new_df.columns.str.lower()

                    df = pd.concat([df, new_df])
                    df = df.drop_duplicates(subset=[time_col], keep="last")
                else:
                    # existing file doesn't contain valid date values -> do full fetch
                    print(f"Existing file found but no valid '{time_col}' column or no rows. Doing full fetch...")
                    df = pd.DataFrame()
            else:
                print(f"Existing file found but time column '{time_col}' not present. Doing full fetch...")
                df = pd.DataFrame()

        if (not file_name.is_file()) or df.empty:
            # Full fetch from remote server
            print(f"File not found or empty. Full fetch...")
            df = yf.download(quote, period="max", auto_adjust=True, multi_level_index=False, session=session)

            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'], format="ISO8601").dt.date
            df.rename({'Date': time_column}, axis=1, inplace=True)
            df.columns = df.columns.str.lower()

            print(f"Full fetch finished.")

        # work with normalized lowercase time column name
        df = df.sort_values(by=time_col)

        # Limit the saved size by only the latest rows
        if download_max_rows:
            df = df.tail(download_max_rows)

        # ensure we write columns back with expected casing (use lowercase headers)
        df.to_csv(file_name, index=False)
        print(f"Finished downloading '{quote}'. Stored {len(df)} rows in '{file_name}'")

    elapsed = datetime.now() - now
    print(f"Finished downloading data in {str(elapsed).split('.')[0]}")

    return df


if __name__ == '__main__':
    main()
