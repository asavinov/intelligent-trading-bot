from datetime import timedelta, datetime
from dateutil import parser

#import math
#import os.path
#import json
#import time

import pandas as pd

import click

from binance import Client

from common.utils import klines_to_df, binance_freq_from_pandas
from service.App import *
from pathlib import Path

"""
Download from binance
"""

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Retrieving historic klines from binance server.

    Client.get_historical_klines
    """
    load_config(config_file)

    time_column = App.config["time_column"]
    data_path = Path(App.config["data_folder"])
    download_max_rows = App.config.get("download_max_rows", 0)

    now = datetime.now()

    freq = App.config["freq"]  # Pandas frequency
    print(f"Pandas frequency: {freq}")

    freq = binance_freq_from_pandas(freq)
    print(f"Binance frequency: {freq}")

    save = True

    # Use public endpoints for klines when keys are placeholder/empty (no auth required for historical klines)
    api_key = App.config.get("api_key")
    api_secret = App.config.get("api_secret")
    def _is_placeholder(x: str | None) -> bool:
        if not x:
            return True
        ux = str(x).strip().upper()
        # Treat common placeholders as empty to avoid sending invalid headers to public endpoints
        return ux in {"DEMO_API_KEY_FOR_TESTING", "DEMO_SECRET_FOR_TESTING", "TEST", "PLACEHOLDER", "NONE", "NULL", ""}

    if _is_placeholder(api_key):
        api_key = None
    if _is_placeholder(api_secret):
        api_secret = None

    App.client = Client(api_key=api_key, api_secret=api_secret)

    futures = False
    if futures:
        App.client.API_URL = "https://fapi.binance.com/fapi"
        App.client.PRIVATE_API_VERSION = "v1"
        App.client.PUBLIC_API_VERSION = "v1"

    data_sources = App.config["data_sources"]
    for ds in data_sources:
        # Assumption: folder name is equal to the symbol name we want to download
        quote = ds.get("folder")
        if not quote:
            print(f"ERROR. Folder is not specified.")
            continue

        print(f"Start downloading '{quote}' ...")

        file_path = data_path / quote
        file_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

        file_name = (file_path / ("futures" if futures else "klines")).with_suffix(".csv")

        # Get a few latest klines to determine the latest available timestamp
        latest_klines = App.client.get_klines(symbol=quote, interval=freq, limit=5)
        latest_ts = pd.to_datetime(latest_klines[-1][0], unit='ms')

        if file_name.is_file():
            # Load the existing data in order to append newly downloaded data
            df = pd.read_csv(file_name)
            # ensure time column is parsed as datetime (tolerant parse)
            try:
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            except Exception:
                df[time_column] = pd.to_datetime(df[time_column].astype(str), errors='coerce')

            # Choose an older existing point to allow some overlap when fetching new data.
            # If the dataframe is small, fall back to the earliest timestamp or a default start date.
            if not df.empty:
                if len(df) >= 5:
                    oldest_point = df[time_column].iloc[-5]
                else:
                    # use the first available timestamp in the file
                    oldest_point = df[time_column].iloc[0]
            else:
                oldest_point = datetime(2017, 1, 1)

            print(f"File found. Downloaded data for {quote} and {freq} since {str(latest_ts)} will be appended to the existing file {file_name}")
        else:
            # No existing data so we will download all available data and store as a new file
            df = pd.DataFrame()

            oldest_point = datetime(2017, 1, 1)

            print(f"File not found. All data will be downloaded and stored in newly created file for {quote} and {freq}.")

        #delta_minutes = (latest_ts - oldest_point).total_seconds() / 60
        #binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
        #delta_lines = math.ceil(delta_minutes / binsizes[freq])

        # === Download from the remote server using binance client
        klines = App.client.get_historical_klines(
            symbol=quote,
            interval=freq,
            start_str=oldest_point.isoformat(),
            #end_str=latest_ts.isoformat()  # fetch everything up to now
        )

        df = klines_to_df(klines, df)

        # Remove last row because it represents a non-complete kline (the interval not finished yet)
        df = df.iloc[:-1]

        # Limit the saved size by only the latest rows
        if download_max_rows:
            df = df.tail(download_max_rows)

        if save:
            df.to_csv(file_name)

        print(f"Finished downloading '{quote}'. Stored {len(df)} rows in '{file_name}'")

    elapsed = datetime.now() - now
    print(f"Finished downloading data in {str(elapsed).split('.')[0]}")

    return df


if __name__ == '__main__':
    main()
