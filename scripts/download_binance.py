from datetime import timedelta, datetime
from dateutil import parser

#import math
#import os.path
#import json
#import time

import pandas as pd

import click

from binance import Client

from common.utils import binance_freq_from_pandas
from inputs.collector_binance import klines_to_df, column_types
from service.App import *

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

    client_args = App.config.get("client_args", {})
    if App.config.get("api_key"):
        client_args["api_key"] = App.config.get("api_key")
    if App.config.get("api_secret"):
        client_args["api_secret"] = App.config.get("api_secret")
    App.client = Client(**client_args)

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
            df[time_column] = pd.to_datetime(df[time_column], format='ISO8601')
            df = df.astype(column_types)
            df = df.set_index('timestamp', inplace=False, drop=False)

            # oldest_point = parser.parse(data["timestamp"].iloc[-1])
            oldest_point = df["timestamp"].iloc[-5]  # Use an older point so that new data will overwrite old data

            print(f"File found. Downloaded data for {quote} and {freq} since {str(latest_ts)} will be appended to the existing file {file_name}")
        else:
            # No existing data so we will download all available data and store as a new file
            df = None

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

        df_new = klines_to_df(klines)

        if df is None:
            df = df_new
        else:
            df = pd.concat([df, df_new])

            # Drop duplicates
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            # df = df[~df.index.duplicated(keep='last')]  # alternatively, drop duplicates in index

        # Remove last row because it represents a non-complete kline (the interval not finished yet)
        df = df.iloc[:-1]

        # Limit the saved size by only the latest rows
        if download_max_rows:
            df = df.tail(download_max_rows)

        if save:
            df.to_csv(file_name, index=False)

        print(f"Finished downloading '{quote}'. Stored {len(df)} rows in '{file_name}'")

    elapsed = datetime.now() - now
    print(f"Finished downloading data in {str(elapsed).split('.')[0]}")

    return df


if __name__ == '__main__':
    main()
