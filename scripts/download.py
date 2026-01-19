from datetime import datetime, date, timedelta
from pathlib import Path

import click

from common.types import Venue
from service.App import *

"""
Download raw data for the specified venu and store udpates in the corresponding files.
If a file exists then new data will be appended (by overwriting some latest records).
If a file does not exist then all data will be downloaded and the file will be created.

The real connection and data retrieval is performed by venue-specific functions.
"""

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    """
    load_config(config_file)

    time_column = App.config["time_column"]
    data_path = Path(App.config["data_folder"])
    pandas_freq = App.config["freq"]

    download_max_rows = App.config.get("download_max_rows", 0)

    now = datetime.now()

    venue = App.config.get("venue")
    venue = Venue(venue)

    data_sources = App.config["data_sources"]

    if venue == Venue.BINANCE:
        from inputs.download_binance import download_binance
        download_binance(App.config, data_sources)

    elif venue == Venue.YAHOO:
        from inputs.download_yahoo import download_yahoo
        download_yahoo(App.config, data_sources)

    elif venue == Venue.MT5:
        from inputs.download_mt5 import download_mt5
        download_mt5(App.config, data_sources)

    else:
        if not venue:
            print(f"ERROR. Venue is not specified.")
        else:
            print(f"ERROR. Unknown venue {venue} or downloader for the venue not implemented.")

    elapsed = datetime.now() - now
    print(f"")
    print(f"Finished downloading {len(data_sources)} data sources from {venue} in {str(elapsed).split('.')[0]}")

if __name__ == '__main__':
    main()
