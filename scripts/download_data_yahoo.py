from datetime import datetime, date, timedelta

import click

import yfinance as yfin

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

    data_path = Path(App.config["data_folder"])
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return

    now = datetime.now()
    today = date.today()

    data_sources = App.config["data_sources"]
    for data_source in data_sources:
        # Assumption: folder name is equal to the symbol name we want to download
        quote = data_source.get("folder")
        if not quote:
            print(f"ERROR. Folder is not specified.")
            continue

        # If file name is not specified then use symbol name as file name
        file = data_source.get("file", quote)
        if not file:
            file = quote

        print(f"Start downloading '{quote}' ...")

        in_file_path = data_path / quote / file

        if in_file_path.with_suffix(".csv").is_file():
            df = pd.read_csv(in_file_path.with_suffix(".csv"), parse_dates=["Date"])
            #df['Date'] = pd.to_datetime(df['Date'])  # "2022-06-07" iso format
            df['Date'] = df['Date'].dt.date
            last_date = df.iloc[-1]['Date']

            # === Download from the remote server
            new_df = yfin.download(quote, last_date - timedelta(days=5))  # Download somewhat more than we need
            # ===

            new_df = new_df.reset_index()
            new_df['Date'] = pd.to_datetime(new_df['Date']).dt.date
            df = pd.concat([df, new_df])
            df = df.drop_duplicates(subset=["Date"], keep="last")

        else:
            print(f"File not found. Full fetch...")
            df = yfin.download(quote, date(1990, 1, 1))
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            print(f"Full fetch finished.")

        df = df.sort_values(by="Date")

        in_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(in_file_path.with_suffix(".csv"), index=False)
        print(f"Stored in '{in_file_path}'")

    elapsed = datetime.now() - now
    #print(f"Finished downloading data in {int(elapsed.total_seconds())} seconds.")
    print(f"Finished downloading data in {str(elapsed).split('.')[0]}")

    return df


if __name__ == '__main__':
    main()
