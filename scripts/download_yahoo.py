from datetime import datetime, date, timedelta

import click

import yfinance as yf

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
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return

    now = datetime.now()

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

        in_file_path = (data_path / quote / file).with_suffix(".csv")

        if in_file_path.is_file():
            df = pd.read_csv(in_file_path, parse_dates=[time_column])
            #df['Date'] = pd.to_datetime(df['Date'])  # "2022-06-07" iso format
            df[time_column] = df[time_column].dt.date
            last_date = df.iloc[-1][time_column]

            # === Download from the remote server
            new_df = yf.download(quote, last_date - timedelta(days=5))  # Download somewhat more than we need

            new_df = new_df.reset_index()
            new_df['Date'] = pd.to_datetime(new_df['Date']).dt.date
            del new_df['Close']
            new_df.rename({'Adj Close': 'Close', 'Date': time_column}, axis=1, inplace=True)
            new_df.columns = new_df.columns.str.lower()

            df = pd.concat([df, new_df])
            df = df.drop_duplicates(subset=[time_column], keep="last")

        else:
            print(f"File not found. Full fetch...")

            # === Download from the remote server
            df = yf.download(quote, date(1990, 1, 1))

            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            del df['Close']
            df.rename({'Adj Close': 'Close', 'Date': time_column}, axis=1, inplace=True)
            df.columns = df.columns.str.lower()

            print(f"Full fetch finished.")

        df = df.sort_values(by=time_column)

        in_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(in_file_path, index=False)
        print(f"Stored in '{in_file_path}'")

    elapsed = datetime.now() - now
    print(f"Finished downloading data in {str(elapsed).split('.')[0]}")

    return df


if __name__ == '__main__':
    main()
