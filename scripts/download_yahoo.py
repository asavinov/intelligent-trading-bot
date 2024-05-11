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

        file_path = data_path / quote
        file_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

        file_name = (file_path / file).with_suffix(".csv")

        if file_name.is_file():
            df = pd.read_csv(file_name, parse_dates=[time_column], date_format="ISO8601")
            #df['Date'] = pd.to_datetime(df['Date'], format="ISO8601")  # "2022-06-07" iso format
            df[time_column] = df[time_column].dt.date
            last_date = df.iloc[-1][time_column]

            # === Download from the remote server
            # Download more data than we need and then overwrite the older data
            new_df = yf.download(quote, period="5d", auto_adjust=True)

            new_df = new_df.reset_index()
            new_df['Date'] = pd.to_datetime(new_df['Date'], format="ISO8601").dt.date
            #del new_df['Close']
            #new_df.rename({'Adj Close': 'Close'}, axis=1, inplace=True)
            new_df.rename({'Date': time_column}, axis=1, inplace=True)
            new_df.columns = new_df.columns.str.lower()

            df = pd.concat([df, new_df])
            df = df.drop_duplicates(subset=[time_column], keep="last")

        else:
            print(f"File not found. Full fetch...")

            # === Download from the remote server
            #df = yf.download(quote, date(1990, 1, 1), auto_adjust=True)
            df = yf.download(quote, period="max", auto_adjust=True)

            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'], format="ISO8601").dt.date
            #del df['Close']
            #df.rename({'Adj Close': 'Close'}, axis=1, inplace=True)
            df.rename({'Date': time_column}, axis=1, inplace=True)
            df.columns = df.columns.str.lower()

            print(f"Full fetch finished.")

        df = df.sort_values(by=time_column)

        df.to_csv(file_name, index=False)
        print(f"Stored in '{file_name}'")

    elapsed = datetime.now() - now
    print(f"Finished downloading data in {str(elapsed).split('.')[0]}")

    return df


if __name__ == '__main__':
    main()
