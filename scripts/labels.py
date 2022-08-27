from pathlib import Path

import pandas as pd
import click

from service.App import *
from scripts.features import generate_feature_set
from common.label_generation_highlow import *
from common.label_generation_topbot import *

"""
This script will load a feature file (or any file with close price), and add
top-bot columns according to the label parameter, by finally storing both input
data and the labels in the output file (can be the same file as input).
"""


#
# Parameters
#
class P:
    in_nrows = 100_000_000
    tail_rows = int(2.5 * 525_600)  # Process only this number of last rows


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Load a file with close price (typically feature matrix),
    compute top-bottom labels, add them to the data, and store to output file.
    """
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    #
    # Load merged data with regular time series
    #
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol

    file_path = (data_path / App.config.get("feature_file_name")).with_suffix(".csv")
    if not file_path.is_file():
        print(f"Data file does not exist: {file_path}")
        return

    print(f"Loading data from source data file {file_path}...")
    df = pd.read_csv(file_path, parse_dates=[time_column], nrows=P.in_nrows)
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:]
    df = df.reset_index(drop=True)

    #
    # Generate derived features
    #
    label_sets = App.config.get("label_sets", [])
    if not label_sets:
        print(f"ERROR: no label sets defined. Nothing to process.")
        return
        # By default, we generate standard labels
        #label_sets = [{"column_prefix": "", "generator": "highlow", "feature_prefix": ""}]

    # Apply all feature generators to the data frame which get accordingly new derived columns
    # The feature parameters will be taken from App.config (depending on generator)
    print(f"Start generating labels for {len(df)} input records.")

    all_features = []
    for fs in label_sets:
        df, new_features = generate_feature_set(df, fs, last_rows=0)
        all_features.extend(new_features)

    print(f"Finished generating labels.")

    print(f"Number of NULL values:")
    print(df[all_features].isnull().sum().sort_values(ascending=False))

    #
    # Store feature matrix in output file
    #
    out_file_name = App.config.get("matrix_file_name")
    out_path = (data_path / out_file_name).with_suffix(".csv").resolve()

    print(f"Storing file with labels. {len(df)} records and {len(df.columns)} columns in output file...")
    df.to_csv(out_path, index=False, float_format="%.4f")

    #
    # Store labels
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f'"{f}"' for f in all_features] ) + "\n\n")

    print(f"Stored {len(all_features)} labels in output file {out_path}")

    elapsed = datetime.now() - now
    print(f"Finished label generation in {str(elapsed).split('.')[0]}")

    print(f"Output file location: {out_path}")


if __name__ == '__main__':
    main()
