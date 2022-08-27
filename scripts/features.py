from typing import Tuple
from pathlib import Path
import click

import numpy as np
import pandas as pd

from service.App import *
from common.feature_generation import *
from common.label_generation_highlow import generate_labels_highlow
from common.label_generation_topbot import generate_labels_topbot

#
# Parameters
#
class P:
    in_nrows = 50_000_000  # Load only this number of records
    tail_rows = int(2.5 * 525_600)  # Process only this number of last rows


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    #
    # Load merged data with regular time series
    #
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol

    file_path = (data_path / App.config.get("merge_file_name")).with_suffix(".csv")
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
    feature_sets = App.config.get("feature_sets", [])
    if not feature_sets:
        print(f"ERROR: no feature sets defined. Nothing to process.")
        return
        # By default, we generate standard kline features
        #feature_sets = [{"column_prefix": "", "generator": "klines", "feature_prefix": ""}]

    # Apply all feature generators to the data frame which get accordingly new derived columns
    # The feature parameters will be taken from App.config (depending on generator)
    print(f"Start generating features for {len(df)} input records.")

    all_features = []
    for fs in feature_sets:
        fs_now = datetime.now()
        print(f"Start generator {fs.get('generator')}...")
        df, new_features = generate_feature_set(df, fs, last_rows=0)
        all_features.extend(new_features)
        fs_elapsed = datetime.now() - fs_now
        print(f"Finished generator {fs.get('generator')}. Features: {len(new_features)}. Time: {str(fs_elapsed).split('.')[0]}")

    print(f"Finished generating features.")

    print(f"Number of NULL values:")
    print(df[all_features].isnull().sum().sort_values(ascending=False))

    #
    # Store feature matrix in output file
    #
    out_file_name = App.config.get("feature_file_name")
    out_path = (data_path / out_file_name).with_suffix(".csv").resolve()

    print(f"Storing feature matrix with {len(df)} records and {len(df.columns)} columns in output file...")
    df.to_csv(out_path, index=False, float_format="%.4f")
    #df.to_parquet(out_path.with_suffix('.parquet'), engine='auto', compression=None, index=None, partition_cols=None)

    #
    # Store features
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f'"{f}"' for f in all_features] ) + "\n\n")

    print(f"Stored {len(all_features)} features in output file {out_path}")

    elapsed = datetime.now() - now
    print(f"Finished feature generation in {str(elapsed).split('.')[0]}")

    print(f"Output file location: {out_path}")


def generate_feature_set(df: pd.DataFrame, fs: dict, last_rows: int) -> Tuple[pd.DataFrame, list]:
    """
    Apply the specified resolved feature generator to the input data set.
    """

    #
    # Select columns from the data set to be processed by the feature generator
    #
    cp = fs.get("column_prefix")
    if cp:
        cp = cp + "_"
        f_cols = [col for col in df if col.startswith(cp)]
        f_df = df[f_cols]  # Alternatively: f_df = df.loc[:, df.columns.str.startswith(cf)]
        # Remove prefix because feature generators are generic (a prefix will be then added to derived features before adding them back to the main frame)
        f_df = f_df.rename(columns=lambda x: x[len(cp):] if x.startswith(cp) else x)  # Alternatively: f_df.columns = f_df.columns.str.replace(cp, "")
    else:
        f_df = df[df.columns.to_list()]  # We want to have a different data frame object to add derived featuers and then join them back to the main frame with prefix

    #
    # Resolve and apply feature generator functions from the configuration
    #
    generator = fs.get("generator")
    if generator == "binance_main":
        features = generate_features_binance_main(
            f_df, use_differences=False,
            base_window=App.config["base_window"], windows=App.config["averaging_windows"],
            area_windows=App.config["area_windows"], last_rows=last_rows
        )
    elif generator == "binance_secondary":
        features = generate_features_binance_secondary(
            f_df, use_differences=False,
            base_window=App.config["base_window"], windows=App.config["averaging_windows"],
            area_windows=App.config["area_windows"], last_rows=last_rows
        )
    elif generator == "futures":
        features = generate_features_futures(f_df)
    elif generator == "depth":
        features = generate_features_depth(f_df)
    elif generator == "yahoo_main":
        features = generate_features_yahoo_main(
            f_df, use_differences=False,
            base_window=App.config["base_window"], windows=App.config["averaging_windows"],
            area_windows=App.config["area_windows"], last_rows=last_rows
        )
    elif generator == "yahoo_secondary":
        features = generate_features_yahoo_secondary(
            f_df, use_differences=False,
            base_window=App.config["base_window"], windows=App.config["averaging_windows"],
            area_windows=App.config["area_windows"], last_rows=last_rows
        )
    elif generator == "area_features":
        area_windows = App.config["area_windows"]
        if not area_windows:
            area_windows = [60, 120, 180, 300]
        # Numeric features which is a ratio between areas over and under the latest price
        features = add_area_ratio(f_df, is_future=False, column_name="close", windows=area_windows, suffix="_area_past")
    elif generator == "tsfresh":
        tsfresh_windows = App.config["tsfresh_windows"]
        features = generate_features_tsfresh(f_df, column_name="close", windows=tsfresh_windows, last_rows=last_rows)

    # Labels
    elif generator == "highlow":
        horizon = App.config["highlow_horizon"]

        # Binary labels whether max has exceeded a threshold or not
        print(f"Generating 'highlow' labels with horizon {horizon}...")
        features = generate_labels_highlow(f_df, horizon=horizon)

        print(f"Finished generating 'highlow' labels. {len(features)} labels generated.")
    elif generator == "topbot":
        column_name = App.config.get("topbot_column_name", "close")

        top_level_fracs = [0.02, 0.03, 0.04, 0.05, 0.06]
        bot_level_fracs = [-x for x in top_level_fracs]

        features = generate_labels_topbot(f_df, column_name, top_level_fracs, bot_level_fracs)
    elif generator == "area_labels":
        area_windows = App.config["area_windows_labels"]
        if not area_windows:
            area_windows = [60, 120, 180, 300]
        # Numeric label which is a ratio between areas over and under the latest price
        features = add_area_ratio(f_df, is_future=True, column_name="close", windows=area_windows, suffix = "_area_future")
    else:
        print(f"Unknown feature generator {generator}")
        return

    #
    # Add generated features to the main data frame with all other columns and features
    #
    f_df = f_df[features]
    fp = fs.get("feature_prefix")
    if fp:
        f_df = f_df.add_prefix(fp + "_")

    new_features = f_df.columns.to_list()

    df = df.join(f_df)  # Attach all derived features to the main frame

    return df, new_features


if __name__ == '__main__':
    main()
