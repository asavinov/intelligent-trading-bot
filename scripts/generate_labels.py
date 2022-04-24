from pathlib import Path
from typing import Union

import click
import pandas as pd

from service.App import *
from common.feature_generation_rolling_agg import *
from common.label_generation import *
from common.label_generation_top_bot import *

"""
This script will load a feature file (or any file with close price), and add
top-bot columns according to the label parameter, by finally storing both input
data and the labels in the output file (can be the same file as input).

Note that high-low labels are generated along with features.
"""


#
# Parameters
#
class P:
    label_sets = ["top-bot"]  # Possible values: "high-low", "top-bot"

    in_nrows = 100_000_000


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Load a file with close price (typically feature matrix),
    compute top-bottom labels, add them to the data, and store to output file.
    """
    load_config(config_file)

    freq = "1m"
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return

    config_file_modifier = App.config.get("config_file_modifier")
    config_file_modifier = ("-" + config_file_modifier) if config_file_modifier else ""

    start_dt = datetime.now()

    #
    # Load input data (normally feature matrix but not necessarily)
    #
    in_file_suffix = App.config.get("feature_file_modifier")

    in_file_name = f"{in_file_suffix}{config_file_modifier}.csv"
    in_path = (data_path / in_file_name).resolve()

    print(f"Loading data from feature file {str(in_path)}...")
    in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)
    print(f"Finished loading {len(in_df)} records with {len(in_df.columns)} columns.")

    # Filter (for debugging)
    #df = df.iloc[-one_year:]

    labels = []

    #
    # Generate labels (always the same, currently based on kline data which must be therefore present)
    #
    if "high-low" in P.label_sets:
        horizon = App.config["high_low_horizon"]

        # Binary labels whether max has exceeded a threshold or not
        print(f"Generating 'high-low' labels with horizon {horizon}...")
        labels += generate_labels_thresholds(in_df, horizon=horizon)

        # Numeric label which is a ratio between areas over and under the latest price
        print(f"Generating ration labels with horizon...")
        labels += add_area_ratio(in_df, is_future=True, column_name="close", windows=[60, 120, 180, 300], suffix = "_area_future")

        print(f"Finished generating 'high-low' labels. {len(labels)} labels generated.")

    #
    # top-bot labels
    #
    if "top-bot" in P.label_sets:
        column_name = App.config.get("top_bot_column_name", "close")

        top_level_fracs = [0.02, 0.03, 0.04, 0.05, 0.06]
        bot_level_fracs = [-x for x in top_level_fracs]

        # Tolerance 0.0025
        tolerance_frac = 0.0025
        top_labels = ['top2_025', 'top3_025', 'top4_025', 'top5_025', 'top6_025']
        bot_labels = ['bot2_025', 'bot3_025', 'bot4_025', 'bot5_025', 'bot6_025']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.005
        tolerance_frac = 0.005
        top_labels = ['top2_05', 'top3_05', 'top4_05', 'top5_05', 'top6_05']
        bot_labels = ['bot2_05', 'bot3_05', 'bot4_05', 'bot5_05', 'bot6_05']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.0075
        tolerance_frac = 0.0075
        top_labels = ['top2_075', 'top3_075', 'top4_075', 'top5_075', 'top6_075']
        bot_labels = ['bot2_075', 'bot3_075', 'bot4_075', 'bot5_075', 'bot6_075']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.01
        tolerance_frac = 0.01
        top_labels = ['top2_1', 'top3_1', 'top4_1', 'top5_1', 'top6_1']
        bot_labels = ['bot2_1', 'bot3_1', 'bot4_1', 'bot5_1', 'bot6_1']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.0125
        tolerance_frac = 0.0125
        top_labels = ['top2_125', 'top3_125', 'top4_125', 'top5_125', 'top6_125']
        bot_labels = ['bot2_125', 'bot3_125', 'bot4_125', 'bot5_125', 'bot6_125']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.015
        tolerance_frac = 0.015
        top_labels = ['top2_15', 'top3_15', 'top4_15', 'top5_15', 'top6_15']
        bot_labels = ['bot2_15', 'bot3_15', 'bot4_15', 'bot5_15', 'bot6_15']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.0175
        tolerance_frac = 0.0175
        top_labels = ['top2_175', 'top3_175', 'top4_175', 'top5_175', 'top6_175']
        bot_labels = ['bot2_175', 'bot3_175', 'bot4_175', 'bot5_175', 'bot6_175']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.02
        tolerance_frac = 0.02
        top_labels = ['top2_2', 'top3_2', 'top4_2', 'top5_2', 'top6_2']
        bot_labels = ['bot2_2', 'bot3_2', 'bot4_2', 'bot5_2', 'bot6_2']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.025
        tolerance_frac = 0.025
        top_labels = ['top2_25', 'top3_25', 'top4_25', 'top5_25', 'top6_25']
        bot_labels = ['bot2_25', 'bot3_25', 'bot4_25', 'bot5_25', 'bot6_25']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

        # Tolerance 0.03
        tolerance_frac = 0.03
        top_labels = ['top2_3', 'top3_3', 'top4_3', 'top5_3', 'top6_3']
        bot_labels = ['bot2_3', 'bot3_3', 'bot4_3', 'bot5_3', 'bot6_3']

        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        print(f"Top labels computed: {top_labels}")
        labels += add_extremum_features(in_df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

    # Save in output file
    out_file_suffix = App.config.get("matrix_file_modifier")

    out_file_name = f"{out_file_suffix}{config_file_modifier}.csv"
    out_path = (data_path / out_file_name).resolve()

    print(f"Storing file with labels. {len(in_df)} records and {len(in_df.columns)} columns in output file...")
    in_df.to_csv(out_path, index=False, float_format="%.4f")

    #
    # Store labels
    #
    out_file_name = f"{out_file_suffix}{config_file_modifier}.txt"
    out_path = (data_path / out_file_name).resolve()

    with open(out_path, "a+") as f:
        f.write(", ".join([f"'{l}'" for l in labels] ) + "\n")

    print(f"Stored {len(labels)} labels in output file {out_path}")

    elapsed = datetime.now() - start_dt
    print(f"Finished label generation in {int(elapsed.total_seconds())} seconds")
    print(f"Output file location: {out_path}")


if __name__ == '__main__':
    main()
