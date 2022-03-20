from pathlib import Path
from typing import Union
import pandas as pd

from common.label_generation_top_bot import *

"""
This script will load a feature file (or any file with close price), and add
top-bot columns according to the label parameter, by finally storing both input
data and the labels in the output file (can be the same file as input).

Note that high-low labels are generated along with features.
"""


data_path = Path(r"C:\DATA2\BITCOIN\GENERATED\BTCUSDT")
feature_file = "BTCUSDT-1m-features.csv"  # Input file with features. Only close will be used
extremum_file = "BTCUSDT-1m-features-exremums.csv"  # Output with all top-bot labels


def add_extremum_labels():
    """
    Load a file with close price (typically feature matrix),
    compute top-bottom labels, add them to the data, and store to output file.
    """

    df = pd.read_csv(data_path / feature_file)
    print(f"Feature matrix loaded from file: {data_path / feature_file}. Length {len(df)}")

    # Filter (for debugging)
    #df = df.iloc[-one_year:]

    top_level_fracs = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    bot_level_fracs = [-x for x in top_level_fracs]

    # Tolerance 0.01
    top_labels1 = ['top6_1', 'top7_1', 'top8_1', 'top9_1', 'top10_1', 'top11_1', 'top12_1']
    bot_labels1 = ['bot6_1', 'bot7_1', 'bot8_1', 'bot9_1', 'bot10_1', 'bot11_1', 'bot12_1']

    label_names = add_extremum_features(df, column_name='close', level_fracs=top_level_fracs, tolerance_frac=0.01, out_names=top_labels1)
    print(f"Top labels computed: {label_names}")
    label_names = add_extremum_features(df, column_name='close', level_fracs=bot_level_fracs, tolerance_frac=0.01, out_names=bot_labels1)
    print(f"Bottom labels computed: {label_names}")

    # Tolerance 0.02
    top_labels2 = ['top6_2', 'top7_2', 'top8_2', 'top9_2', 'top10_2', 'top11_2', 'top12_2']
    bot_labels2 = ['bot6_2', 'bot7_2', 'bot8_2', 'bot9_2', 'bot10_2', 'bot11_2', 'bot12_2']

    label_names = add_extremum_features(df, column_name='close', level_fracs=top_level_fracs, tolerance_frac=0.02, out_names=top_labels2)
    print(f"Top labels computed: {label_names}")
    label_names = add_extremum_features(df, column_name='close', level_fracs=bot_level_fracs, tolerance_frac=0.02, out_names=bot_labels2)
    print(f"Bottom labels computed: {label_names}")

    # Tolerance 0.03
    top_labels3 = ['top6_3', 'top7_3', 'top8_3', 'top9_3', 'top10_3', 'top11_3', 'top12_3']
    bot_labels3 = ['bot6_3', 'bot7_3', 'bot8_3', 'bot9_3', 'bot10_3', 'bot11_3', 'bot12_3']

    label_names = add_extremum_features(df, column_name='close', level_fracs=top_level_fracs, tolerance_frac=0.03, out_names=top_labels3)
    print(f"Top labels computed: {label_names}")
    label_names = add_extremum_features(df, column_name='close', level_fracs=bot_level_fracs, tolerance_frac=0.03, out_names=bot_labels3)
    print(f"Bottom labels computed: {label_names}")

    # Save in output file
    df.to_csv(data_path / extremum_file, index=False)
    print(f"Feature matrix stored in file: {data_path / extremum_file}. Length {len(df)}")

    pass


if __name__ == '__main__':
    add_extremum_labels()
