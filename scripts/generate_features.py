import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle

import numpy as np
import pandas as pd

from common.feature_generation import *
from common.label_generation import *

"""
Compute derived features from source data and store then in an output file.
This file can be then used to train models and tune hyper-parameters.
It will generate *all* features as defined in the corresponding procedures (so that we should select only the necessary ones for prediction).
It will generate all labels as defined in the procedure (so that only the necessary labels should be chosen later for prediction).
!!! Make sure that *last* dates are the same - otherwise some last data will be empty which is worse for training
"""

#
# Parameters
#
class P:
    feature_sets = ["kline", "futur"]

    in_path_name = r"C:\DATA2\BITCOIN\GENERATED"
    in_file_name = r"BTCUSDT-1m.csv"
    in_nrows = 10_000_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"BTCUSDT-1m-features"


def main(args=None):
    in_df = None

    start_dt = datetime.now()

    #
    # Load historic data
    #
    print(f"Loading data from source file...")
    in_path = Path(P.in_path_name).joinpath(P.in_file_name)
    in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)

    print(f"Finished loading {len(in_df)} records with {len(in_df.columns)} columns.")

    # For testing only
    #in_df = in_df.tail(1_000)

    #
    # Generate derived features
    #

    if "kline" in P.feature_sets:
        print(f"Generating klines features...")
        k_features = generate_features(in_df)
        print(f"Finished generating {len(k_features)} kline features")
    else:
        k_features = []

    if "futur" in P.feature_sets:
        print(f"Generating futur features...")
        f_features = generate_features_futur(in_df)
        print(f"Finished generating {len(f_features)} futur features")
    else:
        f_features = []

    if "depth" in P.feature_sets:
        print(f"Generating depth features...")
        d_features = generate_features_depth(in_df)
        print(f"Finished generating {len(f_features)} depth features")
    else:
        d_features = []

    #
    # Generate labels (always the same, currently based on kline data which must be therefore present)
    #
    print(f"Generating labels...")
    labels = []

    # Binary labels whether max has exceeded a threshold or not
    labels += generate_labels_thresholds(in_df, horizon=180)

    # Numeric label which is ration between areas over and under the latest price
    labels += add_area_ratio(in_df, is_future=True, column_name="close", windows=[60, 120, 180, 300], suffix = "_area_future")

    print(f"Finished generating {len(labels)} labels")

    #
    # Store feature matrix in output file
    #
    print(f"Storing feature matrix with {len(in_df)} records and {len(in_df.columns)} columns in output file...")

    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
    out_path = out_path.joinpath(P.out_file_name)

    in_df.to_csv(out_path.with_suffix('.csv'), index=False, float_format="%.4f")

    #in_df.to_parquet(out_path.with_suffix('.parquet'), engine='auto', compression=None, index=None, partition_cols=None)

    elapsed = datetime.now() - start_dt
    print(f"Finished feature generation in {int(elapsed.total_seconds())} seconds")

if __name__ == '__main__':
    main(sys.argv[1:])
