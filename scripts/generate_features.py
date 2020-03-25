import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle

import numpy as np
import pandas as pd

from trade.utils import *
from trade.feature_generation import *
from trade.label_generation import *

"""
Compute derived features from source data and store then in an output file.
This file can be then used to train models and tune hyper-parameters.
"""

#
# Parameters
#
class P:
    in_path_name = r"C:\DATA2\BITCOIN"
    in_file_name = r"BTCUSDT-1m-data.csv"
    in_nrows = 100_000_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"_BTCUSDT-1m-data-features_1"

    features_horizon = 300  # Features are generated using this past window length
    labels_horizon = 60  # Labels are generated using this number of steps ahead

def main(args=None):
    in_df = None

    start_dt = datetime.now()

    #
    # Load historic data
    #
    print(f"Loading data from source file...")
    in_path = Path(P.in_path_name).joinpath(P.in_file_name)
    in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)

    #
    # Generate features (from past data)
    #
    print(f"Generating features...")
    features = generate_features(in_df)

    #
    # Generate labels (from future data)
    #
    print(f"Generating labels...")
    labels = generate_labels_thresholds(in_df, horizon=60)

    #
    # Store feature matrix in output file
    #
    print(f"Storing feature matrix in output file...")

    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
    out_path = out_path.joinpath(P.out_file_name)

    in_df.to_csv(out_path.with_suffix('.csv'), index=False, float_format="%.4f")

    #in_df.to_parquet(out_path.with_suffix('.parq'), engine='auto', compression=None, index=None, partition_cols=None)

    elapsed = datetime.now() - start_dt
    print(f"Finished feature generation in {int(elapsed.total_seconds())} seconds")

if __name__ == '__main__':
    main(sys.argv[1:])
