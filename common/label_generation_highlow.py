from __future__ import annotations  # Eliminates problem with type annotations like list[int]
import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

from common.utils import *
from common.feature_generation import *
from common.feature_generation_rolling_agg import *

"""
Label generation.
(True) labels are features computed from future data but stored as properties of the current row (in contrast to normal features which are computed from past data).
"""


def generate_labels_highlow(df, horizon):
    """
    Generate (compute) a number of labels similar to other derived features but using future data.
    This function is used only in training.
    For prediction, the labels are computed by the models (not from future data which is absent).

    We use the following conventions and dimensions for generating binary labels:
    - Threshold is used to compare all values of some parameter, for example, 0.5 or 2.0 (sign is determined from the context)
    - Greater or less than the threshold. Note that the threshold has a sign which however is determined from the context
    - High or low column to compare with the threshold. Note that relative deviations from the close are used.
      Hence, high is always positive and low is always negative.
    - horizon which determines the future window used to compute all or one
    Thus, a general label is computed via the condition: [all or one] [relative high or low] [>= or <=] threshold
    However, we do not need all combinations of parameters but rather only some of them which are grouped as follows:
    - high >= large_threshold - at least one higher than threshold: 0.5, 1.0, 1.5, 2.0, 2.5
    - high <= small_threshold - all lower than threshold: 0.1, 0.2, 0.3, 0.4
    - low >= -small_threshold - all higher than threshold: 0.1, 0.2, 0.3, 0.4
    - low <= -large_threshold - at least one lower than (negative) threshold: 0.5, 1.0, 1.5, 2.0, 2.5
    Accordingly, we encode the labels as follows (60 is horizon):
    - high_xx (xx is threshold): for big xx - high_xx means one is larger, for small xx - all are less
    - low_xx (xx is threshold): for big xx - low_xx means one is larger, for small xx - all are less
    """
    labels = []
    windows = [horizon]

    # Max high for horizon relative to close (normally positive but can be negative)
    labels += add_future_aggregations(df, "high", np.max, windows=windows, suffix='_max', rel_column_name="close", rel_factor=100.0)
    high_column_name = "high_max_"+str(horizon)  # Example: high_max_180

    # Max high crosses (is over) the threshold
    labels += add_threshold_feature(df, high_column_name, thresholds=[1.0, 1.5, 2.0, 2.5, 3.0], out_names=["high_10", "high_15", "high_20", "high_25", "high_30"])
    # Max high does not cross (is under) the threshold
    labels += add_threshold_feature(df, high_column_name, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], out_names=["high_01", "high_02", "high_03", "high_04", "high_05"])

    # Min low for horizon relative to close (normally negative but can be positive)
    labels += add_future_aggregations(df, "low", np.min, windows=windows, suffix='_min', rel_column_name="close", rel_factor=100.0)
    low_column_name = "low_min_"+str(horizon)  # Example: low_min_180

    # Min low does not cross (is over) the negative threshold
    labels += add_threshold_feature(df, low_column_name, thresholds=[-0.1, -0.2, -0.3, -0.4, -0.5], out_names=["low_01", "low_02", "low_03", "low_04", "low_05"])
    # Min low crosses (is under) the negative threshold
    labels += add_threshold_feature(df, low_column_name, thresholds=[-1.0, -1.5, -2.0, -2.5, -3.0], out_names=["low_10", "low_15", "low_20", "low_25", "low_30"])

    #
    # Ratio high_to_low_window
    #
    # Set negative to 0
    df[high_column_name] = df[high_column_name].clip(lower=0)
    # Set positive to 0
    df[low_column_name] = df[low_column_name].clip(upper=0)
    df[low_column_name] = df[low_column_name] * -1
    # Ratio between max high and min low in [-1,+1]. +1 means min is 0. -1 means high is 0
    column_sum = df[high_column_name] + df[low_column_name]
    ratio_column_name = "high_to_low_"+str(horizon)
    ratio_column = df[high_column_name] / column_sum  # in [0,1]
    df[ratio_column_name] = (ratio_column * 2) - 1

    return labels


def generate_labels_sim(df, horizon):
    """Currently not used."""
    labels = []

    # Max high
    add_future_aggregations(df, "high", np.max, horizon, suffix='_max', rel_column_name="close", rel_factor=100.0)

    # Max high crosses (is over) the threshold
    labels += add_threshold_feature(df, "high_max_180", thresholds=[2.0], out_names=["high_20"])
    # Max high does not cross (is under) the threshold
    labels += add_threshold_feature(df, "high_max_180", thresholds=[0.2], out_names=["high_02"])

    # Min low
    add_future_aggregations(df, "low", np.min, horizon, suffix='_min', rel_column_name="close", rel_factor=100.0)

    # Min low does not cross (is over) the negative threshold
    labels += add_threshold_feature(df, "low_min_180", thresholds=[-0.2], out_names=["low_02"])
    # Min low crosses (is under) the negative threshold
    labels += add_threshold_feature(df, "low_min_180", thresholds=[-2.0], out_names=["low_20"])

    return labels


def generate_labels_regressor(df, horizon):
    """Labels for regression. Currently not used."""
    labels = []

    # Max high relative to close in percent
    labels += add_future_aggregations(df, "high", np.max, horizon, suffix='_max', rel_column_name="close", rel_factor=100.0)
    # "high_max"

    # Min low relative to close in percent (negative values)
    labels += add_future_aggregations(df, "low", np.min, horizon, suffix='_min', rel_column_name="close", rel_factor=100.0)
    # "low_min"

    return labels


if __name__ == "__main__":
    pass
