import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

from common.utils import *
from common.gen_features import *
from common.gen_features_rolling_agg import *

"""
Label generation. Labels are features which are used for training.
In forecasting, they are typically computed from future values as 
opposed to normal features computed from past values.
"""


def generate_labels_highlow(df, horizon):
    """
    Generate (compute) a number of labels similar to other derived features but using future data.
    This function is used before training to generate true labels.

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


def generate_labels_highlow2(df, config: dict):
    """
    Generate multiple increase/decrease labels which are typically used for training.

    :param df:
    :param horizon:
    :return:
    """
    column_names = config.get('columns')
    close_column = column_names[0]
    high_column = column_names[1]
    low_column = column_names[2]

    function = config.get('function')
    if not isinstance(function, str):
        raise ValueError(f"Wrong type of the 'function' parameter: {type(function)}")
    if function not in ['high', 'low']:
        raise ValueError(f"Unknown function name {function}. Only 'high' or 'low' are possible")

    tolerance = config.get('tolerance')  # Fraction of the level/threshold

    thresholds = config.get('thresholds')  # List of thresholds which are growth/drop in percent
    if not isinstance(thresholds, list):
        thresholds = [thresholds]

    if function == 'high':
        thresholds = [abs(t) for t in thresholds]
        price_columns = [high_column, low_column]
    elif function == 'low':
        thresholds = [-abs(t) for t in thresholds]
        price_columns = [low_column, high_column]

    tolerances = [round(-t*tolerance, 6) for t in thresholds]  # Tolerance have opposite sign

    horizon = config.get('horizon')  # Length of history to be analyzed

    names = config.get('names')  # For example, ['first_high_10', 'first_high_15'] for two tolerances
    if len(names) != len(thresholds):
        raise ValueError(f"'highlow2' Label generator: for each threshold value one name has to be provided.")

    labels = []
    for i, threshold in enumerate(thresholds):
        first_cross_labels(df, horizon, [threshold, tolerances[i]], close_column, price_columns, names[i])
        labels.append(names[i])

    print(f"Highlow2 labels computed: {labels}")

    return df, labels


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


def _first_location_of_crossing_threshold(df, horizon, threshold, close_column_name, price_column_name):
    """
    First location of crossing the threshold.
    For each point, take its close price, and then find the distance (location, index)
    to the _first_ future point with high or low price higher or lower, respectively
    than the close price.

    If the location (index) is 0 then it is the next point. If location (index) is NaN,
    then the price does not cross the specified threshold during the horizon
    (or there is not enough data, e.g., at the end of the series). Therefore, this
    function can be used to find whether the price will cross the threshold at all
    during the specified horizon.

    The function is somewhat similar to the tsfresh function first_location_of_maximum
    or minimum. The difference is that this function does not search for maximum but rather
    first cross of the threshold.

    Horizon specifies how many points are considered after this point and without this point.

    Threshold is increase or decrease coefficient, say, 50.0 means 50% increase with respect to
    the current close price.
    """

    def fn_high(x):
        if len(x) < 2:
            return np.nan
        p = x[0, 0]  # Reference price
        p_threshold = p*(1+(threshold/100.0))  # Cross line
        idx = np.argmax(x[1:, 1] > p_threshold)  # First index where price crosses the threshold

        # If all False, then index is 0 (first element of constant series) and we are not able to distinguish it from first element being True
        # If index is 0 and first element False (under threshold) then NaN (not exceeds)
        if idx == 0 and x[1, 1] <= p_threshold:
            return np.nan
        return idx

    def fn_low(x):
        if len(x) < 2:
            return np.nan
        p = x[0, 0]  # Reference price
        p_threshold = p*(1+(threshold/100.0))  # Cross line
        idx = np.argmax(x[1:, 1] < p_threshold)  # First index where price crosses the threshold

        # If all False, then index is 0 (first element of constant series) and we are not able to distinguish it from first element being True
        # If index is 0 and first element False (under threshold) then NaN (not exceeds)
        if idx == 0 and x[1, 1] >= p_threshold:
            return np.nan
        return idx

    # Window df will include the current row as well as horizon of past rows with 0 index starting from the oldest row and last index with the current row
    rl = df[[close_column_name, price_column_name]].rolling(horizon + 1, min_periods=(horizon // 2), method='table')

    if threshold > 0:
        df_out = rl.apply(fn_high, raw=True, engine='numba')
    elif threshold < 0:
        df_out = rl.apply(fn_low, raw=True, engine='numba')
    else:
        raise ValueError(f"Threshold cannot be zero.")

    # Because rolling apply processes past records while we need future records
    df_out = df_out.shift(-horizon)

    # For some unknown reason (bug?), rolling apply (with table and numba) returns several columns rather than one column
    out_column = df_out.iloc[:, 0]

    return out_column


def first_cross_labels(df, horizon, thresholds, close_column, price_columns, out_column):
    """
    Produce one boolean column which is true if the price crosses the first threshold
    but does not cross the second threshold in the opposite direction before that.

    For example, if columns are (high, low) and thresholds are [5.0, -1.0]
    then the result is true if price increases by 5% but never decreases lower than 1% during this growth.

    If columns are (low, high) and thresholds are [-5.0, 1.0]
    the result is true if price decreases by 5% but never increases higher than 1% before that.
    """

    # High label - find first (forward) index like +5 of the value exceeds the threshold. Or 0/nan if not found within window
    df["first_idx_column"] = _first_location_of_crossing_threshold(df, horizon, thresholds[0], close_column, price_columns[0])

    # Low label - find first (forward) index like +6 of the value lower than threshold. Or 0/nan if not found within window
    df["second_idx_column"] = _first_location_of_crossing_threshold(df, horizon, thresholds[1], close_column, price_columns[1])

    # The final value is chosen from these two whichever is smaller (as absolute value), that is, closer to this point
    def is_high_true(x):
        if np.isnan(x[0]):
            return False
        elif np.isnan(x[1]):
            return True
        else:
            return x[0] <= x[1]  # If the first cross point is closer to this point than the second one

    df[out_column] = df[["first_idx_column", "second_idx_column"]].apply(is_high_true, raw=True, axis=1)

    # Indexes are not needed anymore
    df.drop(columns=['first_idx_column', 'second_idx_column'], inplace=True)

    return out_column


if __name__ == "__main__":
    pass
