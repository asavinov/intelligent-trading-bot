from __future__ import annotations  # Eliminates problem with type annotations like list[int]
import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

"""
Feature/label generation.
These features are computed using explict transformations.
Labels are features computed from future data but stored as properties of the current row (in contrast to normal features which are computed from past data).
"""


def generate_features(df, use_differences=False):
    """
    Generate derived features by adding them as new columns to the data frame.
    This (same) function must be used for both training and prediction.
    If we use it for training to produce models then this same function has to be used before applying these models.
    """
    # Parameters of moving averages
    windows = [1, 2, 5, 20, 60, 180]
    base_window = 300

    features = []
    to_drop = []

    if use_differences:
        df['close'] = to_diff(df['close'])
        df['volume'] = to_diff(df['volume'])
        df['trades'] = to_diff(df['trades'])

    # close mean
    to_drop += add_past_aggregations(df, 'close', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'close', np.mean, windows, '', to_drop[-1], 100.0)
    # ['close_1', 'close_2', 'close_5', 'close_20', 'close_60', 'close_180']

    # close std
    to_drop += add_past_aggregations(df, 'close', np.std, base_window)  # Base column
    features += add_past_aggregations(df, 'close', np.std, windows, '_std', to_drop[-1], 100.0)
    # ['close_std_1', 'close_std_2', 'close_std_5', 'close_std_20', 'close_std_60', 'close_std_180']

    # volume mean
    to_drop += add_past_aggregations(df, 'volume', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'volume', np.mean, windows, '', to_drop[-1], 100.0)
    # ['volume_1', 'volume_2', 'volume_5', 'volume_20', 'volume_60', 'volume_180']

    # Span: high-low difference
    df['span'] = df['high'] - df['low']
    to_drop += add_past_aggregations(df, 'span', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'span', np.mean, windows, '', to_drop[-1], 100.0)
    # ['span_1', 'span_2', 'span_5', 'span_20', 'span_60', 'span_180']

    # Number of trades
    to_drop += add_past_aggregations(df, 'trades', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'trades', np.mean, windows, '', to_drop[-1], 100.0)
    # ['trades_1', 'trades_2', 'trades_5', 'trades_20', 'trades_60', 'trades_180']

    # tb_base_av / volume varies around 0.5 in base currency
    df['tb_base'] = df['tb_base_av'] / df['volume']
    to_drop.append('tb_base')
    to_drop += add_past_aggregations(df, 'tb_base', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'tb_base', np.mean, windows, '', to_drop[-1], 100.0)
    # ['tb_base_1', 'tb_base_2', 'tb_base_5', 'tb_base_20', 'tb_base_60', 'tb_base_180']

    # tb_quote_av / quote_av varies around 0.5 in quote currency
    df['tb_quote'] = df['tb_quote_av'] / df['quote_av']
    to_drop.append('tb_quote')
    to_drop += add_past_aggregations(df, 'tb_quote', np.mean, base_window, suffix='')  # Base column
    features += add_past_aggregations(df, 'tb_quote', np.mean, windows, '', to_drop[-1], 100.0)
    # ['tb_quote_1', 'tb_quote_2', 'tb_quote_5', 'tb_quote_20', 'tb_quote_60', 'tb_quote_180']

    df.drop(columns=to_drop, inplace=True)

    return features


#
# Auxiliary feature functions
#

def to_diff_NEW(sr):
    # TODO: Use an existing library function to compute difference
    #   We used it in fast hub for computing datetime difference - maybe we can use it for numeric diffs
    pass


def to_diff(sr):
    """
    Convert the specified input column to differences.
    Each value of the output series is equal to the difference between current and previous values divided by the current value.
    """

    def diff_fn(x):  # ndarray. last element is current row and first element is most old historic value
        return 100 * (x[1] - x[0]) / x[0]

    diff = sr.rolling(window=2, min_periods=2).apply(diff_fn, raw=True)
    return diff


def add_past_aggregations(df, column_name: str, fn, windows: Union[int, list[int]], suffix=None,
                          rel_column_name: str = None, rel_factor: float = 1.0):
    return _add_aggregations(df, False, column_name, fn, windows, suffix, rel_column_name, rel_factor)


def add_future_aggregations(df, column_name: str, fn, windows: Union[int, list[int]], suffix=None,
                            rel_column_name: str = None, rel_factor: float = 1.0):
    return _add_aggregations(df, True, column_name, fn, windows, suffix, rel_column_name, rel_factor)


def _add_aggregations(df, is_future: bool, column_name: str, fn, windows: Union[int, list[int]], suffix=None,
                      rel_column_name: str = None, rel_factor: float = 1.0):
    """
    Compute moving aggregations over past or future values of the specified base column using the specified windows.

    Windowing. Window size is the number of elements to be aggregated.
    For past aggregations, the current value is always included in the window.
    For future aggregations, the current value is not included in the window.

    Naming. The result columns will start from the base column name then suffix is used and then window size is appended (separated by underscore).
    If suffix is not provided then it is function name.
    The produced names will be returned as a list.

    Relative values. If the base column is provided then the result is computed as a relative change.
    If the coefficient is provided then the result is multiplied by it.

    The result columns are added to the data frame (and their names are returned).
    The length of the data frame is not changed even if some result values are None.
    """

    column = df[column_name]

    if isinstance(windows, int):
        windows = [windows]

    if rel_column_name:
        rel_column = df[rel_column_name]

    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:
        # Aggregate
        ro = column.rolling(window=w, min_periods=max(1, w // 10))
        feature = ro.apply(fn, raw=True)

        # Convert past aggregation to future aggregation
        if is_future:
            feature = feature.shift(periods=-w)

        # Normalize
        feature_name = column_name + suffix + '_' + str(w)
        features.append(feature_name)
        if rel_column_name:
            df[feature_name] = rel_factor * (feature - rel_column) / rel_column
        else:
            df[feature_name] = rel_factor * feature

    return features


def add_threshold_feature(df, column_name: str, thresholds: list, out_names: list):
    """

    :param df:
    :param column_name: Column with values to compare with the thresholds
    :param thresholds: List of thresholds. For each of them an output column will be generated
    :param out_names: List of output column names (same length as thresholds)
    :return: List of output column names
    """

    for i, threshold in enumerate(thresholds):
        out_name = out_names[i]
        if threshold > 0.0:  # Max high
            if abs(threshold) >= 0.75:  # Large threshold
                df[out_name] = df[column_name] >= threshold  # At least one high is greater than the threshold
            else:  # Small threshold
                df[out_name] = df[column_name] <= threshold  # All highs are less than the threshold
        else:  # Min low
            if abs(threshold) >= 0.75:  # Large negative threshold
                df[out_name] = df[column_name] <= threshold  # At least one low is less than the (negative) threshold
            else:  # Small threshold
                df[out_name] = df[column_name] >= threshold  # All lows are greater than the (negative) threshold

    return out_names


# TODO: DEPRCATED: check that it is not used. Or refactor by using no apply (direct computation of relative) and remove row filter
def ___add_label_column(df, window, threshold, max_column_name='<HIGH>', ref_column_name='<CLOSE>',
                        out_column_name='label'):
    """
    Add a goal column to the dataframe which stores a label computed from future data.
    We take the column with maximum values and find its maximum for the specified window.
    Then we find relative deviation of this maximum from the value in the reference column.
    Finally, we compare this relative deviation with the specified threashold and write either 1 or 0 into the output.
    The resulted column with 1s and 0s is attached to the dataframe.
    """

    ro = df[max_column_name].rolling(window=window, min_periods=window)

    max = ro.max()  # Aggregate

    df['max'] = max.shift(periods=-window)  # Make it future max value (will be None if not enough history)

    # count = df.count()
    # labelnacount = df['label'].isna().sum()
    # nacount =  df.isna().sum()
    df.dropna(subset=['max', ref_column_name], inplace=True)  # Number of nans (at the end) is equal to the window size

    # Compute relative max value
    def relative_max_fn(row):
        # if np.isnan(row['max']) or np.isnan(row['<CLOSE>']):
        #    return None
        return 100 * (row['max'] - row[ref_column_name]) / row[ref_column_name]  # Percentage

    df['max'] = df.apply(relative_max_fn, axis=1)

    # Whether it exceeded the threshold
    df[out_column_name] = df.apply(
        lambda row: 1 if row['max'] > threshold else (0 if row['max'] <= threshold else None), axis=1)

    # Uncomment to use relative max as a numeric label
    # df['label'] = df['max']

    # df.drop(columns=['max'], inplace=True)  # Not needed anymore

    return df


def klines_to_df(klines: list):
    """
    Convert a list of klines to a data frame.
    """

    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                       'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df.set_index('timestamp', inplace=True)

    return df

# =====
# Labels

def generate_labels_thresholds(df, horizon=60):
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
    - high_60_xx (xx is threshold): for big xx - high_xx means one is larger, for small xx - all are less
    - low_60_xx (xx is threshold): for big xx - low_xx means one is larger, for small xx - all are less
    """
    labels = []
    to_drop = []

    # Max high
    to_drop += add_future_aggregations(df, "high", np.max, horizon, suffix='_max', rel_column_name="close", rel_factor=100.0)

    # Max high crosses (is over) the threshold
    labels += add_threshold_feature(df, "high_max_60", thresholds=[1.0, 1.5, 2.0, 2.5], out_names=["high_60_10", "high_60_15", "high_60_20", "high_60_25"])
    # Max high does not cross (is under) the threshold
    labels += add_threshold_feature(df, "high_max_60", thresholds=[0.1, 0.2, 0.3, 0.4], out_names=["high_60_01", "high_60_02", "high_60_03", "high_60_04"])

    # Min low
    to_drop += add_future_aggregations(df, "low", np.min, horizon, suffix='_min', rel_column_name="close", rel_factor=100.0)

    # Min low does not cross (is over) the negative threshold
    labels += add_threshold_feature(df, "low_min_60", thresholds=[-0.1, -0.2, -0.3, -0.4], out_names=["low_60_01", "low_60_02", "low_60_03", "low_60_04"])
    # Min low crosses (is under) the negative threshold
    labels += add_threshold_feature(df, "low_min_60", thresholds=[-1.0, -1.5, -2.0, -2.5], out_names=["low_60_10", "low_60_15", "low_60_20", "low_60_25"])

    df.drop(columns=to_drop, inplace=True)

    return labels

def generate_labels_sim(df, horizon):
    labels = []

    # Max high
    add_future_aggregations(df, "high", np.max, horizon, suffix='_max', rel_column_name="close", rel_factor=100.0)

    # Max high crosses (is over) the threshold
    labels += add_threshold_feature(df, "high_max_60", thresholds=[2.0], out_names=["high_60_20"])
    # Max high does not cross (is under) the threshold
    labels += add_threshold_feature(df, "high_max_60", thresholds=[0.2], out_names=["high_60_02"])

    # Min low
    add_future_aggregations(df, "low", np.min, horizon, suffix='_min', rel_column_name="close", rel_factor=100.0)

    # Min low does not cross (is over) the negative threshold
    labels += add_threshold_feature(df, "low_min_60", thresholds=[-0.2], out_names=["low_60_02"])
    # Min low crosses (is under) the negative threshold
    labels += add_threshold_feature(df, "low_min_60", thresholds=[-2.0], out_names=["low_60_20"])

    return labels

def generate_labels_regressor(df, horizon):
    labels = []

    # Max high relative to close in percent
    labels += add_future_aggregations(df, "high", np.max, horizon, suffix='_max', rel_column_name="close", rel_factor=100.0)
    # "high_max_60"

    # Min low relative to close in percent (negative values)
    labels += add_future_aggregations(df, "low", np.min, horizon, suffix='_min', rel_column_name="close", rel_factor=100.0)
    # "low_min_60"

    return labels


if __name__ == "__main__":
    pass
