import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

from trade.utils import *

"""
Feature/label generation.
These features are computed using explict transformations.
(True) labels are features computed from future data but stored as properties of the current row (in contrast to normal features which are computed from past data).
(Currently) feature/label generation is not based on (explicit) models - all parameters are hard-coded.
Also, no parameter training is performed.
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


if __name__ == "__main__":
    pass
