from __future__ import annotations  # Eliminates problem with type annotations like list[int]
import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors


"""
Signals are binary features. 
However, they are not trained but rather found using grid search by checking their overall performance during trading for some period
"""


# TODO: Define method for training models (optimal parameters) by using grid search and performance evaluation
#   It is done similar to model training but using grid search instead and performance function instead of loss function

# IMPROVEMENT:
# In future, signaling means quantitative measure of probabilistic profitability of moving left or right.
# It can be expressed in the relative allocation between left and right sides like 70% and 30% or maybe some weights.
# This relative allocation is then used to determine quantities to be really moved between left and right. After trade, our own allocation has to correspond to this ideal one.
# In other words, signal is not boolean, but rather always numeric and relative.
# One problem here is that the sides are not equal and have very different weights. Yet, we could try to ignore this.
# If we consider the sides equal, then we act more like a market maker.

def generate_signals(df, models: dict):
    """
    Use predicted labels in the data frame to decide whether to buy or sell.
    Use rule-based approach by comparing the predicted scores with some thresholds.
    The decision is made for the last row only but we can use also previous data.

    TODO: In future, values could be functions which return signal 1 or 0 when applied to a row

    :param df: data frame with features which will be used to generate signals
    :param models: dict where key is a signal name which is also an output column name and value a dict of parameters of the model
    :return: A number of binary columns will be added each corresponding to one signal and having same name
    """

    # Define one function for each signal type.
    # A function applies a predicates by using the provided parameters and qualifies this row as true or false
    # TODO: Access to model parameters and row has to be rubust and use default values (use get instead of [])

    def all_higher_fn(row, model):
        keys = model.keys()
        for field, value in model.items():
            if row.get(field) >= value:
                continue
            else:
                return 0
        return 1

    def all_lower_fn(row, model):
        keys = model.keys()
        for field, value in model.items():
            if row.get(field) <= value:
                continue
            else:
                return 0
        return 1

    for signal, model in models.items():
        # Choose function which implements (knows how to generate) this signal
        fn = None
        if signal == "buy":
            fn = all_higher_fn
        elif signal == "sell":
            fn = all_lower_fn
        else:
            print("ERROR: Wrong use. Unexpected signal name.")

        # Model will be passed as the second argument (the first one is the row)
        df[signal] = df.apply(fn, axis=1, args=[model])

    return models.keys()

