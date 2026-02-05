import logging
import dateparser
import pytz
from datetime import datetime, timezone, timedelta
from typing import Union, List
import json
from decimal import *

import numpy as np
import pandas as pd

from sklearn import metrics

from apscheduler.triggers.cron import CronTrigger

from common.gen_features import *

#
# Decimals
#

def to_decimal(value):
    """Convert to a decimal with the required precision. The value can be string, float or decimal."""
    # Possible cases: string, 4.1-e7, float like 0.1999999999999 (=0.2), Decimal('4.1E-7')

    # App.config["trade"]["symbol_info"]["baseAssetPrecision"]

    n = 8
    rr = Decimal(1) / (Decimal(10) ** n)  # Result: 0.00000001
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_DOWN)
    return ret

def round_str(value, digits):
    rr = Decimal(1) / (Decimal(10) ** digits)  # Result for 8 digits: 0.00000001
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_HALF_UP)
    return f"{ret:.{digits}f}"

def round_down_str(value, digits):
    rr = Decimal(1) / (Decimal(10) ** digits)  # Result for 8 digits: 0.00000001
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_DOWN)
    return f"{ret:.{digits}f}"

#
# Interval arithmetic (pandas)
#

def pandas_get_interval(freq: str, timestamp: int=None):
    """
    Find a discrete interval for the given timestamp and return its start and end.

    :param freq: pandas frequency
    :param timestamp: milliseconds (13 digits)
    :return: triple (start, end)
    """
    if not timestamp:
        timestamp = int(datetime.now(timezone.utc).timestamp())  # seconds (10 digits)
        # Alternatively: timestamp = int(datetime.utcnow().replace(tzinfo=pytz.utc).timestamp())
    elif isinstance(timestamp, datetime):
        timestamp = int(timestamp.replace(tzinfo=pytz.utc).timestamp())
    elif isinstance(timestamp, int):
        pass
    else:
        ValueError(f"Error converting timestamp {timestamp} to millis. Unknown type {type(timestamp)} ")

    # Interval length for the given frequency
    interval_length_sec = pandas_interval_length_ms(freq) / 1000

    start = (timestamp // interval_length_sec) * interval_length_sec
    end = start + interval_length_sec

    return int(start*1000), int(end*1000)

def pandas_interval_length_ms(freq: str):
    return int(pd.Timedelta(freq).to_pytimedelta().total_seconds() * 1000)

def get_interval_count_from_start_dt(freq: str, start_dt):
    """How many whole intervals are from the specified start datetime and now."""
    interval_length_td = pd.Timedelta(freq).to_pytimedelta()
    now = datetime.now(timezone.utc)
    interval_count = (now - start_dt) // interval_length_td  # How many whole intervals
    return interval_count + 2

def get_start_dt_for_interval_count(freq: str, interval_count: int):
    """Start datetime for the specified number of whole intervals back. Result is not aligned with the reaster."""
    interval_length_td = pd.Timedelta(freq).to_pytimedelta()
    period_length_td = interval_length_td * (interval_count + 1)
    now = datetime.now(timezone.utc)
    start_dt = (now - period_length_td)
    return start_dt

#
# Date and time
#

def freq_to_CronTrigger(freq: str):
    # Add small time after interval end to get a complete interval from the server
    if freq.endswith("min"):
        if freq[:-3] == "1":
            trigger = CronTrigger(minute="*", second="1", timezone="UTC")
        else:
            trigger = CronTrigger(minute="*/" + freq[:-3], second="1", timezone="UTC")
    elif freq.endswith("h"):
        if freq[:-1] == "1":
            trigger = CronTrigger(hour="*", minute="0", second="2", timezone="UTC")
        else:
            trigger = CronTrigger(hour="*/" + freq[:-1], minute="0", second="2", timezone="UTC")
    elif freq.endswith("D"):
        if freq[:-1] == "1":
            trigger = CronTrigger(day="*", second="5", timezone="UTC")
        else:
            trigger = CronTrigger(day="*/" + freq[:-1], second="5", timezone="UTC")
    elif freq.endswith("W"):
        if freq[:-1] == "1":
            trigger = CronTrigger(week="*", second="10", timezone="UTC")
        else:
            trigger = CronTrigger(day="*/" + freq[:-1], second="10", timezone="UTC")
    elif freq.endswith("MS"):
        if freq[:-2] == "1":
            trigger = CronTrigger(month="*", second="30", timezone="UTC")
        else:
            trigger = CronTrigger(month="*/" + freq[:-1], second="30", timezone="UTC")
    else:
        raise ValueError(f"Cannot convert frequency '{freq}' to cron.")

    return trigger

def now_timestamp():
    """
    """
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def find_index(df: pd.DataFrame, date_str: str, column_name: str = "timestamp"):
    """
    Return index of the record with the specified datetime string.

    :return: row id in the input data frame which can be then used in iloc function
    :rtype: int
    """
    d = dateparser.parse(date_str)
    try:
        res = df[df[column_name] == d]
    except TypeError:  # "Cannot compare tz-naive and tz-aware datetime-like objects"
        # Change timezone (set UTC timezone or reset timezone)
        if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
            d = d.replace(tzinfo=pytz.utc)
        else:
            d = d.replace(tzinfo=None)

        # Repeat
        res = df[df[column_name] == d]

    if res is None or len(res) == 0:
        raise ValueError(f"Cannot find date '{date_str}' in the column '{column_name}'. Either it does not exist or wrong format")

    id = res.index[0]

    return id

def notnull_tail_rows(df):
    """Maximum number of tail rows without nulls."""

    nan_df = df.isnull()
    nan_cols = nan_df.any()  # Series with columns having at least one NaN
    nan_cols = nan_cols[nan_cols].index.to_list()
    if len(nan_cols) == 0:
        return len(df)

    # Indexes of last NaN for all columns and then their minimum
    tail_rows = nan_df[nan_cols].values[::-1].argmax(axis=0).min()

    return tail_rows

#
# System etc.
#

def resolve_generator_name(gen_name: str):
    """
    Resolve the specified name to a function reference.
    Fully qualified name consists of module name and function name separated by a colon,
    for example:  'mod1.mod2.mod3:my_func'.

    Example: fn = resolve_generator_name("common.gen_features_topbot:generate_labels_topbot3")
    """

    mod_and_func = gen_name.split(':', 1)
    mod_name = mod_and_func[0] if len(mod_and_func) > 1 else None
    func_name = mod_and_func[-1]

    if not mod_name:
        return None

    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return None
    if mod is None:
        return None

    try:
        func = getattr(mod, func_name)
    except AttributeError as e:
        return None

    return func

#
# Data processing
#

def double_columns(df, shifts: List[int]):
    """
    Use previous rows as features appended to this row. This allows us to move history to the current time.
    One limitation is that this function will duplicate *all* features and only using the explicitly specified list of offsets.
    """
    if not shifts:
        return df
    df_list = [df.shift(shift) for shift in shifts]
    df_list.insert(0, df)
    max_shift = max(shifts)

    # Shift and add same columns
    df_out = pd.concat(df_list, axis=1)  # keys=('A', 'B')

    return df_out

def append_rows(df, new_df):
    """
    Append all rows from the second new data frame to the first old data frame by overwriting existing rows if any.

    Notes:
    - The rows are appended individually (row-by-row in the loop). If the second (new) data frame is large then this operation might be inefficient.
    - No validity check are performed like column overlaps or resulting index gaps.
    - If it is necessary to retain continuous index then additional validations have to be done (before or after this operation).
    """
    for idx, row in new_df.iterrows():
        df.loc[idx] = row
    return df

def append_df_drop_concat(df, new_df):

    # Drop explicitly last rows in the overlap range
    #df_wo_overlap = df.drop(new_df.index[:], errors="ignore")  # ignore errors in case of missing indexes (for new rows). Set in_place if necessary

    # Drop explicitly last rows in the overlap range
    df_wo_overlap = df[:new_df.index[0]]  # Select only records till the first index in the new frame. Assume the rows in the new frame are ordered
    df_wo_overlap = df_wo_overlap.iloc[:-1]  # Remove last element because slicing above includes the range right side

    # Drop explicitly last rows in the overlap range
    #last_idx = df.index.get_loc(new_df.index[0])
    #df_wo_overlap = df.iloc[:last_idx]

    df3 = pd.concat([df_wo_overlap, new_df])  # Append new rows with overlap removed above

    return df3

def append_df_combine_update(df, new_df):
    # The first old frame has priority and its values will be always retained if available (non-null).
    # Only if an old value is null or a new row was appended, the new values from new frame is used.
    # Update null (and only null) elements with values in the same location in other (if any, that is, null is not used for overwriting).
    df2 = df.combine_first(new_df)
    # Yet, we want to have new values overwrite even old non-null values so enforce complete overwriting (inplace)
    df2.update(new_df)
    return df2

def merge_data_sources(data_sources: list, time_column: str, freq: str, merge_interpolate: bool):
    """
    Create one data frame by merging multiple source data frames on the specified time column by generating a common time raster.

    :param data_sources: list of dicts where each dict describes one data source and stores its data in df
    :param time_column: column name with timestamps
    :param freq: pandas frequency for the common time raster like 1min, 1h etc.
    :param merge_interpolate: if True then the missing values will be interpolated
    :return: data frame with all data merged on timestamps
    """
    for ds in data_sources:
        df = ds.get("df")

        if time_column in df.columns:
            df = df.set_index(time_column)
        elif df.index.name == time_column:
            pass
        else:
            print(f"ERROR: Timestamp column is absent.")
            return

        # Add prefix if not already there
        if ds['column_prefix']:
            #df = df.add_prefix(ds['column_prefix']+"_")
            df.columns = [
                ds['column_prefix']+"_"+col if not col.startswith(ds['column_prefix']+"_") else col
                for col in df.columns
            ]

        ds["start"] = df.first_valid_index()  # df.index[0]
        ds["end"] = df.last_valid_index()  # df.index[-1]

        ds["df"] = df

    #
    # Create common (main) index and empty data frame
    #
    range_start = min([ds["start"] for ds in data_sources])
    range_end = min([ds["end"] for ds in data_sources])

    # Generate a discrete time raster according to the (pandas) frequency parameter
    index = pd.date_range(range_start, range_end, freq=freq)

    df_out = pd.DataFrame(index=index)
    df_out.index.name = time_column
    df_out.insert(0, time_column, df_out.index)  # Repeat index as a new column

    for ds in data_sources:
        # Note that timestamps must have the same semantics, for example, start of kline (and not end of kline)
        # If different data sets have different semantics for timestamps, then data must be shifted accordingly
        df_out = df_out.join(ds["df"])

    # Interpolate numeric columns
    if merge_interpolate:
        num_columns = df_out.select_dtypes((float, int)).columns.tolist()
        for col in num_columns:
            df_out[col] = df_out[col].interpolate()

    return df_out

#
# Analysis
#

def compute_scores(y_true, y_hat):
    """Compute several scores and return them as dict."""
    y_true = y_true.astype(int)
    y_hat_class = np.where(y_hat.values > 0.5, 1, 0)

    try:
        auc = metrics.roc_auc_score(y_true, y_hat.fillna(value=0))
    except ValueError:
        auc = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

    try:
        ap = metrics.average_precision_score(y_true, y_hat.fillna(value=0))
    except ValueError:
        ap = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

    f1 = metrics.f1_score(y_true, y_hat_class)
    precision = metrics.precision_score(y_true, y_hat_class)
    recall = metrics.recall_score(y_true, y_hat_class)

    scores = dict(auc=auc, ap=ap, f1=f1, precision=precision, recall=recall,)

    scores = {key: round(float(value), 3) for (key, value) in scores.items()}

    return scores

def compute_scores_regression(y_true, y_hat):
    """Compute regression scores. Input columns must have numeric data type."""

    try:
        mae = metrics.mean_absolute_error(y_true, y_hat)
    except ValueError:
        mae = np.nan

    try:
        mape = metrics.mean_absolute_percentage_error(y_true, y_hat)
    except ValueError:
        mape = np.nan

    try:
        r2 = metrics.r2_score(y_true, y_hat)
    except ValueError:
        r2 = np.nan

    #
    # How good it is in predicting the sign (increase of decrease)
    #

    y_true_class = np.where(y_true.values > 0.0, +1, -1)
    y_hat_class = np.where(y_hat.values > 0.0, +1, -1)

    try:
        auc = metrics.roc_auc_score(y_true_class, y_hat_class)
    except ValueError:
        auc = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

    try:
        ap = metrics.average_precision_score(y_true_class, y_hat_class)
    except ValueError:
        ap = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

    f1 = metrics.f1_score(y_true_class, y_hat_class)
    precision = metrics.precision_score(y_true_class, y_hat_class)
    recall = metrics.recall_score(y_true_class, y_hat_class)

    scores = dict(
        mae=mae, mape=mape, r2=r2,
        auc=auc, ap=ap, f1=f1, precision=precision, recall=recall,
    )

    scores = {key: round(float(value), 3) for (key, value) in scores.items()}

    return scores
