from pathlib import Path
from typing import Union
import json
import pickle
from datetime import datetime, date, timedelta
import queue

import numpy as np
import pandas as pd

from common.utils import *
from inputs.collector_binance import klines_to_df, column_types
from common.classifiers import *
from common.model_store import *
from common.generators import generate_feature_set
from common.generators import predict_feature_set

from scripts.merge import *
from scripts.features import *

import logging
log = logging.getLogger('analyzer')


class Analyzer:
    """
    In-memory database which represents the current state of the (trading) environment including its history.
    """

    def __init__(self, config: dict, model_store: ModelStore):
        """
        Create a new operation object using its definition.

        :param config: Initialization parameters defining what is in the database including its persistent parameters and schema
        """

        self.config = config
        self.model_store = model_store

        #
        # Data shape and parameters
        #

        # Minimum length of the data frame determined by such criteria as the necessary history to compute new values or how much we want to output
        self.min_window_length = self.config["predict_length"] + self.config["features_horizon"]
        # After growing larger than maximum the array will be truncated (back to the minimum required size) by removing older records
        self.max_window_length = self.min_window_length + 15

        # How many records to request in addition to the necessary missing records
        self.append_overlap_records = self.config.get("append_overlap_records", 5)

        # How many tail records store empty/wrong values and must be recomputed.
        # Initially it is equal to the data size (all records have to be recomputed). -1 (or too large) means that all records will be re-computed in batch mode (rather than stream mode).
        # After each analysis it is set to 0 which means that all features were evaluated and data is up-to-date.
        # After appending new records it again set to non-0 value by indicating that the data state needs to be re-evaluated.
        self.dirty_records = -1

        self.is_train = config.get("train")
        if self.is_train:
            print(f"WARNING: Train mode is specified although the server is intended for running in predict mode")

        #
        # Data frame with all the data (source and derived) where rows are appended and their (derived) columns are computed
        #

        # All explicitly declared in config derived feature columns
        train_features = self.config.get("train_features", [])
        train_features_dtypes = {k: 'float64' for k in train_features}  # Same data type

        # All explicitly declared in config label columns if in train mode
        labels = self.config.get("labels", [])
        labels_dtypes = {k: 'float64' for k in labels}  # Same data type

        # Combine all raw columns, derived features and (if train mode) label columns
        all_columns_dtypes = column_types | train_features_dtypes
        if self.is_train:
            all_columns_dtypes = all_columns_dtypes | labels_dtypes

        self.df = pd.DataFrame(columns=all_columns_dtypes).astype(all_columns_dtypes)
        self.df = self.df.set_index('timestamp', inplace=False, drop=False)  # timestamp column in the index and also as a column for convenience
        # Now the data frame is initialized and regular updates will append rows to it followed by analysis (computation of derived features)

        self.previous_df = None  # For validation

    #
    # Data state operations
    #

    def get_size(self):
        return len(self.df)

    def get_last_kline(self):
        if len(self.df) > 0:
            return self.df.iloc[-1]
        else:
            return None

    def get_last_kline_dt(self):
        """Open time of the last kline. It is simultaneously kline id. Add 1m if the end is needed."""
        if len(self.df) > 0:
            return self.df.index[-1]
        else:
            return None

    def get_missing_klines_count(self):
        """
        The number of complete discrete intervals between the last available kline and current timestamp.
        The interval length is determined by the frequency parameter.
        """
        last_kline_dt = self.get_last_kline_dt()
        if not last_kline_dt:
            return self.min_window_length

        freq = self.config["freq"]
        interval_length_td = pd.Timedelta(freq).to_pytimedelta()

        now = datetime.now(timezone.utc)
        intervals_count = (now-last_kline_dt) // interval_length_td  # How many whole intervals

        return intervals_count + self.append_overlap_records

    def append_klines(self, data: dict):
        """
        Convert the input klines to data frames, merge them and append to the main data frame by overwriting existing klines in case of overlap.

        :param data: Dict of lists with symbol as a key, and list of klines for this symbol as a value.
            Example: { 'BTCUSDT': [ [], [], [] ] }
        :type dict:
        """
        freq = self.config["freq"]
        interval_length_ms = pandas_interval_length_ms(freq)
        interval_length_td = pd.Timedelta(freq).to_pytimedelta()

        #
        # Convert source data (klines) into data frames for each data source
        #
        data_sources = self.config.get("data_sources", [])
        for ds in data_sources:
            if ds.get("file") == "klines":
                try:
                    ds_symbol = ds.get("folder")
                    klines = data.get(ds_symbol)
                    df = klines_to_df(klines)

                    # Validate
                    if df.isnull().any().any():
                        null_columns = {k: v for k, v in df.isnull().any().to_dict().items() if v}
                        log.warning(f"Null in source data found. Columns with Null: {null_columns}")
                    # TODO: We might receive empty strings or 0s in numeric data - how can we detect them?
                    # TODO: Check that timestamps in 'close_time' are strictly consecutive
                except Exception as e:
                    log.error(f"Error in klines_to_df method: {e}. Length klines: {len(klines)}")
                    return
            else:
                log.error("Unknown data sources. Currently only 'klines' is supported. Check 'data_sources' in config, key 'file'")
                return
            ds["df"] = df

        #
        # Merge multiple dfs in one df with prefixes and common regular time index
        #
        df = merge_data_sources(data_sources)

        #
        # Append to the main data frame (by overwriting existing overlap)
        #
        self.previous_df = self.df.tail(10).copy()

        initial_df_len = len(self.df)
        appended_df_len = len(df)
        self.df = append_df_drop_concat(self.df, df)
        result_df_len = len(self.df)
        overwritten_rows_len = (initial_df_len + appended_df_len) - result_df_len  # 0 if exact concatenation, positive if overlap, negative if gap (error)

        #
        # Compute and set new dirty count (how many to re-evaluate, the current number might be more than 0, e.g., if we append without evaluation)
        #
        if self.dirty_records < 0:
            pass  # No change: still all records have to be re-computed
        elif initial_df_len == 0:
            self.dirty_records = -1  # All records are new therefore all re-compute
        else:
            # For example, we already had 10 records dirty and 4 were deleted (overwritten by new records)
            # All appended rows have to be re-computed. Plus the non-overwritten rows from the old dirty rows (normally 0)
            self.dirty_records = appended_df_len + max(0, self.dirty_records-overwritten_rows_len)
            if self.dirty_records >= result_df_len:
                self.dirty_records = -1

    def analyze(self):
        """
        Compute derived columns for dirty records and add them to the data frame:
        1. Evaluate derived features
        2. Evaluate trained ML features
        3. Evaluate signal features
        """
        symbol = self.config["symbol"]

        if self.dirty_records == 0:
            log.warning(f"Analysis function called with 0 dirty records. Exit because there is nothing to do. Normally should not happen.")
            return

        # It is a parameter passed to generators which indicates the exact (small) number of last rows to re-evaluate to avoid full-evaluation for performance reasons
        last_rows = self.dirty_records if self.dirty_records > 0 else 0

        last_kline_dt = self.get_last_kline_dt()
        last_kline_ts_str = str(pd.to_datetime(last_kline_dt, unit='ms', utc=True))

        log.info(f"Analyze {symbol}. Last kline timestamp: {last_kline_ts_str}")

        #
        # 1. Generate all derived features (NaNs are possible due to limited history)
        #

        feature_sets = self.config.get("feature_sets", [])
        feature_columns = []
        for fs in feature_sets:
            df, feats = generate_feature_set(self.df, fs, self.config, self.model_store, last_rows=last_rows)
            self.df = df
            feature_columns.extend(feats)

        #
        # 2. Apply ML models and generate intelligent indicators
        #

        train_features = self.config["train_features"]

        # Shorten the data frame by selecting last rows for which to do predictions
        tail_rows = notnull_tail_rows(self.df[train_features])  # How many last rows have all non-null values
        predict_size = tail_rows if not last_rows else min(tail_rows, last_rows)
        predict_features_df = self.df.tail(predict_size)

        predict_features_df = predict_features_df[train_features]

        # Validation
        if predict_features_df.isnull().any().any():
            null_columns = {k: v for k, v in predict_features_df.isnull().any().to_dict().items() if v}
            log.error(f"Null in predict_df found. Columns with Null: {null_columns}")
            return

        train_feature_sets = self.config.get("train_feature_sets", [])
        predict_labels_df = pd.DataFrame(index=predict_features_df.index)
        predict_label_columns = []
        for fs in train_feature_sets:
            fs_df, feats = predict_feature_set(predict_features_df, fs, self.config, self.model_store)
            predict_labels_df = pd.concat([predict_labels_df, fs_df], axis=1)
            predict_label_columns.extend(feats)

        # Attach all predicted label columns (only for last rows) to the main data frame
        self.df = self.df.combine_first(predict_labels_df)  # Attach new columns
        self.df.update(predict_labels_df)  # Overwrite older values with newly computed values

        #
        # 3. Signals
        #

        signal_sets = self.config.get("signal_sets", [])
        signal_columns = []
        for fs in signal_sets:
            df, feats = generate_feature_set(self.df, fs, self.config, self.model_store, last_rows=last_rows)
            self.df = df  # TODO: Signal features should be computed in the same way as normal (pre-ML) features
            signal_columns.extend(feats)

        #
        # Append the new rows to the main data frame with all previously computed data
        #

        # Log signal values
        row = self.get_last_kline()  # Last row stores the latest values we need
        scores = ", ".join([f"{x}={row[x]:+.3f}" if isinstance(row[x], float) else f"{x}={str(row[x])}" for x in signal_columns])
        log.info(f"Analyze finished. Close: {int(row['close']):,} Signals: {scores}")

        #
        # Validation: newly retrieved and computed values should be (almost) equal to those computed previously in the overlap area
        #
        check_row_count = 3  # These last rows should be correctly computed (particularly, have enough history in case of aggregation)
        num_cols = self.previous_df.select_dtypes((float, int)).columns.tolist()
        # Loop over several last newly computed data rows
        # Skip last row because it should not exist, and before the last row because its kline is frequently updated after retrieval
        for r in range(2, min(check_row_count, len(self.df))):
            idx = self.df.index[-r-1]

            if idx not in self.previous_df.index:
                continue

            # Compare all numeric values of the previously retrieved and newly retrieved rows for the same time
            old_row = self.previous_df[num_cols].loc[idx]
            new_row = self.df[num_cols].loc[idx]
            comp_idx = np.isclose(old_row, new_row)
            if not np.all(comp_idx):
                log.warning(f"Newly computed row is not equal to the previously computed row for '{idx}'. NEW: {new_row[~comp_idx].to_dict()}. OLD: {old_row[~comp_idx].to_dict()}")

        self.dirty_records = 0  # Everything is computed

        # Remove too old rows
        if len(self.df) > self.max_window_length:
            self.df = self.df.tail(self.max_window_length)


if __name__ == "__main__":
    pass
