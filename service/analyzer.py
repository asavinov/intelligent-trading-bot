from pathlib import Path
from typing import Union
import json
import pickle
from datetime import datetime, date, timedelta
import queue

import numpy as np
import pandas as pd

from common.utils import *
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

    Properties of klines:
    - "timestamp" is a left border of the interval like "2017-08-17 04:00:00"
    - "close_time" is a right border of the interval in ms (last millisecond) like "1502942459999" equivalent to "2017-08-17 04:00::59.999"
    """

    def __init__(self, config: dict, model_store: ModelStore):
        """
        Create a new operation object using its definition.

        :param config: Initialization parameters defining what is in the database including its persistent parameters and schema
        """

        self.config = config
        self.model_store = model_store

        #
        # Raw input data
        # Klines are stored as a dict of lists. Key is a symbol and the list is a list of latest kline records
        # One kline record is a list of values (not dict) as returned by API: open time, open, high, low, close, volume etc.
        #
        self.klines = {}

        #
        # Data frame with all the data (source and derived) where rows are appended and their (derived) columns are computed
        #
        self.df = None

        # Minimum length of the data frame determined by such criteria as the necessary history to compute new values or how much we want to output
        self.min_window_length = self.config["predict_length"] + self.config["features_horizon"]
        # After growing larger than maximum the array will be truncated (back to the minimum required size) by removing older records
        self.max_window_length = self.min_window_length + 15
        # How many records to request in addition to the necessary missing records
        self.append_overlap_records = self.config.get("append_overlap_records", 5)
        # How many tail records must be recomputed. It is a variable field
        # It is set to 0 after each analysis and then again set to non-0 value after new records are appended
        # If equal to -1 (or too large) then all records will be re-computed in batch mode (rather than stream mode)
        self.dirty_records = -1

    #
    # Data state operations
    #

    def get_klines_count(self, symbol):
        return len(self.klines.get(symbol, []))

    def get_last_kline(self, symbol):
        if self.get_klines_count(symbol) > 0:
            return self.klines.get(symbol)[-1]
        else:
            return None

    def get_last_kline_ts(self, symbol):
        """Open time of the last kline. It is simultaneously kline id. Add 1m if the end is needed."""
        last_kline = self.get_last_kline(symbol=symbol)
        if not last_kline:
            return 0
        last_kline_ts = last_kline[0]
        return last_kline_ts

    def get_missing_klines_count(self, symbol):
        """
        The number of complete discrete intervals between the last available kline and current timestamp.
        The interval length is determined by the frequency parameter.
        """
        last_kline_ts = self.get_last_kline_ts(symbol)
        if not last_kline_ts:
            return self.min_window_length

        freq = self.config["freq"]
        interval_length = pd.Timedelta(freq).to_pytimedelta()

        now = datetime.now(timezone.utc)
        last_kline = datetime.fromtimestamp(last_kline_ts // 1000, timezone.utc)

        intervals_count = (now-last_kline) // interval_length

        return intervals_count + self.append_overlap_records

    def append_klines(self, data: dict):
        """
        Append new (latest) klines for the specified symbols to the list.
        Existing klines for the symbol and timestamp will be overwritten.

        :param data: Dict of lists with symbol as a key, and list of klines for this symbol as a value.
            Example: { 'BTCUSDT': [ [], [], [] ] }
        :type dict:
        """
        now_ts = now_timestamp()
        freq = self.config["freq"]
        interval_length_ms = pandas_interval_length_ms(freq)

        max_new_kline_size = max([len(v) for v in data.values()])
        is_initialized = False
        min_deleted_overlaps = max_new_kline_size

        for symbol, new_klines in data.items():

            existing_klines = self.klines.get(symbol)
            if existing_klines is None:  # Cold start: if no data for the symbol then initialize
                self.klines[symbol] = []
                existing_klines = self.klines.get(symbol)
                is_initialized = True

            # Find overlap and delete
            # Overlap are existing klines which are younger than the first new kline
            ts = new_klines[0][0]  # Very first timestamp of the new data
            # same_kline = next((x for x in klines_data if x[0] == ts), None)
            existing_indexes = [i for i, x in enumerate(existing_klines) if x[0] >= ts]
            if existing_indexes:  # If there is overlap with new klines
                start = min(existing_indexes)
                num_deleted = len(existing_klines) - start
                min_deleted_overlaps = min(min_deleted_overlaps, num_deleted)
                del existing_klines[start:]  # Delete starting from the first kline in new data (which will be added below)
                if len(new_klines) < num_deleted:  # Validation: it is expected that we will add same or more klines than deleted
                    log.error("More klines is deleted by new klines added, than we actually add. Something working with timestamps and storage logic.")

            # Append new klines to the (instead of and in addition to the deleted above)
            existing_klines.extend(new_klines)

            # Remove old klines from the tail to limit its length
            to_delete = len(existing_klines) - self.max_window_length
            if to_delete > 0:
                del existing_klines[:to_delete]

            # Check validity. It has to be an ordered time series with certain frequency
            for i, kline in enumerate(existing_klines):
                ts = kline[0]
                if i > 0:
                    if ts - prev_ts != interval_length_ms:
                        log.error("Wrong sequence of klines. They are expected to be a regular time series with 1m frequency.")
                prev_ts = kline[0]

            # Debug message about the last received kline end and current ts (which must be less than 1m - rather small delay)
            log.debug(f"Stored klines. Total {len(existing_klines)} in db. Last kline end: {self.get_last_kline_ts(symbol)+interval_length_ms}. Current time: {now_ts}")

        # This number of records are new and have to be re-computed. -1 means all records
        if self.dirty_records < 0:
            pass  # No change: still all records have to be re-computed
        elif is_initialized:
            self.dirty_records = -1  # All records are new therefore all re-compute
        else:
            # For example, we already had 10 records dirty and 4 were deleted (overwritten by new records)
            old_dirty = self.dirty_records - min_deleted_overlaps
            self.dirty_records = max(max_new_kline_size, max_new_kline_size+old_dirty)

    def analyze(self):
        """
        Compute derived columns in the data frame.
        1. Convert klines to df
        2. Derive (compute) features (use same function as for model training)
        3. Derive (predict) labels by applying models trained for each label
        4. Generate buy/sell signals by applying rule models trained for best overall trade performance
        """
        symbol = self.config["symbol"]

        if self.dirty_records == 0:
            log.warning(f"Analysis function called with 0 dirty records. Exit because there is nothing to do. Normally should not happen.")
            return

        last_rows = self.dirty_records if self.dirty_records > 0 else 0

        last_kline_ts = self.get_last_kline_ts(symbol)
        last_kline_ts_str = str(pd.to_datetime(last_kline_ts, unit='ms'))

        log.info(f"Analyze {symbol}. Last kline timestamp: {last_kline_ts_str}")

        #
        # Convert source data (klines) into data frames for each data source
        #
        data_sources = self.config.get("data_sources", [])
        for ds in data_sources:
            if ds.get("file") == "klines":
                try:
                    klines = self.klines.get(ds.get("folder"))
                    df = binance_klines_to_df(klines)

                    # Validate
                    source_columns = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']
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
        # 1.
        # MERGE multiple dfs in one df with prefixes and common regular time index
        #
        df = merge_data_sources(data_sources)

        #
        # 2.
        # Generate all necessary derived features (NaNs are possible due to limited history)
        #
        feature_sets = self.config.get("feature_sets", [])
        feature_columns = []
        for fs in feature_sets:
            df, feats = generate_feature_set(df, fs, self.config, self.model_store, last_rows=last_rows)
            feature_columns.extend(feats)

        # Shorten the data frame. Only several last rows will be needed and not the whole data context
        if last_rows:
            df = df.iloc[-last_rows:]

        features = self.config["train_features"]
        # Exclude rows with at least one NaN in feature columns
        tail_rows = notnull_tail_rows(df[features])
        df = df.tail(tail_rows)

        #
        # 3.
        # Apply ML models and generate score columns
        #

        # Select row for which to do predictions
        predict_df = df[features]
        if predict_df.isnull().any().any():
            null_columns = {k: v for k, v in predict_df.isnull().any().to_dict().items() if v}
            log.error(f"Null in predict_df found. Columns with Null: {null_columns}")
            return

        train_feature_sets = self.config.get("train_feature_sets", [])
        score_df = pd.DataFrame(index=predict_df.index)
        train_feature_columns = []
        for fs in train_feature_sets:
            fs_df, feats, _ = predict_feature_set(predict_df, fs, self.config, self.model_store)
            score_df = pd.concat([score_df, fs_df], axis=1)
            train_feature_columns.extend(feats)

        # Attach all predicted features to the main data frame
        df = pd.concat([df, score_df], axis=1)

        #
        # 4.
        # Signals
        #
        signal_sets = self.config.get("signal_sets", [])
        signal_columns = []
        for fs in signal_sets:
            df, feats = generate_feature_set(df, fs, self.config, self.model_store, last_rows=last_rows)
            signal_columns.extend(feats)

        #
        # Append the new rows to the main data frame with all previously computed data
        #

        # Log signal values
        row = df.iloc[-1]  # Last row stores the latest values we need
        scores = ", ".join([f"{x}={row[x]:+.3f}" if isinstance(row[x], float) else f"{x}={str(row[x])}" for x in signal_columns])
        log.info(f"Analyze finished. Close: {int(row['close']):,} Signals: {scores}")

        if self.df is None or len(self.df) == 0:
            self.df = df
            self.dirty_records = 0  # Everything is computed
            return

        # Validation: newly retrieved and computed values should be equal to those computed previously
        check_row_count = 3  # These last rows must be correctly computed (particularly, have enough history in case of aggregation)
        num_cols = df.select_dtypes((float, int)).columns.tolist()
        # Loop over several last newly computed data rows
        # Skip last row because it should not exist, and before the last row because its kline is frequently updated after retrieval
        for r in range(2, min(check_row_count, len(df))):
            idx = df.index[-r-1]

            if idx not in self.df.index:
                continue

            # Compare all numeric values of the previously retrieved and newly retrieved rows for the same time
            old_row = self.df[num_cols].loc[idx]
            new_row = df[num_cols].loc[idx]
            comp_idx = np.isclose(old_row, new_row)
            if not np.all(comp_idx):
                log.warning(f"Newly computed row is not equal to the previously computed row for '{idx}'. NEW: {new_row[~comp_idx].to_dict()}. OLD: {old_row[~comp_idx].to_dict()}")

        # Append newly computed rows to the main data frame by overwriting latest existing rows in case of overlap
        if self.dirty_records > 0:
            self.df = df.tail(self.dirty_records).combine_first(self.df)
        else:
            self.df = df.combine_first(self.df)

        self.dirty_records = 0  # Everything is computed

        # Remove too old rows
        if len(self.df) > self.max_window_length:
            self.df = self.df.tail(self.max_window_length)


if __name__ == "__main__":
    pass
