from pathlib import Path
from typing import Union
import json
import pickle
from datetime import datetime, date, timedelta
import queue

import numpy as np
import pandas as pd

from service.App import *
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

    def __init__(self, config):
        """
        Create a new operation object using its definition.

        :param config: Initialization parameters defining what is in the database including its persistent parameters and schema
        """

        self.config = config

        #
        # Data state
        #

        # Klines are stored as a dict of lists. Key is a symbol and the list is a list of latest kline records
        # One kline record is a list of values (not dict) as returned by API: open time, open, high, low, close, volume etc.
        self.klines = {}

        self.queue = queue.Queue()

        #
        # Load models
        #
        symbol = App.config["symbol"]
        data_path = Path(App.config["data_folder"]) / symbol

        model_path = Path(App.config["model_folder"])
        if not model_path.is_absolute():
            model_path = data_path / model_path
        model_path = model_path.resolve()

        labels = App.config["labels"]
        algorithms = App.config["algorithms"]
        self.models = load_models(model_path, labels, algorithms)

        # Load latest transaction and (simulated) trade state
        App.transaction = load_last_transaction()

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
            return App.config["features_horizon"]

        freq = App.config["freq"]
        now = datetime.utcnow()
        last_kline = datetime.utcfromtimestamp(last_kline_ts // 1000)
        interval_length = pd.Timedelta(freq).to_pytimedelta()
        intervals_count = (now-last_kline) // interval_length

        intervals_count += 2
        return intervals_count

    def store_klines(self, data: dict):
        """
        Store latest klines for the specified symbols.
        Existing klines for the symbol and timestamp will be overwritten.

        :param data: Dict of lists with symbol as a key, and list of klines for this symbol as a value.
            Example: { 'BTCUSDT': [ [], [], [] ] }
        :type dict:
        """
        now_ts = now_timestamp()
        freq = App.config["freq"]
        interval_length_ms = pandas_interval_length_ms(freq)

        for symbol, klines in data.items():
            # If symbol does not exist then create
            klines_data = self.klines.get(symbol)
            if klines_data is None:
                self.klines[symbol] = []
                klines_data = self.klines.get(symbol)

            ts = klines[0][0]  # Very first timestamp of the new data

            # Find kline with this or younger timestamp in the database
            # same_kline = next((x for x in klines_data if x[0] == ts), None)
            existing_indexes = [i for i, x in enumerate(klines_data) if x[0] >= ts]
            #print(f"===>>> Existing tss: {[x[0] for x in klines_data]}")
            #print(f"===>>> New tss: {[x[0] for x in klines]}")
            #print(f"===>>> {symbol} Overlap {len(existing_indexes)}. Existing Indexes: {existing_indexes}")
            if existing_indexes:  # If there is overlap with new klines
                start = min(existing_indexes)
                num_deleted = len(klines_data) - start
                del klines_data[start:]  # Delete starting from the first kline in new data (which will be added below)
                if len(klines) < num_deleted:  # It is expected that we add same or more klines than deleted
                    log.error("More klines is deleted by new klines added, than we actually add. Something woring with timestamps and storage logic.")

            # Append new klines
            klines_data.extend(klines)

            # Remove too old klines
            kline_window = App.config["features_horizon"]
            to_delete = len(klines_data) - kline_window
            if to_delete > 0:
                del klines_data[:to_delete]

            # Check validity. It has to be an ordered time series with certain frequency
            for i, kline in enumerate(self.klines.get(symbol)):
                ts = kline[0]
                if i > 0:
                    if ts - prev_ts != interval_length_ms:
                        log.error("Wrong sequence of klines. They are expected to be a regular time series with 1m frequency.")
                prev_ts = kline[0]

            # Debug message about the last received kline end and current ts (which must be less than 1m - rather small delay)
            log.debug(f"Stored klines. Total {len(klines_data)} in db. Last kline end: {self.get_last_kline_ts(symbol)+interval_length_ms}. Current time: {now_ts}")

    def store_depth(self, depths: list, freq):
        """
        Persistently store order books from the input list. Each entry is one response from order book request for one symbol.
        Currently the order books are directly stored in a file (for this symbol) and not in this object.

        :param depths: List of dicts where each dict is an order book with such fields as 'asks', 'bids' and 'symbol' (symbol is added after loading).
        :type list:
        """

        # File name like TRADE_HOME/COLLECT/DEPTH/depth-BTCUSDT-5s.txt
        TRADE_DATA = "."  # TODO: We need to read it from the environment. It could be data dir or docker volume.
        # BASE_DIR = Path(__file__).resolve().parent.parent
        # BASE_DIR = Path.cwd()

        for depth in depths:
            # TODO: The result might be an exception or some other object denoting bad return (timeout, cancelled etc.)

            symbol = depth["symbol"]

            path = Path(TRADE_DATA).joinpath(App.config["collector"]["folder"])
            path = path.joinpath(App.config["collector"]["depth"]["folder"])
            path.mkdir(parents=True, exist_ok=True)  # Ensure that dir exists

            file_name = f"depth-{symbol}-{freq}"
            file = Path(path, file_name).with_suffix(".txt")

            # Append to the file (create if it does not exist)
            json_line = json.dumps(depth)
            with open(file, 'a+') as f:
                f.write(json_line + "\n")

    def store_queue(self):
        """
        Persistently store the queue data to one or more files corresponding to the stream (event) type, symbol (and frequency).

        :return:
        """
        #
        # Get all the data from the queue
        #
        events = {}
        item = None
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty as ee:
                break
            except:
                break

            if item is None:
                break

            c = item.get("e")  # Channel
            if not events.get(c):  # Insert if does not exit
                events[c] = {}
            symbols = events[c]

            s = item.get("s")  # Symbol
            if not symbols.get(s):  # Insert if does not exit
                symbols[s] = []
            data = symbols[s]

            data.append(item)

            self.queue.task_done()  # TODO: Do we really need this?

        # File name like TRADE_HOME/COLLECT/DEPTH/depth-BTCUSDT-5s.txt
        TRADE_DATA = "."  # TODO: We need to read it from the environment. It could be data dir or docker volume.
        # BASE_DIR = Path(__file__).resolve().parent.parent
        # BASE_DIR = Path.cwd()

        path = Path(TRADE_DATA).joinpath(App.config["collector"]["folder"])
        path = path.joinpath(App.config["collector"]["stream"]["folder"])
        path.mkdir(parents=True, exist_ok=True)  # Ensure that dir exists

        now = datetime.utcnow()
        #rotate_suffix = f"{now:%Y}{now:%m}{now:%d}"  # Daily files
        rotate_suffix = f"{now:%Y}{now:%m}"  # Monthly files

        #
        # Get all the data from the queue and store in file
        #
        for c, symbols in events.items():
            for s, data in symbols.items():
                file_name = f"{c}-{s}-{rotate_suffix}"
                file = Path(path, file_name).with_suffix(".txt")

                # Append to the file (create if it does not exist)
                data = [json.dumps(event) for event in data]
                data_str = "\n".join(data)
                with open(file, 'a+') as f:
                    f.write(data_str + "\n")

    #
    # Analysis (features, predictions, signals etc.)
    #

    def analyze(self, ignore_last_rows=False):
        """
        1. Convert klines to df
        2. Derive (compute) features (use same function as for model training)
        3. Derive (predict) labels by applying models trained for each label
        4. Generate buy/sell signals by applying rule models trained for best overall trade performance
        """
        symbol = App.config["symbol"]

        # Features, predictions, signals etc. have to be computed only for these last rows (for performance reasons)
        last_rows = App.config["features_last_rows"]

        last_kline_ts = self.get_last_kline_ts(symbol)
        last_kline_ts_str = str(pd.to_datetime(last_kline_ts, unit='ms'))

        log.info(f"Analyze {symbol}. Last kline timestamp: {last_kline_ts_str}")

        #
        # Convert source data (klines) into data frames for each data source
        #
        data_sources = App.config.get("data_sources", [])
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
        feature_sets = App.config.get("feature_sets", [])
        feature_columns = []
        for fs in feature_sets:
            df, feats = generate_feature_set(df, fs, last_rows=last_rows if not ignore_last_rows else 0)
            feature_columns.extend(feats)

        # Shorten the data frame. Only several last rows will be needed and not the whole data context
        if not ignore_last_rows:
            df = df.iloc[-last_rows:]

        features = App.config["train_features"]
        # Exclude rows with at least one NaN
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

        train_feature_sets = App.config.get("train_feature_sets", [])
        if not train_feature_sets:
            log.error(f"ERROR: no train feature sets defined. Nothing to process.")
            return

        # Apply all train feature generators to the data frame by generating predicted columns
        score_df = pd.DataFrame(index=predict_df.index)
        train_feature_columns = []
        for fs in train_feature_sets:
            fs_df, feats, _ = predict_feature_set(predict_df, fs, App.config, self.models)
            score_df = pd.concat([score_df, fs_df], axis=1)
            train_feature_columns.extend(feats)

        # Attach all predicted features to the main data frame
        df = pd.concat([df, score_df], axis=1)

        #
        # 4.
        # Signals
        #
        signal_sets = App.config.get("signal_sets", [])
        if not signal_sets:
            log.error(f"ERROR: no signal sets defined. Nothing to process.")
            return

        # Apply all feature generators to the data frame which get accordingly new derived columns
        signal_columns = []
        for fs in signal_sets:
            df, feats = generate_feature_set(df, fs, last_rows=last_rows if not ignore_last_rows else 0)
            signal_columns.extend(feats)

        #
        # Append the new rows to the main data frame with all previously computed data
        #

        # Log signal values
        row = df.iloc[-1]  # Last row stores the latest values we need
        scores = ", ".join([f"{x}={row[x]:+.3f}" if isinstance(row[x], float) else f"{x}={str(row[x])}" for x in signal_columns])
        log.info(f"Analyze finished. Close: {int(row['close']):,} Signals: {scores}")

        if App.df is None or len(App.df) == 0:
            App.df = df
            return

        # Test if newly retrieved and computed values are equal to the previous ones
        check_row_count = 3  # These last rows must be correctly computed (particularly, have enough history in case of aggregation)
        num_cols = df.select_dtypes((float, int)).columns.tolist()
        # Loop over several last newly computed data rows
        # Skip last row because it should not exist, and before the last row because its kline is frequently updated after retrieval
        for r in range(2, check_row_count):
            idx = df.index[-r-1]

            if idx not in App.df.index:
                continue

            # Compare all numeric values of the previously retrieved and newly retrieved rows for the same time
            old_row = App.df[num_cols].loc[idx]
            new_row = df[num_cols].loc[idx]
            comp_idx = np.isclose(old_row, new_row)
            if not np.all(comp_idx):
                log.warning(f"Newly computed row is not equal to the previously computed row for '{idx}'. NEW: {new_row[~comp_idx].to_dict()}. OLD: {old_row[~comp_idx].to_dict()}")

        # Append new rows to the main data frame
        App.df = df.tail(check_row_count).combine_first(App.df)

        # Remove too old rows
        features_horizon = App.config["features_horizon"]
        if len(App.df) > features_horizon + 15:
            App.df = App.df.tail(features_horizon)


if __name__ == "__main__":
    pass
