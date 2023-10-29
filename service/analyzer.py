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
from common.feature_generation import *
from common.signal_generation import *
from common.model_store import *

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

        score_aggregation_sets = App.config['score_aggregation_sets']
        all_labels = list(itertools.chain.from_iterable([x.get("buy_labels") + x.get("sell_labels") for x in score_aggregation_sets]))

        self.models = {label: load_model_pair(model_path, label) for label in all_labels}

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
        now_ts = now_timestamp()
        last_kline_ts = self.get_last_kline_ts(symbol)
        if not last_kline_ts:
            return App.config["features_horizon"]
        end_of_last_kline = last_kline_ts + 60_000  # Plus 1m because kline timestamp is

        minutes = (now_ts - end_of_last_kline) / 60_000
        minutes += 2
        return int(minutes)

    def store_klines(self, data: dict):
        """
        Store latest klines for the specified symbols.
        Existing klines for the symbol and timestamp will be overwritten.

        :param data: Dict of lists with symbol as a key, and list of klines for this symbol as a value.
            Example: { 'BTCUSDT': [ [], [], [] ] }
        :type dict:
        """
        now_ts = now_timestamp()

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
                    if ts - prev_ts != 60_000:
                        log.error("Wrong sequence of klines. They are expected to be a regular time series with 1m frequency.")
                prev_ts = kline[0]

            # Debug message about the last received kline end and current ts (which must be less than 1m - rather small delay)
            log.debug(f"Stored klines. Total {len(klines_data)} in db. Last kline end: {self.get_last_kline_ts(symbol)+60_000}. Current time: {now_ts}")

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

    def analyze(self):
        """
        1. Convert klines to df
        2. Derive (compute) features (use same function as for model training)
        3. Derive (predict) labels by applying models trained for each label
        4. Generate buy/sell signals by applying rule models trained for best overall trade performance
        """
        symbol = App.config["symbol"]

        last_kline_ts = self.get_last_kline_ts(symbol)
        last_kline_ts_str = str(pd.to_datetime(last_kline_ts, unit='ms'))

        log.info(f"Analyze {symbol}. Last kline timestamp: {last_kline_ts_str}")

        #
        # 1.
        # MERGE: Produce a single data frame with înput data from all sources
        #
        data_sources = App.config.get("data_sources", [])
        if not data_sources:
            data_sources = [{"folder": App.config["symbol"], "file": "klines", "column_prefix": ""}]

        # Read data from online sources into data frames
        for ds in data_sources:
            if ds.get("file") == "klines":
                try:
                    klines = self.klines.get(ds.get("folder"))
                    df = klines_to_df(klines)

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

        # Merge in one df with prefixes and common regular time index
        df = merge_data_sources(data_sources)

        #
        # 2.
        # Generate all necessary derived features (NaNs are possible due to short history)
        #

        # We want to generate features only for the last rows (for performance reasons)
        # Therefore, determine how many last rows we will actually need
        score_aggregation_sets = App.config['score_aggregation_sets']
        all_windows = [sa_set.get("parameters", {}).get("window", 0) for sa_set in score_aggregation_sets]
        last_rows = max(all_windows) + 1

        feature_sets = App.config.get("feature_sets", [])
        if not feature_sets:
            log.error(f"ERROR: no feature sets defined. Nothing to process.")
            return
            # By default, we generate standard kline features
            #feature_sets = [{"column_prefix": "", "generator": "itblib", "feature_prefix": ""}]

        # Apply all feature generators to the data frame which get accordingly new derived columns
        # The feature parameters will be taken from App.config (depending on generator)
        for fs in feature_sets:
            df, _ = generate_feature_set(df, fs, last_rows=last_rows)

        App.feature_df = df  # These data might be needed later on, for example, visualization

        df = df.iloc[-last_rows:]  # For signal generation, only several last rows

        #
        # 3.
        # Apply ML models and generate score columns
        #

        # kline feature set
        features = App.config["train_features"]
        predict_df = df[features]
        if predict_df.isnull().any().any():
            null_columns = {k: v for k, v in predict_df.isnull().any().to_dict().items() if v}
            log.error(f"Null in predict_df found. Columns with Null: {null_columns}")
            return

        # Do prediction by applying all models (for the score columns declared in config) to the data
        algorithms = App.config.get("algorithms")
        score_df = pd.DataFrame(index=predict_df.index)
        try:
            for score_column_name, model_pair in self.models.items():

                label, algo_name = score_to_label_algo_pair(score_column_name)
                model_config = get_algorithm(algorithms, algo_name)  # Get algorithm description from the algo store
                algo_type = model_config.get("algo")

                if algo_type == "gb":
                    df_y_hat = predict_gb(model_pair, predict_df, model_config)
                elif algo_type == "nn":
                    df_y_hat = predict_nn(model_pair, predict_df, model_config)
                elif algo_type == "lc":
                    df_y_hat = predict_lc(model_pair, predict_df, model_config)
                elif algo_type == "svc":
                    df_y_hat = predict_svc(model_pair, predict_df, model_config)
                else:
                    raise ValueError(f"Unknown algorithm type '{algo_type}'")

                score_df[score_column_name] = df_y_hat

        except Exception as e:
            log.error(f"Error in predict: {e}: '{score_column_name=}', '{algo_name=}")
            return

        # This df contains only one (last) record
        df = df.join(score_df)
        #df = pd.concat([predict_df, score_df], axis=1)

        #
        # 4.
        # Aggregate and post-process
        #
        score_aggregation_sets = App.config['score_aggregation_sets']
        # Temporary (post-processed) columns for each aggregation set
        buy_column = 'aggregated_buy_score'
        sell_column = 'aggregated_sell_score'
        score_column_names = []
        for i, sa_set in enumerate(score_aggregation_sets):

            buy_labels = sa_set.get("buy_labels")
            sell_labels = sa_set.get("sell_labels")
            if set(buy_labels + sell_labels) - set(df.columns):
                missing_labels = list(set(buy_labels + sell_labels) - set(df.columns))
                print(f"ERROR: Some buy/sell labels from config are not present in the input data. Missing labels: {missing_labels}")
                return

            parameters = sa_set.get("parameters", {})
            # Aggregate predictions of different algorithms separately for buy and sell
            aggregate_scores(df, parameters, buy_column, buy_labels)  # Output is buy column
            aggregate_scores(df, parameters, sell_column, sell_labels)  # Output is sell column

            score_column = sa_set.get("column")
            score_column_names.append(score_column)

            # Here we want to take into account relative values of buy and sell scores
            # Mutually adjust two independent scores with opposite buy/sell semantics
            combine_scores(df, parameters, buy_column, sell_column, score_column)
        # Delete temporary columns
        del df[buy_column]
        del df[sell_column]

        #
        # 5.
        # Apply rule to last row
        #
        trade_model = App.config['trade_model']
        if trade_model.get('rule_name') == 'two_dim_rule':
            apply_rule_with_score_thresholds_2(df, score_column_names, trade_model)
        else:  # Default one dim rule
            apply_rule_with_score_thresholds(df, score_column_names, trade_model)

        #
        # 6.
        # Collect results and create signal object
        #
        row = df.iloc[-1]  # Last row stores the latest values we need

        close_price = row["close"]
        close_time = row.name+timedelta(minutes=1)  # Add 1 minute because timestamp is start of the interval

        trade_scores = [row[col] for col in score_column_names]

        signal_column_names = trade_model.get("signal_columns")
        buy_signal_column = signal_column_names[0]
        sell_signal_column = signal_column_names[1]
        buy_signal = row[buy_signal_column]
        sell_signal = row[sell_signal_column]

        signal = dict(
            side="",
            trade_score=trade_scores,
            buy_signal=buy_signal, sell_signal=sell_signal,
            close_price=close_price, close_time=close_time
        )

        if any(x is None or not np.isfinite(x) for x in trade_scores):
            log.warning(f"Null or infinite/nan score found: {trade_scores=}. Score ignored. No signal.")
            pass  # Something is wrong with the computation results
        elif buy_signal and sell_signal:  # Both signals are true - should not happen
            signal["side"] = "BOTH"
        elif buy_signal:
            signal["side"] = "BUY"
        elif sell_signal:
            signal["side"] = "SELL"
        else:
            signal["side"] = ""

        App.signal = signal

        scores = [f"{x:+.3f}" for x in trade_scores]
        log.info(f"Analyze finished. Signal: {signal['side']}. Trade scores: {scores}. Price: {int(close_price):,}")


if __name__ == "__main__":
    pass
