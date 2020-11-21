from pathlib import Path
from typing import Union
import json
import pickle
from datetime import datetime, date
import queue

import numpy as np
import pandas as pd

from common.utils import *
from trade.App import App
from trade.feature_prediction import *
from trade.feature_generation import *
from trade.signal_generation import *

import logging
log = logging.getLogger('DB')

class Database:
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
        # Start a thread for storing data
        #


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
        """Open time of the last kline. It is simultaniously kline id. Add 1m if the end is needed."""
        last_kline = self.get_last_kline(symbol=symbol)
        if not last_kline:
            return 0
        last_kline_ts = last_kline[0]
        return last_kline_ts

    def get_missing_klines_count(self, symbol):
        now_ts = now_timestamp()
        last_kline_ts = self.get_last_kline_ts(symbol)
        if not last_kline_ts:
            return App.config["trade"]["analysis"]["kline_window"]
        end_of_last_kline = last_kline_ts + 60_000  # Plus 1m

        minutes = (now_ts - end_of_last_kline) / 60_000
        minutes += 2
        return int(minutes)

    def store_klines(self, data: dict):
        """
        Store latest klines for the specified symbols.
        Existing klines for the symbol and timestamp will be replaced.

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
            kline_window = App.config["trade"]["analysis"]["kline_window"]
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

            path = Path(TRADE_DATA).joinpath(App.config["collect"]["folder"])
            path = path.joinpath(App.config["collect"]["depth"]["folder"])
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

        path = Path(TRADE_DATA).joinpath(App.config["collect"]["folder"])
        path = path.joinpath(App.config["collect"]["stream"]["folder"])
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

    def analyze(self, symbol):
        """
        1. Convert klines to df
        2. Derive (compute) features (use same function as for model training)
        3. Derive (predict) labels by applying models trained for each label
        4. Generate buy/sell signals by applying rule models trained for best overall trade performance
        """
        klines = self.klines.get(symbol)
        last_kline_ts = self.get_last_kline_ts(symbol)

        log.info(f"Analyze {symbol}. {len(klines)} klines in the database. Last kline timestamp: {last_kline_ts}")

        #
        # 1.
        # Produce a data frame with source data
        #
        df = klines_to_df(klines)

        #
        # 2.
        # generate_features(): get data in some form, generate feature matrix by apply some transformations and returning a df
        # the data frame will get additional columns with features
        #
        features_out = generate_features(df)

        # Now we have as many additional columns as we have defined derived features

        #
        # 3.
        # predict_labels(): get df with feature matrix, apply models and generate predictive features, and return a df
        # the data frame will get additional columns with predicted features
        #

        # Prepare (load) models (models are identified by label names and algorithm name)
        labels = App.config["trade"]["analysis"]["labels"]
        model_path = App.config["trade"]["analysis"]["folder"]
        models = {}
        for label in labels:
            model_file_name = label + "_gb"
            model_file = Path(model_path, model_file_name).with_suffix(".pkl")
            if not Path(model_file).exists():
                log.error(f"Model file does not exist: {model_file}")
                return

            with open(model_file, 'rb') as f:
                model = pickle.load(f)

            models[label + "_gb"] = model

        # Prepare input
        features = App.config["trade"]["analysis"]["features"]
        df = df[features]
        # Drop NaN because of limited history and many empty values
        df = df.dropna(subset=features)

        # Do prediction by applying models.
        # New, predicted columns will be added to features. They will be named as in dict
        labels_df = predict_labels(df, models)

        # Shift each score column
        if App.config["trade"]["parameters"]["use_previous_scores"]:
            for label in labels:
                label_prev = label + "_prev"
                labels_df[label_prev] = labels_df[label].shift(1)

        labels_df = labels_df.iloc[1:]  # Drop first row which has NaN for previous (absent) scores

        # Now df will have additionally as many new columns as we have defined labels (and corresponding models)

        #
        # 4.
        # generate_signals(): get prediction features and return buy/sell signals with asset symbol and amount
        # get a dict with the decisions based on the last row of the data frame
        #
        # Currently signals are generated manually for the last row only so we do not use this function
        #signals_out = generate_signals(df, models)

        # Currently the best model is hard-coded. In future, it will be loaded from file
        model = {"threshold_buy_10": 0.26, "threshold_buy_20": 0.07, "percentage_sell_price": 1.018, "sell_timeout": 70}
        threshold_buy_10 = model["threshold_buy_10"]
        threshold_buy_20 = model["threshold_buy_20"]
        threshold_buy_10_prev = float(model.get("threshold_buy_10_prev", 0.0))
        threshold_buy_20_prev = float(model.get("threshold_buy_20_prev", 0.0))
        # Object parameters (label prediction scores)
        row = labels_df.iloc[-1]
        high_60_10_gb = row["high_60_10_gb"]
        high_60_20_gb = row["high_60_20_gb"]

        # Apply model parameters and generate a signal for the current row
        if high_60_10_gb >= threshold_buy_10 and high_60_20_gb >= threshold_buy_20:
            is_buy_signal = True
        else:
            is_buy_signal = False

        if is_buy_signal and App.config["trade"]["parameters"]["use_previous_scores"]:
            high_60_10_gb_prev = row["high_60_10_gb_prev"]
            high_60_20_gb_prev = row["high_60_20_gb_prev"]
            if high_60_10_gb_prev >= threshold_buy_10_prev and high_60_20_gb_prev >= threshold_buy_20_prev:
                is_buy_signal = True
            else:
                is_buy_signal = False

        App.config["trade"]["state"]["buy_signal_scores"] = [high_60_10_gb, high_60_20_gb]
        App.config["trade"]["state"]["buy_signal"] = is_buy_signal


if __name__ == "__main__":
    db = Database(None)
    db.queue.put({"e": "depth", "s": "BTCUSDT", "price": 12.34})
    db.queue.put({"e": "kline", "s": "ETHBTC", "price": 23.45})
    db.queue.put({"e": "kline", "s": "ETHBTC", "price": 45.67})

    db.store_queue("5s")

    pass
