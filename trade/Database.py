from pathlib import Path
from typing import Union
import json
import pickle
from datetime import datetime, date
import queue

import numpy as np
import pandas as pd

from trade.App import App
from trade.utils import *
from trade.feature_prediction import *
from trade.feature_generation import *
from trade.signal_generation import *

import logging
log = logging.getLogger('database')

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
        self.klines = {}  # It is a dict of lists where key is a symbol and the list is a list of latest kline records
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
            self.klines.get(symbol)[-1]
        else:
            return None

    def get_missing_klines_count(self, symbol):
        # TODO: Depending on the now time, return how many klines are needed
        return 1

    def store_klines(self, data: dict):
        """
        Store latest klines for the specified symbols.
        Existing klines for the symbol and timestamp will be replaced.

        :param data: Dict of lists with symbol as a key, and list of klines for this symbol as a value.
            Example: { 'BTCUSDT': [ [], [], [] ] }
        :type dict:
        """
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

            # Check validity. It has to be a time series with certain frequency
            for i, kline in enumerate(self.klines.get(symbol)):
                ts = kline[0]
                if i > 0:
                    if ts - prev_ts != 60_000:
                        log.error("Wrong sequence of klines. They are expected to be a regular time series with 1m frequency.")
                prev_ts = kline[0]

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
        2. Derive (compute) features (same as for model training)
        3. Derive (predict) labels by applying a model trained for each label
        4. Generate buy/sell signals
        """
        klines = self.klines.get(symbol)

        print(f"===>>> Analyze {symbol}. {len(klines)} klines in the database. Last timestamp: {klines[-1][0]}")

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
        # generate_predictions(): get df with feature matrix, apply models and generate predictive features, and return a df
        # the data frame will get additional columns with predicted features
        #

        # Prepare (load) models (models are identified by label names)
        labels = App.config["trade"]["analysis"]["labels"]
        model_path = App.config["trade"]["analysis"]["folder"]
        models = {}
        for label in labels:
            model_file_name = label
            model_file = Path(model_path, model_file_name).with_suffix(".pkl")
            if not Path(model_file).exists():
                log.error(f"Model file does not exist: {model_file}")
                return

            model = pickle.load(model_file)

            models[label] = model

        # Prepare input
        features = App.config["trade"]["analysis"]["features"]
        df = df[features].dropna(subset=features, inplace=True)

        # Do prediction by applying models
        labels_out = predict_labels(df, models)

        # Now df will have additionally as many new columns as we have defined labels (and corresponding models)

        #
        # 4.
        # generate_signals(): get prediction features and return buy/sell signals with asset symbol and amount
        # get a dict with the decisions based on the last row of the data frame
        #
        signals = App.config["trade"]["analysis"]["signals"]
        models = {}
        for signal in signals:
            model = {"param1": 0.0, "param2": 0.0}
            models[label] = model

        signals_out = generate_signals(df, models)


if __name__ == "__main__":
    db = Database(None)
    db.queue.put({"e": "depth", "s": "BTCUSDT", "price": 12.34})
    db.queue.put({"e": "kline", "s": "ETHBTC", "price": 23.45})
    db.queue.put({"e": "kline", "s": "ETHBTC", "price": 45.67})

    db.store_queue("5s")

    pass
