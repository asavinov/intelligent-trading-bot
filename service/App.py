from pathlib import Path
from typing import Union
import json
from datetime import datetime, date, timedelta
import re

import pandas as pd

PACKAGE_ROOT = Path(__file__).parent.parent
#PACKAGE_PARENT = '..'
#SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
#sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
#PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


class App:
    """Globally visible variables."""

    # System
    loop = None  # asyncio main loop
    sched = None  # Scheduler

    analyzer = None  # Store and analyze data

    # Connector client
    client = None

    # WebSocket for push notifications
    bm = None
    conn_key = None  # Socket

    #
    # State of the server (updated after each interval)
    #
    # State 0 or None or empty means ok. String and other non empty objects mean error
    error_status = 0  # Networks, connections, exceptions etc. what does not allow us to work at all
    server_status = 0  # If server allow us to trade (maintenance, down etc.)
    account_status = 0  # If account allows us to trade (funds, suspended etc.)
    trade_state_status = 0  # Something wrong with our trading logic (wrong use, inconsistent state etc. what we cannot recover)

    df = None  # Data from the latest analysis

    # Trade simulator
    transaction = None
    # Trade binance
    status = None  # BOUGHT, SOLD, BUYING, SELLING
    order = None  # Latest or current order
    order_time = None  # Order submission time

    # Available assets for trade
    # Can be set by the sync/recover function or updated by the trading algorithm
    base_quantity = "0.04108219"  # BTC owned (on account, already bought, available for trade)
    quote_quantity = "1000.0"  # USDT owned (on account, available for trade)

    #
    # Trader. Status data retrieved from the server. Below are examples only.
    #
    system_status = {"status": 0, "msg": "normal"}  # 0: normal，1：system maintenance
    symbol_info = {}
    account_info = {}

    #
    # Constant configuration parameters
    #
    config = {
        # Binance
        "api_key": "",
        "api_secret": "",

        # Telegram
        "telegram_bot_token": "",  # Source address of messages
        "telegram_chat_id": "",  # Destination address of messages

        #
        # Conventions for the file and column names
        #
        "merge_file_name": "data.csv",
        "feature_file_name": "features.csv",
        "matrix_file_name": "matrix.csv",
        "predict_file_name": "predictions.csv",  # predict, predict-rolling
        "signal_file_name": "signals.csv",
        "signal_models_file_name": "signal_models",

        "model_folder": "MODELS",

        "time_column": "timestamp",

        # File locations
        "data_folder": "C:/DATA_ITB",  # Location for all source and generated data/models

        # ==============================================
        # === DOWNLOADER, MERGER and (online) READER ===

        # Symbol determines sub-folder and used in other identifiers
        "symbol": "BTCUSDT",  # BTCUSDT ETHUSDT ^gspc

        # This parameter determines time raster (granularity) for the data
        # It is pandas frequency
        "freq": "1min",

        # This list is used for downloading and then merging data
        # "folder" is symbol name for downloading. prefix will be added column names during merge
        "data_sources": [],

        # ==========================
        # === FEATURE GENERATION ===

        # What columns to pass to which feature generator and how to prefix its derived features
        # Each executes one feature generation function applied to columns with the specified prefix
        "feature_sets": [],

        # ========================
        # === LABEL GENERATION ===

        "label_sets": [],

        # ===========================
        # === MODEL TRAIN/PREDICT ===
        #     predict off-line and on-line

        "label_horizon": 0,  # This number of tail rows will be excluded from model training
        "train_length": 0,  # train set maximum size. algorithms may decrease this length

        # List all features to be used for training/prediction by selecting them from the result of feature generation
        # The list of features can be found in the output of the feature generation (but not all must be used)
        # Currently the same feature set for all algorithms
        "train_features": [],

        # Labels to be used for training/prediction by all algorithms
        # List of available labels can be found in the output of the label generation (but not all must be used)
        "labels": [],

        # Algorithms and their configurations to be used for training/prediction
        "algorithms": [],

        # ===========================
        # ONLINE (PREDICTION) PARAMETERS
        # Minimum history length required to compute derived features
        "features_horizon": 10,

        # ===============
        # === SIGNALS ===

        "signal_sets": [],

        # =====================
        # === NOTIFICATIONS ===

        "score_notification_model": {},
        "diagram_notification_model": {},

        # ===============
        # === TRADING ===
        "trade_model": {
            "no_trades_only_data_processing": False,  # in market or out of market processing is excluded (all below parameters ignored)
            "test_order_before_submit": False,  # Send test submit to the server as part of validation
            "simulate_order_execution": False,  # Instead of real orders, simulate their execution (immediate buy/sell market orders and use high price of klines for limit orders)

            "percentage_used_for_trade": 99,  # in % to the available USDT quantity, that is, we will derive how much BTC to buy using this percentage
            "limit_price_adjustment": 0.005,  # Limit price of orders will be better than the latest close price (0 means no change, positive - better for us, negative - worse for us)
        },

        "train_signal_model": {},

        # =====================
        # === BINANCE TRADER ===
        "base_asset": "",  # BTC ETH
        "quote_asset": "",

        # ==================
        # === COLLECTORS ===
        "collector": {
            "folder": "DATA",
            "flush_period": 300,  # seconds
            "depth": {
                "folder": "DEPTH",
                "symbols": ["BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"],
                "limit": 100,  # Legal values (depth): '5, 10, 20, 50, 100, 500, 1000, 5000' <100 weight=1
                "freq": "1min",  # Pandas frequency
            },
            "stream": {
                "folder": "STREAM",
                # Stream formats:
                # For kline channel: <symbol>@kline_<interval>, Event type: "e": "kline", Symbol: "s": "BNBBTC"
                # For depth channel: <symbol>@depth<levels>[@100ms], Event type: NO, Symbol: NO
                # btcusdt@ticker
                "channels": ["kline_1m", "depth20"],  # kline_1m, depth20, depth5
                "symbols": ["BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"],
                # "BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"
            }
        },
    }


def data_provider_problems_exist():
    if App.error_status != 0:
        return True
    if App.server_status != 0:
        return True
    return False


def problems_exist():
    if App.error_status != 0:
        return True
    if App.server_status != 0:
        return True
    if App.account_status != 0:
        return True
    if App.trade_state_status != 0:
        return True
    return False


def load_config(config_file):
    if config_file:
        config_file_path = PACKAGE_ROOT / config_file
        with open(config_file_path, encoding='utf-8') as json_file:
            #conf_str = json.load(json_file)
            conf_str = json_file.read()

            # Remove everything starting with // and till the line end
            conf_str = re.sub(r"//.*$", "", conf_str, flags=re.M)

            conf_json = json.loads(conf_str)
            App.config.update(conf_json)


def load_last_transaction():
    transaction_file = Path("transactions.txt")
    t_dict = dict(timestamp=str(datetime.now()), price=0.0, profit=0.0, status="")
    if transaction_file.is_file():
        with open(transaction_file, "r") as f:
            line = ""
            for line in f:
                pass
        if line:
            t_dict = dict(zip("timestamp,price,profit,status".split(","), line.strip().split(",")))
            t_dict["price"] = float(t_dict["price"])
            t_dict["profit"] = float(t_dict["profit"])
            #t_dict = json.loads(line)
    else:  # Create file with header
        pass
        #with open(transaction_file, 'a+') as f:
        #    f.write("timestamp,price,profit,status\n")
    return t_dict


def load_all_transactions():
    transaction_file = Path("transactions.txt")
    df = pd.read_csv(transaction_file, names="timestamp,price,profit,status".split(","), header=None)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    return df


if __name__ == "__main__":
    pass
