from pathlib import Path
from typing import Union
import json
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

    signal = None,  # Latest signal "BUY", "SELL"

    #
    # State of the server (updated after each interval)
    #
    # State 0 or None or empty means ok. String and other non empty objects mean error
    error_status = 0  # Networks, connections, exceptions etc. what does not allow us to work at all
    server_status = 0  # If server allow us to trade (maintenance, down etc.)
    account_status = 0  # If account allows us to trade (funds, suspended etc.)
    trade_state_status = 0  # Something wrong with our trading logic (wrong use, inconsistent state etc. what we cannot recover)

    # Trade status
    transaction = None
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
        "actions": ["notify"],  # Values: notify, trade

        # Binance
        "api_key": "",
        "api_secret": "",

        # Telegram
        "telegram_bot_token": "",  # Source address of messages
        "telegram_chat_id": "",  # Destination address of messages

        #
        # Naming conventions
        #
        "merge_file_name": "data",
        "feature_file_name": "features",
        "matrix_file_name": "matrix",
        "predict_file_name": "predictions",  # predict, predict-rolling
        "signal_file_name": "signals",
        "signal_models_file_name": "signal_models",

        "time_column": "timestamp",

        # File locations
        "data_folder": "C:/DATA_ITB",  # Location for all source and generated data/models

        # ==============================================
        # === DOWNLOADER, MERGER and (online) READER ===

        # Symbol determines sub-folder and used in other identifiers
        "symbol": "BTCUSDT",  # BTCUSDT ETHUSDT ^gspc

        # This parameter determines the time raster (granularity) for the data
        # Currently 1m for binance, and 1d for yahoo are supported (only workdays)
        "freq": "1m",

        # This list is used for downloading and then merging data
        # "folder" is symbol name for downloading. prefix will be added column names during merge
        "data_sources": [
            {"folder": "BTCUSDT", "file": "klines", "column_prefix": ""}
        ],

        # ==========================
        # === FEATURE GENERATION ===

        # What columns to pass to which feature generator and how to prefix its derived features
        # Each executes one feature generation function applied to columns with the specified prefix
        "feature_sets": [
            {"column_prefix": "", "generator": "binance_main", "feature_prefix": ""}
        ],
        # Parameters of some feature generators
        # They influence generated feature names (below)
        "base_window": 360,
        "averaging_windows": [1, 10, 60],
        "area_windows": [10, 60],

        # ========================
        # === LABEL GENERATION ===

        "label_sets": [
            {"column_prefix": "", "generator": "highlow", "feature_prefix": ""},
        ],
        # highlow label parameter: max (of high) and min (of low) for this horizon ahead
        "highlow_horizon": 60,  # 1 hour prediction

        # ===========================
        # === MODEL TRAIN/PREDICT ===
        #     predict off-line and on-line

        # This number of tail rows will be excluded from model training
        "label_horizon": 60,
        "train_length": int(0.5 * 525_600),  # train set maximum size. algorithms may decrease this length

        # List all features to be used for training/prediction by selecting them from the result of reature generation
        # Remove: "_std_1", "_trend_1"
        "train_features": [
            "close_1", "close_10", "close_60",
            "close_std_10", "close_std_60",
            "volume_1", "volume_10", "volume_60",
            "span_1", "span_10", "span_60",
            "trades_1", "trades_10", "trades_60",
            "tb_base_1", "tb_base_10", "tb_base_60",
            "close_area_10", "close_area_60",
            "close_trend_10", "close_trend_60",
            "volume_trend_10", "volume_trend_60"
        ],

        # Models (for each algorithm) will be trained for these target labels
        "labels": [
            "high_10", "high_15", "high_20", "high_25", "high_30",
            #"high_01", "high_02", "high_03", "high_04", "high_05",
            #"low_01", "low_02", "low_03", "low_04", "low_05",
            "low_10", "low_15", "low_20", "low_25", "low_30"
        ],

        # algorithm names defined in the model store
        "algorithms": ["lc"],

        # ONLINE (PREDICTION) PARAMETERS
        # Minimum history length required to compute derived features
        # It is used in online mode where we need to maintain data window of this size or larger
        # Take maximum aggregation windows from feature generation code (and add something to be sure that we have all what is needed)
        # Basically, should be equal to base_window
        "features_horizon": 10180,

        # =========================
        # === SIGNAL GENERATION ===

        # These predicted columns (scores) will be used for generating buy/sell signals
        "buy_labels": ["high_10_lc", "high_15_lc", "high_20_lc"],
        "sell_labels": ["low_10_lc", "low_15_lc", "low_20_lc"],

        # It defines how signal scores, trade signals, and notification signals will be generated
        # from point-wise prediction scores for two groups of labels
        "signal_model": {
            # First, aggregation in group over various algorithms and label parameters
            "buy_point_threshold": None,  # Second, produce boolean column (optional)
            "buy_window": 3,  # Third, aggregate in time
            # Now we have the final score
            "buy_signal_threshold": 0.65,  # To decide whether to buy/sell after all aggregations/combinations
            "buy_notify_threshold": 0.05,  # To decide whether to notify (can be an option of individual users/consumers)

            "combine": "",  # "no_combine", "relative", "difference"  Find relative/difference

            "sell_point_threshold": None,
            "sell_window": 3,
            "sell_signal_threshold": 0.65,
            "sell_notify_threshold": 0.05,

            "trade_icon_step": 0.1,  # For each step, one icon added
            "notify_frequency_minutes": 10,  # 1m, 5m, 10m, 15m etc. Minutes will be divided by this number
        },

        # =====================
        # === TRADER SERVER ===
        "base_asset": "",  # BTC ETH
        "quote_asset": "",

        "trader": {
            # For debugging: determine what parts of code will be executed
            "no_trades_only_data_processing": False,  # in market or out of market processing is excluded (all below parameters ignored)
            "test_order_before_submit": False,  # Send test submit to the server as part of validation
            "simulate_order_execution": False,  # Instead of real orders, simulate their execution (immediate buy/sell market orders and use high price of klines for limit orders)

            "percentage_used_for_trade": 99,  # in % to the available USDT quantity, that is, we will derive how much BTC to buy using this percentage
            "limit_price_adjustment": -0.0001,  # Limit price of orders will be better than the latest close price (0 means no change, positive - better for us, negative - worse for us)

            # Signal model (trade strategy) - currently NOT USED
            "sell_timeout": 70,  # Seconds
            "percentage_sell_price": 1.018,  # our planned profit per trade via limit sell order (part of the model)
        },

        # ==================
        # === COLLECTORS ===
        "collector": {
            "folder": "DATA",
            "flush_period": 300,  # seconds
            "depth": {
                "folder": "DEPTH",
                "symbols": ["BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"],
                "limit": 100,  # Legal values (depth): '5, 10, 20, 50, 100, 500, 1000, 5000' <100 weight=1
                "freq": "1m",  # Binance standard frequency: 5s, 1m etc.
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


if __name__ == "__main__":
    pass
