from pathlib import Path
from typing import Union
import json

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
        "signal_file_name": "performance",

        "time_column": "timestamp",

        # File locations
        "data_folder": "",  # It is needed for model training

        # ==============================================
        # === DOWNLOADER, MERGER and (online) READER ===

        # Symbol determines sub-folder and used in other identifiers
        # It is used as a name of the output matrix (so it is not really any traded symbol but is arbitrary name but typically we use name of the symbol we want to predict/trade)
        "symbol": "BTCUSDT",  # BTCUSDT ETHUSDT ^gspc
        # It will be appended to generated file names and denotes config (like config name).

        # This parameter determines the time raster (granularity) for the data.
        # Currently 1m for binance, and 1d for yahoo are supported (only workdays)
        "freq": "1m",

        # Specify which data sources to merge into one file with all source columns
        # Format: (folder, file, prefix) - (symbol, data source, column prefix)
        # These descriptors are also used to retrieve data by collector in online mode - folder name is symbol name

        "data_sources": [
            {"folder": "BTCUSDT", "file": "klines", "column_prefix": ""},
        ],
        # Example for Yahoo 1d inputs
        #"data_sources": [
        #    {"folder": "^gspc", "file": "", "column_prefix": ""},
        #    {"folder": "^vix", "file": "", "column_prefix": "^vix"},
        #    {"folder": "^tnx", "file": "", "column_prefix": "^tnx"},
        #],
        #"data_sources": [
        #    {"folder": "BTCUSDT", "file": "klines", "column_prefix": "btc"},
        #    {"folder": "ETHUSDT", "file": "klines", "column_prefix": "eth"},
        #],

        # ==========================
        # === FEATURE GENERATION ===

        # What columns to pass to which feature generator and how to prefix its derived features
        # Each executes one feature generation function applied to columns with the specified prefix
        "feature_sets": [
            {"column_prefix": "btc", "generator": "klines", "feature_prefix": "btc"},
            {"column_prefix": "eth", "generator": "klines", "feature_prefix": "eth"},
        ],
        #"feature_sets": [
        #    {"column_prefix": "", "generator": "yahoo_main", "feature_prefix": ""},
        #    {"column_prefix": "^vix", "generator": "yahoo_secondary", "feature_prefix": "^vix"},
        #    {"column_prefix": "^tnx", "generator": "yahoo_secondary", "feature_prefix": "^tnx"},
        #],
        # Parameters of klines feature generator
        # If these are changed, then feature names (below) will also have to be changed

        # v0.3.0
        "base_window": 10080,  # 1 week
        "averaging_windows": [1, 10, 30, 180, 720, 1440],
        "area_windows": [10, 30, 180, 720, 1440],

        # "base_window": 40320,  # 4 weeks
        # "averaging_windows": [1, 20, 60, 360, 1440, 10080],
        # "area_windows": [20, 60, 360, 1440, 10080],

        # "base_window": 20160,  # 2 weeks
        # "averaging_windows": [1, 20, 60, 180, 720, 2880],
        # "area_windows": [60, 120, 180, 720, 2880],
        # "averaging_windows": [1, 30, 120, 360, 1440, 4320],
        # "area_windows": [30, 120, 360, 1440, 4320],
        # "averaging_windows": [1, 60, 180, 720, 2880, 5760],
        # "area_windows": [60, 180, 720, 2880, 5760],

        #"base_window": 40,
        #"averaging_windows": [2, 12, 20],
        #"area_windows": [12, 20],

        # ========================
        # === LABEL GENERATION ===

        "label_sets": [
            {"column_prefix": "btc", "generator": "topbot", "feature_prefix": ""},
            #{"column_prefix": "", "generator": "highlow", "feature_prefix": ""},
        ],
        # highlow label parameter: max (of high) and min (of low) for this horizon ahead
        "highlow_horizon": 10,  # 10 (2 weeks) for yahoo, 1440 for BTC
        "topbot_column_name": "close",

        # ===========================
        # === MODEL TRAIN/PREDICT ===
        #     predict off-line and on-line

        # This number of tail rows will be excluded from model training
        "label_horizon": 0,
        "train_length": int(2.0 * 525_600),  # train set maximum size. algorithms may decrease this length

        # One model is for each (label, train_features, algorithm)
        # Feature column names returned by the feature generators
        # They are used by train/predict
        "train_features": [
            'btc_close_1', 'btc_close_10', 'btc_close_30', 'btc_close_180', 'btc_close_720', 'btc_close_1440',
            'btc_close_std_10', 'btc_close_std_30', 'btc_close_std_180', 'btc_close_std_720', 'btc_close_std_1440',
            'btc_volume_1', 'btc_volume_10', 'btc_volume_30', 'btc_volume_180', 'btc_volume_720', 'btc_volume_1440',
            'btc_span_1', 'btc_span_10', 'btc_span_30', 'btc_span_180', 'btc_span_720', 'btc_span_1440',
            'btc_trades_1', 'btc_trades_10', 'btc_trades_30', 'btc_trades_180', 'btc_trades_720', 'btc_trades_1440',
            'btc_tb_base_1', 'btc_tb_base_10', 'btc_tb_base_30', 'btc_tb_base_180', 'btc_tb_base_720', 'btc_tb_base_1440',
            #'btc_close_area_10', 'btc_close_area_30', 'btc_close_area_180', 'btc_close_area_720', 'btc_close_area_1440',
            'btc_close_trend_10', 'btc_close_trend_30', 'btc_close_trend_180', 'btc_close_trend_720', 'btc_close_trend_1440',
            'btc_volume_trend_10', 'btc_volume_trend_30', 'btc_volume_trend_180', 'btc_volume_trend_720', 'btc_volume_trend_1440',

            'eth_close_1', 'eth_close_10', 'eth_close_30', 'eth_close_180', 'eth_close_720', 'eth_close_1440',
            'eth_close_std_10', 'eth_close_std_30', 'eth_close_std_180', 'eth_close_std_720', 'eth_close_std_1440',
            'eth_volume_1', 'eth_volume_10', 'eth_volume_30', 'eth_volume_180', 'eth_volume_720', 'eth_volume_1440',
            'eth_span_1', 'eth_span_10', 'eth_span_30', 'eth_span_180', 'eth_span_720', 'eth_span_1440',
            'eth_trades_1', 'eth_trades_10', 'eth_trades_30', 'eth_trades_180', 'eth_trades_720', 'eth_trades_1440',
            'eth_tb_base_1', 'eth_tb_base_10', 'eth_tb_base_30', 'eth_tb_base_180', 'eth_tb_base_720', 'eth_tb_base_1440',
            #'eth_close_area_10', 'eth_close_area_30', 'eth_close_area_180', 'eth_close_area_720', 'eth_close_area_1440',
            'eth_close_trend_10', 'eth_close_trend_30', 'eth_close_trend_180', 'eth_close_trend_720', 'eth_close_trend_1440',
            'eth_volume_trend_10', 'eth_volume_trend_30', 'eth_volume_trend_180', 'eth_volume_trend_720', 'eth_volume_trend_1440'
        ],

        # algorithm descriptors from model store
        "algorithms": ["nn"],  # gb, nn, lc - these are names from the model store which stores all the necessary parameters for each algorithm

        # Models (for each algorithm) will be trained for these target labels
        "labels": [
            "bot2_025", "bot2_05", "bot2_075", "bot2_1", "bot2_125", "bot2_15",
            "top2_025", "top2_05", "top2_075", "top2_1", "top2_125", "top2_15",
        ],
        "_labels": [
            "bot3_025", "bot3_05", "bot3_075", "bot3_1", "bot3_125", "bot3_15", "bot3_175",
            "top3_025", "top3_05", "top3_075", "top3_1", "top3_125", "top3_15", "top3_175",

            "bot4_025", "bot4_05", "bot4_075", "bot4_1", "bot4_125", "bot4_15", "bot4_175", "bot4_2",
            "top4_025", "top4_05", "top4_075", "top4_1", "top4_125", "top4_15", "top4_175", "top4_2",

            "bot5_025", "bot5_05", "bot5_075", "bot5_1", "bot5_125", "bot5_15", "bot5_175", "bot5_2", "bot5_25",
            "top5_025", "top5_05", "top5_075", "top5_1", "top5_125", "top5_15", "top5_175", "top5_2", "top5_25",

            'high_max_180',  # Maximum high (relative)

            "high_10", "high_15", "high_20", "high_25", "high_30",
            "low_10", "low_15", "low_20", "low_25", "low_30",

            'low_min_180',  # Minimum low (relative)
            'low_01', 'low_02', 'low_03', 'low_04',  # Always above
            'low_10', 'low_15', 'low_20', 'low_25',  # At least one time below

            'high_to_low_180',

            'close_area_future_60', 'close_area_future_120', 'close_area_future_180', 'close_area_future_300',
        ],

        # ONLINE (PREDICTION) PARAMETERS
        # Minimum history length required to compute derived features
        # It is used in online mode where we need to maintain data window of this size or larger
        # Take maximum aggregation windows from feature generation code (and add something to be sure that we have all what is needed)
        # Basically, should be equal to base_window
        "features_horizon": 10180,

        # =========================
        # === SIGNAL GENERATION ===

        # These are predicted columns <label, train_features, algorithm> as well as model (pair) names
        "buy_labels": [],
        "sell_labels": [],
        "_buy_labels": ["bot4_1_k_lc", "bot4_15_k_lc", "bot4_2_k_lc", "bot4_25_k_lc", "bot4_3_k_lc"],
        "_sell_labels": ["top4_1_k_lc", "top4_15_k_lc", "top4_2_k_lc", "top4_25_k_lc", "top4_3_k_lc"],

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
        with open(config_file_path) as json_file:
            config_json = json.load(json_file)
            App.config.update(config_json)


if __name__ == "__main__":
    pass
