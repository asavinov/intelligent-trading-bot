from pathlib import Path
from typing import Union
import json


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

        # File locations
        "data_folder": "",  # It is needed for model training
        # Symbol determines sub-folder and used in other identifiers
        "symbol": "",  # BTCUSDT ETHUSDT
        # It will be appended to generated file names and denotes config (like config name).
        # TODO: In future, we could/should change it semantics, and store the generated files in the folder with this name, rather than append a modifier to file names
        "config_file_modifier": "",

        # File name conventions
        "merge_file_modifier": "data",
        "feature_file_modifier": "features",
        "matrix_file_modifier": "matrix",
        "predict_file_modifier": "predict",  # Or predict-rolling
        "signal_file_modifier": "performance",

        # =====================================
        # === MERGER and (online) COLLECTOR ===

        # Specify which data sources to merge into one file with all source columns
        # Format: (folder, file, prefix) - (symbol, data source, column prefix)
        # These descriptors are also used to retrieve data by collector in online mode - folder name is symbol name
        #"data_sources": [
        #    {"folder": "BTCUSDT", "file": "klines", "column_prefix": ""},
        #],
        "data_sources": [
            {"folder": "BTCUSDT", "file": "klines", "column_prefix": ""},
            {"folder": "ETHUSDT", "file": "klines", "column_prefix": "eth"},
        ],

        # ==========================
        # === FEATURE GENERATION ===

        # What columns to pass to which feature generator and how to prefix its derived features
        "feature_sets": [
            {"column_prefix": "", "generator": "klines", "feature_prefix": ""},
            {"column_prefix": "eth", "generator": "klines", "feature_prefix": "eth"},
        ],
        # Parameters of klines feature generator
        # If these are changed, then feature names (below) will also have to be changed
        "base_window_kline": 40320,
        "windows_kline": [1, 60, 360, 1440, 4320, 10080],
        "area_windows_kline": [60, 360, 1440, 4320, 10080],

        # ========================
        # === LABEL GENERATION ===

        "high_low_horizon": 1440,  # Parameter of labels: computing max and min for this horizon ahead

        # ===========================
        # === MODEL TRAIN/PREDICT ===
        #     off-line and on-line

        # Minimum history required to compute derived features
        # It is used in online mode where we need to maintain data window of this size or larger
        # Take maximum aggregation windows from feature generation code (and add something to be sure that we have all what is needed)
        # Basically, should be equal to base_window_kline
        "features_horizon": 40420,

        # One model is for each (label, train_features, algorithm)

        # Feature column names returned by the klines feature generator
        # They are used by train/predict
        "features_kline": [
            "close_1", "close_60", "close_360", "close_1440", "close_4320", "close_10080",
            "close_std_60", "close_std_360", "close_std_1440", "close_std_4320", "close_std_10080",  # Removed "std_1"
            "volume_1", "volume_60", "volume_360", "volume_1440", "volume_4320", "volume_10080",
            "span_1", "span_60", "span_360", "span_1440", "span_4320", "span_10080",
            "trades_1", "trades_60", "trades_360", "trades_1440", "trades_4320", "trades_10080",
            "tb_base_1", "tb_base_60", "tb_base_360", "tb_base_1440", "tb_base_4320", "tb_base_10080",
            "tb_quote_1", "tb_quote_60", "tb_quote_360", "tb_quote_1440", "tb_quote_4320", "tb_quote_10080",
            "close_area_60", "close_area_360", "close_area_1440", "close_area_4320", "close_area_10080",
            "close_trend_60", "close_trend_360", "close_trend_1440", "close_trend_4320", "close_trend_10080",

            'eth_close_1', 'eth_close_60', 'eth_close_360', 'eth_close_1440', 'eth_close_4320', 'eth_close_10080',
            'eth_close_std_60', 'eth_close_std_360', 'eth_close_std_1440', 'eth_close_std_4320', 'eth_close_std_10080',
            'eth_volume_1', 'eth_volume_60', 'eth_volume_360', 'eth_volume_1440', 'eth_volume_4320', 'eth_volume_10080',
            'eth_span_1', 'eth_span_60', 'eth_span_360', 'eth_span_1440', 'eth_span_4320', 'eth_span_10080',
            'eth_trades_1', 'eth_trades_60', 'eth_trades_360', 'eth_trades_1440', 'eth_trades_4320', 'eth_trades_10080',
            'eth_tb_base_1', 'eth_tb_base_60', 'eth_tb_base_360', 'eth_tb_base_1440', 'eth_tb_base_4320', 'eth_tb_base_10080',
            'eth_tb_quote_1', 'eth_tb_quote_60', 'eth_tb_quote_360', 'eth_tb_quote_1440', 'eth_tb_quote_4320', 'eth_tb_quote_10080',
            'eth_close_area_60', 'eth_close_area_360', 'eth_close_area_1440', 'eth_close_area_4320', 'eth_close_area_10080',
            'eth_close_trend_60', 'eth_close_trend_360', 'eth_close_trend_1440', 'eth_close_trend_4320', 'eth_close_trend_10080',
        ],
        "features_kline_eth": [
            'eth_close_1', 'eth_close_60', 'eth_close_360', 'eth_close_1440', 'eth_close_4320', 'eth_close_10080',
            'eth_close_std_60', 'eth_close_std_360', 'eth_close_std_1440', 'eth_close_std_4320', 'eth_close_std_10080',
            'eth_volume_1', 'eth_volume_60', 'eth_volume_360', 'eth_volume_1440', 'eth_volume_4320', 'eth_volume_10080',
            'eth_span_1', 'eth_span_60', 'eth_span_360', 'eth_span_1440', 'eth_span_4320', 'eth_span_10080',
            'eth_trades_1', 'eth_trades_60', 'eth_trades_360', 'eth_trades_1440', 'eth_trades_4320', 'eth_trades_10080',
            'eth_tb_base_1', 'eth_tb_base_60', 'eth_tb_base_360', 'eth_tb_base_1440', 'eth_tb_base_4320', 'eth_tb_base_10080',
            'eth_tb_quote_1', 'eth_tb_quote_60', 'eth_tb_quote_360', 'eth_tb_quote_1440', 'eth_tb_quote_4320', 'eth_tb_quote_10080',
            'eth_close_area_60', 'eth_close_area_360', 'eth_close_area_1440', 'eth_close_area_4320', 'eth_close_area_10080',
            'eth_close_trend_60', 'eth_close_trend_360', 'eth_close_trend_1440', 'eth_close_trend_4320', 'eth_close_trend_10080',
        ],

        # Used to select columns for training by adding together different feature sets (the lists are defined elswhere in variables like "features_kline")
        "train_features": ["kline"],
        # algorithm descriptors from model store
        "algorithms": ["nn"],  # gb, nn, lc - definitions of their parameters are in model store

        # This will be excluded from model training
        "label_horizon": 0,

        # These labels are specified manually and are produced by label generator
        # Models will be trained for these models
        "labels": [
            "bot2_025", "bot2_05", "bot2_075", "bot2_1", "bot2_125", "bot2_15",
            "top2_025", "top2_05", "top2_075", "top2_1", "top2_125", "top2_15",

            "bot3_025", "bot3_05", "bot3_075", "bot3_1", "bot3_125", "bot3_15", "bot3_175",
            "top3_025", "top3_05", "top3_075", "top3_1", "top3_125", "top3_15", "top3_175",

            "bot4_025", "bot4_05", "bot4_075", "bot4_1", "bot4_125", "bot4_15", "bot4_175", "bot4_2",
            "top4_025", "top4_05", "top4_075", "top4_1", "top4_125", "top4_15", "top4_175", "top4_2",

            "bot5_025", "bot5_05", "bot5_075", "bot5_1", "bot5_125", "bot5_15", "bot5_175", "bot5_2", "bot5_25",
            "top5_025", "top5_05", "top5_075", "top5_1", "top5_125", "top5_15", "top5_175", "top5_2", "top5_25",
        ],
        "_labels": [
            "bot4_1", "bot4_15", "bot4_2", "bot4_25", "bot4_3",
            "top4_1", "top4_15", "top4_2", "top4_25", "top4_3",

            "bot5_1", "bot5_15", "bot5_2", "bot5_25", "bot5_3",
            "top5_1", "top5_15", "top5_2", "top5_25", "top5_3",
            "bot6_1", "bot6_15", "bot6_2", "bot6_25", "bot6_3",
            "top6_1", "top6_15", "top6_2", "top6_25", "top6_3",
            "bot7_1", "bot7_15", "bot7_2", "bot7_25", "bot7_3",
            "top7_1", "top7_15", "top7_2", "top7_25", "top7_3",
            "bot8_1", "bot8_15", "bot8_2", "bot8_25", "bot8_3",
            "top8_1", "top8_15", "top8_2", "top8_25", "top8_3",

            'high_max_180',  # Maximum high (relative)

            "high_10", "high_15", "high_20", "high_25", "high_30",
            "low_10", "low_15", "low_20", "low_25", "low_30",

            'low_min_180',  # Minimum low (relative)
            'low_01', 'low_02', 'low_03', 'low_04',  # Always above
            'low_10', 'low_15', 'low_20', 'low_25',  # At least one time below

            'high_to_low_180',

            'close_area_future_60', 'close_area_future_120', 'close_area_future_180', 'close_area_future_300',
        ],

        # =========================
        # === SIGNAL GENERATION ===

        # These are predicted columns <label, train_features, algorithm> as well as model (pair) names
        "buy_labels": ["bot4_1_k_nn", "bot4_15_k_nn", "bot4_2_k_nn", "bot4_25_k_nn", "bot4_3_k_nn", "bot5_1_k_nn", "bot5_15_k_nn", "bot5_2_k_nn", "bot5_25_k_nn", "bot5_3_k_nn"],
        "sell_labels": ["top4_1_k_nn", "top4_15_k_nn", "top4_2_k_nn", "top4_25_k_nn", "top4_3_k_nn", "top5_1_k_nn", "top5_15_k_nn", "top5_2_k_nn", "top5_25_k_nn", "top5_3_k_nn"],
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
