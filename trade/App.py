from typing import Union
import json

class App:
    """
    Static global parameters visible and accessible from all parts of the application.
    """

    """
    Approach 1: direct access to globals from other modules:
    from trade.App import App
    api_key = App.api_key
    """

    #
    # Transient data
    #
    client = None
    loop = None  # asyncio main loop
    sched = None  # Scheduler
    database = None
    log = None
    bm = None
    conn_key = None  # Socket

    #
    # Persistent configuration
    #
    config = {
        "command": "collect",  # "collect" "trade"

        "api_key": "***REMOVED***",
        "api_secret": "***REMOVED***",

        "trade": {
            "folder": "DATA",
            "symbols": ["BTCUSDT"],  # Symbols to be traded. Currently only one (the first symbol will be traded)
            "analysis": {  # Same for all symbols
                "folder": "MODELS",
                "features": [],
                "labels": [],
                "signals": ["buy", "sell"],
            },
            "signals": {
                "sell_timeout": 90,  # Seconds
            },
            "state": {  # Current state updated after each trade session
                "in_market": False,  # If True then we bought symbol and there is sell order. If False then we have money.
                "buy_signal": False,
                "sell_signal": False,
                "buy_order": None,
                "sell_order": None,
                "assets": {},
            },
        },

        "collect": {
            "folder": "DATA",
            "flush_period": 300,  # seconds
            "depth": {
                "folder": "DEPTH",
                "symbols": ["BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"],
                "limit": 100,  # Legal values: '5, 10, 20, 50, 100, 500, 1000, 5000' <100 weight=1
                "freq": "1m",  # Binance frequency: 5s, 1m etc.
            },
            "stream": {
                "folder": "STREAM",
                # Stream formats:
                # For kline channel: <symbol>@kline_<interval>, Event type: "e": "kline", Symbol: "s": "BNBBTC"
                # For depth channel: <symbol>@depth<levels>[@100ms], Event type: NO, Symbol: NO
                # btcusdt@ticker
                "channels": ["kline_1m", "depth20"],  # kline_1m, depth20, depth5
                "symbols": ["BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"],  # "BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"
            }
        },
    }

    """
    Approach 2: Indirect access via static methods
    port = App.conf("MYSQL_PORT")
    App.set("username", "hi")
    """
    __conf = {
        "username": "",
        "password": "",
        "MYSQL_PORT": 3306,
        "MYSQL_DATABASE": 'mydb',
        "MYSQL_DATABASE_TABLES": ['tb_users', 'tb_groups']
    }
    __setters = ["username", "password"]  # A list of (configuration) names which can be set

    @staticmethod
    def conf(name):
        return App.__conf[name]

    @staticmethod
    def set(name, value):
        if name in App.__setters:
            App.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")

    # TODO: Lock for synchronization of access to shared resources
    # INFO: use queue instead of asynchio https://docs.python.org/3/library/queue.html

class Debug:
    parameter_debug = 234


if __name__ == "__main__":
    pass
