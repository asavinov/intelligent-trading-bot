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

        # === DATA COLLECTION SERVER ===
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

        # === TRADE SERVER ===
        "trade": {
            #
            # Status data retrieved from the server. Below are examples only.
            #
            "system_status": {"status": 0, "msg": "normal"},  # 0: normal，1：system maintenance
            "symbol_info": {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"],
                "icebergAllowed": True,
                "ocoAllowed": True,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
                "filters": [],
            },
            "account_info": {
                "makerCommission": 15,
                "takerCommission": 15,
                "buyerCommission": 0,
                "sellerCommission": 0,
                "canTrade": True,
                "canWithdraw": True,
                "canDeposit": True,
                "balances": [
                    {"asset": "BTC", "free": "4723846.89208129", "locked": "0.00000000"},
                    {"asset": "LTC", "free": "4763368.68006011", "locked": "0.00000000"},
                ]
            },

            #
            # Static parameters of trade
            #
            "folder": "DATA",
            "symbol": "BTCUSDT",
            "base_asset": "BTC",
            "quote_asset": "USDT",

            "analysis": {  # Same for all symbols
                "folder": "MODELS",
                "features": [],
                "labels": [],
            },

            "parameters": {
                "test_order_before_submit": True,  # Send test submit to the server as part of validation
                "simulate_order_execution": True,  # Instead of real orders, simulate their execution (immediate buy/sell market orders and use high price of klines for limit orders)
                "sell_timeout": 90,  # Seconds
                "percentage_used_for_trade": 90,  # in % to the available USDT quantity, that is, we will derive how much BTC to buy using this percentage
                "percentage_sell_price": 1.0,  # in % to the buy price, that is, our planned profit per trade
            },

            #
            # Dynamic state being changed regularly
            #
            "state": {  # Current state updated after each trade session
                "error": None,

                # Server state (error etc.)
                "server_alive": True,

                # What we possess. Can be set by the sync/recover function or updated by the trade algorithm
                "base_quantity": "0.04108219",  # BTC owned (on account, already bought, available for trade)
                "quote_quantity": "1000.0",  # USDT owned (on account, available for trade)

                # Set by data retrieval and other update/sync functions
                "latest_price": "8000.0",
                # Here we might have latest buy or ask price etc.

                # Set by analysis procedure like signals
                "buy_signal": False,
                "sell_signal": False,

                # State. Can be initialized (if necessary), e.g., by sync function
                "in_market": False,  # True if we bought and posses BTC and False otherwise. Note that we might fail in creating a sell order.

                "buy_order": None,  # Latest order used to buy BTC or None if not in the market
                "in_market_price": "8000.0",  # Price of latest executed buy order used to enter the market.

                "sell_order": None,  # Latest active limit (or market in case of force sell) sell order. Must exist if in market - error if it does not. Can be updated by sync function.
                # NOTE: if we have an active limit order then we still own (base or quote) but we should not use them
                #   This "reserved" quantity can be determined from the active (not yet filled) order if necessary
                # NOTE: The avilable quantities are initialized and then can change during trading (e.g., decrease if we lose)
            },
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
