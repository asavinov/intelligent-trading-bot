import os
import sys
import argparse
import math, time
from datetime import datetime
from decimal import *

import pandas as pd
import asyncio

from apscheduler.schedulers.background import BackgroundScheduler

from binance.client import Client
from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from binance.enums import *

from common.utils import *
from trade.App import App
from trade.Database import *

import logging
log = logging.getLogger('signaler')
logging.basicConfig(filename='signaler.log', level=logging.DEBUG)  # filename='example.log', - parameter in App




async def sync_signaler_task():
    """
    It is a highest level task which is added to the event loop and executed normally every 1 minute and then it calls other tasks.
    """
    symbol = App.config["trader"]["symbol"]
    startTime, endTime = get_interval("1m")
    now_ts = now_timestamp()

    log.info(f"===> Start signaler task. Timestamp {now_ts}. Interval [{startTime},{endTime}].")

    #
    # 0. Check server state (if necessary)
    #
    if problems_exist():
        await update_state_and_health_check()
        if problems_exist():
            log.error(f"There are problems with connection, server, account or consistency.")
            return

    #
    # 1. Ensure that we are up-to-date with klines
    #
    res = await sync_data_collector_task()

    if res > 0:
        return

    # Now the local database is up-to-date with latest (klines) data from the market and hence can use for analysis

    #
    # 2. Derive features by using latest (up-to-date) daa from local db
    #

    # Generate features, generate predictions, generate signals
    # We use latest trained models (they are supposed to be periodically re-trained)
    App.database.analyze(symbol)

    # Now we have a list of signals and can make trade decisions using trading logic and trade

    last_kline_ts = App.database.get_last_kline_ts(symbol)
    if last_kline_ts + 60_000 != startTime:
        log.error(f"Problem during analysis. Last kline end ts {last_kline_ts + 60_000} not equal to start of current interval {startTime}.")

    is_buy_signal = App.config["trader"]["state"]["buy_signal"]
    buy_signal_scores = App.config["trader"]["state"]["buy_signal_scores"]
    log.debug(f"Analysis finished. Buy signal: {is_buy_signal} with scores {buy_signal_scores}")
    if is_buy_signal:
        log.debug(f"\n==============  BUY SIGNAL  ==============. Scores: {buy_signal_scores}\n")

    log.info(f"<=== End signaler task.")

#
# Server and account info
#

async def update_state_and_health_check():
    """
    Request information about the current state of the account (balances), order (buy and sell), server state.
    This function is called when we want to get complete real (true) state, for example, after re-start or network problem.
    It sets our state by requesting information from the server.
    """
    symbol = App.config["trader"]["symbol"]

    # Get server state (ping) and trade status (e.g., trade can be suspended on some symbol)
    system_status = App.client.get_system_status()
    #{
    #    "status": 0,  # 0: normal，1：system maintenance
    #    "msg": "normal"  # normal or System maintenance.
    #}
    if not system_status or system_status.get("status") != 0:
        App.config["trader"]["state"]["server_status"] = 1
        return 1
    App.config["trader"]["state"]["server_status"] = 0

    # "orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]
    # "isSpotTradingAllowed": True

    # Ping the server

    # Check time synchronization
    #server_time = App.client.get_server_time()
    #time_diff = int(time.time() * 1000) - server_time['serverTime']
    # TODO: Log large time differences (or even trigger time synchronization if possible)

    # Get symbol info
    symbol_info = App.client.get_symbol_info(symbol)
    App.config["trader"]["symbol_info"] = symbol_info
    if not symbol_info or symbol_info.get("status") != "TRADING":
        App.config["trader"]["state"]["server_status"] = 1
        return 1
    App.config["trader"]["state"]["server_status"] = 0

    # Get account trading status (it can be blocked/suspended, e.g., too many orders)
    account_info = App.client.get_account()
    if not account_info or not account_info.get("canTrade"):
        App.config["trader"]["state"]["account_status"] = 1
        return 1
    App.config["trader"]["state"]["account_status"] = 0

    # Get current balances (available funds)
    #balance = App.client.get_asset_balance(asset=App.config["trader"]["base_asset"])
    balance = next((b for b in account_info.get("balances", []) if b.get("asset") == App.config["trader"]["base_asset"]), {})
    App.config["trader"]["state"]["base_quantity"] = Decimal(balance.get("free", "0.00000000"))

    #balance = App.client.get_asset_balance(asset=App.config["trader"]["quote_asset"])
    balance = next((b for b in account_info.get("balances", []) if b.get("asset") == App.config["trader"]["quote_asset"]), {})
    App.config["trader"]["state"]["quote_quantity"] = Decimal(balance.get("free", "0.00000000"))

    # Get current active orders
    #orders = App.client.get_all_orders(symbol=symbol, limit=10)  # All orders
    orders = App.client.get_open_orders(symbol=symbol)
    if len(orders) == 0:  # No open orders
        App.config["trader"]["state"]["sell_order"] = None  # Forget about our sell order
    elif len(orders) == 1:
        order = orders[0]
        if order["side"] == "BUY":
            App.config["trader"]["state"]["trade_state_status"] = "Buy order still open. Market buy order have to be executed immediately."
            return 1
        elif order["side"] == "SELL":
            # It is our limit sell order. We are expected to be in market (check it) and assets should be as expected.
            # Check that this order exists and update its status
            pass
    else:
        App.config["trader"]["state"]["trade_state_status"] = "More than 1 active order. There cannot be more than 1 active order."
        return 1

    App.config["trader"]["state"]["trade_state_status"] = 0

    return 0

def problems_exist():
    if App.config["trader"]["state"]["error_status"] != 0:
        return True
    if App.config["trader"]["state"]["server_status"] != 0:
        return True
    if App.config["trader"]["state"]["account_status"] != 0:
        return True
    if App.config["trader"]["state"]["trade_state_status"] != 0:
        return True
    return False

#
# Request/update market data
#

# Load order book (order book could be requested along with klines)
# order_book = App.client.get_order_book(symbol="BTCUSDT", limit=100)  # 100-1_000
# order_book_ticker = App.client.get_orderbook_ticker(symbol="BTCUSDT")  # dict: "bidPrice", "bidQty", "askPrice", "askQty",
# print(order_book_ticker)

async def sync_data_collector_task():
    """
    Collect latest data.
    After executing this task our local (in-memory) data state is up-to-date.
    Hence, we can do something useful like data analysis and trading.

    Limitations and notes:
    - Currently, we can work only with one symbol
    - We update only local state by loading latest data. If it is necessary to initialize the db then another function should be used.
    """

    symbol = App.config["trader"]["symbol"]
    symbols = [symbol]  # In future, we might want to collect other data, say, from other cryptocurrencies

    # Request newest data
    # We do this in any case in order to update our state (data, orders etc.)
    missing_klines_count = App.database.get_missing_klines_count(symbol)

    #coros = [request_klines(sym, "1m", 5) for sym in symbols]
    tasks = [asyncio.create_task(request_klines(sym, "1m", missing_klines_count+1)) for sym in symbols]

    results = {}
    timeout = 5  # Seconds to wait for the result

    # Process responses in the order of arrival
    for fut in asyncio.as_completed(tasks, timeout=timeout):
        # Get the results
        res = None
        try:
            res = await fut
        except TimeoutError as te:
            log.warning(f"Timeout {timeout} seconds when requesting kline data.")
            return 1
        except Exception as e:
            log.warning(f"Exception when requesting kline data.")
            return 1

        # Add to the database (will overwrite existing klines if any)
        if res and res.keys():
            results.update(res)
            try:
                added_count = App.database.store_klines(res)
            except Exception as e:
                log.error(f"Error storing kline result in the database. Exception: {e}")
                return 1
        else:
            log.error("Received empty or wrong result from klines request.")
            return 1

    return 0

async def request_klines(symbol, freq, limit):
    """
    Request klines data from the service for one symbol. Maximum the specified number of klines will be returned.

    :return: Dict with the symbol as a key and a list of klines as a value. One kline is also a list.
    """
    now_ts = now_timestamp()

    startTime, endTime = get_interval(freq)

    klines = []
    try:
        # INFO:
        # - startTime: include all intervals (ids) with same or greater id: if within interval then excluding this interval; if is equal to open time then include this interval
        # - endTime: include all intervals (ids) with same or smaller id: if equal to left border then return this interval, if within interval then return this interval
        # - It will return also incomplete current interval (in particular, we could collect approximate klines for higher frequencies by requesting incomplete intervals)
        klines = App.client.get_klines(symbol=symbol, interval=freq, limit=limit, endTime=now_ts)
        # Return: list of lists, that is, one kline is a list (not dict) with items ordered: timestamp, open, high, low, close etc.
    except BinanceRequestException as bre:
        # {"code": 1103, "msg": "An unknown parameter was sent"}
        log.error(f"BinanceRequestException while requesting klines: {bre}")
        return {}
    except BinanceAPIException as bae:
        # {"code": 1002, "msg": "Invalid API call"}
        log.error(f"BinanceAPIException while requesting klines: {bae}")
        return {}
    except Exception as e:
        log.error(f"Exception while requesting klines: {e}")
        return {}

    #
    # Post-process
    #

    # Find latest *full* (completed) interval in the result list.
    # The problem is that the result also contains the current (still running) interval which we want to exclude
    klines_full = [kl for kl in klines if kl[0] < startTime]

    last_full_kline = klines_full[-1]
    last_full_kline_ts = last_full_kline[0]

    if last_full_kline_ts != startTime - 60_000:
        log.error(f"UNEXPECTED RESULT: Last full kline timestamp {last_full_kline_ts} is not equal to previous full interval start {startTime - 60_000}. Maybe some results are missing and there are gaps.")

    # Return all received klines with the symbol as a key
    return {symbol: klines_full}

#
# Main procedure. Initialize everything
#

def start_signlaer():
    #
    # Validation
    #
    symbol = App.config["trader"]["symbol"]

    log.info(f"Initializing signaler server. Trade symbol {symbol}. ")

    #
    # Connect to the server and update/initialize our system state
    #
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    App.database = Database(None)

    App.loop = asyncio.get_event_loop()

    # Do one time server check and state update
    try:
        App.loop.run_until_complete(update_state_and_health_check())
    except:
        pass
    if problems_exist():
        log.error(f"Problems found. Check server, symbol, account or system state.")
        return

    log.info(f"Finished updating state and health check.")

    # Do one time data update (cold start)
    try:
        App.loop.run_until_complete(sync_data_collector_task())
    except:
        pass
    if problems_exist():
        log.error(f"Problems found. Check server, symbol, account or system state.")
        return

    log.info(f"Finished updating data.")

    #
    # Register schedulers
    #

    # INFO: Scheduling:
    #     - https://medium.com/greedygame-engineering/an-elegant-way-to-run-periodic-tasks-in-python-61b7c477b679
    #     - https://schedule.readthedocs.io/en/stable/ https://github.com/dbader/schedule - 6.6k
    #     - https://github.com/agronholm/apscheduler/blob/master/docs/index.rst - 2.1k
    #       - https://apscheduler.readthedocs.io/en/latest/modules/schedulers/asyncio.html
    #     - https://docs.python.org/3/library/sched.html

    App.sched = BackgroundScheduler(daemon=False)  # Daemon flag is passed to Thread (False means the program will not exit until all Threads are finished)
    #logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

    App.sched.add_job(
        # We register a normal Python function as a call back.
        # The only role of this function is to add an asyncio task to the event loop
        # INFO: Creating/adding asyncio tasks from another thread
        # - https://docs.python.org/3/library/asyncio-task.html#scheduling-from-other-threads
        # - App.loop.call_soon_threadsafe(sync_responder)  # This works, but takes a normal funciton (not awaitable), which has to call coroutine: eventLoop.create_task(coroutine())
        lambda: asyncio.run_coroutine_threadsafe(sync_signaler_task(), App.loop),
        trigger='cron',
        #second='*/30',
        minute='*',
        id='sync_signaler_task'
    )

    App.sched.start()  # Start scheduler (essentially, start the thread)

    log.info(f"Scheduler started.")

    #
    # Start event loop
    #
    try:
        App.loop.run_forever()  # Blocking. Run until stop() is called
    except KeyboardInterrupt:
        log.info(f"KeyboardInterrupt.")
        pass
    finally:
        App.loop.close()
        log.info(f"Event loop closed.")
        App.sched.shutdown()
        log.info(f"Scheduler shutdown.")

    return 0

if __name__ == "__main__":
    # Short version of start_trader (main procedure)
    App.database = Database(None)
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])
    App.loop = asyncio.get_event_loop()
    try:
        log.debug("Start in debug mode.")
        log.info("Start testing in main.")
        App.loop.run_until_complete(update_state_and_health_check())

        #App.loop.run_until_complete(check_limit_sell_order())

        App.loop.run_until_complete(sync_data_collector_task())

        App.database.analyze("BTCUSDT")

        #App.loop.run_until_complete(sync_signaler_task())
    except BinanceAPIException as be:
        # IP is not registred in binance
        # BinanceAPIException: APIError(code=-2015): Invalid API-key, IP, or permissions for action
        # APIError(code=-1021): Timestamp for this request was 1000ms ahead of the server's time.
        print(be)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(f"Exception {e}")
    finally:
        log.info(f"Finished.")
        App.loop.close()
        #App.sched.shutdown()

    pass
