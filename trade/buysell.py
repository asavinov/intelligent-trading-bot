import os
import sys
import argparse
import math, time
from datetime import datetime
import pandas as pd
import asyncio

from apscheduler.schedulers.background import BackgroundScheduler

from binance.client import Client
from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds

from trade.utils import *
from trade.App import App
from trade.Database import *

import logging
log = logging.getLogger('buysell')


# TODO: We need to understand how to trade many different symbols. Should we do it completely independently starting from data requests and ending with trades?
#   Or we should assume that one service instance trades only one symbol. It is easier.
#   A better solution is that we trade one symbol always, but have either different strategies or two opposite directions.


async def sync_data_collect_task():
    """
    Collect latest data.
    After executing this task our local (in-memory) data state is up-to-date.
    Hence, we can do something useful like data analysis and trading.

    Limitations and notes:
    - Currently, we can work only with one symbol
    - We update only local state by loading latest data. If it is necessary to initialize the db then another function should be used.
    """

    symbols = App.config["trade"]["symbols"]
    symbol = symbols[0]  # Currently, we trade only one symbol (one pair)

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

async def sync_trade_task():
    """It will be executed for each trade normally every 1 minute."""
    log.info(f"===> Start trade task.")

    symbols = App.config["trade"]["symbols"]

    #
    # 1. Ensure that we are up-to-date
    #
    res = await sync_data_collect_task()

    if res > 0:
        return

    # TODO: In future, we might request more data like order book or latest transactions or state of our own orders

    # Now the local database is up-to-date with latest data from the market and hence can use for analysis

    #
    # 2. Derive features by using latest (up-to-date) daa from local db
    #

    # Generate features, generate predictions, generate signals
    # We use latest trained models (they are supposed to be periodically re-trained)
    for symbol in symbols:
        App.database.analyze(symbol)

    # Now we have a list of signals and can make trade decisions using trading logic and trade

    #
    # 4.
    # Main trade logic depends on the order status (open orders)
    #
    in_market = App.config["trade"]["state"]["in_market"]

    #
    # In market. Sell mode (trying to sell). An active limit order exists (but low funds)
    #
    if in_market:
        order_status = await update_sell_order_status()
        sell_order = App.config["trade"]["state"]["sell_order"]

        if sell_order is None or sell_order.get("status", 0) == 0 or order_status is None or order_status == 0:
            # No sell order exists or some problem
            # TODO: Need to recover by checking funds, updating/initializing/reseting complete trade state
            #   We cannot trade because it is not clear what happend with the sell order: no connection, wrong order state, order rejected etc.
            pass
        elif order_status == "REJECTED":
            # TODO: Create new sell order: either market (force sell) or repeat or compute new price
            # TODO: Recover.
            # Some problem which has to be logged. Yet, the order does not exist, btc is not sold so new limit order has to be created.
            pass
        elif order_status == "CANCELED" or order_status == "PENDING_CANCEL":
            # TODO: Recover:
            pass
        elif order_status == "EXPIRED":
            # TODO: Recover:
            #  Force sell etc.
            #  Create new sell order: either market (force sell) or repeat or compute new price
            pass
        elif order_status == "PARTIALLY_FILLED":
            # Do nothing. Wait further until the rest is filled (alternatively, we could force sell it)
            pass
        if order_status == "FILLED":
            # TODO: Log fulfilled transaction
            sell_order = None
            App.config["trade"]["state"]["sell_order"] = None
            in_market = False
        else:  # order_status == "NEW"
            # Order still exists and is active

            # TODO: Check timeout by comparing current time with the order start time
            sell_timeout = App.config["trade"]["signals"]["sell_timeout"]  # Seconds
            creation_ts = sell_order.get("creation_time", 0)
            now_ts = now_timestamp()
            if now_ts - creation_ts >= sell_timeout * 1_000:
                is_timeout = True
            else:
                is_timeout = False

            if is_timeout:
                # Force cell by converting into market order oder updating the limit price
                is_sold = await force_sell_limit_order()
                in_market = False
                # TODO: Log transaction and update other state (available amount etc.)

    #
    # In money. Buy mode (trying to buy). No orders exist (but there are funds)
    #
    if not in_market:
        # TODO: Check from data if there is a buy signal
        is_buy_signal = False  # App.config["trade"]["state"]["buy_signal"]

        if is_buy_signal:
            # Create, parameterize, submit and confirm execution of market buy order (enter market)
            is_bought = await create_execute_buy_order()
            # Create, parameters, submit limit sell order
            success = create_sell_order()

    log.info(f"<=== End trade task.")

async def update_sell_order_status(order_id):
    """
    Get order execution status for the specified order.
    The function is used to learn if the order has been filled (success) or is still waiting (also normal) or something is wrong.

    ASSUMPTIONS and notes:
    - Status codes: NEW PARTIALLY_FILLED FILLED CANCELED PENDING_CANCEL(currently unused) REJECTED EXPIRED
    - only one or no orders can be active currently, but in future there can be many orders
    - if no order id(s) is provided then retrieve all existing orders

    :param order_id:
    :return:
    """
    symbols = App.config["trade"]["symbols"]
    symbol = symbols[0]

    # Get currently active order and id (if any)
    sell_order = App.config["trade"]["state"]["sell_order"]
    sell_order_id = sell_order.get("orderId", 0) if sell_order else 0
    if sell_order_id == 0:
        # TODO: Maybe retrieve all existing (sell, limit) orders
        return None

    # ===
    # Retrieve order
    order = App.client.get_order(symbol=symbol, orderId=sell_order_id)

    # Impose and overwrite the new order information
    if order is None:
        return None
    else:
        sell_order.update(order)

    # Now order["status"] contains the latest status of the order
    return sell_order["status"]

async def cancel_sell_order():
    """
    Kill existing sell order. It is a blocking request, that is, it waits for the end of the operation.
    Info: DELETE /api/v3/order - cancel order
    """
    symbols = App.config["trade"]["symbols"]
    symbol = symbols[0]

    # Get currently active order and id (if any)
    sell_order = App.config["trade"]["state"]["sell_order"]
    sell_order_id = sell_order.get("orderId", 0) if sell_order else 0
    if sell_order_id == 0:
        # TODO: Maybe retrieve all existing (sell, limit) orders
        return None

    # ===
    cancel_response = App.client.cancel_sell_order(symbol=symbol, orderId=sell_order_id)

    if cancel_response["status"] == "CANCELED":
        sell_order.update(cancel_response)
        return True
    else:
        # TODO: Maybe get all open orders and kill them
        #  Or check the status of this order in a loop until we get cancelled status
        #  Or retrieve a list of all active orders as an indication of success and ensure that it is empty
        return False

# TODO:
async def force_sell_limit_order():
    sell_order = App.config["trade"]["state"]["sell_order"]
    sell_order_id = sell_order.get("orderId", 0) if sell_order else 0
    if sell_order_id == 0:
        # TODO: Maybe retrieve all existing (sell, limit) orders
        return None

    # Kill existing order
    is_cancelled = await cancel_sell_order()
    if not is_cancelled:
        # TODO: Log error. Will try to do the same on the next cycle.
        return False

    # Forget about this order (no need to log it)
    App.config["trade"]["state"]["sell_order"] = None

    # Create a new market sell order with the whole possessed amount to sell
    is_executed = await create_execute_sell_order()
    if not is_executed:
        # TODO: Log error. Will try to do the same on the next cycle.
        return False

    # Update state

    pass

async def create_execute_sell_order():
    """
    Sell all available btc currently possessed using a market sell order.
    It is a blocking request until everything is sold.
    The function determines the total quantity of btc we possess and then creates a market order.
    """

    # Determine total quantity we have on our account
    asset = "BTC"
    quantity = App.client.get_asset_balance(asset=asset)

    # Create and submit market sell order
    sell_order = {"id": "my_id", "amount": quantity, "side": side, "order_status": order_status}


# TODO:
async def create_execute_buy_order():
    #
    # Submit all orders
    #
    # INFO: order types: LIMIT MARKET STOP_LOSS STOP_LOSS_LIMIT TAKE_PROFIT TAKE_PROFIT_LIMIT LIMIT_MAKER
    # INFO: order side: BUY SELL
    # INFO: timeInForce: GTC IOC FOK
    # GET /api/v3/account - we use it to get "balances": [ {"asset": "BTC", "free": "123.456}, {} ]

    balance = App.client.get_asset_balance(asset='BTC')

    for symbol in symbols:
        signal = signals[symbol]
        if signal not in ["buy", "sell"]:
            continue  # Nothing to do

        # Load order book (order book could be requested along with klines)
        # order_book = App.client.get_order_book(symbol="BTCUSDT", limit=100)  # 100-1_000
        # order_book_ticker = App.client.get_orderbook_ticker(symbol="BTCUSDT")  # dict: "bidPrice", "bidQty", "askPrice", "askQty",
        # print(order_book_ticker)

        # BUY SELL
        if signal == "buy":
            side = "BUY"
        elif signal == "sell":
            side = "SELL"
        type = "LIMIT"  # LIMIT MARKET STOP_LOSS STOP_LOSS_LIMIT TAKE_PROFIT TAKE_PROFIT_LIMIT LIMIT_MAKER
        timeInForce = "GTC"  # GTC IOC FOK
        quantity = 0.002  # Determine from market data (like order book), or market oder or what is available
        price = 0.0  # Determine from market data (like order book), or market order
        #newClientOrderId = 123  # Auto generated if not sent
        newOrderRespType = "FULL"  # ACK, RESULT, or FULL (default for MARKET and LIMIT)
        timestamp = 123  # Mandatory
        binance_order = {"id": newClientOrderId, "amount": quantity, "side": side, "order_status": order_status}
        # TODO: Submit binance order

        # Create our own order record for tracking/logging
        order = {"id": newClientOrderId, "symbol": symbol, "created": "", "binance_order": binance_order}
        orders.append(order)

# TODO:
async def create_limit_sell_order():
    pass


async def get_existing_orders():
    # GET /api/v3/openOrders - get current open orders
    # GET /api/v3/allOrders - get all orders: active, canceled, or filled
    # Alternatively, we could retrieve all (open) orders which is probably more reliable because we cannot lose any order
    # In this case, we get a list of orders and update our complete status
    # orders = App.client.get_all_orders(symbol=symbol, limit=10)
    # open_orders = App.client.get_open_orders(symbol=symbol)  # It seems that by "open" orders they mean "NEW" or "PARTIALLY_FILLED"
    pass








async def request_klines(symbol, freq, limit):
    """
    Request klines data from the service for one symbol. Maximum the specified number of klines will be returned.

    :return: Dict with the symbol as a key and a list of klines as a value. One kline is also a list.
    """
    requestTime = now_timestamp()

    startTime, endTime = get_interval(freq)

    klines = []
    try:
        # INFO:
        # - startTime: include all intervals (ids) with same or greater id: if within interval then excluding this interval; if is equal to open time then include this interval
        # - endTime: include all intervals (ids) with same or smaller id: if equal to left border then return this interval, if within interval then return this interval
        # - It will return also incomplete current interval (in particular, we could collect approximate klines for higher frequencies by requesitng incomplete intervals)
        klines = App.client.get_klines(symbol=symbol, interval=freq, limit=limit, endTime=requestTime)
    except BinanceRequestException as bre:
        # {"code": 1103, "msg": "An unknown parameter was sent"}
        try:
            # Collect information about the error
            bre.code
            bre.msg
        except:
            pass
    except BinanceAPIException as bae:
        # {"code": 1002, "msg": "Invalid API call"}
        try:
            # Collect information about the error
            bae.code
            bae.message
            # depth['msg'] = bae.msg  # Does not exist
        except:
            pass

    responseTime = now_timestamp()

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

def start_trade():
    #
    # Validity check
    #
    symbols = App.config["trade"]["symbols"]
    if len(symbols) != 1:
        print("ERROR: Currently only one symbol can be traded. Exit.")
        return

    #
    # Initialize data state, connections and listeners
    #
    App.database = Database(None)
    # TODO: Load available historic klines either from file or from service

    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

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
        lambda: asyncio.run_coroutine_threadsafe(sync_trade_task(), App.loop),
        trigger='cron',
        #second='*/30',
        minute='*',
        id='sync_trade_task'
    )

    App.sched.start()  # Start scheduler (essentially, start the thread)

    #
    # Start event loop
    #

    App.loop = asyncio.get_event_loop()
    try:
        #App.loop.run_until_complete(work())
        App.loop.run_forever()  # Blocking. Run until stop() is called
    except KeyboardInterrupt:
        pass
    finally:
        print("===> Closing Loop")
        App.loop.close()
        App.sched.shutdown()

    return 0

if __name__ == "__main__":
    App.database = Database(None)
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])
    App.loop = asyncio.get_event_loop()
    try:
        App.loop.run_until_complete(sync_trade_task())
    except KeyboardInterrupt:
        pass
    finally:
        print("===> Closing Loop")
        App.loop.close()
        App.sched.shutdown()

    pass
