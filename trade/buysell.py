import os
import sys
import argparse
import math, time
from datetime import datetime
import decimal

import pandas as pd
import asyncio

from apscheduler.schedulers.background import BackgroundScheduler

from binance.client import Client
from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from binance.enums import *

from trade.utils import *
from trade.App import App
from trade.Database import *

import logging
log = logging.getLogger('buysell')


async def sync_trade_task():
    """
    It is a highest level task which is added to the event loop and executed normally every 1 minute and then it calls other tasks.
    """
    log.info(f"===> Start trade task.")

    symbol = App.config["trade"]["symbol"]

    #
    # 0. Check server state (if necessary)
    #

    #
    # 1. Ensure that we are up-to-date with klines
    #
    res = await sync_data_collect_task()

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

    #
    # 4.
    # Main trade logic depends on the order status (no or one open order created before)
    #
    in_market = App.config["trade"]["state"]["in_market"]

    #
    # In market. Sell mode (trying to sell). An active limit order is supposed to exist (and hence low funds)
    #
    if in_market:

        # ---
        # Check that we are really in market (the order could have been sold already)
        # We must get status even if the order has been filled
        order_status = await update_sell_order_status()
        sell_order = App.config["trade"]["state"]["sell_order"]

        if sell_order is None or (not sell_order.get("status")) == 0 or (not order_status):
            # No sell order exists or some problem
            # TODO (RECOVER ERROR, suspend): Need to recover by checking funds, updating/initializing/reseting complete trade state
            #   We cannot trade because it is not clear what happend with the sell order:
            #   - no connection,
            #   - wrong order state,
            #   - order rejected etc.
            #   First, we need to check connection (like ping), then server status, then our own account status, then funds, orders etc.
            #     The result is a recovered initialized state or some error which is then used to suspend trade (suspended state will be then used to regularly try to recover again)
            pass

        elif order_status == "REJECTED":
            # TODO (ERROR, suspend): Create new sell order: either market (force sell) or repeat or compute new price
            # TODO: Recover: either possible or suspend state with some reason like cannot submit orders because they are rejected.
            #   Some problem which has to be logged. Yet, the order does not exist, btc is not sold so new limit order has to be created.
            pass

        elif order_status == "CANCELED" or order_status == "PENDING_CANCEL":
            # TODO: Why it can be cancelled? If only we can do this, then it is normal.
            pass

        elif order_status == "EXPIRED":
            # TODO:
            #  Try exit market again. Create new sell order:
            #  - either market (force sell)
            #  - or repeat old limit order
            #  - or limit order with new updated price
            pass

        elif order_status == "PARTIALLY_FILLED":
            # Do nothing. Wait further until the rest is filled (alternatively, we could force sell it)
            pass

        elif order_status == "FILLED":  # Success: order filled
            # TODO (LOG): Log fulfilled transaction
            sell_order = None
            App.config["trade"]["state"]["sell_order"] = None
            in_market = False

        # elif order_status == "NEW"
        else:  # Order still exists and is active
            # TODO: Check timeout by comparing current time with the order start time
            sell_timeout = App.config["trade"]["parameters"]["sell_timeout"]  # Seconds
            creation_ts = sell_order.get("creation_time", 0)
            now_ts = now_timestamp()
            if now_ts - creation_ts >= sell_timeout * 1_000:
                is_timeout = True
            else:
                is_timeout = False

            if is_timeout:
                # ---
                # Force cell by converting into market order oder updating the limit price
                is_sold = await force_sell()
                if is_sold:
                    # TODO (LOG): Log fulfilled transaction
                    sell_order = None
                    App.config["trade"]["state"]["sell_order"] = None
                    in_market = False
                else:
                    # TODO (ERROR, suspend)
                    pass

    #
    # In money. Buy mode (trying to buy). No orders are known to exist (but there are funds)
    #
    if not in_market:
        # TODO: Check from data if there is a buy signal
        #   App.config["trade"]["state"]["buy_signal"]
        is_buy_signal = False

        if is_buy_signal:

            # ---
            # Create, parameterize, submit and confirm execution of market buy order (enter market)
            is_bought = await new_market_buy_order()

            # ---
            # Create, parameters, submit limit sell order
            success = new_limit_sell_order()

    log.info(f"<=== End trade task.")

#
# Server and account info
#

async def update_server_account_state():
    """
    Request information about the current state of the account (balances), order (buy and sell), server state.
    This function is called when we want to get complete real (true) state, for example, after re-start or network problem.
    It sets our state by requesting information from the server.
    """
    symbol = App.config["trade"]["symbol"]

    # Get server state (ping) and trade status (e.g., trade can be suspended on some symbol)
    system_status = App.client.get_system_status()

    server_time = App.client.get_server_time()
    time_diff = int(time.time() * 1000) - server_time['serverTime']

    # Get symbol info
    symbol_info = App.client.get_symbol_info(symbol)
    App.config["trade"]["symbol_info"] = symbol_info
    # NOTE: for us, an important field is "status": "TRADING"
    # recvWindow=6000000

    # Get account trading status (it can be blocked/suspended, e.g., too many orders)
    account_info = App.client.get_account()
    App.config["trade"]["account_info"] = account_info
    # In addition: account_status = client.get_account_status()
    # NOTE: for, an important field is "canTrade": True

    # Get current balances (available funds)
    #balance = App.client.get_asset_balance(asset=App.config["trade"]["base_asset"])
    balance = next((b for b in account_info.get("balances", []) if b.get("asset") == App.config["trade"]["base_asset"]), {})
    App.config["trade"]["state"]["base_quantity"] = balance.get("free", "0.00000000")

    #balance = App.client.get_asset_balance(asset=App.config["trade"]["quote_asset"])
    balance = next((b for b in account_info.get("balances", []) if b.get("asset") == App.config["trade"]["quote_asset"]), {})
    App.config["trade"]["state"]["quote_quantity"] = balance.get("free", "0.00000000")

    # Get current active orders
    #orders = App.client.get_all_orders(symbol=symbol, limit=10)  # All orders
    orders = App.client.get_open_orders(symbol=symbol)
    if len(orders) == 0:  # No open orders
        App.config["trade"]["state"]["sell_order"] = None  # Forget about our sell order
    elif len(orders) == 1:
        order = orders[0]
        if order["side"] == "BUY":
            App.config["trade"]["state"]["error"] = "Buy order still open. Market buy order have to be executed immediately."
        elif order["side"] == "SELL":
            # It is our limit sell order. We are expected to be in market (check it) and assets should be as expected.
            # Check that this order exists and update its status
            pass
    else:
        App.config["trade"]["state"]["error"] = "More than 1 active order. There cannot be more than 1 active order."

#
# Request/update market data
#

# Load order book (order book could be requested along with klines)
# order_book = App.client.get_order_book(symbol="BTCUSDT", limit=100)  # 100-1_000
# order_book_ticker = App.client.get_orderbook_ticker(symbol="BTCUSDT")  # dict: "bidPrice", "bidQty", "askPrice", "askQty",
# print(order_book_ticker)

async def sync_data_collect_task():
    """
    Collect latest data.
    After executing this task our local (in-memory) data state is up-to-date.
    Hence, we can do something useful like data analysis and trading.

    Limitations and notes:
    - Currently, we can work only with one symbol
    - We update only local state by loading latest data. If it is necessary to initialize the db then another function should be used.
    """

    symbol = App.config["trade"]["symbol"]
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

#
# Order and asset status
#

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
    symbol = App.config["trade"]["symbol"]

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

async def get_existing_orders():
    # GET /api/v3/openOrders - get current open orders
    # GET /api/v3/allOrders - get all orders: active, canceled, or filled
    # Alternatively, we could retrieve all (open) orders which is probably more reliable because we cannot lose any order
    # In this case, we get a list of orders and update our complete status
    # orders = App.client.get_all_orders(symbol=symbol, limit=10)
    # open_orders = App.client.get_open_orders(symbol=symbol)  # It seems that by "open" orders they mean "NEW" or "PARTIALLY_FILLED"
    pass


# INFO: order types: LIMIT MARKET STOP_LOSS STOP_LOSS_LIMIT TAKE_PROFIT TAKE_PROFIT_LIMIT LIMIT_MAKER
# INFO: order side: BUY SELL
# INFO: timeInForce: GTC IOC FOK
# GET /api/v3/account - we use it to get "balances": [ {"asset": "BTC", "free": "123.456}, {} ]
#balance = App.client.get_asset_balance(asset='BTC')

#
# Cancel and liquidation orders
#

async def cancel_sell_order():
    """
    Kill existing sell order. It is a blocking request, that is, it waits for the end of the operation.
    Info: DELETE /api/v3/order - cancel order
    """
    symbol = App.config["trade"]["symbol"]

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

async def force_sell():
    """
    Force sell available btc and exit market.
    We kill an existing limit sell order (if any) and then create a new sell market order.
    """
    symbol = App.config["trade"]["symbol"]

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
    is_executed = await new_market_sell_order()
    if not is_executed:
        # TODO: Log error. Will try to do the same on the next cycle.
        return False

    # Update state

    pass

#
# Order creation
#

# Order parameters
#type = "LIMIT"  # LIMIT MARKET STOP_LOSS STOP_LOSS_LIMIT TAKE_PROFIT TAKE_PROFIT_LIMIT LIMIT_MAKER
#timeInForce = "GTC"  # GTC IOC FOK
#quantity = 0.002  # Determine from market data (like order book), or market oder or what is available
#price = 0.0  # Determine from market data (like order book), or market order
##newClientOrderId = 123  # Auto generated if not sent
#newOrderRespType = "FULL"  # ACK, RESULT, or FULL (default for MARKET and LIMIT)
#timestamp = 123  # Mandatory

async def new_market_buy_order():
    """
    Submit a new market buy order. Wait until it is executed.
    """
    symbol = App.config["trade"]["symbol"]

    #
    # Determine BTC quantity to buy depending on how much USDT we have and what is the latest price taking into account possible price increase
    #
    latest_price = App.config["trade"]["state"]["latest_price"]

    quote_quantity = App.config["trade"]["state"]["quote_quantity"]
    percentage_used_for_trade = App.config["trade"]["parameters"]["percentage_used_for_trade"]
    quantity = (quote_quantity * percentage_used_for_trade) / 100.0 / latest_price
    # Alternatively, we can pass quoteOrderQty in USDT (how much I want to spend)

    decimal.getcontext().rounding = decimal.ROUND_DOWN
    decimal.getcontext().prec = App.config["trade"]["symbol_info"]["baseAssetPrecision"]
    quantity = decimal.Decimal(quantity)

    #
    # Execute order
    #
    order_spec = dict(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, timeInForce=TIME_IN_FORCE_GTC, quantity=quantity)
    # newOrderRespType (ACK, RESULT, or FULL) for market and limit order defaults to FULL

    order = execute_order(order_spec)

    # Process response

    #
    # Store/log order object in our records
    #
    App.config["trade"]["state"]["buy_order"] = order
    App.config["trade"]["state"]["in_market"] = True
    App.config["trade"]["state"]["in_market_price"] = order.get("price", None)

    # TODO: Update available funds

async def new_market_sell_order():
    """
    Sell all available btc currently possessed using a market sell order.
    It is a blocking request until everything is sold.
    The function determines the total quantity of btc we possess and then creates a market order.
    """
    symbol = App.config["trade"]["symbol"]

    #
    # We want to sell all BTC we own.
    #
    quantity = App.config["trade"]["state"]["base_quantity"]

    decimal.getcontext().rounding = decimal.ROUND_DOWN
    decimal.getcontext().prec = App.config["trade"]["symbol_info"]["baseAssetPrecision"]
    quantity = decimal.Decimal(quantity)

    #
    # Execute order
    #
    order_spec = dict(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, timeInForce=TIME_IN_FORCE_GTC, quantity=quantity)

    order = execute_order(order_spec)

    # Process response

    #
    # Store/log order object in our records (only after confirmation of success)
    #
    # TODO: Store new order
    # TODO: Update available funds

async def new_limit_sell_order():
    """
    Create a new limit sell order with the amount we current have (and have just bought).
    The amount is total amount and price is determined according to our strategy (either fixed increase or increase depending on the signal).
    """
    symbol = App.config["trade"]["symbol"]

    #
    # We want to sell all BTC we own but the price is derived from previous (buy) order price
    #
    quantity = App.config["trade"]["state"]["base_quantity"]

    decimal.getcontext().rounding = decimal.ROUND_DOWN
    decimal.getcontext().prec = App.config["trade"]["symbol_info"]["baseAssetPrecision"]
    quantity = decimal.Decimal(quantity)

    in_market_price = App.config["trade"]["state"]["in_market_price"]
    price = in_market_price * (100.0 + App.config["trade"]["parameters"]["percentage_sell_price"]) / 100.0
    price = decimal.Decimal(price)

    #
    # Execute order
    #
    order_spec = dict(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT, timeInForce=TIME_IN_FORCE_GTC, quantity=quantity, price=price)

    order = execute_order(order_spec)

    # Process response

    #
    # Store/log order object in our records (only after confirmation of success)
    #
    # TODO: Store new order
    # TODO: Update available funds

def execute_order(order: dict):
    """
    Validate and execute order.

    Results/influences:
    - output type
    - App.state, say, created order or sold order or latest prices
    - Error state like suspended trade, timeout state (e.g., to check execution of some order), no connection state or complete recovery state update
      Note that in the case of recover, we could find any asset and order state, that is, existing order or no funds etc.

    For validation errors, ...
    For market orders, ...
    For limit orders,  ...
    For simulated orders, ...
    """

    # TODO: Check validity, e.g., against filters (min, max) and our own limits

    if App.config["trade"]["parameters"]["test_order_before_submit"]:
        test_response = App.client.create_test_order(**order)  # Returns {} if ok, but what if error? Exception?

    if App.config["trade"]["parameters"]["simulate_order_execution"]:
        # TODO:
        #   For market buy order, use latest close price (slightly worse)
        #   For market sell order, use latest close price (slightly worse)
        #   For limit sell order, simply make a submission record, and then, during regualr check, check the latest high price of kline to determine if it is filled or not.
        pass

    else:
        order = App.client.create_order(**order)  # Response FULL: "status": "FILLED", (there can be many fills with different partial quantities, prices and comissions)

        if not order or not order.get("status"):
            pass  # TODO: Error. Recover.

        if order.get("status") != "FILLED":
            order = wait_until_filled(order)

        # If filled, then return ...
        # If timeout or error, then ...

    return order

async def wait_until_filled(order):
    """Regularly check the status of this order until it is filled and return the original order updated with new responses."""
    symbol = order.get("symbol", "")
    orderId = order.get("orderId", "")

    response = order
    while not response or response.get("status") != "FILLED":
        response = App.client.get_order(symbol=symbol, orderId=orderId)
        # TODO: sleep
        # TODO: Return error after some number of attempts

    order.update(response)

    return order

#
# Main procedure. Initialize everything
#

def start_trade():
    #
    # Validity check
    #
    symbol = App.config["trade"]["symbol"]

    # TODO: Retrieve exchange info and store (overwrite) it App.config["trade"]["symbol_info"]
    #   Check the necessary working status of the server (it can be part of a general validation procedure like check_server_state):
    #     "status": "TRADING"
    #     "orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]
    #     "isSpotTradingAllowed": True

    symbol = App.config["trade"]["symbol"]

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
        App.loop.run_until_complete(update_server_account_state())
        #App.loop.run_until_complete(sync_trade_task())
    except KeyboardInterrupt:
        pass
    finally:
        print("===> Closing Loop")
        App.loop.close()
        App.sched.shutdown()

    pass
