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
    symbol = symbols[0]  # Further we work with only one symbol

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

async def sync_order_status_task():
    """
    OPTIMIZATION: we could do it concurrently with other requests like klines because it belongs to the state where we update current state
    ASSUMPTIONS and notes:
    - Status codes: NEW PARTIALLY_FILLED FILLED CANCELED PENDING_CANCEL(currently unused) REJECTED EXPIRED
    - only one order can be active: either buy market order or sell limit order (or sell market order in exceptional cases)
    - only one symbol (pair) is traded
    - we either have to know current order ids, or request all running/active order ids

    :return:
    """
    sell_order = App.config["trade"]["state"]["sell_order"]
    buy_order = App.config["trade"]["state"]["buy_order"]

    # Determine currently active order id
    order_id = sell_order.get("orderId", 0) if sell_order else (buy_order.get("orderId", 0) if buy_order else None)
    # Send a request to check order status
    order = None
    if order_id and order_id != 0:
        order = App.client.get_order(symbol=symbol, orderId=order_id)

    # Alternatively, we could retrieve all (open) orders which is probably more reliable because we cannot lose any order
    # In this case, we get a list of orders and update our complete status
    # orders = App.client.get_all_orders(symbol=symbol, limit=10)
    # open_orders = App.client.get_open_orders(symbol=symbol)  # It seems that by "open" orders they mean "NEW" or "PARTIALLY_FILLED"

    # Note that we also need to get information about filled (executed) orders for logging purposes

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
    # 3. Update status of existing orders
    #
    res = await sync_order_status_task()

    # Now we have up-to-date order (and assert) status and can choose trade mode

    #
    # 4.
    # Main trade logic depends on the order status (open orders)
    #

    # Sell mode (no funds or low funds by partial fill)
    # There is a submitted order (not executed).
    # Our logic is that we never buy until everything is sold (until full exit)
    # Also, sell time in our strategy takes long time because we use limit orders with expiration by passively waiting for execution
    # *Waiting* for filling a sell order is what we do most of the time
    # Our goal is to cacth the moment when it is filled itself or expires automatically or we adjust/fill it manually after time out
    if sell_order:
        if order["status"] == "FILLED":
            # TODO: Log the order
            # Update our state and forget the order (trade cycle finished)
            sell_order = None
            App.config["trade"]["state"]["sell_order"] = None
            # We are now in buy mode since we have funds
        elif order["status"] == "PARTIALLY_FILLED":
            pass  # It is absolutely possible for limit orders.
            # Wait further untill the rest is filled (alternatively, we could force sell it)
        elif order["status"] == "REJECTED":
            pass  # Some problem which has to be logged.
            # TODO: Create new sell order: either market (force sell) or repeat or compute new price
        elif order["status"] == "EXPIRED":
            pass  # For example, if it is day-long order
            # TODO: Create new sell order: either market (force sell) or repeat or compute new price
        else:  # Order still exists. Groom.
            # In our strategy, check time out condition and force sell it by killing and creating a market sell order

            # Check how old the order is by comparing its creation time and current time with our time out (parameter of the strategy)
            sell_timeout = App.config["trade"]["signals"]["sell_timeout"]  # Seconds
            creation_ts = sell_order.get("creation_time", 0)
            now_ts = now_timestamp()
            if now_ts - creation_ts >= sell_timeout * 1_000:
                # Kill order
                cancel_response = App.client.cancel_order(symbol=symbol, orderId=order_id)
                if cancel_response["status"] == "CANCELED":
                    # Check funds
                    balance = App.client.get_asset_balance(asset='BTC')
                    quantity = order["origQty"]
                    if balance != quantity:
                        pass  # TODO: Either funds not updated yet after cancelling or something wrong.

                    # TODO: Create a market sell order with the same parameters as the cancelled order
                    # Take new order parameters from the cancelled sell order but make market order (not limit order)

                    #  Check if it is executed immediately. If not, then wait for the next cycle.
                    # Note that this same logic will be used if no orders found and no funds, when we force sell what we have
                    pass
                else:
                    # Do nothing. We will check again on the next cycle. Note that we will not have orders but we still will have to sell
                    # Alternatively, wait and check again either original or this cancel order id
                    pass




    if buy_order:
        # TODO: How to send a request to check order status?
        # Update the status of the order (executed or not)
        # If executed, then probably we do not need it anymore and can store it in log and clean the state
        # TODO: If executed, then we need to create a sell order immediately
        # If not executed, then (in our strategy) something is wrong - warning or cancel
        pass


    if sell_order is None:  # Buy mode (funds available) - no sell order but there might be buy order submitted
        # If there exist a buy order (for whatever reason), then do nothing (check its status again in the next cycle).
        # Currently this should not happen, so warning.

        buy_signal = App.config["trade"]["state"]["buy_signal"]

        # Check the buy signal
        if buy_signal:
            # TODO: How to submit a new buy order. Submit a (market) buy order and continue (will be checked on the next cycle).
            pass
        else:  # No buy signal
            pass  # No-op



    # Principles of our strategy:
    # - Buy order is market order, sell order is a limit order
    # - Only 3 cases: buy order running, sell order running, no orders running (buy and sell orders running is impossible)
    #   - if running orders,
    #     - buy order is running: no-op (wait for the result by checking order status regularly maybe more frequently), maybe price adjustment
    #     - sell order running: check if adjustment is needed, time out with (market price *adjustment*), or cancel
    # - if no running orders then two situations:
    #   - we have money (and can buy), if buy signal, then immediately new *buy* order (for market price), otherwise no-op
    #   - we have coins (and can sell), immediately new limit *sell* order ideally right after buy order executed (everything else is controlled via its limit price)









    #
    # Submit all orders
    #
    # INFO: order types: LIMIT MARKET STOP_LOSS STOP_LOSS_LIMIT TAKE_PROFIT TAKE_PROFIT_LIMIT LIMIT_MAKER
    # INFO: order side: BUY SELL
    # INFO: timeInForce: GTC IOC FOK
    # GET /api/v3/account - we use it to get "balances": [ {"asset": "BTC", "free": "123.456}, {} ]
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

    #
    # Track order status
    #
    param_num_order_checks = 3
    param_check_order_delay = 10
    #for i in range(param_num_order_checks):
    #    asyncio.sleep(param_check_order_delay)
    #    for order in orders:
    #        order_id = order.get("order_id")
    #        # TODO: Send binance request to check order status
    #        order["status"] = "executed"
    #        order["execution_price"] = 0.002
    #        # TODO: We need to record all possible situations: partial execution or some special status
    asyncio.sleep(param_check_order_delay)

    #
    # Cancel open orders
    #
    # INFO: order status: NEW PARTIALLY_FILLED FILLED CANCELED PENDING_CANCEL REJECTED EXPIRED
    # GET /api/v3/openOrders - get current open orders
    # GET /api/v3/allOrders - get all orders: active, canceled, or filled
    # DELETE /api/v3/order - cancel order
    for order in orders:
        order_status = order.get("order_status", "")
        if order_status == "executed":
            pass
        elif order_status == "running":
            # TODO: Cancel binance order
            order["order_status"] = "cancelled"
        else:  # All other statuse
            log.warning(f"Unexpected order status {order_status}")
            pass

    #
    # Update state, log, clean this trade session
    #
    for order in orders:
        order_status = order.get("order_status", "")
        quantity = order.get("amount", 0.0)
        symbol = order.get("symbol", "")
        base_amount = 0.0
        quote_amount = 0.0
        if order_status == "executed":
            # TODO: Update our balance of assets
            #   We need to know base asset and quote asset
            #   Then we need to know base asset volume and quote asset volume
            #   Maybe we can simply request the status of our assets from binance
            pass

    orders.clear()

    log.info(f"<=== End trade task.")

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
