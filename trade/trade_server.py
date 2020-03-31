import os
import sys
import argparse
import math, time
from datetime import datetime
from _decimal import *

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
log = logging.getLogger('TRADE')
logging.basicConfig(filename='temp.log', level=logging.DEBUG)  # filename='example.log', - parameter in App

# TODO: after analysis: set buy signal true or false, actually it has to be stored in dataframe and we can retrieve it and store in the config field

# test runs:
# - run in data collection, analysis, signal generation only, excluding real trade (maybe only test orders) but reporting signals (and maybe timeouts of sells)
#   goals: check data collection/updates/discards work, check stream processing works with new data periodic analyses.
#   no trades (order creation etc.) - only analysis logic

# TODO: Add high level exception catchers too all calls where we have binance client calls (which may well fail so that they do not crash the server - see main, e.g., IP wrong)

#   - train model(s) and copy them to the folder - document the steps and develop standard procedures
#   - train signal model and copy to the folder (so that we can re-train it later) - document the steps and develop standard procedures
#   - configure App.config: feature list, label list etc.
#   - check that logging works and configure debug level for logging

# - run in order simulation mode
#   simulate orders by generating their specifications and then assume they are executed with any (valid) results
#   goals: check that orders get valid parameters like quantity and price

# TODO:
# - check/ensure that now_timestamp return UTC millis
# - surround all python-binance calls with try-catch
# - check if timeout for sell orders works correctly

async def sync_trade_task():
    """
    It is a highest level task which is added to the event loop and executed normally every 1 minute and then it calls other tasks.
    """
    symbol = App.config["trade"]["symbol"]
    startTime, endTime = get_interval("1m")
    now_ts = now_timestamp()

    log.info(f"===> Start trade task. Timestamp {now_ts}. Interval [{startTime},{endTime}].")

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
    # Get balances and determine whether we are in market or in cash
    #
    await update_account_state()
    base_quantity = App.config["trade"]["state"]["base_quantity"]
    quote_quantity = App.config["trade"]["state"]["quote_quantity"]
    if base_quantity > 0.00000010:
        App.config["trade"]["state"]["in_market"] = True
        in_market = True
    else:
        App.config["trade"]["state"]["in_market"] = False
        in_market = False

    last_kline_ts = App.database.get_last_kline_ts(symbol)
    if last_kline_ts + 60_000 != startTime:
        log.error(f"Problem during analysis. Last kline end ts {last_kline_ts + 60_000} not equal to start of current interval {startTime}.")

    is_buy_signal = App.config["trade"]["state"]["buy_signal"]
    buy_signal_scores = App.config["trade"]["state"]["buy_signal_scores"]
    log.debug(f"Analysis finished. BTC: {base_quantity:.8f}. USDT: {quote_quantity:.8f}. In market {in_market}. Buy signal: {is_buy_signal} with scores {buy_signal_scores}")
    if is_buy_signal:
        log.debug(f"\n==============  BUY SIGNAL  ==============. Scores: {buy_signal_scores}\n")

    if App.config["trade"]["parameters"]["no_trades_only_data_processing"]:
        log.info(f"<=== End trade task. Only data operations performed (trading is disabled).")
        return

    #
    # 4. In market. Trying to sell. An active limit order is supposed to exist (and hence low funds)
    #
    if in_market:
        await in_market_trade()  # Check status of an existing limit sell order

    #
    # 5. In money. Try to buy (more) because there are funds
    #
    if not in_market:
        await out_of_market_trade()

    log.info(f"<=== End trade task.")

async def in_market_trade():
    """Check the existing limit sell order if it has been filled."""

    # ---
    # Check that we are really in market (the order could have been sold already)
    # We must get status even if the order has been filled
    order_status = await update_sell_order_status()

    sell_order = App.config["trade"]["state"]["sell_order"]

    if not sell_order or not sell_order.get("status"):
        # No sell order exists or some problem
        # TODO (RECOVER ERROR, suspend): Need to recover by checking funds, updating/initializing/reseting complete trade state
        #   We cannot trade because it is not clear what happend with the sell order:
        #   - no connection,
        #   - wrong order state,
        #   - order rejected etc.
        #   First, we need to check connection (like ping), then server status, then our own account status, then funds, orders etc.
        #     The result is a recovered initialized state or some error which is then used to suspend trade (suspended state will be then used to regularly try to recover again)
        pass

    elif order_status == ORDER_STATUS_REJECTED:
        log.error(f"Wrong state or use: limit sell order rejected. Force sell.")
        await force_sell()
        pass

    elif order_status == ORDER_STATUS_CANCELED or order_status == ORDER_STATUS_PENDING_CANCEL:
        log.error(f"Wrong state or use: limit sell order cancelled. Force sell.")
        await force_sell()
        pass

    elif order_status == ORDER_STATUS_EXPIRED:
        log.error(f"Wrong state or use: limit sell order expired. Force sell.")
        await force_sell()
        pass

    elif order_status == ORDER_STATUS_PARTIALLY_FILLED:
        # Do nothing. Wait further until the rest is filled (alternatively, we could force sell it)
        pass

    elif order_status == ORDER_STATUS_FILLED:  # Success: order filled
        log.info(f"Limit sell order filled. {sell_order}")
        sell_order = None
        App.config["trade"]["state"]["sell_order"] = None
        in_market = False

    # elif order_status == ORDER_STATUS_NEW
    else:  # Order still exists and is active
        now_ts = now_timestamp()

        sell_timeout = App.config["trade"]["parameters"]["sell_timeout"]  # Seconds
        creation_ts = App.config["trade"]["state"]["sell_order_time"]

        if (now_ts - creation_ts) >= (sell_timeout * 1_000):
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

async def out_of_market_trade():
    """If buy signal, then enter the market and immediately creating a limit sell order."""

    # Result of analysis
    is_buy_signal = App.config["trade"]["state"]["buy_signal"]

    if not is_buy_signal:
        return

    # ---
    # Create, parameterize, submit and confirm execution of market buy order (enter market)
    buy_order = await new_market_buy_order()
    if not buy_order:
        log.error(f"Problem creating market buy order (empty response).")
        return

    # Give some time to the server to process the transaction
    await asyncio.sleep(2)

    # ---
    # Retrieve latest account state (important for making buy market order)
    await update_account_state()
    base_quantity = App.config["trade"]["state"]["base_quantity"]
    quote_quantity = App.config["trade"]["state"]["quote_quantity"]
    if base_quantity < 0.00000010:
        log.error(f"Problem or wrong state: attempt to create a limit sell order while the base quantity is 0. Base quantity: {base_quantity}. Quote quantity: {quote_quantity}")
        return

    # ---
    # Create, parameterize, submit limit sell order
    sell_order = new_limit_sell_order()
    if not sell_order:
        log.error(f"Problem creating limit sell order (empty response).")
        return

#
# Server and account info
#

async def update_account_state():

    balance = App.client.get_asset_balance(asset=App.config["trade"]["base_asset"])
    App.config["trade"]["state"]["base_quantity"] = Decimal(balance.get("free", "0.00000000"))

    balance = App.client.get_asset_balance(asset=App.config["trade"]["quote_asset"])
    App.config["trade"]["state"]["quote_quantity"] = Decimal(balance.get("free", "0.00000000"))

    pass

async def update_state_and_health_check():
    """
    Request information about the current state of the account (balances), order (buy and sell), server state.
    This function is called when we want to get complete real (true) state, for example, after re-start or network problem.
    It sets our state by requesting information from the server.
    """
    symbol = App.config["trade"]["symbol"]

    # Get server state (ping) and trade status (e.g., trade can be suspended on some symbol)
    system_status = App.client.get_system_status()
    #{
    #    "status": 0,  # 0: normal，1：system maintenance
    #    "msg": "normal"  # normal or System maintenance.
    #}
    if not system_status or system_status.get("status") != 0:
        App.config["trade"]["state"]["server_status"] = 1
        return 1
    App.config["trade"]["state"]["server_status"] = 0

    # "orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]
    # "isSpotTradingAllowed": True

    # Ping the server

    # Check time synchronization
    #server_time = App.client.get_server_time()
    #time_diff = int(time.time() * 1000) - server_time['serverTime']
    # TODO: Log large time differences (or even trigger time synchronization if possible)

    # Get symbol info
    symbol_info = App.client.get_symbol_info(symbol)
    App.config["trade"]["symbol_info"] = symbol_info
    if not symbol_info or symbol_info.get("status") != "TRADING":
        App.config["trade"]["state"]["server_status"] = 1
        return 1
    App.config["trade"]["state"]["server_status"] = 0

    # Get account trading status (it can be blocked/suspended, e.g., too many orders)
    account_info = App.client.get_account()
    if not account_info or not account_info.get("canTrade"):
        App.config["trade"]["state"]["account_status"] = 1
        return 1
    App.config["trade"]["state"]["account_status"] = 0

    # Get current balances (available funds)
    #balance = App.client.get_asset_balance(asset=App.config["trade"]["base_asset"])
    balance = next((b for b in account_info.get("balances", []) if b.get("asset") == App.config["trade"]["base_asset"]), {})
    App.config["trade"]["state"]["base_quantity"] = Decimal(balance.get("free", "0.00000000"))

    #balance = App.client.get_asset_balance(asset=App.config["trade"]["quote_asset"])
    balance = next((b for b in account_info.get("balances", []) if b.get("asset") == App.config["trade"]["quote_asset"]), {})
    App.config["trade"]["state"]["quote_quantity"] = Decimal(balance.get("free", "0.00000000"))

    # Get current active orders
    #orders = App.client.get_all_orders(symbol=symbol, limit=10)  # All orders
    orders = App.client.get_open_orders(symbol=symbol)
    if len(orders) == 0:  # No open orders
        App.config["trade"]["state"]["sell_order"] = None  # Forget about our sell order
    elif len(orders) == 1:
        order = orders[0]
        if order["side"] == "BUY":
            App.config["trade"]["state"]["trade_state_status"] = "Buy order still open. Market buy order have to be executed immediately."
            return 1
        elif order["side"] == "SELL":
            # It is our limit sell order. We are expected to be in market (check it) and assets should be as expected.
            # Check that this order exists and update its status
            pass
    else:
        App.config["trade"]["state"]["trade_state_status"] = "More than 1 active order. There cannot be more than 1 active order."
        return 1

    App.config["trade"]["state"]["trade_state_status"] = 0

    return 0

def problems_exist():
    if App.config["trade"]["state"]["error_status"] != 0:
        return True
    if App.config["trade"]["state"]["server_status"] != 0:
        return True
    if App.config["trade"]["state"]["account_status"] != 0:
        return True
    if App.config["trade"]["state"]["trade_state_status"] != 0:
        return True
    return False

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
    if not sell_order_id:
        log.error(f"Wrong state or use: check sell order status cannot find the order id.")
        return None

    # ===
    # Retrieve order from the server
    order = App.client.get_order(symbol=symbol, orderId=sell_order_id)

    # Impose and overwrite the new order information
    if not order:
        return order
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

#
# Order creation
#

async def new_market_buy_order():
    """
    Submit a new market buy order. Wait until it is executed.
    """
    symbol = App.config["trade"]["symbol"]

    #
    # Get latest market parameters
    #
    last_kline = App.database.get_last_kline(symbol)
    last_close_price = last_kline[4]  # Close price of kline has index 4 in the list
    if not last_close_price:
        log.error(f"Cannot determine last close price in order to create a market buy order.")
        return None

    symbol_ticker = App.client.get_symbol_ticker(symbol=symbol)
    last_price = symbol_ticker.get("price", None)
    if not last_price:
        log.error(f"Cannot determine last price in order to create a market buy order.")
        return None

    if abs((last_price - last_close_price) / last_close_price) > 0.005:  # Change more than 0.5% since analysis
        log.info(f"Price changed more than 0.5% since last kline and analysis.")
        return None

    #
    # Determine BTC quantity to buy depending on how much USDT we have and what is the latest price taking into account possible price increase
    #
    quote_quantity = App.config["trade"]["state"]["quote_quantity"]
    percentage_used_for_trade = App.config["trade"]["parameters"]["percentage_used_for_trade"]
    quantity = (quote_quantity * percentage_used_for_trade) / 100.0 / last_close_price
    quantity = to_decimal(quantity)
    # Alternatively, we can pass quoteOrderQty in USDT (how much I want to spend)

    #
    # Execute order
    #
    order_spec = dict(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, timeInForce=TIME_IN_FORCE_GTC, quantity=quantity)
    # newOrderRespType (ACK, RESULT, or FULL) for market and limit order defaults to FULL

    order = execute_order(order_spec)

    # Process response
    if not order or order.get("status") != ORDER_STATUS_FILLED:
        return order

    #
    # Store/log order object in our records
    #
    App.config["trade"]["state"]["in_market"] = True

    App.config["trade"]["state"]["buy_order"] = order
    App.config["trade"]["state"]["buy_order_price"] = Decimal(order.get("price", "0.00000000"))

    App.config["trade"]["state"]["base_quantity"] += quantity  # Increase BTC
    App.config["trade"]["state"]["quote_quantity"] -= percentage_used_for_trade  # Decrease USDT

    return order

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
    quantity = to_decimal(quantity)

    #
    # Execute order
    #
    order_spec = dict(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, timeInForce=TIME_IN_FORCE_GTC, quantity=quantity)

    order = execute_order(order_spec)

    # Process response
    if not order:
        return order

    #
    # Store/log order object in our records (only after confirmation of success)
    #

    return order

async def new_limit_sell_order():
    """
    Create a new limit sell order with the amount we current have (and have just bought).
    The amount is total amount and price is determined according to our strategy (either fixed increase or increase depending on the signal).
    """
    symbol = App.config["trade"]["symbol"]
    now_ts = now_timestamp()

    #
    # We want to sell all BTC we own but the price is derived from previous (buy) order price
    #
    quantity = App.config["trade"]["state"]["base_quantity"]
    quantity = to_decimal(quantity)

    buy_order_price = App.config["trade"]["state"]["buy_order_price"]
    price = buy_order_price * App.config["trade"]["parameters"]["percentage_sell_price"]
    price = to_decimal(price)

    #
    # Execute order
    #
    order_spec = dict(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT, timeInForce=TIME_IN_FORCE_GTC, quantity=quantity, price=price)
    # Alternatively, use ORDER_TYPE_LIMIT_MAKER

    order = execute_order(order_spec)

    # Process response
    if not order:
        App.config["trade"]["state"]["sell_order"] = order
        App.config["trade"]["state"]["sell_order_time"] = 0
        return order

    #
    # Store/log order object in our records (only after confirmation of success)
    #
    App.config["trade"]["state"]["sell_order"] = order
    App.config["trade"]["state"]["sell_order_time"] = now_ts

    return order

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
        order = App.client.create_order(**order)

        if not order or not order.get("status"):
            return order

    return order

async def wait_until_filled(order):
    """Regularly check the status of this order until it is filled and return the original order updated with new responses."""
    symbol = order.get("symbol", "")
    orderId = order.get("orderId", "")

    response = order
    while not response or response.get("status") != ORDER_STATUS_FILLED:
        response = App.client.get_order(symbol=symbol, orderId=orderId)
        # TODO: sleep
        # TODO: Return error after some number of attempts

    order.update(response)

    return order

def to_decimal(value):
    """Convert to a decimal with the required precision. The value can be string, float or decimal."""
    # App.config["trade"]["symbol_info"]["baseAssetPrecision"]
    n = 8
    rr = Decimal(1) / (Decimal(10) ** n)  # Result: 0.00000001
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_DOWN)
    return ret

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
    order = App.client.cancel_order(symbol=symbol, orderId=sell_order_id)

    return order

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
    order = await cancel_sell_order()
    if not order or order.get("status") != ORDER_STATUS_CANCELED:
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
# Test procedures
#

async def check_limit_sell_order():
    """It will really create a limit sell order and then immedialtely cancel this order."""

    App.config["trade"]["state"]["base_quantity"] = 0.001  # How much
    App.config["trade"]["state"]["buy_order_price"] = Decimal("10_000.00000000")  # Some percent will be added to this price to compute limit

    # Create limit sell order (with high price)
    # Store what it returns and whether it has important information
    sell_order = await new_limit_sell_order()
    # INFO: return of the limit sell order creation:
    #{
    #    'symbol': 'BTCUSDT',
    #    'orderId': 1508649440,
    #    'orderListId': -1,
    #    'clientOrderId': 'NwWxgIItFFqRn6Tl60B7lq',
    #    'transactTime': 1584391594040,
    #    'price': '10100.00000000',
    #    'origQty': '0.00100000',
    #    'executedQty': '0.00000000',
    #    'cummulativeQuoteQty': '0.00000000',
    #    'status': 'NEW',
    #    'timeInForce': 'GTC',
    #    'type': 'LIMIT',
    #    'side': 'SELL',
    #    'fills': []
    #}

    # TODO: Retrieve available order by recovering the account state
    # INFO: return of the same order by get_open_orders:
    #{
    #    'symbol': 'BTCUSDT',
    #    'orderId': 1508649440,
    #    'orderListId': -1,
    #    'clientOrderId': 'NwWxgIItFFqRn6Tl60B7lq',
    #    'price': '10100.00000000',
    #    'origQty': '0.00100000',
    #    'executedQty': '0.00000000',
    #    'cummulativeQuoteQty': '0.00000000',
    #    'status': 'NEW',
    #    'timeInForce': 'GTC',
    #    'type': 'LIMIT',
    #    'side': 'SELL',
    #    'stopPrice': '0.00000000',
    #    'icebergQty': '0.00000000',
    #    'time': 1584391594040,
    #    'updateTime': 1584391594040,
    #    'isWorking': True,
    #    'origQuoteOrderQty': '0.00000000'
    #}

    # Kill an existing (limit sell) order. Use data from the created sell order
    # Store what it returns and whether it has important information
    cancel_order = await cancel_sell_order()
    # INFO: Return from cancel order:
    #{
    #    'symbol': 'BTCUSDT',
    #    'origClientOrderId': '8TxPBSaXR0tPCy2lOboV3h',
    #    'orderId': 1508747727,
    #    'orderListId': -1,
    #    'clientOrderId': 'dsmdBSCizc71pfeo8jTGLo',
    #    'price': '10100.00000000',
    #    'origQty': '0.00100000',
    #    'executedQty': '0.00000000',
    #    'cummulativeQuoteQty': '0.00000000',
    #    'status': 'CANCELED',
    #    'timeInForce': 'GTC',
    #    'type': 'LIMIT',
    #    'side': 'SELL'
    #}

    pass

#
# Main procedure. Initialize everything
#

def start_trade():
    #
    # Validation
    #
    symbol = App.config["trade"]["symbol"]

    log.info(f"Initializing trade server. Trade symbol {symbol}. ")

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
        App.loop.run_until_complete(sync_data_collect_task())
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
        lambda: asyncio.run_coroutine_threadsafe(sync_trade_task(), App.loop),
        trigger='cron',
        #second='*/30',
        minute='*',
        id='sync_trade_task'
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
    # Short version of start_trade (main procedure)
    App.database = Database(None)
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])
    App.loop = asyncio.get_event_loop()
    try:
        log.debug("Start in debug mode.")
        log.info("Start testing in main.")
        App.loop.run_until_complete(update_state_and_health_check())

        #App.loop.run_until_complete(check_limit_sell_order())

        App.loop.run_until_complete(sync_data_collect_task())

        App.database.analyze("BTCUSDT")

        #App.loop.run_until_complete(sync_trade_task())
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
