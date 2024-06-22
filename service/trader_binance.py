import os
import sys
import argparse
import math, time
from datetime import datetime
from decimal import *

import pandas as pd
import asyncio

from binance.client import Client
from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from binance.enums import *

from service.App import *
from common.utils import *
from service.analyzer import *
from service.notifier_trades import get_signal

import logging


log = logging.getLogger('trader')
logging.basicConfig(
    filename="trader.log",  # parameter in App
    level=logging.DEBUG,
    #format = "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    format = "%(asctime)s %(levelname)s %(message)s",
    #datefmt = '%Y-%m-%d %H:%M:%S',
)


async def main_trader_task():
    """
    It is a highest level task which is added to the event loop and executed normally every 1 minute and then it calls other tasks.
    """
    symbol = App.config["symbol"]
    freq = App.config["freq"]
    startTime, endTime = pandas_get_interval(freq)
    now_ts = now_timestamp()

    trade_model = App.config.get("trade_model", {})

    signal = get_signal()
    signal_side = signal.get("side")
    close_price = signal.get("close_price")
    close_time = signal.get("close_time")

    log.info(f"===> Start trade task. Timestamp {now_ts}. Interval [{startTime},{endTime}].")

    #
    # Sync trade status, check running orders (orders, account etc.)
    #
    status = App.status

    if status == "BUYING" or status == "SELLING":
        # We expect that an order was created before and now we need to check if it still exists or was executed
        # -----
        order_status = await update_order_status()

        order = App.order
        # If order status executed then change the status
        # Status codes: NEW PARTIALLY_FILLED FILLED CANCELED PENDING_CANCEL(currently unused) REJECTED EXPIRED

        if not order or not order_status:
            # No sell order exists or some problem
            # TODO: Recover, reset, init/sync state (cannot trade because wrong happened with the order or connection or whatever)
            #   check connection (like ping), then server status, then our own account status, then funds, orders etc.
            # -----
            await update_trade_status()
            log.error(f"Bad order or order status {order}. Full reset/init needed.")
            return
        if order_status == ORDER_STATUS_FILLED:
            log.info(f"Limit order filled. {order}")
            if status == "BUYING":
                print(f"===> BOUGHT: {order}")
                App.status = "BOUGHT"
            elif status == "SELLING":
                print(f"<=== SOLD: {order}")
                App.status = "SOLD"
            log.info(f'New trade mode: {App.status}')
        elif order_status == ORDER_STATUS_REJECTED or order_status == ORDER_STATUS_EXPIRED or order_status == ORDER_STATUS_CANCELED:
            log.error(f"Failed to fill order with order status {order_status}")
            if status == "BUYING":
                App.status = "SOLD"
            elif status == "SELLING":
                App.status = "BOUGHT"
            log.info(f'New trade mode: {App.status}')
        elif order_status == ORDER_STATUS_PENDING_CANCEL:
            return  # Currently do nothing. Check next time.
        elif order_status == ORDER_STATUS_PARTIALLY_FILLED:
            pass  # Currently do nothing. Check next time.
        elif order_status == ORDER_STATUS_NEW:
            pass  # Wait further for execution
        else:
            pass  # Order still exists and is active
    elif status == "BOUGHT" or status == "SOLD":
        pass  # Do nothing
    else:
        log.error(f"Wrong status value {status}.")

    #
    # Prepare. Kill or update existing orders (if necessary)
    #
    status = App.status

    # If not sold for 1 minute, then kill and then a new order will be created below if there is signal
    # Essentially, this will mean price adjustment (if a new order of the same direction will be created)
    # In future, we might kill only after some timeout
    if status == "BUYING" or status == "SELLING":  # Still not sold for 1 minute
        # -----
        order_status = await cancel_order()
        if not order_status:
            # Cancel exception (the order still exists) or the order was filled and does not exist
            await update_trade_status()
            return
        await asyncio.sleep(1)  # Wait for a second till the balance is updated
        if status == "BUYING":
            App.status = "SOLD"
        elif status == "SELLING":
            App.status = "BOUGHT"

    #
    # Trade by creating orders
    #
    status = App.status

    if signal_side == "BUY":
        print(f"===> BUY SIGNAL {signal}: ")
    elif signal_side == "SELL":
        print(f"<=== SELL SIGNAL: {signal}")
    else:
        print(f"PRICE: {close_price:.2f}")

    # Update account balance etc. what is needed for trade
    # -----
    await update_account_balance()

    if status == "SOLD" and signal_side == "BUY":
        # -----
        await new_limit_order(side=SIDE_BUY)

        if trade_model.get("no_trades_only_data_processing"):
            print("SKIP TRADING due to 'no_trades_only_data_processing' parameter True")
            # Never change status if orders not executed
        else:
            App.status = "BUYING"
    elif status == "BOUGHT" and signal_side == "SELL":
        # -----
        await new_limit_order(side=SIDE_SELL)

        if trade_model.get("no_trades_only_data_processing"):
            print("SKIP TRADING due to 'no_trades_only_data_processing' parameter True")
            # Never change status if orders not executed
        else:
            App.status = "SELLING"

    log.info(f"<=== End trade task.")

    return


#
# Order and asset status
#

async def update_trade_status():
    """Read the account state and set the local state parameters."""
    # GET /api/v3/openOrders - get current open orders
    # GET /api/v3/allOrders - get all orders: active, canceled, or filled

    symbol = App.config["symbol"]

    # -----
    try:
        open_orders = App.client.get_open_orders(symbol=symbol)  # By "open" orders they probably mean "NEW" or "PARTIALLY_FILLED"
        # orders = App.client.get_all_orders(symbol=symbol, limit=10)
    except Exception as e:
        log.error(f"Binance exception in 'get_open_orders' {e}")
        return

    if not open_orders:
        # -----
        await update_account_balance()

        last_kline = App.analyzer.get_last_kline(symbol)
        last_close_price = to_decimal(last_kline[4])  # Close price of kline has index 4 in the list

        base_quantity = App.base_quantity  # BTC
        btc_assets_in_usd = base_quantity * last_close_price  # Cost of available BTC in USD

        usd_assets = App.quote_quantity  # USD

        if usd_assets >= btc_assets_in_usd:
            App.status = "SOLD"
        else:
            App.status = "BOUGHT"

    elif len(open_orders) == 1:
        order = open_orders[0]
        if order.get("side") == SIDE_SELL:
            App.status = "SELLING"
        elif order.get("side") == SIDE_BUY:
            App.status = "BUYING"
        else:
            log.error(f"Neither SELL nor BUY side of the order {order}.")
            return None

    else:  # Many orders
        log.error(f"Wrong state. More than one open order. Fix manually.")
        return None


async def update_order_status():
    """
    Update information about the current order and return its execution status.

    ASSUMPTIONS and notes:
    - Status codes: NEW PARTIALLY_FILLED FILLED CANCELED PENDING_CANCEL(currently unused) REJECTED EXPIRED
    - only one or no orders can be active currently, but in future there can be many orders
    - if no order id(s) is provided then retrieve all existing orders
    """
    symbol = App.config["symbol"]

    # Get currently active order and id (if any)
    order = App.order
    order_id = order.get("orderId", 0) if order else 0
    if not order_id:
        log.error(f"Wrong state or use: check order status cannot find the order id.")
        return None

    # -----
    # Retrieve order from the server
    try:
        new_order = App.client.get_order(symbol=symbol, orderId=order_id)
    except Exception as e:
        log.error(f"Binance exception in 'get_order' {e}")
        return

    # Impose and overwrite the new order information
    if new_order:
        order.update(new_order)
    else:
        return None

    # Now order["status"] contains the latest status of the order
    return order["status"]


async def update_account_balance():
    """Get available assets (as decimal)."""

    try:
        balance = App.client.get_asset_balance(asset=App.config["base_asset"])
    except Exception as e:
        log.error(f"Binance exception in 'get_asset_balance' {e}")
        return

    App.base_quantity = Decimal(balance.get("free", "0.00000000"))  # BTC

    try:
        balance = App.client.get_asset_balance(asset=App.config["quote_asset"])
    except Exception as e:
        log.error(f"Binance exception in 'get_asset_balance' {e}")
        return

    App.quote_quantity = Decimal(balance.get("free", "0.00000000"))  # USD

    pass


#
# Cancel and liquidation orders
#

async def cancel_order():
    """
    Kill existing sell order. It is a blocking request, that is, it waits for the end of the operation.
    Info: DELETE /api/v3/order - cancel order
    """
    symbol = App.config["symbol"]

    # Get currently active order and id (if any)
    order = App.order
    order_id = order.get("orderId", 0) if order else 0
    if order_id == 0:
        # TODO: Maybe retrieve all existing (sell, limit) orders
        return None

    # -----
    try:
        log.info(f"Cancelling order id {order_id}")
        new_order = App.client.cancel_order(symbol=symbol, orderId=order_id)
    except Exception as e:
        log.error(f"Binance exception in 'cancel_order' {e}")
        return None

    # TODO: There is small probability that the order will be filled just before we want to kill it
    #   We need to somehow catch and process this case
    #   If we get an error (say, order does not exist and cannot be killed), then after error returned, we could do trade state reset

    # Impose and overwrite the new order information
    if new_order:
        order.update(new_order)
    else:
        return None

    # Now order["status"] contains the latest status of the order
    return order["status"]


#
# Order creation
#

async def new_limit_order(side):
    """
    Create a new limit sell order with the amount we current have.
    The amount is total amount and price is determined according to our strategy (either fixed increase or increase depending on the signal).
    """
    symbol = App.config["symbol"]
    now_ts = now_timestamp()

    trade_model = App.config.get("trade_model", {})

    #
    # Find limit price (from signal, last kline and adjustment parameters)
    #
    last_kline = App.analyzer.get_last_kline(symbol)
    last_close_price = to_decimal(last_kline[4])  # Close price of kline has index 4 in the list
    if not last_close_price:
        log.error(f"Cannot determine last close price in order to create a market buy order.")
        return None

    price_adjustment = trade_model.get("limit_price_adjustment")
    if side == SIDE_BUY:
        price = last_close_price * Decimal(1.0 - price_adjustment)  # Adjust price slightly lower
    elif side == SIDE_SELL:
        price = last_close_price * Decimal(1.0 + price_adjustment)  # Adjust price slightly higher

    price_str = round_str(price, 2)
    price = Decimal(price_str)  # We will use the adjusted price for computing quantity

    #
    # Find quantity
    #
    if side == SIDE_BUY:
        # Find how much quantity we can buy for all available USD using the computed price
        quantity = App.quote_quantity  # USD
        percentage_used_for_trade = trade_model.get("percentage_used_for_trade")
        quantity = (quantity * percentage_used_for_trade) / Decimal(100.0)  # Available for trade
        quantity = quantity / price  # BTC to buy
        # Alternatively, we can pass quoteOrderQty in USDT (how much I want to spend)
    elif side == SIDE_SELL:
        # All available BTCs
        quantity = App.base_quantity  # BTC

    quantity_str = round_down_str(quantity, 6)

    #
    # Execute order
    #
    order_spec = dict(
        symbol=symbol,
        side=side,
        type=ORDER_TYPE_LIMIT,  # Alternatively, ORDER_TYPE_LIMIT_MAKER
        timeInForce=TIME_IN_FORCE_GTC,
        quantity=quantity_str,
        price=price_str,
    )

    if trade_model.get("no_trades_only_data_processing"):
        print(f"NOT executed order spec: {order_spec}")
    else:
        order = execute_order(order_spec)

    #
    # Store/log order object in our records (only after confirmation of success)
    #
    App.order = order
    App.order_time = now_ts

    return order


def execute_order(order: dict):
    """Validate and submit order"""

    trade_model = App.config.get("trade_model", {})

    # TODO: Check validity, e.g., against filters (min, max) and our own limits

    if trade_model.get("test_order_before_submit"):
        try:
            log.info(f"Submitting test order: {order}")
            test_response = App.client.create_test_order(**order)  # Returns {} if ok. Does not check available balances - only trade rules
        except Exception as e:
            log.error(f"Binance exception in 'create_test_order' {e}")
            # TODO: Reset/resync whole account
            return

    if trade_model.get("simulate_order_execution"):
        # TODO: Simply store order so that later we can check conditions of its execution
        print(order)
        pass
    else:
        # -----
        # Submit order
        try:
            log.info(f"Submitting order: {order}")
            order = App.client.create_order(**order)
        except Exception as e:
            log.error(f"Binance exception in 'create_order' {e}")
            return

        if not order or not order.get("status"):
            return None

    return order
