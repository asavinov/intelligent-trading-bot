from decimal import *
from enum import Enum

import pandas as pd
import asyncio

import MetaTrader5 as mt5

from service.App import *
from common.utils import *
from common.model_store import *
from outputs.notifier_trades import get_signal
from service.mt5 import connect_mt5

import logging

log = logging.getLogger('trader')


async def trader_mt5(df: pd.DataFrame, model: dict, config: dict, model_store: ModelStore):
    """It is a highest level task which is added to the event loop and executed normally every frequency specified(e.g 1h) and then it calls other tasks.

    This function implements the main trading logic for the MetaTrader 5 platform.
    It handles order placement, order status updates, account balance updates, and signal processing.

    Parameters:
    ----------
        df (pd.DataFrame): The DataFrame containing the trading data.
        model (dict): The model configuration dictionary.
        config (dict): The general configuration dictionary.
    """
    # Connect to trading account (same as before)
    mt5_account_id = App.config.get("mt5_account_id")
    mt5_password = App.config.get("mt5_password")
    mt5_server = App.config.get("mt5_server")
    if mt5_account_id and mt5_password and mt5_server:
        authorized = connect_mt5(int(mt5_account_id), password=str(mt5_password), server=str(mt5_server))
        if not authorized:
            log.error(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            return

    symbol = config["symbol"]
    freq = config["freq"]
    startTime, endTime = pandas_get_interval(freq)
    now_ts = now_timestamp()

    buy_signal_column = model.get("buy_signal_column")
    sell_signal_column = model.get("sell_signal_column")

    signal = get_signal(buy_signal_column, sell_signal_column)
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
        if order_status == MT5OrderStatus.FILLED.value:
            log.info(f"Limit order filled. {order}")
            if status == "BUYING":
                print(f"===> BOUGHT: {order}")
                App.status = "BOUGHT"
            elif status == "SELLING":
                print(f"<=== SOLD: {order}")  # TODO: Check if it is correct
                App.status = "SOLD"
            log.info(f'New trade mode: {App.status}')
        elif order_status in [MT5OrderStatus.REJECTED.value, MT5OrderStatus.EXPIRED.value, MT5OrderStatus.CANCELED.value]:
            log.error(f"Failed to fill order with order status {order_status}")
            if status == "BUYING":
                App.status = "SOLD"
            elif status == "SELLING":
                App.status = "BOUGHT"
            log.info(f'New trade mode: {App.status}')
        elif order_status == MT5OrderStatus.PENDING_CANCEL.value:
            return  # Currently do nothing. Check next time.   
        elif order_status == mt5.ORDER_STATE_PARTIAL:
            pass  # Currently do nothing. Check next time.
        elif order_status == mt5.ORDER_STATE_PLACED: # try ORDER_STATE_STARTED
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
        await new_limit_order(side=mt5.ORDER_TYPE_BUY_LIMIT)

        if model.get("no_trades_only_data_processing"):
            print("SKIP TRADING due to 'no_trades_only_data_processing' parameter True")
            # Never change status if orders not executed
        else:
            App.status = "BUYING"
    elif status == "BOUGHT" and signal_side == "SELL": 
        # -----
        await new_limit_order(symbol, side=mt5.ORDER_TYPE_SELL)

        if model.get("no_trades_only_data_processing"):
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

    symbol = App.config["symbol"]

    # -----
    # Get all open orders
    open_orders = mt5.orders_get(symbol=symbol)

    if open_orders is None:
        log.error(f"MT5 error in 'orders_get' {mt5.last_error()}")
        return

    if not open_orders:
        # -----
        await update_account_balance()

        last_kline = App.analyzer.get_last_kline(symbol)
        last_close_price = to_decimal(last_kline[4])  # Close price of kline has index 4 in the list (0-based)

        base_quantity = App.account_info.base_quantity  # BTC
        btc_assets_in_usd = base_quantity * last_close_price  # Cost of available BTC in USD

        usd_assets = App.account_info.quote_quantity  # USD

        if usd_assets >= btc_assets_in_usd:
            App.status = "SOLD"
        else:
            App.status = "BOUGHT"

    elif len(open_orders) == 1:
        order = open_orders[0]  # TODO: Check if it is correct
        if order.type == mt5.ORDER_TYPE_SELL_LIMIT:
            App.status = "SELLING"
        elif order.type == mt5.ORDER_TYPE_BUY_LIMIT:
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

    # Get currently active order and id (if any)
    order = App.order
    order_id = order.get("orderId", 0) if order else 0
    if not order_id:
        log.error(f"Wrong state or use: check order status cannot find the order id.")
        return None

    # -----
    # Retrieve order from the server
    order_info = mt5.orders_get(ticket=order_id)
    if order_info is None or len(order_info) == 0:
        log.error(f"MT5 error in 'orders_get' {mt5.last_error()}")
        return None
    new_order = order_info[0]

    # Impose and overwrite the new order information
    if new_order: 
        order.update(new_order._asdict())
    else:  # TODO: Check if it is correct
        return None 

    # Now order["status"] contains the latest status of the order
    return order["status"]


async def update_account_balance():
    """Get available assets (as decimal)."""

    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        log.error(f"MT5 error in 'account_info' {mt5.last_error()}")
        return

    # Get free margin
    margin_free = account_info.margin_free
    App.account_info.quote_quantity = Decimal(margin_free)  # USD

    # Get positions
    positions = mt5.positions_get()
    if positions is not None and len(positions) > 0:
        position = positions[0]
        App.account_info.base_quantity = Decimal(position.volume)  # BTC

    pass


#
# Cancel and liquidation orders
#

async def cancel_order():
    """
    Kill existing sell order. It is a blocking request, that is, it waits for the end of the operation.
    Info: DELETE /api/v3/order - cancel order
    """
    
    # Get currently active order and id (if any)
    order = App.order
    order_id = order.get("orderId", 0) if order else 0
    if order_id == 0:
        # TODO: Maybe retrieve all existing (sell, limit) orders
        return None

    # -----
    log.info(f"Cancelling order id {order_id}")
    request = {
        "action": mt5.TRADE_ACTION_REMOVE, 
        "order": order_id,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"MT5 error in 'order_send' {mt5.last_error()}")
        return None
    new_order = result
    # TODO: There is small probability that the order will be filled just before we want to kill it
    #   We need to somehow catch and process this case
    #   If we get an error (say, order does not exist and cannot be killed), then after error returned, we could do trade state reset

    # Impose and overwrite the new order information
    if new_order:
        order.update(new_order)
    else:
        return None  # TODO: Check if it is correct

    # Now order["state"] contains the latest status of the order
    return order["state"]


#
# Order creation (MT5)
#

async def new_limit_order(symbol, side):
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
    if not last_close_price or last_close_price == 0:
        log.error(f"Cannot determine last close price in order to create a market buy order.")
        return None

    price_adjustment = trade_model.get("limit_price_adjustment")
    if side == mt5.ORDER_TYPE_BUY_LIMIT: 
        price = last_close_price * Decimal(1.0 - price_adjustment)  # Adjust price slightly lower
    elif side == mt5.ORDER_TYPE_SELL_LIMIT:
        price = last_close_price * Decimal(1.0 + price_adjustment)  # Adjust price slightly higher 

    price_str = round_str(price, 2)
    price = Decimal(price_str)  # We will use the adjusted price for computing quantity

    #
    # Find quantity
    #
    if side == mt5.ORDER_TYPE_BUY_LIMIT: 
        # Find how much quantity we can buy for all available USD using the computed price
        quantity = App.account_info.quote_quantity  # USD
        percentage_used_for_trade = trade_model.get("percentage_used_for_trade")
        quantity = (quantity * percentage_used_for_trade) / Decimal(100.0)  # Available for trade
        quantity = quantity / price  # BTC to buy
        # Alternatively, we can pass quoteOrderQty in USDT (how much I want to spend)
    elif side == mt5.ORDER_TYPE_SELL_LIMIT:
        # All available BTCs
        quantity = App.account_info.base_quantity  # BTC

    quantity_str = round_down_str(quantity, 6)

    #
    # Execute order
    #
    order_spec = dict( 
        symbol=symbol,
        type=side,
        volume=float(quantity_str),
        price=float(price_str),
        type_time=mt5.ORDER_TIME_GTC, 
        type_filling=mt5.ORDER_FILLING_IOC,
        deviation=20,
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
        log.info(f"Submitting test order: {order}")
        # TODO: Check if it is possible to test order in MT5
        pass

    if trade_model.get("simulate_order_execution"):
        # TODO: Simply store order so that later we can check conditions of its execution
        print(order)
        pass
    else:
        # Submit order
        log.info(f"Submitting order: {order}")
        request = { 
            "action": mt5.TRADE_ACTION_DEAL, 
            **order
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"MT5 error in 'order_send' {mt5.last_error()}")
            return None
        return result._asdict()


class MT5OrderStatus(Enum):
    NEW = mt5.ORDER_STATE_PLACED
    PARTIALLY_FILLED = mt5.ORDER_STATE_PARTIAL
    FILLED = mt5.ORDER_STATE_FILLED
    CANCELED = mt5.ORDER_STATE_CANCELED
    PENDING_CANCEL = mt5.ORDER_STATE_REQUEST_CANCEL
    REJECTED = mt5.ORDER_STATE_REJECTED
    EXPIRED = mt5.ORDER_STATE_EXPIRED
