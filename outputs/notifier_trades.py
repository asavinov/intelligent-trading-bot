import os
import sys
from datetime import timedelta, datetime

import asyncio

import pandas as pd
import requests

from service.App import *
from common.utils import *

import logging
log = logging.getLogger('notifier')

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

transaction_file = Path("transactions.txt")


async def trader_simulation():
    """
    Very simple trade strategy where we only buy and sell using the whole available amount
    """
    symbol = App.config["symbol"]

    signal = get_signal()
    signal_side = signal.get("side")
    close_price = signal.get("close_price")
    close_time = signal.get("close_time")

    # Previous transaction: BUY (we are currently selling) or SELL (we are currently buying)
    t_status = App.transaction.get("status")
    t_price = App.transaction.get("price")
    if signal_side == "BUY" and (not t_status or t_status == "SELL"):
        profit = t_price - close_price if t_price else 0.0
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="BUY")
    elif signal_side == "SELL" and (not t_status or t_status == "BUY"):
        profit = close_price - t_price if t_price else 0.0
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="SELL")
    else:
        return None

    # Save this transaction
    App.transaction = t_dict
    with open(transaction_file, 'a+') as f:
        f.write(",".join([f"{v:.2f}" if isinstance(v, float) else str(v) for v in t_dict.values()]) + "\n")

    log.info(f"Trade simulator transaction: {t_dict}")

    return t_dict


async def send_transaction_message(transaction):

    profit, profit_percent, profit_descr, profit_percent_descr = await generate_transaction_stats()

    if transaction.get("status") == "SELL":
        message = "⚡💰 *SOLD: "
    elif transaction.get("status") == "BUY":
        message = "⚡💰 *BOUGHT: "
    else:
        log.error(f"ERROR: Should not happen")

    message += f" Profit: {profit_percent:.2f}% {profit:.2f}₮*"

    bot_token = App.config["telegram_bot_token"]
    chat_id = App.config["telegram_chat_id"]
    try:
        url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message
        response = requests.get(url)
        response_json = response.json()
        if not response_json.get('ok'):
            log.error(f"Error sending notification.")
    except Exception as e:
        log.error(f"Error sending notification: {e}")

    #
    # Send stats about previous transactions (including this one)
    #
    if transaction.get("status") == "SELL":
        message = "↗ *LONG transactions stats (4 weeks)*\n"
    elif transaction.get("status") == "BUY":
        message = "↘ *SHORT transactions stats (4 weeks)*\n"
    else:
        log.error(f"ERROR: Should not happen")

    message += f"🔸sum={profit_percent_descr['count'] * profit_percent_descr['mean']:.2f}% 🔸count={int(profit_percent_descr['count'])}\n"
    message += f"🔸mean={profit_percent_descr['mean']:.2f}% 🔸std={profit_percent_descr['std']:.2f}%\n"
    message += f"🔸min={profit_percent_descr['min']:.2f}% 🔸median={profit_percent_descr['50%']:.2f}% 🔸max={profit_percent_descr['max']:.2f}%\n"

    try:
        url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message
        response = requests.get(url)
        response_json = response.json()
        if not response_json.get('ok'):
            log.error(f"Error sending notification.")
    except Exception as e:
        log.error(f"Error sending notification: {e}")


async def generate_transaction_stats():
    """Here we assume that the latest transaction is saved in the file and this function computes various properties."""

    df = pd.read_csv(transaction_file, parse_dates=[0], header=None, names=["timestamp", "close", "profit", "status"], date_format="ISO8601")

    mask = (df['timestamp'] >= (datetime.now() - timedelta(weeks=4)))
    df = df[max(mask.idxmax()-1, 0):]  # We add one previous row to use the previous close

    df["prev_close"] = df["close"].shift()
    df["profit_percent"] = df.apply(lambda x: 100.0*x["profit"]/x["prev_close"], axis=1)

    df = df.iloc[1:]  # Remove the first row which was added to compute relative profit

    long_df = df[df["status"] == "SELL"]
    short_df = df[df["status"] == "BUY"]

    #
    # Determine properties of the latest transaction
    #

    # Sample output:
    # BTC, LONG or SHORT
    # sell price 24,000 (now), buy price (datetime) 23,000
    # profit abs: 1,000.00,
    # profit rel: 3.21%

    last_transaction = df.iloc[-1]
    transaction_dt = last_transaction["timestamp"]
    transaction_type = last_transaction["status"]
    profit = last_transaction["profit"]
    profit_percent = last_transaction["profit_percent"]

    #
    # Properties of last period of trade
    #

    if transaction_type == "SELL":
        df2 = long_df
    elif transaction_type == "BUY":
        df2 = short_df

    # Sample output for abs profit
    # sum 1,200.00, mean 400.00, median 450.00, std 250.00, min -300.0, max 1200.00

    profit_sum = df2["profit"].sum()
    profit_descr = df2["profit"].describe()  # count, mean, std, min, 50% max

    profit_percent_sum = df2["profit_percent"].sum()
    profit_percent_descr = df2["profit_percent"].describe()  # count, mean, std, min, 50% max

    return profit, profit_percent, profit_descr, profit_percent_descr


def get_signal():
    """From the last row, produce and return an object with parameters important for trading."""
    freq = App.config["freq"]

    df = App.df
    row = df.iloc[-1]  # Last row stores the latest values we need

    interval_length = pd.Timedelta(freq).to_pytimedelta()
    close_time = row.name + interval_length  # Add interval length because timestamp is start of the interval
    close_price = row["close"]

    model = App.config["trade_model"]
    buy_signal_column = model.get("buy_signal_column")
    sell_signal_column = model.get("sell_signal_column")
    buy_signal = row[buy_signal_column]
    sell_signal = row[sell_signal_column]

    if buy_signal and sell_signal:  # Both signals are true - should not happen
        signal_side = "BOTH"
    elif buy_signal:
        signal_side = "BUY"
    elif sell_signal:
        signal_side = "SELL"
    else:
        signal_side = ""

    signal = {"side": signal_side, "close_price": close_price, "close_time": close_time}

    return signal


if __name__ == '__main__':
    pass
