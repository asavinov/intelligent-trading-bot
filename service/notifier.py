import os
import sys
import asyncio
import requests

from service.App import *
from common.utils import *

import logging
log = logging.getLogger('notifier')


async def notify_telegram():
    await simulate_trade()

    symbol = App.config["symbol"]
    base_asset =  App.config["base_asset"]
    quote_asset =  App.config["quote_asset"]

    status = App.status
    signal = App.signal
    signal_side = signal.get("side")
    close_price = signal.get('close_price')
    buy_score = signal.get('buy_score')
    sell_score = signal.get('sell_score')
    close_time = signal.get('close_time')

    model = App.config["signal_model"]
    buy_notify_threshold = model["buy_notify_threshold"]
    sell_notify_threshold = model["sell_notify_threshold"]
    buy_signal_threshold = model["buy_signal_threshold"]
    sell_signal_threshold = model["sell_signal_threshold"]
    trade_icon_step = model.get("trade_icon_step", 0)
    notify_frequency_minutes = model.get("notify_frequency_minutes", 1)

    # Crypto Currency Symbols: https://github.com/yonilevy/crypto-currency-symbols
    if base_asset == "BTC":
        symbol_sign = "₿"
    elif base_asset == "ETH":
        symbol_sign = "Ξ"
    else:
        symbol_sign = base_asset

    # Notification logic:
    # 1. Trade signal in the case it is suggested to really buy or sell: BUY or SELL and one corresponding score
    # 2. Notification signal simply to provide information (separate criteria): both scores
    # Icons: down: 📉, ⬇ ⬇️🔴 (red), up:  📈, ⬆,  ⬆️ ↗️ 🟢 (green)
    message = ""
    if signal_side == "BUY":
        score_steps = (np.abs(buy_score - buy_signal_threshold) // trade_icon_step) if trade_icon_step else 0
        message = "🟢"*int(score_steps+1) + f" *BUY: {symbol_sign} {int(close_price):,} Buy score: {buy_score:+.2f}*"
    elif signal_side == "SELL":
        score_steps = (np.abs(sell_score - sell_signal_threshold) // trade_icon_step) if trade_icon_step else 0
        message = "🔴"*int(score_steps+1) + f" *SELL: {symbol_sign} {int(close_price):,} Sell score: {-sell_score:+.2f}*"
    elif (close_time.minute % notify_frequency_minutes) == 0:  # Info message with custom frequency
        if buy_score > sell_score:
            message = f"{symbol_sign} {int(close_price):,} 📈{buy_score:+.2f}"
        else:
            message = f"{symbol_sign} {int(close_price):,} 📉{-sell_score:+.2f}"
    message = message.replace("+", "%2B")  # For Telegram to display plus sign

    if not message:
        return
    if buy_score < buy_notify_threshold and sell_score < sell_notify_threshold:
        return  # Do not send notifications with low notification threshold (also no buy/sell notifications)

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


async def simulate_trade():
    symbol = App.config["symbol"]
    base_asset = App.config["base_asset"]
    quote_asset = App.config["quote_asset"]

    status = App.status
    signal = App.signal
    signal_side = signal.get("side")
    close_price = signal.get('close_price')
    buy_score = signal.get('buy_score')
    sell_score = signal.get('sell_score')
    close_time = signal.get('close_time')

    t_status = App.transaction.get("status")
    t_price = App.transaction.get("price")
    if signal_side == "BUY" and (not t_status or t_status == "BUYING"):
        profit = t_price - close_price if t_price else 0.0
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="SELLING")
    elif signal_side == "SELL" and (not t_status or t_status == "SELLING"):
        profit = close_price - t_price if t_price else 0.0
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="BUYING")
    else:
        return

    # Save this transaction
    App.transaction = t_dict
    transaction_file = Path("transactions.txt")
    with open(transaction_file, 'a+') as f:
        f.write(",".join([str(v) for v in t_dict.values()]) + "\n")


if __name__ == '__main__':
    pass
