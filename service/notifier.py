import os
import sys
import asyncio
import requests

from service.App import *
from common.utils import *


async def notify_telegram():
    symbol = App.config["symbol"]
    base_asset =  App.config["base_asset"]
    quote_asset =  App.config["quote_asset"]

    model = App.config["signal_model"]
    buy_notify_threshold = model["buy_notify_threshold"]
    sell_notify_threshold = model["sell_notify_threshold"]

    status = App.status
    signal = App.signal
    signal_side = signal.get("side")
    close_price = signal.get('close_price')
    buy_score = signal.get('buy_score')
    sell_score = signal.get('sell_score')

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
    if signal_side == "BUY":
        message = f"📈 *BUY*: {symbol_sign} {int(close_price):,} Buy score: {buy_score:+.2f}"
    elif signal_side == "SELL":
        message = f"📉 *SELL*: {symbol_sign} {int(close_price):,} Sell score: {sell_score:+.2f}"
    elif buy_score >= buy_notify_threshold or sell_score >= sell_notify_threshold:
        message = f"SCORE: {symbol_sign} {int(close_price):,} 📈 {buy_score:+.3f}, 📉 {sell_score:+.3f}"
    else:
        message = ""
    message = message.replace("+", "%2B")  # For Telegram to display plus sign

    # Number of icons. How many steps of the score
    """
    score_step_length = 0.05
    score_steps = np.abs(score) // score_step_length

    if score_steps < notification_threshold:
        return

    if score > 0:
        sign = "📈" * int(score_steps - notification_threshold + 1)  # 📈 >
    elif score < 0:
        sign = "📉" * int(score_steps - notification_threshold + 1)  # 📉 <
    else:
        sign = ""
    """

    #if signal_side in ["BUY", "SELL"]:
    #    message = f"*{message}. SIGNAL: {signal_side}*"

    bot_token = App.config["telegram_bot_token"]
    chat_id = App.config["telegram_chat_id"]

    url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message

    try:
        response = requests.get(url)
        response_json = response.json()
        if not response_json.get('ok'):
            print(f"Error sending notification.")
    except Exception as e:
        print(f"Error sending notification: {e}")


if __name__ == '__main__':
    pass
