import os
import sys
import asyncio
import requests

from service.App import *
from common.utils import *


async def notify_telegram():
    status = App.status
    signal = App.signal
    notification_threshold = App.config["signaler"]["notification_threshold"]
    symbol = App.config["symbol"]
    base_asset =  App.config["base_asset"]
    quote_asset =  App.config["quote_asset"]
    close_price = signal.get('close_price')

    signal_side = signal.get("side")
    score = signal.get('score')

    # How many steps of the score
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

    # Crypto Currency Symbols: https://github.com/yonilevy/crypto-currency-symbols
    if base_asset == "BTC":
        symbol_sign = "₿"
    elif base_asset == "ETH":
        symbol_sign = "Ξ"
    else:
        symbol_sign = base_asset

    message = f"{symbol_sign} {int(close_price):,} {sign} Score: {score:+.2f}"
    message = message.replace("+", "%2B")  # For Telegram to display plus sign

    #if signal_side in ["BUY", "SELL"]:
    #    message = f"*{message}. SIGNAL: {signal_side}*"

    bot_token = App.config["telegram_bot_token"]
    chat_id = App.config["telegram_chat_id"]

    url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message

    try:
        response = requests.get(url)
        #response_json = response.json()
    except Exception as e:
        print(f"Error sending notification: {e}")


if __name__ == '__main__':
    pass
