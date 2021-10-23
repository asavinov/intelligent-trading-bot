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

    signal_side = signal.get("side")
    score = signal.get('score')

    if -notification_threshold < score < notification_threshold:
        return

    if score < -0.5:
        sign = "<<<<<"
    elif score < -0.4:
        sign = "<<<<"
    elif score < -0.3:
        sign = "<<<"
    elif score < -0.2:
        sign = "<<"
    elif score < -0.1:
        sign = "<"
    elif score < +0.1:
        sign = ""
    elif score < +0.2:
        sign = ">"
    elif score < +0.3:
        sign = ">>"
    elif score < +0.4:
        sign = ">>>"
    elif score < +0.5:
        sign = ">>>>"
    else:
        sign = ">>>>>"

    message = f"{sign} {score:+.2f}. {symbol} {int(signal.get('close_price')):,}"

    if signal.get('side') in ["BUY", "SELL"]:
        message = f"*{message}. SIGNAL: {signal.get('side')}*"

    bot_token = App.config["telegram_bot_token"]
    chat_id = App.config["telegram_chat_id"]

    url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message

    response = requests.get(url)
    response_json = response.json()
