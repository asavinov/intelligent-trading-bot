import os
import sys
import asyncio
import requests

from service.App import *
from common.utils import *


async def notify_telegram():
    status = App.status
    signal = App.signal
    signal_side = signal.get("side")
    score = signal.get('score')

    sign = "+++>>>" if score > 0 else "---<<<"

    message = f"{sign} {score:+.2f}. PRICE: {int(signal.get('close_price'))}. STATUS: {status}"

    bot_token = App.config["telegram_bot_token"]
    chat_id = App.config["telegram_chat_id"]

    url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message

    response = requests.get(url)
    response_json = response.json()
