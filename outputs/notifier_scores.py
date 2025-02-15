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


async def send_score_notification():
    symbol = App.config["symbol"]
    freq = App.config["freq"]
    model = App.config["score_notification_model"]

    score_column_names = model.get("score_column_names")
    if not score_column_names:
        log.error(f"Empty list of score columns in score notifier. At least one column name with a score has to be provided in config. Ignore")
        return

    df = App.df
    row = df.iloc[-1]  # Last row stores the latest values we need

    interval_length = pd.Timedelta(freq).to_pytimedelta()
    close_time = row.name + interval_length  # Add interval length because timestamp is start of the interval
    close_price = row["close"]
    trade_scores = [row[col] for col in score_column_names]
    trade_score_primary = trade_scores[0]
    trade_score_secondary = trade_scores[1] if len(trade_scores) > 1 else None

    # Determine the band for the current score
    if trade_score_primary > 0:
        bands = model.get("positive_bands", [])
        band_no, band = next(((i, x) for i, x in enumerate(bands) if trade_score_primary <= x.get("edge")), (len(bands), None))
    else:
        bands = model.get("negative_bands", [])
        band_no, band = next(((i, x) for i, x in enumerate(bands) if trade_score_primary >= x.get("edge")), (len(bands), None))

    if not band:
        log.error(f"Notification band for the score {trade_score_primary} not found. Check the list of bands in config. Ignore")
        return

    #
    # To message or not to message depending on score value and time
    #

    # Determine if the band was changed since the last time
    prev_band_no = model.get("prev_band_no")
    band_up = prev_band_no is not None and prev_band_no < band_no
    band_dn = prev_band_no is not None and prev_band_no > band_no
    model["prev_band_no"] = band_no  # Store for the next time in the config section

    if band.get("frequency"):
        new_to_time_interval = close_time.minute % band.get("frequency") == 0
    else:
        new_to_time_interval = False

    # Send only if one of these conditions is true  or entered new time interval (current time)
    notification_is_needed = (
        (model.get("notify_band_up") and band_up) or  # entered a higher band (absolute score increased)
        (model.get("notify_band_dn") and band_dn) or  # returned to a lower band (absolute score decreased)
        new_to_time_interval  # new time interval is started like 10 minutes
    )

    if not notification_is_needed:
        return  # Nothing important happened: within the same band and same time interval

    #
    # Build a message with parameters from the current band
    #

    # Crypto Currency Symbols: https://github.com/yonilevy/crypto-currency-symbols
    if symbol == "BTCUSDT":
        symbol_char = "₿"
    elif symbol == "ETHUSDT":
        symbol_char = "Ξ"
    else:
        symbol_char = symbol

    if band_up:
        band_change_char = "↑"
    elif band_dn:
        band_change_char = "↓"
    else:
        band_change_char = ""

    primary_score_str = f"{trade_score_primary:+.2f} {band_change_char} "
    secondary_score_str = f"{trade_score_secondary:+.2f}" if trade_score_secondary is not None else ''

    message = f"{band.get('sign', '')} {symbol_char} {int(close_price):,} Score: {primary_score_str} {secondary_score_str} {band.get('text', '')}"
    if band.get("bold"):
        message = "*" + message + "*"

    message = message.replace("+", "%2B")  # For Telegram to display plus sign

    #
    # Send notification
    #
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


if __name__ == '__main__':
    pass
