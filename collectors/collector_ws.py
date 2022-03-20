import os
import sys
import argparse
import math, time
from datetime import datetime, date
import time

import pandas as pd
import asyncio

from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from binance.client import Client
from binance.websockets import BinanceSocketManager

from common.utils import *
from service.App import *
from service.analyzer import *

import logging
log = logging.getLogger('collector_ws')

#
# Subscribe to a stream and receive the events with updates
# The received data is stored in the corresponding files
#

# TODO:
#  Create websocket and subscribe to klines and depth high frequency data
#  Create event queue for processing incoming events. Writings to one file must be sequential - not in parallel.
#    But writes in different files can be independent tasks.
#    How to ensure sequential tasks? Essentially, incoming events are not allowed to overlap somewhere down the pipeline (at the end).
#  Store two streams (klines and depth) for all listed symbols in test files
#  Check for:
#   - recover after lost/bad connection (errors)
#   - continuation of connection (confirmation responses) - chect if it is done automatically by the client


def process_message(msg):
    if msg is None:
        print(f"Empty message received")
        return
    if not isinstance(msg, dict):
        print(f"Message received is not dict")
        return
    if len(msg.keys()) != 2:
        print(f"Message received has unexpected length. Message: {msg}")
        return

    error = msg.get('e')
    if error is not None:
        error_message = msg.get('m')
        print(f"Connection error: {error_message}")
        return

    stream = msg.get('stream')
    if stream is None:
        print(f"Empty stream received. Message: {msg}")
        # TODO: Check what happens and maybe reconnect
        return
    stream_symbol, stream_channel = tuple(stream.split("@"))

    event = msg.get('data')
    if event is None:
        print(f"Empty event received. Message {msg}")
        # TODO: Check what happens and maybe reconnect
        return

    event_channel = event.get('e')
    if event_channel == 'error':
        # close and restart the socket
        return
    if event_channel is None:
        event["e"] = stream_channel

    event_symbol = event.get('s')
    if event_symbol is None:
        event["s"] = stream_symbol.upper()

    event_ts = event.get('E')
    if event_ts is None:
        event["E"] = int(datetime.utcnow().timestamp() * 1000)

    #print(f"Event symbol: {event_symbol}, Event channel: {event_channel}")

    # Submit a task to our main event queue
    App.analyzer.queue.put(event)


def start_collector_ws():
    print(f"Start collecting data using WebSocket streams.")

    #
    # Initialize data state, connections and listeners
    #
    App.analyzer = Analyzer(None)

    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    #
    # Register websocket listener
    #

    # INFO:
    # Trades:
    # start_aggtrade_socket: individual trades aggregated for a single taker order
    # start_trade_socket: individual trades (not aggregated)

    # Klines:
    # start_kline_socket: 1s xor 2s (not clear), updates for the current kline of size 1m, 3m, 5m and so on. kline is either closed or not. question: it is like tumbling window or sliding window.
    # bm.start_kline_socket('BNBBTC', process_message, interval=KLINE_INTERVAL_30MINUTE)
    # +++We can store the updates as is, and then post-process by extracting shorter klines from long kline updates

    # Tickers:
    # start_miniticker_socket: 1s updates for rolling kline of fixed length 24h for one symbol
    # - there is a version for all symbols (which have changed)
    # start_symbol_ticker_socket: same as miniticker but with more information
    # start_ticker_socket: for all
    # All tickers send stats for the last 24h - not interesting (we can get this via klines)

    # start_depth_socket: 1s or 100ms, levels 5, 10, 20
    # bm.start_depth_socket('BNBBTC', process_message, depth=BinanceSocketManager.WEBSOCKET_DEPTH_20)
    # - there is also diff depth
    # +++

    # start_multiplex_socket: trade, kline, ticker, depth streams (but not user stream)

    # start_user_socket: account/order/trade updates (socket is kept alive automatically)

    # List of streams
    channels = App.config["collector"]["stream"]["channels"]
    print(f"Channels: {channels}")

    # List of symbols
    symbols = App.config["collector"]["stream"]["symbols"]
    print(f"Channels: {symbols}")

    streams = []
    for c in channels:
        for s in symbols:
            stream = s.lower() + "@" + c.lower()
            streams.append(stream)
    print(f"Streams: {streams}")

    App.bm = BinanceSocketManager(App.client, user_timeout=BinanceSocketManager.DEFAULT_USER_TIMEOUT)
    App.conn_key = App.bm.start_multiplex_socket(streams, process_message)
    App.bm.start()
    print(f"Subscribed to the streams.")

    #
    # Start event loop
    #

    """
    App.loop = asyncio.get_event_loop()
    try:
        App.loop.run_forever()  # Blocking. Run until stop() is called
    except KeyboardInterrupt:
        pass
    finally:
        print("===> Closing Loop")
        App.loop.close()
        App.sched.shutdown()
    """

    # Periodically call db-store
    saving_period = App.config["collector"]["flush_period"]
    try:
        while True:
            time.sleep(saving_period)
            event_count = App.analyzer.queue.qsize()
            if event_count > 0:
                print(f"Storing {event_count} events.")
                App.analyzer.store_queue()
            else:
                # Reconnect
                print(f"No incoming messages. Trying to reconnect.")
                reconnect_pause = 30
                time.sleep(reconnect_pause)
                if App.bm is not None:
                    try:
                        App.bm.close()
                    except:
                        pass

                try:
                    App.bm = BinanceSocketManager(App.client, user_timeout=BinanceSocketManager.DEFAULT_USER_TIMEOUT)
                    App.conn_key = App.bm.start_multiplex_socket(streams, process_message)
                    App.bm.start()
                except:
                    print(f"Exception while reconnecting. Will try next time.")
    except KeyboardInterrupt:
        pass

    if App.bm is not None:
        App.bm.close()

    print(f"End collecting data using WebSocket streams.")

    return 0


if __name__ == "__main__":
    start_collector_ws()

    pass
