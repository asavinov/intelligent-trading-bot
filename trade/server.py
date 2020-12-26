import os
import sys
import argparse
import math, time
from datetime import datetime
from decimal import *

import pandas as pd
import asyncio

from apscheduler.schedulers.background import BackgroundScheduler

from binance.client import Client
from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from binance.enums import *

from common.utils import *
from trade.App import *
from trade.Database import *

import signaler
import trader

#
# Main procedure
#

def start_signaler():
    #
    # Validation
    #
    symbol = App.config["trader"]["symbol"]

    log.info(f"Initializing signaler server. Trade symbol {symbol}. ")

    #
    # Connect to the server and update/initialize our system state
    #
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    App.database = Database(None)

    App.loop = asyncio.get_event_loop()

    # Do one time server check and state update
    try:
        App.loop.run_until_complete(signaler.data_provider_health_check())
    except:
        pass
    if data_provider_problems_exist():
        log.error(f"Problems with the data provider server found.")
        return

    log.info(f"Finished updating state and health check.")

    # Do one time data update (cold start)
    try:
        App.loop.run_until_complete(signaler.sync_data_collector_task())
    except:
        pass
    if data_provider_problems_exist():
        log.error(f"Problems with the data provider server found.")
        return

    log.info(f"Finished updating data.")

    #
    # Register schedulers
    #

    # INFO: Scheduling:
    #     - https://medium.com/greedygame-engineering/an-elegant-way-to-run-periodic-tasks-in-python-61b7c477b679
    #     - https://schedule.readthedocs.io/en/stable/ https://github.com/dbader/schedule - 6.6k
    #     - https://github.com/agronholm/apscheduler/blob/master/docs/index.rst - 2.1k
    #       - https://apscheduler.readthedocs.io/en/latest/modules/schedulers/asyncio.html
    #     - https://docs.python.org/3/library/sched.html

    App.sched = BackgroundScheduler(daemon=False)  # Daemon flag is passed to Thread (False means the program will not exit until all Threads are finished)
    #logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

    App.sched.add_job(
        # We register a normal Python function as a call back.
        # The only role of this function is to add an asyncio task to the event loop
        # INFO: Creating/adding asyncio tasks from another thread
        # - https://docs.python.org/3/library/asyncio-task.html#scheduling-from-other-threads
        # - App.loop.call_soon_threadsafe(sync_responder)  # This works, but takes a normal funciton (not awaitable), which has to call coroutine: eventLoop.create_task(coroutine())
        lambda: asyncio.run_coroutine_threadsafe(signaler.sync_signaler_task(), App.loop),
        trigger='cron',
        #second='*/30',
        minute='*',
        id='sync_signaler_task'
    )

    App.sched.start()  # Start scheduler (essentially, start the thread)

    log.info(f"Scheduler started.")

    #
    # Start event loop
    #
    try:
        App.loop.run_forever()  # Blocking. Run until stop() is called
    except KeyboardInterrupt:
        log.info(f"KeyboardInterrupt.")
        pass
    finally:
        App.loop.close()
        log.info(f"Event loop closed.")
        App.sched.shutdown()
        log.info(f"Scheduler shutdown.")

    return 0

#
# Main procedure
#

def start_trader():
    #
    # Validation
    #
    symbol = App.config["trader"]["symbol"]

    log.info(f"Initializing trade server. Trade symbol {symbol}. ")

    #
    # Connect to the server and update/initialize our system state
    #
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    App.database = Database(None)

    App.loop = asyncio.get_event_loop()

    # Do one time server check and state update
    try:
        App.loop.run_until_complete(trader.update_state_and_health_check())
    except:
        pass
    if problems_exist():
        log.error(f"Problems found. Check server, symbol, account or system state.")
        return

    log.info(f"Finished updating state and health check.")

    log.info(f"Finished updating data.")

    #
    # Register schedulers
    #

    # INFO: Scheduling:
    #     - https://medium.com/greedygame-engineering/an-elegant-way-to-run-periodic-tasks-in-python-61b7c477b679
    #     - https://schedule.readthedocs.io/en/stable/ https://github.com/dbader/schedule - 6.6k
    #     - https://github.com/agronholm/apscheduler/blob/master/docs/index.rst - 2.1k
    #       - https://apscheduler.readthedocs.io/en/latest/modules/schedulers/asyncio.html
    #     - https://docs.python.org/3/library/sched.html

    App.sched = BackgroundScheduler(daemon=False)  # Daemon flag is passed to Thread (False means the program will not exit until all Threads are finished)
    #logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

    App.sched.add_job(
        # We register a normal Python function as a call back.
        # The only role of this function is to add an asyncio task to the event loop
        # INFO: Creating/adding asyncio tasks from another thread
        # - https://docs.python.org/3/library/asyncio-task.html#scheduling-from-other-threads
        # - App.loop.call_soon_threadsafe(sync_responder)  # This works, but takes a normal funciton (not awaitable), which has to call coroutine: eventLoop.create_task(coroutine())
        lambda: asyncio.run_coroutine_threadsafe(trader.sync_trader_task(), App.loop),
        trigger='cron',
        #second='*/30',
        minute='*',
        id='sync_trader_task'
    )

    App.sched.start()  # Start scheduler (essentially, start the thread)

    log.info(f"Scheduler started.")

    #
    # Start event loop
    #
    try:
        App.loop.run_forever()  # Blocking. Run until stop() is called
    except KeyboardInterrupt:
        log.info(f"KeyboardInterrupt.")
        pass
    finally:
        App.loop.close()
        log.info(f"Event loop closed.")
        App.sched.shutdown()
        log.info(f"Scheduler shutdown.")

    return 0

if __name__ == "__main__":
    start_signaler()
    os.exit()

    # Short version of start_trader (main procedure) for testing/debug purposes
    App.database = Database(None)
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])
    App.loop = asyncio.get_event_loop()
    try:
        log.debug("Start in debug mode.")
        log.info("Start testing in main.")

        App.loop.run_until_complete(signaler.data_provider_health_check())

        App.loop.run_until_complete(signaler.sync_data_collector_task())

        App.database.analyze("BTCUSDT")

        #App.loop.run_until_complete(sync_signaler_task())

    except BinanceAPIException as be:
        # IP is not registred in binance
        # BinanceAPIException: APIError(code=-2015): Invalid API-key, IP, or permissions for action
        # APIError(code=-1021): Timestamp for this request was 1000ms ahead of the server's time.
        print(be)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(f"Exception {e}")
    finally:
        log.info(f"Finished.")
        App.loop.close()
        #App.sched.shutdown()

    pass
