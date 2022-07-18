from datetime import datetime
from decimal import *
import click

import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from service.App import *
from common.utils import *
from service.collector import *
from service.analyzer import *
from service.notifier import *
from service.trader import *

import logging

log = logging.getLogger('server')


#
# Main procedure
#

async def main_task():
    """This task will be executed regularly according to the schedule"""
    res = await main_collector_task()
    if res:
        return res

    # TODO: Validation
    #last_kline_ts = App.analyzer.get_last_kline_ts(symbol)
    #if last_kline_ts + 60_000 != startTime:
    #    log.error(f"Problem during analysis. Last kline end ts {last_kline_ts + 60_000} not equal to start of current interval {startTime}.")

    # Generate signals (derived features, predictions)
    try:
        analyze_task = await App.loop.run_in_executor(None, App.analyzer.analyze)
    except Exception as e:
        print(f"Error while analyzing data: {e}")
        return
    # Signal is stored in App.signal

    if "notify" in App.config["actions"]:
        notify_task = App.loop.create_task(notify_telegram())

    # Now we have a list of signals and can make trade decisions using trading logic and trade
    if "trade" in App.config["actions"]:
        trade_task = App.loop.create_task(main_trader_task())

    return


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def start_server(config_file):

    load_config(config_file)

    symbol = App.config["symbol"]

    print(f"Initializing server. Trade pair: {symbol}. ")

    #getcontext().prec = 8

    #
    # Validation
    #

    #
    # Connect to the server and update/initialize the system state
    #
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    App.analyzer = Analyzer(App.config)

    App.loop = asyncio.get_event_loop()

    # Do one time server check and state update
    try:
        App.loop.run_until_complete(data_provider_health_check())
    except Exception as e:
        print(f"Problems during health check (connectivity, server etc.) {e}")

    if data_provider_problems_exist():
        print(f"Problems during health check (connectivity, server etc.)")
        return

    print(f"Finished health check (connection, server status etc.)")

    # Do one time data load (cold start)
    try:
        App.loop.run_until_complete(sync_data_collector_task())
        # First call may take a while because of big batch and hence we make second call into to get the (possible) newest klines not received by the first call
        App.loop.run_until_complete(sync_data_collector_task())
    except Exception as e:
        print(f"Problems during initial data collection. {e}")

    if data_provider_problems_exist():
        print(f"Problems during initial data collection.")
        return

    print(f"Finished initial data collection.")

    # Initialize trade status (account, balances, orders etc.)
    if "trade" in App.config["actions"]:
        try:
            App.loop.run_until_complete(update_trade_status())
        except Exception as e:
            print(f"Problems trade status sync. {e}")

        if data_provider_problems_exist():
            print(f"Problems trade status sync.")
            return

        print(f"Finished trade status sync (account, balances etc.)")
        print(f"Balance: {App.config['base_asset']} = {str(App.base_quantity)}")
        print(f"Balance: {App.config['quote_asset']} = {str(App.quote_quantity)}")

    #
    # Register scheduler
    #

    App.sched = AsyncIOScheduler()
    # logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

    App.sched.add_job(
        main_task,
        trigger='cron',
        # second='*/30',
        minute='*',
        id='main_task'
    )

    App.sched.start()  # Start scheduler (essentially, start the thread)

    print(f"Scheduler started.")

    #
    # Start event loop
    #
    try:
        App.loop.run_forever()  # Blocking. Run until stop() is called
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt.")
    finally:
        App.loop.close()
        print(f"Event loop closed.")
        App.sched.shutdown()
        print(f"Scheduler shutdown.")

    return 0


if __name__ == "__main__":
    start_server()
