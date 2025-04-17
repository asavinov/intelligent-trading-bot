from datetime import datetime
from decimal import *
import asyncio

import click

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from binance import Client

from service.App import *
from common.utils import *
from common.generators import output_feature_set
from service.analyzer import *

from inputs.collector_binance import main_collector_task, data_provider_health_check, sync_data_collector_task

from outputs.notifier_trades import *
from outputs.notifier_scores import *
from outputs.notifier_diagram import *
from outputs.trader_binance import trader_binance, update_trade_status

import logging
log = logging.getLogger('server')

logging.basicConfig(
    filename="server.log",
    level=logging.DEBUG,
    #format = "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    format = "%(asctime)s %(levelname)s %(message)s",
    #datefmt = '%Y-%m-%d %H:%M:%S',
)

#
# Main procedure
#

async def main_task():
    """This task will be executed regularly according to the schedule"""

    #
    # 1. Execute input adapters to receive new data from data source(s)
    #

    try:
        res = await main_collector_task()
    except Exception as e:
        log.error(f"Error in main_collector_task function: {e}")
        return
    if res:
        log.error(f"Error in main_collector_task function: {res}")
        return res

    # TODO: Validation
    #last_kline_ts = App.analyzer.get_last_kline_ts(symbol)
    #if last_kline_ts + 60_000 != startTime:
    #    log.error(f"Problem during analysis. Last kline end ts {last_kline_ts + 60_000} not equal to start of current interval {startTime}.")

    #
    # 2. Apply transformations (merge, features, prediction scores, signals) and generate new data columns
    #

    try:
        analyze_task = await App.loop.run_in_executor(None, App.analyzer.analyze)
    except Exception as e:
        log.error(f"Error in analyze function: {e}")
        return

    #
    # 3. Execute output adapter which send the results of analysis to consumers
    #

    # Execute all output set entries
    output_sets = App.config.get("output_sets", [])
    for os in output_sets:
        try:
            await output_feature_set(App.df, os, App.config)
        except Exception as e:
            log.error(f"Error in output function: {e}")
            return

    return


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def start_server(config_file):

    load_config(config_file)

    symbol = App.config["symbol"]
    freq = App.config["freq"]

    log.info(f"Initializing server. Trade pair: {symbol}. ")

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
        log.error(f"Problems during health check (connectivity, server etc.) {e}")

    if data_provider_problems_exist():
        log.error(f"Problems during health check (connectivity, server etc.)")
        return

    log.info(f"Finished health check (connection, server status etc.)")

    # Cold start: load initial data, do complete analysis
    try:
        App.loop.run_until_complete(sync_data_collector_task())
        # First call may take some time because of big initial size and hence we make the second call to get the (possible) newest klines
        App.loop.run_until_complete(sync_data_collector_task())

        # Analyze all received data (and not only last few rows) so that we have full history
        App.analyzer.analyze(ignore_last_rows=True)
    except Exception as e:
        log.error(f"Problems during initial data collection. {e}")

    if data_provider_problems_exist():
        log.error(f"Problems during initial data collection.")
        return

    log.info(f"Finished initial data collection.")

    # TODO: Only for binance output and if it has been defined
    # Initialize trade status (account, balances, orders etc.) in case we are going to really execute orders
    if App.config.get("trade_model", {}).get("trader_binance"):
        try:
            App.loop.run_until_complete(update_trade_status())
        except Exception as e:
            log.error(f"Problems trade status sync. {e}")

        if data_provider_problems_exist():
            log.error(f"Problems trade status sync.")
            return

        log.info(f"Finished trade status sync (account, balances etc.)")
        log.info(f"Balance: {App.config['base_asset']} = {str(App.account_info.base_quantity)}")
        log.info(f"Balance: {App.config['quote_asset']} = {str(App.account_info.quote_quantity)}")

    #
    # Register scheduler
    #

    App.sched = AsyncIOScheduler()
    # logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

    trigger = freq_to_CronTrigger(freq)

    App.sched.add_job(
        main_task,
        trigger=trigger,
        id='main_task'
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
    finally:
        App.loop.close()
        log.info(f"Event loop closed.")
        App.sched.shutdown()
        log.info(f"Scheduler shutdown.")

    return 0


if __name__ == "__main__":
    start_server()
