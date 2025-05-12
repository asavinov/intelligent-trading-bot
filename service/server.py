
from decimal import *
import asyncio

import click

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from binance import Client

from common.types import Venue
from service.App import *
from common.utils import *
from common.generators import output_feature_set
from service.analyzer import *
from service.mt5 import connect_mt5

from inputs import get_collector_functions

from outputs.notifier_trades import *
from outputs.notifier_scores import *
from outputs.notifier_diagram import *
from outputs import get_trader_functions


import logging

log = logging.getLogger('server')

logging.basicConfig(
    filename="server.log",
    level=logging.DEBUG,
    #format = "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    format = "%(asctime)s %(levelname)s %(message)s",
    #datefmt = '%Y-%m-%d %H:%M:%S',
)

# Get the collector functions based on the collector type

#
# Main procedure
#
async def main_task():
    """This task will be executed regularly according to the schedule"""

    #
    # 1. Execute input adapters to receive new data from data source(s)
    #
    venue = App.config.get("venue")
    venue = Venue(venue)
    main_collector_task, _, _ = get_collector_functions(venue)

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
    venue = App.config.get("venue")
    try:
        if venue is not None:
            venue = Venue(venue)
    except ValueError as e:
        log.error(f"Invalid venue specified in config: {venue}. Error: {e}. Currently these values are supported: {[e.value for e in Venue]}")
        return
    
    _, data_provider_health_check, sync_data_collector_task = get_collector_functions(venue)
    trader_funcs = get_trader_functions(venue)
    
    log.info(f"Initializing server. Venue: {venue.value}. Trade pair: {symbol}. Frequency: {freq}")
    
    #getcontext().prec = 8

    #
    # Validation
    #

    #
    # Connect to the server and update/initialize the system state
    #
    if venue == Venue.BINANCE:
        App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])
    
    if venue == Venue.MT5:
        authorized = connect_mt5(mt5_account_id=int(App.config.get("mt5_account_id")), mt5_password=str(App.config.get("mt5_password")), mt5_server=str(App.config.get("mt5_server")))
        if not authorized:
            log.error(f"Failed to connect to MT5. Check credentials and server details.")
            return
        App.client = mt5  

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
            App.loop.run_until_complete(trader_funcs['update_trade_status']())
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

    App.sched._eventloop = App.loop
    App.sched.start()  # Start scheduler (essentially, start the thread)

    log.info(f"Scheduler started.")
    
    #
    # Start event loop and scheduler
    #
    try:
        App.loop.run_forever()  # Blocking. Run until stop() is called
    except KeyboardInterrupt:
        log.info(f"KeyboardInterrupt.")
    finally:
        log.info("Shutting down...")
        # Graceful shutdown
        if App.sched and App.sched.running:
             App.sched.shutdown()
             log.info(f"Scheduler shutdown.")
        # Stop the loop if it's still running (e.g., if shutdown initiated by signal other than KeyboardInterrupt)
        if App.loop.is_running():
             App.loop.stop()
             log.info("Event loop stop requested.")
        # Close the loop
        # Allow pending tasks to complete before closing (optional but good practice)
        # You might need to run loop.run_until_complete(asyncio.sleep(0.1)) or similar
        # if loop.stop() doesn't immediately halt everything.
        App.loop.close()
        log.info(f"Event loop closed.")
        # Shutdown MT5 connection if it was initialized
        if venue == Venue.MT5:
            mt5.shutdown()
            log.info("MT5 connection shutdown.")

    return 0


if __name__ == "__main__":
    start_server()
