
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
            await output_feature_set(App.analyzer.df, os, App.config, App.model_store)
        except Exception as e:
            log.error(f"Error in output function: {e}")
            return

    return


async def main_collector_task():
    """
    It is a highest level task which is added to the event loop and executed normally every 1 minute and then it calls other tasks.

    # 1) Get missing count from Analyzer
    # 2) Call (async) venue-specific retrieval function and get the raw data (with structure)
    # 3) Append the standard raw data with structure to Analyzer (or convert before)
    """
    venue = App.config.get("venue")
    venue = Venue(venue)
    sync_data_collector_task, data_provider_health_check = get_collector_functions(venue)

    symbol = App.config["symbol"]
    freq = App.config["freq"]
    start_ts, end_ts = pandas_get_interval(freq)
    now_ts = now_timestamp()

    log.info(f"===> Start collector task. Timestamp {now_ts}. Interval [{start_ts},{end_ts}].")

    #
    # 1. Check server state (if necessary)
    #
    if data_provider_problems_exist():
        await data_provider_health_check()
        if data_provider_problems_exist():
            log.error(f"Problems with the data provider server found. No signaling, no trade. Will try next time.")
            return 1

    #
    # 2. Get how much data is missing and request it
    #
    results = await sync_data_collector_task(App.config)
    if results is None:
        log.error(f"Problem getting data from the server. No signaling, no trade. Will try next time.")
        return 1

    #
    # 3. Push data to the analyzer for further processing
    #
    try:
        App.analyzer.append_klines(results)
    except Exception as e:
        log.error(f"Error storing kline result in the database. Exception: {e}")
        return 1

    log.info(f"<=== End collector task.")
    return 0


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def start_server(config_file):

    load_config(config_file)

    App.config["train"] = False  # Server does not train - it only predicts therefore disable train mode

    symbol = App.config["symbol"]
    freq = App.config["freq"]
    venue = App.config.get("venue")
    try:
        if venue is not None:
            venue = Venue(venue)
    except ValueError as e:
        log.error(f"Invalid venue specified in config: {venue}. Error: {e}. Currently these values are supported: {[e.value for e in Venue]}")
        return
    
    sync_data_collector_task, data_provider_health_check = get_collector_functions(venue)
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
        client_args = App.config.get("client_args", {})
        if App.config.get("api_key"):
            client_args["api_key"] = App.config.get("api_key")
        if App.config.get("api_secret"):
            client_args["api_secret"] = App.config.get("api_secret")
        App.client = Client(**client_args)

    if venue == Venue.MT5:
        from service.mt5 import connect_mt5
        authorized = connect_mt5(mt5_account_id=int(App.config.get("mt5_account_id")), mt5_password=str(App.config.get("mt5_password")), mt5_server=str(App.config.get("mt5_server")))
        if not authorized:
            log.error(f"Failed to connect to MT5. Check credentials and server details.")
            return
        App.client = mt5  

    App.model_store = ModelStore(App.config)
    App.model_store.load_models()
    App.analyzer = Analyzer(App.config, App.model_store)

    # Load latest transaction and (simulated) trade state
    App.transaction = load_last_transaction()

    #App.loop = asyncio.get_event_loop()  # In Python 3.12: DeprecationWarning: There is no current event loop
    App.loop = asyncio.new_event_loop()

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
        App.loop.run_until_complete(main_collector_task())
        # The very first call (cold start) may take some time because of big initial size and hence we make the second call to get the (possible) newest klines
        App.loop.run_until_complete(main_collector_task())

        # Analyze all received data (not only last few rows) so that we have full history
        App.analyzer.analyze()
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
