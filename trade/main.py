import os
import sys
import json,re
from pathlib import Path
import argparse
import math, time
from datetime import datetime
import pandas as pd
import asyncio

from binance.client import Client

from trade.utils import *
from trade.App import App
from trade.Database import *
from trade.buysell import *
from trade.collect import *

import logging
log = logging.getLogger('trade')

__version__ = '0.1.0-SNAPSHOT'

"""
async def scheduler():
    while True:
        # sleep until the next whole second
        now = time.time()
        await asyncio.sleep(math.ceil(now) - now)

        # execute any scheduled tasks
        await for task in scheduled_tasks(time.time()):
            await task()

#
# From here: https://stackoverflow.com/questions/51292027/how-to-schedule-a-task-in-asyncio-so-it-runs-at-a-certain-date
#
async def wait_for(dt):
    # sleep until the specified datetime
    while True:
        now = datetime.datetime.now()
        remaining = (dt - now).total_seconds()
        if remaining < 86400:
            break
        # asyncio.sleep doesn't like long sleeps, so don't sleep more
        # than a day at a time
        await asyncio.sleep(86400)
    await asyncio.sleep(remaining)

async def run_at(dt, coro):
    await wait_for(dt)
    return await coro

# Example
async def hello():
    print('hello')

loop = asyncio.get_event_loop()
# print hello ten years after this answer was written
loop.create_task(run_at(datetime.datetime(2028, 7, 11, 23, 36), hello()))
loop.run_forever()
"""

def main(args = None):
    if not args: args = sys.argv[1:]

    programVersion = 'Version ' + __version__
    programDescription = 'Trading application ' + programVersion

    parser = argparse.ArgumentParser(description=programDescription)
    parser.add_argument('-v', '--version', action='version', version=programVersion)

    parser.add_argument('-l', '--log', dest="loglevel", required=False, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Set the logging level (default INFO)")

    parser.add_argument('config_file', type=str, nargs='?', default="", help='Parameter file name')

    arguments = parser.parse_args(args)

    #
    # Configure logging
    #
    logging.basicConfig(stream=sys.stderr, level=arguments.loglevel, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

    #
    # Load configuration into the global parameters
    #
    config_json = None
    if arguments.config_file and Path(arguments.config_file).exists():
        with open(arguments.config_file, 'r', encoding='utf-8') as f:
            json_string = f.read()
            json_string = re.sub("//.*", "", json_string, flags=re.MULTILINE)
            # Alternatively, these two lines to remove comments
            #data = re.sub("//.*?\n", "", data)
            #data = re.sub("/\\*.*?\\*/", "", data)
            config_json = json.loads(json_string)

    # Load into App
    if config_json:
        App.config = config_json
    else:
        log.info(f"No configuration provided or could be loaded. Using default configuration.")

    # Environment
    log.info(programDescription)

    exitcode = 1
    command = App.config["command"]
    try:
        if command == "trade":
            exitcode = start_trade()
        elif command == "collect":
            exitcode = start_collect()
        else:
            print(f"Uknonwn command {command}. Exit")
            exitcode = 1
    except Exception as e:
        log.error(f"Error starting application.")
        log.exception(e)

    logging.shutdown()

    return exitcode

if __name__ == "__main__":

    #App.client = Client(api_key=App.api_key, api_secret=App.api_secret)

    #startTime, endTime = get_interval("1m")
    #startTime = datetime.utcnow().replace(second=0, microsecond=0)
    #endTime = None

    # startTime: include all intervals (ids) with same or greater id: if within interval then excluding this interval; if is equal to open time then include this interval
    # endTime: include all intervals (ids) with same or smaller id: if equal to left border then return this interval, if within interval then return this interval
    # It will return also incomplete current interval (in particular, we could collect approximate klines for higher frequencies by requesitng incomplete intervals)
    #klines = App.client.get_klines(symbol="BTCUSDT", interval="1m", limit=5, endTime=startTime+1)

    #open_ts = klines[-1][0]  # Last line timestamp
    #open_dt = pd.to_datetime(open_ts, unit='ms')

    exitcode = main(sys.argv[1:])

    if(not exitcode):
        exit()
    else:
        exit(exitcode)
