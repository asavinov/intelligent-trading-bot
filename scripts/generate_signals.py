import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

from trade.utils import *

"""
OBSOLETE: Use train_signal_models.py instead for grid search and finding best signal model by loading pre-computed rolling predictions.
  Previously, it was used from the driver start-grid.py. 

This script uses fixed signal parameters (thresholds etc.) to make one pass through the predictions by generating buy/sell signals for each row.
Yet, it does not store signals as new signals but rather uses these signals for trading and evaluating overall performance.
The result of performance evaluation is appended as one line to the output file.
In this sense, it is a trade simulation script for evaluating performance.
"""

#
# Parameters
#
class P:
    in_path_name = r"_TEMP_FEATURES"
    in_file_name = r"_BTCUSDT-1m-data-predictions.csv"
    in_nrows = 100_000_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"_BTCUSDT-1m-data-features_1-signals"

    simulation_start = 100  # 100 172_440 (old data)
    simulation_end = -100  # -100 464_760 (old data)

    threshold_buy_10 = None  # Buy if higher
    threshold_buy_20 = None  # Buy if higher
    knn_threshold_buy_10 = None  # Buy if higher
    knn_threshold_buy_20 = None  # Buy if higher

    threshold_sell = None  # Sell if price increases this factor from the buy price
    forced_sell_length = None

    trade_amount = 1_000.0

    #
    # Parameters which will be changed in the loop and hence need to be reset after each run
    #
    base_amount = 0.0  # Coins
    quote_amount = trade_amount  # Money

    fill_time_nonforced = 0
    fill_time_forced = 0

    total_buy_signal_count = 0  # How many rows satisfy buy signal criteria (in both buy and sell modes)
    buy_signal_count = 0  # How many buy signals in buy mode
    sell_signal_count = 0  # How many time sold for the desired price (excluding forced sales because of time out)

    buy_count = 0  # Really bought (in this approach, equal to buy signals)
    sell_count = 0  # Really sold (in this approach, equal to buy count)

    loss_sales_count = 0
    loss_sales_amount = 0.0


def main(args=None):
    pp = P()  # Here we are guaranteed to have *initial* values which can be then changed in the loop

    in_df = None

    start_dt = datetime.now()

    #
    # Load data with predictions
    #
    print(f"Loading data with predictions from input file...")

    in_path = Path(P.in_path_name).joinpath(P.in_file_name)
    if not in_path.exists():
        print(f"ERROR: Input file does not exist: {in_path}")
        return

    if P.in_file_name.endswith(".csv"):
        in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)
    elif P.in_file_name.endswith(".parq"):
        in_df = pd.read_parquet(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    #
    # Initialize parameters of the loop
    #
    if not P.simulation_start:
        P.simulation_start = 0
    if not P.simulation_end:
        P.simulation_end = len(in_df)
    elif P.simulation_end < 0:
        P.simulation_end = len(in_df) + P.simulation_end

    #
    # Hyper-parameters with defaults
    #
    P.threshold_buy_10 = float(os.getenv("threshold_buy_10", 0.5))
    P.threshold_buy_20 = float(os.getenv("threshold_buy_20", 0.0))
    P.knn_threshold_buy_10 = float(os.getenv("knn_threshold_buy_10", 0.0))
    P.knn_threshold_buy_20 = float(os.getenv("knn_threshold_buy_20", 0.0))

    P.threshold_sell = float(os.getenv("threshold_sell", 1.01))
    P.forced_sell_length = int(os.getenv("forced_sell_length", 60))

    #
    # Main loop over trade sessions
    #
    print(f"Starting simulation loop...")
    buy_price = None
    buy_time = None
    sell_price = None
    for i in range(P.simulation_start, P.simulation_end):

        if i % 10_000 == 0:
            print(f"Processed {i} of {P.simulation_end - P.simulation_start} records.")

        row = in_df.iloc[i]
        close_price = row["close"]
        high_price = row["high"]

        # Check buy criteria for any row (for both buy and sell modes)
        is_buy_signal = False
        if row["high_60_10_gb"] >= P.threshold_buy_10 and row["high_60_20_gb"] >= P.threshold_buy_20 \
                and row["high_60_10_knn"] >= P.knn_threshold_buy_10 and row["high_60_20_knn"] >= P.knn_threshold_buy_20:
            is_buy_signal = True
            pp.total_buy_signal_count += 1

        if pp.base_amount == 0:  # Buy mode: no coins - trying to buy
            # Generate buy signal
            if is_buy_signal:
                pp.buy_signal_count += 1

                # Execute buy signal by doing trade
                pp.base_amount += P.trade_amount / close_price  # Increase coins
                pp.quote_amount -= P.trade_amount  # Decrease money
                pp.buy_count += 1

                buy_price = close_price
                buy_time = i
                sell_price = buy_price * P.threshold_sell

        elif pp.base_amount > 0:  # Sell mode: there are coins - trying to sell
            # Determine if it was sold for our desired price
            if high_price >= sell_price:
                pp.sell_signal_count += 1

                # Execute buy signal by doing trade
                pp.quote_amount += pp.base_amount * sell_price  # Increase money by selling all coins
                pp.base_amount = 0.0
                pp.sell_count += 1
                pp.fill_time_nonforced += (i - buy_time)
            elif (i - buy_time) > P.forced_sell_length:  # Sell time out. Force sell
                # Execute buy signal by doing trade
                pp.quote_amount += pp.base_amount * close_price  # Increase money by selling all coins
                pp.base_amount = 0.0
                pp.sell_count += 1
                pp.fill_time_forced += P.forced_sell_length

                if close_price < buy_price:  # Losses
                    pp.loss_sales_count += 1
                    pp.loss_sales_amount += (buy_price - close_price)

        else:
            print(f"Inconsistent state: both base and quote assets are zeros.")
            return

    #
    # Close. Sell rest base asset if available for the last price
    #
    i = P.simulation_end
    if pp.base_amount > 0.0:  # Check base asset like BTC (we will spend it)
        # Execute buy signal by doing trade
        pp.quote_amount += pp.base_amount * close_price  # Increase money by selling all coins
        pp.base_amount = 0.0
        pp.sell_count += 1
        pp.fill_time_forced += (i - buy_time)

    #
    # Store simulation parameters and performance
    #
    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
    out_path = out_path.joinpath(P.out_file_name)

    precision = pp.sell_signal_count / pp.sell_count if pp.sell_count != 0 else 0.0
    mean_fill_time = pp.fill_time_nonforced / pp.sell_signal_count if pp.sell_signal_count != 0 else P.forced_sell_length
    total_performance = 100.0 * (pp.quote_amount - P.trade_amount) / P.trade_amount
    performance_per_trade = total_performance / pp.buy_count if pp.buy_count != 0 else 0.0
    mean_loss_sale = pp.loss_sales_amount / pp.loss_sales_count if pp.loss_sales_count > 0.0 else 0.0

    out_str = ""
    out_str += f"{P.threshold_buy_10}, {P.threshold_buy_20}, {P.knn_threshold_buy_10}, {P.knn_threshold_buy_20}, "
    out_str += f"{P.threshold_sell}, {P.forced_sell_length}, "
    out_str += f"{pp.total_buy_signal_count}, {pp.buy_signal_count}, {pp.sell_signal_count}, "
    out_str += f"{pp.loss_sales_count}, {mean_loss_sale:.2f}, "
    out_str += f"{precision:.2f}, {mean_fill_time:.2f}, "
    out_str += f"{performance_per_trade:.2f}, {total_performance:.2f}"

    # TODO: Performance per forced vs non-forced trade to see how bad forced trades are
    # TODO: Check what happens if we lose money. We cannot use fixed 1000 for trade
    #    In general, we should be able to get negative performance somehow. Check that it is possible.
    #    Check that always using fixed trade amount is correct. For example, we buy for 1000 and return less or more than 1000 in cash. Then again we use 1000 (decreasing the cash), and return more or less.

    header_str = f"threshold_buy_10,threshold_buy_20,knn_threshold_buy_10,knn_threshold_buy_20," \
                 f"threshold_sell,forced_sell_length," \
                 f"total_buy_signal_count,buy_signal_count,sell_signal_count," \
                 f"loss_sales_count,mean_loss_sale," \
                 f"precision,mean_fill_time," \
                 f"performance_per_trade,total_performance"
    if out_path.with_suffix('.txt').is_file():
        add_header = False
    else:
        add_header = True

    with open(out_path.with_suffix('.txt'), "a+") as f:
        if add_header:
            f.write(header_str + "\n")
        f.write(out_str + "\n")

    print(f"")
    print(f"Simulation finished:")


if __name__ == '__main__':
    main(sys.argv[1:])
