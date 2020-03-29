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
from sklearn.model_selection import ParameterGrid

from trade.utils import *

"""
In fact, it training a hyper-model while the model itself is not trained - it is a rule-based model.
Find best signal generation models using pre-computed (rolling) predict label scores and searching through the threshold grid.
The output is an ordered list of best performing threshold-based signal generation models.

#     - Use rolling predictions as input with scores for each label and algorithm
#     - Parameters: grid of comparison thresholds. each element in the grid defines a binary column of true signal (buy signal)
#     - for each grid element, generate a column of signals, then determine performance
#     - performance is computed by looping through the dataset, determining buy points, then finding sell point using high in klines or close in case of time out, and computing profit for transaction
#     - alternatively, we could store a column of future high along with its id, and a column of future close price (both for our horizon)
"""
# !!! The code below is supposed to be called from outside driver and accepts envvars as hyper-parameters for signals (rules)
#   All other parameters are static
# Approach 1:
# We might modify this procedure as a function which takes P parameters including the hyper-parameters so it can run independently (we will not use this - it is simpler to modify)
#   Note that this function will load the source file and append its result (for hyper-parameter performance) in output file.
#   Then we use this function from the grid driver which will modify hyper-parameters in P
# Approach 2:
# Alternatively, this function could take also input df with (only) necessary columns as input (along with other parameers, so that we avoid loading the same input data)
#   It then does simulation in-memory: input df and output dict with results
#   It will append a column with signals computed from current params but it will be overwritten in next calls
#   The grid driver will make such in-memory calls for each cell by collecting, filtering and storing results.

# Develop a separate function which takes a df with necessary columns as well as trade model (thresholds etc.)
# It returns a dict with trade performance
# Develop a grid driver which will load and prepare data, loop through all trade models by collecting results, store the best models in an output file.


grid_signals = [
    # Production
    #{
    #    'threshold_buy_10': [0.25, 0.26, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.34, 0.35],
    #    'threshold_buy_20': [0.0],
    #    'percentage_sell_price': [1.017, 1.018, 1.019, 1.02],
    #    'sell_timeout': [80, 90, 100],
    #},
    # Debug
    {
        'threshold_buy_10': [0.29, 0.31],
        'threshold_buy_20': [0.0],
        'percentage_sell_price': [1.015, 1.020],
        'sell_timeout': [60],
    },
]

#
# Parameters
#
class P:
    in_path_name = r"_TEMP_FEATURES"
    in_file_name = r"_BTCUSDT-1m-rolling-predictions.csv"
    in_nrows = 100_000_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"_BTCUSDT-1m-signal-models"

    simulation_start = 100  # Default 0
    simulation_end = -100  # Default till end of input data. Negative value is shift from the end

    #
    # Parameters of the whole optimization
    #
    initial_amount = 1_000.0
    performance_weight = 2.0  # 1.0 means all are equal, 2.0 means last has 2 times more weight than first

def main(args=None):

    start_dt = datetime.now()

    #
    # Load data with rolling label score predictions
    #
    print(f"Loading data with label rolling predict scores from input file...")

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

    # Select the necessary interval of data
    if not P.simulation_start:
        P.simulation_start = 0
    if not P.simulation_end:
        P.simulation_end = len(in_df)
    elif P.simulation_end < 0:
        P.simulation_end = len(in_df) + P.simulation_end

    in_df = in_df.iloc[P.simulation_start:P.simulation_end]

    #
    # Loop on all trade hyper-models - one model is one trade (threshold-based) strategy
    #
    grid = ParameterGrid(grid_signals)
    models = list(grid)  # List of model dicts
    performances = []

    for model in models:
        # Set parameters of the model

        print(f"Starting simulation...")
        performance = simulate_trade(in_df, model, P.performance_weight, P.initial_amount)
        print(f"Simulation finished:")

        performances.append(performance)

    #
    # Post-process: sort and filter
    #

    # Column names
    model_keys = models[0].keys()
    performance_keys = performances[0].keys()
    header_str = ",".join(model_keys + performance_keys)

    lines = []
    for i, model in enumerate(models):
        model_values = [f"{v:.2f}" for v in model.values()]
        performance_values = [f"{v:.2f}" for v in performances[i].values()]
        line_str = ",".join(model_values + performance_values)
        lines.append(line_str)

    #
    # Store simulation parameters and performance
    #
    out_path = Path(P.out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
    out_path = out_path.joinpath(P.out_file_name)

    if out_path.with_suffix('.txt').is_file():
        add_header = False
    else:
        add_header = True
    with open(out_path.with_suffix('.txt'), "a+") as f:
        if add_header:
            f.write(header_str + "\n")
        #f.writelines(lines)
        f.write("\n".join(lines))

    pass

def simulate_trade(df, model: dict, performance_weight: float, amount: float):
    """
    It will use 1.0 as initial trade amount and overall performance will be the end amount with respect to the initial one.
    It will always use 1.0 to enter market (for buying) independent of the available (earned or lost) funds.
    It will use the whole data set from start to end.

    :param df:
    :param model:
        threshold_buy_10 - Buy only if score is higher
        threshold_buy_20 - Buy only if score is higher
        percentage_sell_price - how much increase sell price in comparision to buy price (it is our planned profit)
        sell_timeout - Sell using latest close price after this time
    :param amount: initial amount for trade
    :return: Performance record which includes
      overall (weighted) performance,
      sequence of segment performances,
      variance,
      number of plus and minus transactions, number of forced sells etc.
    """
    #
    # Model parameters
    #
    threshold_buy_10 = float(model.get("threshold_buy_10"))
    threshold_buy_20 = float(model.get("threshold_buy_20"))

    percentage_sell_price = float(model.get("percentage_sell_price"))
    sell_timeout = int(model.get("sell_timeout"))

    #
    # Trade parameters variables
    #
    trade_amount = amount
    base_amount = 0.0  # Coins
    quote_amount = trade_amount  # Money

    # TODO: Always invest same amount (constant).
    #  All profits and losses are collected in a separate variable after each sale which determines overall performance.
    #  Separately, accumulate profits/losses also for each segment depending on the current time i (which needs to be partitioned)

    # TODO: Collect all transactions in order to analyze their stats including weighted average profit
    transactions = []  # List of dicts like dict(i=23, is_forced_sell=False, profit=-0.123)

    #
    # Output parameters (statistics)
    #
    total_buy_signal_count = 0  # How many rows satisfy buy signal criteria independent of mode

    buy_count = 0  # Really bought (in this approach, equal to buy signals and equal to number of transactions)
    limit_sell_count = 0  # Really sold using limit price (before timeout)
    forced_sell_count = 0  # Really sold on timeout

    limit_sell_fill_time = 0  # Total (accumulated) time till filled of limit orders
    forced_sell_fill_time = 0  # Total (accumulated) time till filled of forced sell (can be computed using timeout parameter)

    loss_transaction_count = 0  # Number of transactions with losses (can be compared to all transactions)
    loss_transaction_amount = 0.0  # Absolute losses

    #
    # Main loop over trade sessions
    #
    buy_price = None
    buy_time = None
    sell_price = None
    for i in range(0, len(df)):

        if i % 10_000 == 0:
            print(f"Processed {i} of {len(df)} records.")

        row = df.iloc[i]
        close_price = row["close"]
        high_price = row["high"]

        # Check buy criteria for any row (for both buy and sell modes)
        if row["high_60_10_gb"] >= threshold_buy_10 and row["high_60_20_gb"] >= threshold_buy_20:
            is_buy_signal = True
            total_buy_signal_count += 1
        else:
            is_buy_signal = False

        if base_amount == 0:  # Buy mode: no coins - trying to buy
            # Generate buy signal
            if is_buy_signal:
                # Execute buy signal by doing trade
                base_amount += trade_amount / close_price  # Increase coins
                quote_amount -= trade_amount  # Decrease money
                buy_count += 1

                buy_price = close_price
                buy_time = i
                sell_price = buy_price * percentage_sell_price

        elif base_amount > 0:  # Sell mode: there are coins - trying to sell
            # Determine if it was sold for our desired price
            if high_price >= sell_price:
                # Execute sell signal by doing trade
                quote_amount += base_amount * sell_price  # Increase money by selling all coins
                base_amount = 0.0
                limit_sell_count += 1
                limit_sell_fill_time += (i - buy_time)
            elif (i - buy_time) > sell_timeout:  # Sell time out. Force sell
                # Execute sell signal by doing trade
                quote_amount += base_amount * close_price  # Increase money by selling all coins
                base_amount = 0.0
                forced_sell_count += 1
                forced_sell_fill_time += sell_timeout

                if close_price < buy_price:  # If losses
                    loss_transaction_count += 1
                    loss_transaction_amount += (buy_price - close_price)

        else:
            print(f"Inconsistent state: both base and quote assets are zeros.")
            return

    #
    # Close. Sell rest base asset if available for the last price
    #
    i = P.simulation_end
    if base_amount > 0.0:  # Check base asset like BTC (we will spend it)
        # Execute buy signal by doing trade
        quote_amount += base_amount * close_price  # Increase money by selling all coins
        base_amount = 0.0
        forced_sell_count += 1
        forced_sell_fill_time += (i - buy_time)

    #
    # Dervied performance parameters
    #

    # TODO: Weighted average performance

    total_performance = 100.0 * (quote_amount - trade_amount) / trade_amount
    performance_per_trade = total_performance / buy_count if buy_count != 0 else 0.0

    mean_fill_time = limit_sell_fill_time / limit_sell_count if limit_sell_count != 0 else sell_timeout
    mean_loss_sale = loss_transaction_amount / loss_transaction_count if loss_transaction_count > 0.0 else 0.0

    limit_sell_portion = limit_sell_count / buy_count if buy_count != 0 else 0.0
    forced_sell_portion = forced_sell_count / buy_count if buy_count != 0 else 0.0

    performance = dict(
        total_buy_signal_count=total_buy_signal_count,
        buy_count=buy_count,
        limit_sell_count=limit_sell_count,
        forced_sell_count=forced_sell_count,
        limit_sell_fill_time=limit_sell_fill_time,
        forced_sell_fill_time=forced_sell_fill_time,
        loss_transaction_count=loss_transaction_count,
        loss_transaction_amount=loss_transaction_amount,

        # Derived parameters
        total_performance=total_performance,
        performance_per_trade=performance_per_trade,

        mean_fill_time = mean_fill_time,
        mean_loss_sale = mean_loss_sale,

        limit_sell_portion=limit_sell_portion,
        forced_sell_portion=forced_sell_portion,
    )

    return performance


if __name__ == '__main__':
    main(sys.argv[1:])
