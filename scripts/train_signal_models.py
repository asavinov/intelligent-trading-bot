import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle
import click

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import ParameterGrid

from service.App import *
from common.utils import *
from common.signal_generation import *

"""
By signal generation model we mean simple rules with some thresholds as parameters.
Training such a model is performed by brute force by training out all parameters and finding their peformance using back testing.
Back testing is performed using an available feature matrix with all rolling predictions.
The output is an ordered list of top performing threshold-based signal generation models.

One approach is to use two parameters which determine entry and exit thresholds for a position. They need not be symmetric.
These parameters are compared with the final prediction in [-1, +1] which expresses our consolidated future trend. 
We might introduce one or two additional parameters to eliminate jitter if it happens. 
Another way to reduce jitter is to smooth the final score so that it simply does not change quickly.
Yet, if we entry and exist thresholds are significantly different then jitter should not happen.
This approach does not try to optimize each transaction by searching individual best (but rare) entry-exit point.
Instead, it tries to find maximums and minimums by switching the side at these points.
We switch side independent of any other factors - simply because of the future trend and more opportunities on this new side.

NEXT:
!!!- Implement signal generation server with real-time updates and notifying some other
  component which might simulate trade or simply store the logs in a file

- Since we are going to use only kline (no futures), generate rolling predictions for longer period
  - play with other options of NN since we

- Explore how profit depends on month: downward trend (Feb-March) vs. other months.
  - Run back testing on only summer months
- DONE: smooth score and check if the performance is better. simply run same grid with different smooth factors (different definitions of score column).
- alternative to smoothing, generate signal if 2 or more previous values are all higher/lower than threshold
- OR, take generate signal when 2nd time crosses the threshold

- hybrid strategy: instead of simply waiting for signal and change point, we can introduce stop loss or take profit signals.
These signals will allow us to take profit where we see it and it is large rather than wait for something in future.
In other words, such signals are generated from reality (we can earn already now) rather then future.
One way to implement it, is to use this special kind of order (take profit) which will be executed automatically.
Yet, we need to model this logic manually.
"""

grid_signals = [
    {
        "buy_threshold": [
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
            0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
            0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
            0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
        ],  # Buy when higher than this value
        "sell_threshold": [
            -0.10, -0.11, -0.12, -0.13, -0.14, -0.15, -0.16, -0.17, -0.18, -0.19,
            -0.20, -0.21, -0.22, -0.23, -0.24, -0.25, -0.26, -0.27, -0.28, -0.29,
            -0.30, -0.31, -0.32, -0.33, -0.34, -0.35, -0.36, -0.37, -0.38, -0.39,
            -0.40, -0.41, -0.42, -0.43, -0.44, -0.45, -0.46, -0.47, -0.48, -0.49,
        ],  # Sell when lower than this value

        "transaction_fee": [0.005],  # Portion of total transaction amount
        "transaction_price_adjustment": [0.005],  # The real execution price is worse than we assume

        # Per year. 1.0 means all are equal, 3.0 means last has 3 times more weight than first
        "performance_weight": [1.0],
    },
]


#
# Parameters
#
class P:
    feature_sets = ["kline", ]  # "futur"

    in_nrows = 100_000_000

    predict_file_suffix = "features-rolling"  # Examples: features-rolling, features-rolling-scores, predictions
    simulation_start = 263519   # Good start is 2019-06-01 - after it we have stable movement
    simulation_end = -0  # After 2020-11-01 there is sharp growth which we might want to exclude


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    freq = "1m"
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"])
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return
    out_path = Path(App.config["data_folder"])
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

    #
    # Load data with (rolling) label point-wise predictions
    #
    print(f"Loading data with label (rolling) predict scores from input file...")
    start_dt = datetime.now()

    in_file_name = f"{symbol}-{freq}-{P.predict_file_suffix}.csv"
    in_path = data_path / in_file_name
    if not in_path.exists():
        print(f"ERROR: Input file does not exist: {in_path}")
        return

    in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)

    print(f"Predictions loaded. Length: {len(in_df)}. Width: {len(in_df.columns)}")

    #
    # Compute final score (as average over different predictions)
    # This function adds "score" column
    # Important: we need to use the same final score as later in the service
    #
    in_df = generate_score_high_low(in_df, P.feature_sets)

    print(f"Score column generated.")

    #
    # Select data
    #

    # Selecting only needed rows increases performance in several times (~4 times faster)
    use_forecast_score = False
    if use_forecast_score:
        in_df = in_df[["timestamp", "high", "low", "close", "score", "score_forecast_1"]]
    else:
        in_df = in_df[["timestamp", "high", "low", "close", "score"]]

    # Select the necessary interval of data
    if not P.simulation_start:
        P.simulation_start = 0
    if not P.simulation_end:
        P.simulation_end = len(in_df)
    elif P.simulation_end < 0:
        P.simulation_end = len(in_df) + P.simulation_end

    in_df = in_df.iloc[P.simulation_start:P.simulation_end]

    #
    # Loop over all trade hyper-models (possible threshold combinations)
    #
    grid = ParameterGrid(grid_signals)
    models = list(grid)  # List of model dicts
    performances = []
    for i, model in enumerate(models):
        # Set parameters of the model

        start_dt = datetime.now()
        # ---
        performance = simulated_trade_performance(in_df, model)
        # ---
        elapsed = datetime.now() - start_dt
        print(f"Finished simulation {i} / {len(models)} in {elapsed.total_seconds():.1f} seconds.")

        performances.append(performance)

    #
    # Post-process: sort and filter
    #

    # Column names
    model_keys = models[0].keys()
    performance_keys = performances[0].keys()
    header_str = ",".join(list(model_keys) + list(performance_keys))

    lines = []
    for i, model in enumerate(models):
        model_values = [f"{v:.3f}" for v in model.values()]
        performance_values = [f"{v:.2f}" for v in performances[i].values()]
        line_str = ",".join(model_values + performance_values)
        lines.append(line_str)

    #
    # Store simulation parameters and performance
    #
    out_file_name = f"{symbol}-{freq}-signals.txt"
    out_file = (out_path / out_file_name).resolve()

    if out_file.is_file():
        add_header = False
    else:
        add_header = True
    with open(out_file, "a+") as f:
        if add_header:
            f.write(header_str + "\n")
        #f.writelines(lines)
        f.write("\n".join(lines))
        f.write("\n")

    elapsed = datetime.now() - start_dt
    print(f"Finished in {int(elapsed.total_seconds())} seconds.")


def simulated_trade_performance(df, model: dict):
    """
    It will use 1.0 as initial trade amount in USD.
    Overall performance will be the end amount with respect to the initial one.
    It will use the whole data set from start to end.

    Stragegy 1 (non-cumulative): Always use 1.0 to enter market (for buying) independent of the available (earned or lost) funds.
    Strategy 2 (cumulative): Use all currently available funds for trade

    :param df:
    :param model:
    :return: Performance record
    """
    #
    # Model parameters
    #
    buy_threshold = float(model.get("buy_threshold"))
    sell_threshold = float(model.get("sell_threshold"))
    transaction_fee = float(model.get("transaction_fee"))
    transaction_price_adjustment = float(model.get("transaction_price_adjustment"))

    performance_weight = model.get("performance_weight")

    #
    # Statistics of the performance run
    #

    # All transactions will be collected in this list for later analysis
    transactions = []  # List of dicts like dict(i=23, is_forced_sell=False, profit=-0.123)

    # How many signals independent of mode and execution
    buy_signal_count = 0
    sell_signal_count = 0

    #
    # Main loop over trade sessions
    #
    i = 0
    is_buy_mode = True
    for row in df.itertuples(index=True, name="Row"):
        i += 1
        transaction_weight = 1.0 + i * (performance_weight - 1.0) / 525_600  # Increases in time

        # Current market parameters
        close_price = row.close
        if not close_price:  # Missing data
            continue
        high_price = row.high
        low_price = row.high
        timestamp = row.timestamp

        score = row.score
        if not score:  # Missing data
            continue

        use_forecast_score = False
        if use_forecast_score:
            score_forecast_1 = row.score_forecast_1

        #
        # Apply model parameters and generate buy/sell (enter/exit) signal
        #
        previous_transaction = transactions[-1] if len(transactions) > 0 else None
        previous_price = previous_transaction["price"] if previous_transaction else None
        profit = (close_price - previous_price) if previous_price else None

        if score > buy_threshold:  # score > buy_threshold
            buy_signal_count += 1

            if is_buy_mode:  # Buy mode. Enter market by buying BTC
                transaction = dict(
                    side="BUY",
                    price=close_price, quantity=1.0,
                    profit=profit,  # Lower (negative) is better
                    timestamp=timestamp, row=i, weight=transaction_weight,
                )
                transactions.append(transaction)
                is_buy_mode = False

        elif score < sell_threshold:  # score < sell_threshold
            sell_signal_count += 1

            if not is_buy_mode:  # Sell mode. Exit market by selling BTC
                transaction = dict(
                    side="SELL",
                    price=close_price, quantity=1.0,
                    profit=profit,  # Higher (positive) is better
                    timestamp=timestamp, row=i, weight=transaction_weight,
                )
                transactions.append(transaction)
                is_buy_mode = True

        else:
            continue  # No signal. Just wait

    #
    # Remove last transaction if not filled
    #
    if len(transactions) <= 1:
        return {}

    if transactions[-1]["side"] == "BUY":
        del transactions[-1]

    assert len(transactions) % 2 == 0

    #
    # Compute performance parameters from the list of transactions
    #

    sell_transactions = [t for t in transactions if t["side"] == "SELL"]
    sell_profits = [t["profit"] for t in sell_transactions]
    sell_t_count = len(sell_transactions)
    no_months = len(df) / 43_920

    # KPIs
    sell_t_per_month = sell_t_count / no_months

    profit_sum = np.nansum(sell_profits)
    profit_month = profit_sum / no_months
    profit_avg = np.nanmean(sell_profits)
    profit_std = np.nanstd(sell_profits)

    # Percentage of profitable
    profitable_percent = len([t for t in sell_transactions if t["profit"] > 0]) / sell_t_count
    # TODO: Average length (time from buy to sell, that is, difference between sell and previous buy)

    performance = dict(
        profit_sum=profit_sum,
        profit_avg=profit_avg,
        profit_std=profit_std,
        profitable_percent=profitable_percent,

        sell_t_per_month=sell_t_per_month,
        profit_month=profit_month,
    )

    return performance


if __name__ == '__main__':
    main()
