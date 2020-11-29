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

from common.utils import *

"""
By signal generation model we mean simple rules with some thresholds as parameters.
Training such a model is performed by brute force by training out all parameters and finding their peformance using back testing.
Back testing is performed using an available feature matrix with all rolling predictions.
The output is an ordered list of top performing threshold-based signal generation models.

One approach is to use two parameters which determine entry and exit thresholds for a position (BTC). They need not be symmetric.
These parameters are compared with the final prediction in [-1, +1] which expresses our consolidated future trend. 
We might introduce one or two additional parameters to eliminate jitter if it happens. 
Another way to reduce jitter is to smooth the final score so that it simply does not change quickly.
Yet, if we entry and exist thresholds are significantly different then jitter should not happen.
This approach does not try to optimize each transaction by searching individual best (but rare) entry-exit point.
Instead, it tries to find maximums and minimums by switching the side at these points.
We switch side independent of any other factors - simply because of the future trend and more opportunities on this new side.

Performance is determined by how much can be earned with these parameters.
We follow the strategy of switching the side or enter-exit strategy.
Note that this strategy does not have time outs or main asset.
Instead of main asset, we can use non-symmetric entry-exit thresholds which result in having higher
probability of one side/asset than the other.

We can also assign higher weight to recent performance.
Another adjustment is transaction fee which punishes models with many transactions.

Another, original, strategy is to have USD as main asset and then enter (buy BTC) only with 
the purpose to earn while exiting. Note that this strategy could be also modelled 
if the two thresholds are not symmetric.

NEXT:
- Since we are going to use only kline (no futures), generate rolling predictions for longer period
- Explore how profit depends on month: downward tren (Feb-March) vs. other months.
  - Run back testing on only summer months
- DONE: smooth score and check if the performance is better. simply run same grid with different smooth factors (different definitions of score column).
- alternative to smoothing, generate signal if 2 or more previous values are all higher/lower than threshold
- OR, take generate signal when 2nd time crosses the threshold

- simple extrapolation of score (forecast)

- hybrid strategy: instead of simply waiting for signal and change point, we can introduce stop loss or take profit signals.
These signals will allow us to take profit where we see it and it is large rather than wait for something in future.
In other words, such signals are generated from reality (we can earn already now) rather then future.
One way to impelment it, is to use this special kind of order (take profit) which will be executed automatically.
Yet, we need to model this logic manually.
"""

"""
BEST:
without fees or weights (simple mode): 
kf: (0.07, -0.2), profit 8600, per month 960, avg 33, percent 41%, #/month 29
k: (0.14, -0.17), profit 9222, per month 1030, avg 46, percent 41%, #/month 22
k: (0.135, -0.155), profit 9588, per month 1071, avg 44, percent 42%, # 24
+++ k: (0.135, -0.2), profit 9285, per month 1037, avg 53, percent 44%, # 19
k2: (), profit 8180, per month 914, 19, percent 38, 47
k3: (0.13,-0.13), profit 8800, per month 985, avg 39, percent 40%, #/month 25
k5: (0.13,-0.13), profit 8800, per month 985, avg 39, percent 40, # 25 - ???
k10: (0, -0.13), profit 8920, per month 996, avg 19, percent 36, #52
f: (0.07, -0.16), profit 5360, per month 600, avg 8.3, percent 39%, #/month 72

k_nn: (0.18, -0.04), profit 8521, per month 951, avg 10.6, 39%, #90 - many low profit transactions
kf_nn: (0, -0.2), profit 8900, per month 995, avg 11.5, 36%, #86 
kf_lc: (0.03, -0.2), profit 7318, per month 817, avg 18.5, 38%, #44
kf_gb: (0, -0.12), profit 6338, per month 708, avg 9.11, 41%, #77

score as difference (high_k-low_k):
(0.03, -0.05), profit 8388, per month 937, avg 30, 45%, #30

"""

grid_signals = [
    {
        "entry_threshold": [
            0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
            #0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17,
        ],  # Buy BTC when higher than this value
        "exit_threshold": [
            -0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09,
            -0.10, -0.11, -0.12, -0.13, -0.14, -0.15, -0.16, -0.17, -0.18, -0.19, -0.20,
            #-0.14, -0.145, -0.15, -0.155, -0.16, -0.165, -0.17, -0.175, -0.18, -0.185, -0.19, -0.195, -0.20,
        ],  # Sell BTC when lower than this value

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
    in_path_name = r"C:\DATA2\BITCOIN\GENERATED"
    #in_file_name = r"_BTCUSDT-1m-rolling-predictions-no-weights.csv"
    #in_file_name = r"_BTCUSDT-1m-rolling-predictions-with-weights.csv"
    in_file_name = r"BTCUSDT-1m-features-rolling.csv"
    in_nrows = 100_000_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"_BTCUSDT-1m-signals"

    simulation_start = 10  # 10 Default 0, 129_600 (1.5.)
    simulation_end = -10  # -10, -43_199 (1.10.) Default till end of input data. Negative value is shift from the end

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
    elif P.in_file_name.endswith(".parquet"):
        in_df = pd.read_parquet(in_path)
    else:
        print(f"ERROR: Unknown input file extension. Only csv and parquet are supported.")

    #
    # Compute final score (as average over different predictions)
    #
    # high kline: 3 algorithms for all 3 levels
    in_df["high_k"] = \
        in_df["high_10_k_gb"] + in_df["high_10_k_nn"] + in_df["high_10_k_lc"] + \
        in_df["high_15_k_gb"] + in_df["high_15_k_nn"] + in_df["high_15_k_lc"] + \
        in_df["high_20_k_gb"] + in_df["high_20_k_nn"] + in_df["high_20_k_lc"]
    in_df["high_k"] /= 9

    # low kline: 3 algorithms for all 3 levels
    in_df["low_k"] = \
        in_df["low_10_k_gb"] + in_df["low_10_k_nn"] + in_df["low_10_k_lc"] + \
        in_df["low_15_k_gb"] + in_df["low_15_k_nn"] + in_df["low_15_k_lc"] + \
        in_df["low_20_k_gb"] + in_df["low_20_k_nn"] + in_df["low_20_k_lc"]
    in_df["low_k"] /= 9

    # high futur: 3 algorithms for all 3 levels
    in_df["high_f"] = \
        in_df["high_10_f_gb"] + in_df["high_10_f_nn"] + in_df["high_10_f_lc"] + \
        in_df["high_15_f_gb"] + in_df["high_15_f_nn"] + in_df["high_15_f_lc"] + \
        in_df["high_20_f_gb"] + in_df["high_20_f_nn"] + in_df["high_20_f_lc"]
    in_df["high_f"] /= 9

    # low kline: 3 algorithms for all 3 levels
    in_df["low_f"] = \
        in_df["low_10_f_gb"] + in_df["low_10_f_nn"] + in_df["low_10_f_lc"] + \
        in_df["low_15_f_gb"] + in_df["low_15_f_nn"] + in_df["low_15_f_lc"] + \
        in_df["low_20_f_gb"] + in_df["low_20_f_nn"] + in_df["low_20_f_lc"]
    in_df["low_f"] /= 9

    # By algorithm
    in_df["high_k_gb"] = (in_df["high_10_k_gb"] + in_df["high_15_k_gb"] + in_df["high_20_k_gb"]) / 3
    in_df["high_f_gb"] = (in_df["high_10_f_gb"] + in_df["high_15_f_gb"] + in_df["high_20_f_gb"]) / 3

    in_df["low_k_gb"] = (in_df["low_10_k_gb"] + in_df["low_15_k_gb"] + in_df["low_20_k_gb"]) / 3
    in_df["low_f_gb"] = (in_df["low_10_f_gb"] + in_df["low_15_f_gb"] + in_df["low_20_f_gb"]) / 3

    # High and low
    # Both k and f
    #in_df["high"] = (in_df["high_k"] + in_df["high_f"]) / 2
    #in_df["low"] = (in_df["low_k"] + in_df["low_f"]) / 2

    # Only k
    in_df["high"] = (in_df["high_k"]) / 1
    in_df["low"] = (in_df["low_k"]) / 1

    #in_df["high"] = (in_df["high_k_gb"]) / 1
    #in_df["low"] = (in_df["low_k_gb"]) / 1

    # Final score: proportion to the sum
    high_and_low = in_df["high"] + in_df["low"]
    in_df["score"] = ((in_df["high"] / high_and_low) * 2) - 1.0  # in [-1, +1]

    # Final score: abs difference betwee high and low (scaled to [-1,+1] maybe)
    #in_df["score"] = in_df["high"] - in_df["low"]
    from sklearn.preprocessing import StandardScaler
    #in_df["score"] = StandardScaler().fit_transform(in_df["score"])

    #in_df["score"] = in_df["score"].rolling(window=10, min_periods=1).apply(np.nanmean)

    #
    # Select data
    #

    # Selecting only needed rows increases performance in several times (~4 times faster)
    in_df = in_df[["timestamp", "high", "low", "close", "score",]]

    # Select the necessary interval of data
    if not P.simulation_start:
        P.simulation_start = 0
    if not P.simulation_end:
        P.simulation_end = len(in_df)
    elif P.simulation_end < 0:
        P.simulation_end = len(in_df) + P.simulation_end

    in_df = in_df.iloc[P.simulation_start:P.simulation_end]

    #
    # Loop on all trade hyper-models
    #
    grid = ParameterGrid(grid_signals)
    models = list(grid)  # List of model dicts
    performances = []
    for i, model in enumerate(models):
        # Set parameters of the model

        start_dt = datetime.now()
        performance = simulate_trade(in_df, model)
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
        f.write("\n")

    pass


def simulate_trade(df, model: dict):
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
    entry_threshold = float(model.get("entry_threshold"))
    exit_threshold = float(model.get("exit_threshold"))
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
        high_price = row.high
        low_price = row.high
        timestamp = row.timestamp

        score = row.score

        #
        # Table has missing data
        #
        if not (close_price and score):
            continue

        #
        # Apply model parameters and generate buy/sell (enter/exit) signal
        #
        previous_transaction = transactions[-1] if len(transactions) > 0 else None
        previous_price = previous_transaction["price"] if previous_transaction else None
        profit = (close_price - previous_price) if previous_price else None

        if score > entry_threshold:
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

        elif score < exit_threshold:
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
    main(sys.argv[1:])
