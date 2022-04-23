from pathlib import Path
from typing import Union

import click
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay)
from sklearn.model_selection import ParameterGrid

import seaborn as sns
import matplotlib.pyplot as plt

from service.App import *
from common.classifiers import *

from common.label_generation_top_bot import *
from common.signal_generation import *

"""
Input data:
- source columns: close, max, min (for trade simulation)
- point-wise predictions (label-feature_set-algorithm) to produce trade signals using hyper-parameters
Algorithms:
- Standard algorithm for computing trade signal from point-wise predictions (normally aggregation) which uses hyper-parameters
- In loop over all aggregation hyper-parameters, simulate trade over the data range by finding the final performance (profit etc.)
Output:
- Top best hyper-parameters

Notes:
- the script should work with both batch predictions and rolling predictions (better) by
assuming only the necessary input columns.
- the script should support different kinds of aggregation function (it should be easy to replace) 
and their corresponding hyper-parameters (e.g., top-bot and high-low).

TODO:
- DONE. Define generic aggregation functions for transforming point-wise scores to buy-sell score/signals with hyper-parameters
Put these functions in some common module signal_generation::generate_score etc.
- Implement/re-work the main loop and the logic of performance computations. 
- Store output including better performance computation with transaction costs adjustments etc.
- Integrate/use the basic aggregation functions in on-line processing (analyzer) to ensure equivalent results
  Add parameterization where relevant so that it is easy to vary them in future (where they are expected to change, e.g., after re-training and re-optimization)

General pipeline:
- Define programmatically all possible features and all possible labels and produce a feature matrix
- Select various features and labels and find best point-wise performance by varying algorithm hyper-parameters and feature/label parameters
- With fixed feature/label definitions (optimized for best point-wise performance), vary trade signal parameters to maximaze trade performance
- Use the feature/label defintions and trade signal parameters for on-line analysis (guarantee the same features/labels and all hyper-parameters)

Train using extremum labels or prediction  labels and make point-wise predictions.
Define point-wise score aggregation/post-processing parameter space: patience etc.
For all these score aggregation parameters, compute their interval performance score and find best parameters.
Note that it should work for any kind of training and prediction procedure including our old predictions.
That is, it should be possible to take our old rolling_predictions point scores, and feed them into this procedure and find best aggregation parameters.

Repeat this grid search procedure for different definitions of top-bottom labels (level and tolerance). 
Find best level-tolerance which generate best performance for certain performance score parameters.
"""

class P:
    in_nrows = 100_000_000

    start_index = 0
    end_index = None
    # TODO
    simulation_start = 263519   # Good start is 2019-06-01 - after it we have stable movement
    simulation_end = -0  # After 2020-11-01 there is sharp growth which we might want to exclude

    # True if buy and sell hyper-parameters are equal
    # Only buy parameters will be used and sell parameters will be ignored
    buy_sell_equal = True

    topn_to_store = 20

#
# Specify the ranges of signal hyper-parameters
#
grid_signals = [
    {
        "buy_point_threshold": [None], # + np.arange(0.02, 0.20, 0.01).tolist(),  # None means do not use
        "buy_window": [3],  # [5, 6, 7, 8, 9, 10, 11, 12]
        "buy_signal_threshold": [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65],  # 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55],  # [0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65

        # If two groups are equal, then these values are ignored
        "sell_point_threshold": [None], # + np.arange(0.02, 0.20, 0.01).tolist()
        "sell_window": [3],
        "sell_signal_threshold": [0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58],

        "combine": ["no_combine"],  # "no_combine", "difference" (same as no combine or better), "relative" (rather bad)
    },
]


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    The goal is to find how good interval scores can be by performing grid search through
    all aggregation/patience hyper-parameters which generate buy-sell signals on interval level.

    Here we measure performance of trade using top-bottom scores generated using specified aggregation
    parameters (which are searched through a grid). Here lables with true records are not needed.
    In contrast, in another (above) function we do the same search but measure interval-score,
    that is, how many intervals are true and false (either bot or bottom) by comparing with true label.

    General purpose and assumptions. Load any file with two groups of point-wise prediction scores:
    buy score and sell columns. The file must also have columns for trade simulation like close price.
    It can be batch prediction file (one train model and one prediction result) or rolling predictions
    (multiple sequential trains and predictions).
    The script will convert these two buy-sell column groups to boolean buy-sell signals by using
    signal generation hyper-parameters, and then apply trade simulation by computing its overall
    performance. This is done for all simulation parameters from the grid. The results for all
    simulation parameters and their performance are stored in the output file.

    What we do:
        - Load prediction file (batch or rolling) with close/high/low prices for simulation,
        and point-wise predictions for generating signals
        - Read or pre-define input columns we need (like close price) and select them
        - Read point-wise prediction labels we want to use from App.config.
        Maybe split them into buy and sell prediction labels
        - Define explicitly a grid of signal hyper-parameters:
            - equal parameters for buy and sell - will be used for draft search
            - different parameters for buy and sell - will be used for fine-tuning
        - Implementation
            - Main loop over hyper-parameters
            - Produce two columns (and/or one combined column) with signal: first two continuous signals, and second apply final threshold to get binary signals
            - Call performance function which relies on available data: close/high/low and signal column(s)
        - Challenges and use cases:
            - Ideally, it should work for both high-low and top-bot labels and predictions
            - Ideally, it should work for only close price (top-bot) and high/low prices (for high-low use case where we check if max price was higher then we need)
            - Two cases: equal hyper-parameters for buy-sell and different hyper-parameters
            - Two inputs: prediction file and rolling predictions file
            - This implementation should be maximum compatible with the first one (and ideally replace it in future) so follow its structure


    # Follow up:
    # 1) long is more profitable, maybe because of general trend, so exclude the general trend
    # 2) Performance should include number of transactions and maybe price per transaction (deduce)
    # 3) Performance should return # negative transactions
    # 4) Our best results detect the same number of signals as we have in top-bottoms in labels (ideal performance).
    #    However, our performance is significantly lower.
    #    1. Generate table of signals (one row is one signal with timestamp, close price, maybe score etc.)
    #    2. Compare it with labels (top-bottom intervals) which are also represented as a table.
    #    How our signals are positioned relative to intervals?
    #    Take sell signals and position them along with top intervals. How many of them are within intervals?
    #    Function for generating a list of buy-sell signals (part of performance function). Then take these timestamps, and compare with top intervals in label column.
    #    Function for generating interval list: one row is one interval with start/end ts, level etc.
    # 5) Other options:
    # - average several point-wise prediction scores like (10_1+10_2+10_3) / 3
    # - single combined score from top and bot: top_10_1 and bot_10_1 and then use the combined score with max and min thresholds
    """
    load_config(config_file)

    freq = "1m"
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return
    out_path = Path(App.config["data_folder"]) / symbol
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

    config_file_modifier = App.config.get("config_file_modifier")
    config_file_modifier = ("-" + config_file_modifier) if config_file_modifier else ""

    #
    # Load data with (rolling) label point-wise predictions
    #
    in_file_suffix = App.config.get("predict_file_modifier")

    in_file_name = f"{in_file_suffix}{config_file_modifier}.csv"
    in_path = data_path / in_file_name
    if not in_path.exists():
        print(f"ERROR: Input file does not exist: {in_path}")
        return

    print(f"Loading predictions from input file: {in_path}")
    start_dt = datetime.now()
    df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)
    print(f"Predictions loaded. Length: {len(df)}. Width: {len(df.columns)}")

    # Limit size according to parameters start_index end_index
    df = df.iloc[P.start_index:P.end_index]
    df = df.reset_index()

    #
    # Find maximum performance possible based on true labels only
    #
    # Best parameters (just to compute for known parameters)
    #df['buy_signal_column'] = score_to_signal(df[bot_score_column], None, 5, 0.09)
    #df['sell_signal_column'] = score_to_signal(df[top_score_column], None, 10, 0.064)
    #performance_long, performance_short, long_count, short_count, long_profitable, short_profitable, longs, shorts = performance_score(df, 'sell_signal_column', 'buy_signal_column', 'close')
    # TODO: Save maximum performance in output file or print it (use as a reference)

    # Maximum possible on labels themselves
    #performance_long, performance_short, long_count, short_count, long_profitable, short_profitable, longs, shorts = performance_score(df, 'top10_2', 'bot10_2', 'close')

    #
    # Optimization: Compute averages which will be the same for all hyper-parameters
    #
    buy_labels = App.config["buy_labels"]
    sell_labels = App.config["sell_labels"]
    # TODO: Check the existence of these labels in the input file
    #   Check also the existence of some necessary columns like close

    buy_score_column_avg = 'buy_score_column_avg'
    sell_score_column_avg = 'sell_score_column_avg'

    df[buy_score_column_avg] = df[buy_labels].mean(skipna=True, axis=1)
    df[sell_score_column_avg] = df[sell_labels].mean(skipna=True, axis=1)

    if P.buy_sell_equal:
        grid_signals[0]["sell_point_threshold"] = [None]
        grid_signals[0]["sell_window"] = [None]
        grid_signals[0]["sell_signal_threshold"] = [None]

    performances = list()
    for model in tqdm(ParameterGrid(grid_signals)):
        #
        # If equal parameters, then use the first group
        #
        if P.buy_sell_equal:
            model["sell_point_threshold"] = model["buy_point_threshold"]
            model["sell_window"] = model["buy_window"]
            model["sell_signal_threshold"] = model["buy_signal_threshold"]

        #
        # Generate two boolean signal columns from two groups of point-wise score columns using signal model
        # Exactly same procedure has to be used in on-line signal generation by the service
        # It should work for all kinds of point-wise predictions: high-low, top-bottom etc.
        # TODO: We need to encapsulate this step so that it can be re-used in the same form in the service
        #   For example, introduce a higher level function which takes all possible hyper-parameters and then makes these calls and returns two binary signal columns
        #   We need to introduce a signal model. In the service, we transform two groups to two signal scores, and then apply final trade threshold and notification threshold.

        # Produce boolean signal (buy and sell) columns from the current patience parameters
        aggregate_score(df, [buy_score_column_avg], 'buy_score_column', model.get("buy_point_threshold"), model.get("buy_window"))
        aggregate_score(df, [sell_score_column_avg], 'sell_score_column', model.get("sell_point_threshold"), model.get("sell_window"))

        if model.get("combine") == "relative":
            combine_scores_relative(df, 'buy_score_column', 'sell_score_column', 'buy_score_column', 'sell_score_column')
        elif model.get("combine") == "difference":
            combine_scores_difference(df, 'buy_score_column', 'sell_score_column', 'buy_score_column', 'sell_score_column')

        # Final boolean signal using final thresholds
        df['buy_signal_column'] = df['buy_score_column'] >= model.get("buy_signal_threshold")
        df['sell_signal_column'] = df['sell_score_column'] >= model.get("sell_signal_threshold")

        #
        # Simulate trade using close price and two boolean signals
        # Add a pair of two dicts: performance dict and model parameters dict
        #
        performance, long_performance, short_performance = \
            simulated_trade_performance(df, 'sell_signal_column', 'buy_signal_column', 'close')

        performances.append(dict(
            model=model,
            performance=performance,
            long_performance=long_performance,
            short_performance=short_performance
        ))

    #
    # Flatten
    #

    # Sort
    performances = sorted(performances, key=lambda x: x['performance']['profit'], reverse=True)
    performances = performances[:P.topn_to_store]

    # Column names (from one record)
    keys = list(performances[0]['model'].keys()) + \
           list(performances[0]['performance'].keys()) + \
           list(performances[0]['long_performance'].keys()) + \
           list(performances[0]['short_performance'].keys())

    lines = []
    for p in performances:
        record = list(p['model'].values()) + \
                 list(p['performance'].values()) + \
                 list(p['long_performance'].values()) + \
                 list(p['short_performance'].values())
        record = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in record]
        record_str = ",".join(record)
        lines.append(record_str)

    #
    # Store simulation parameters and performance
    #
    out_file_suffix = App.config.get("signal_file_modifier")

    out_file_name = f"{out_file_suffix}{config_file_modifier}.txt"
    out_path = (out_path / out_file_name).resolve()

    if out_path.is_file():
        add_header = False
    else:
        add_header = True
    with open(out_path, "a+") as f:
        if add_header:
            f.write(",".join(keys) + "\n")
        #f.writelines(lines)
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Simulation results stored in: {out_path}. Lines: {len(lines)}.")

    elapsed = datetime.now() - start_dt
    print(f"Finished simulation in {int(elapsed.total_seconds())} seconds.")


def simulated_trade_performance(df, sell_signal_column, buy_signal_column, price_column):
    """
    top_score_column: boolean, true if top is reached - sell signal
    bot_score_column: boolean, true if bottom is reached - buy signal
    price_column: numeric price for computing profit

    return performance: tuple, long and short performance as a sum of differences between two transactions

    The functions switches the mode and searches for the very first signal of the opposite score.
    When found, it again switches the mode and searches for the very first signal of the opposite score.

    Essentially, it is one pass of trade simulation with concrete parameters.
    """
    is_buy_mode = True

    long_profit = 0
    long_transactions = 0
    long_profitable = 0
    longs = list()

    short_profit = 0
    short_transactions = 0
    short_profitable = 0
    shorts = list()

    # The order of columns is important for itertuples
    df = df[[sell_signal_column, buy_signal_column, price_column]]
    for (index, top_score, bot_score, price) in df.itertuples(name=None):
        if is_buy_mode:
            # Check if minimum price
            if bot_score:
                profit = longs[-1][2] - price if len(longs) > 0 else 0
                short_profit += profit
                short_transactions += 1
                if profit > 0:
                    short_profitable += 1
                shorts.append((index, is_buy_mode, price, profit))  # Bought
                is_buy_mode = False
        else:
            # Check if maximum price
            if top_score:
                profit = price - shorts[-1][2] if len(shorts) > 0 else 0
                long_profit += profit
                long_transactions += 1
                if profit > 0:
                    long_profitable += 1
                longs.append((index, is_buy_mode, price, profit))  # Sold
                is_buy_mode = True

    long_performance = dict(
        long_profit=long_profit,
        long_transactions=long_transactions,
        long_profitable=long_profitable / long_transactions if long_transactions else 0.0,
        #longs=longs,
    )
    short_performance = dict(
        short_profit=short_profit,
        short_transactions=short_transactions,
        short_profitable=short_profitable / short_transactions if short_transactions else 0.0,
        #shorts=shorts,
    )

    profit = long_performance['long_profit'] + short_performance['short_profit']
    transactions = long_performance['long_transactions'] + short_performance['short_transactions']
    profitable = long_profitable + short_profitable
    minutes_in_month = 1440 * 30.5
    performance = dict(
        profit=profit,
        transactions=transactions,
        profitable=profitable / transactions if transactions else 0.0,
        profit_per_transaction=profit / transactions if transactions else 0.0,
        profit_per_month=profit / (len(df) / minutes_in_month),
    )

    return performance, long_performance, short_performance


if __name__ == '__main__':
    main()
