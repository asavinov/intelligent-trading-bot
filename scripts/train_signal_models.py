from pathlib import Path
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay)
from sklearn.model_selection import ParameterGrid

from service.App import *
from common.classifiers import *

from common.label_generation_topbot import *
from common.signal_generation import *

"""
Input data:
This script assumes the existence of label prediction scores for a list of labels 
which is computed by some other script (train predict models or (better) rolling predictions).
It also uses the real prices in order to determine if the orders are executed or not 
(currently close prices but it is better to use high and low prices).

Purpose:
The script uses some signal parameters which determine whether to sell or buy based on the current 
label prediction scores. It simulates trade for such signal parameters by running through 
the whole data set. For each such signal parameters, it determines the trade performance 
(overall profit or loss). It then does such simulations for all defined signal parameters
and finally chooses the best performing parameters. These parameters can be then used for real trades.

Notes:
- The simulation is based on some aggregation function which computes the final signal from
multiple label prediction scores. There could be different aggregation logics for example 
finding average value or using pre-defined thresholds or even training some kind of model 
like decision trees
- The signal (aggregation) function assumes that there two kinds of labels: positive (indicating that
the price will go up) and negative (indicating that the price will go down). The are accordingly
stored in two lists in the configuration 
- Tthe script should work with both batch predictions and (better) rolling predictions by
assuming only the necessary columns for predicted label scores and trade columns (close price)
"""

class P:
    in_nrows = 100_000_000

    start_index = 200_000
    end_index = None
    # TODO: currently not used
    simulation_start = 263519   # Good start is 2019-06-01 - after it we have stable movement
    simulation_end = -0  # After 2020-11-01 there is sharp growth which we might want to exclude

    # True if buy and sell hyper-parameters are equal
    # Only buy parameters will be used and sell parameters will be ignored
    buy_sell_equal = False

    # Haw many best performing parameters from the grid to store
    topn_to_store = 20

#
# Specify the ranges of signal hyper-parameters
#
grid_signals = [
    {
        "buy_point_threshold": [None], # + np.arange(0.02, 0.20, 0.01).tolist(),  # None means do not use
        "buy_window": [3],  # [5, 6, 7, 8, 9, 10, 11, 12]
        # [0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65],
        # 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55],
        # [0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65
        # [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]
        "buy_signal_threshold": [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6],
        # 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010
        "buy_slope_threshold": [None],

        # If two groups are equal, then these values are ignored
        "sell_point_threshold": [None], # + np.arange(0.02, 0.20, 0.01).tolist()
        "sell_window": [3],
        "sell_signal_threshold": [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6],
        "sell_slope_threshold": [None],

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
    """
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return
    out_path = Path(App.config["data_folder"]) / symbol
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

    #
    # Load data with (rolling) label point-wise predictions
    #
    file_path = (data_path / App.config.get("predict_file_name")).with_suffix(".csv")
    if not file_path.exists():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading predictions from input file: {file_path}")
    df = pd.read_csv(file_path, parse_dates=[time_column], nrows=P.in_nrows)
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
        grid_signals[0]["sell_slope_threshold"] = [None]

    performances = list()
    for model in tqdm(ParameterGrid(grid_signals)):
        #
        # If equal parameters, then use the first group
        #
        if P.buy_sell_equal:
            model["sell_point_threshold"] = model["buy_point_threshold"]
            model["sell_window"] = model["buy_window"]
            model["sell_signal_threshold"] = model["buy_signal_threshold"]
            model["sell_slope_threshold"] = model["buy_slope_threshold"]

        #
        # Generate two boolean signal columns from two groups of point-wise score columns using signal model
        # Exactly same procedure has to be used in on-line signal generation by the service
        # It should work for all kinds of point-wise predictions: high-low, top-bottom etc.
        # TODO: We need to encapsulate this step so that it can be re-used in the same form in the service
        #   For example, introduce a higher level function which takes all possible hyper-parameters and then makes these calls and returns two binary signal columns
        #   We need to introduce a signal model. In the service, we transform two groups to two signal scores, and then apply final trade threshold and notification threshold.

        # Produce boolean signal (buy and sell) columns from the current patience parameters
        aggregate_score(df, 'buy_score_column', [buy_score_column_avg], model.get("buy_point_threshold"), model.get("buy_window"))
        aggregate_score(df, 'sell_score_column', [sell_score_column_avg], model.get("sell_point_threshold"), model.get("sell_window"))

        if model.get("combine") == "relative":
            combine_scores_relative(df, 'buy_score_column', 'sell_score_column', 'buy_score_column', 'sell_score_column')
        elif model.get("combine") == "difference":
            combine_scores_difference(df, 'buy_score_column', 'sell_score_column', 'buy_score_column', 'sell_score_column')

        #
        # Experimental. Compute slope of the numeric score over model.get("buy_window") and model.get("sell_window")
        #
        from scipy import stats
        from sklearn import linear_model
        def linear_regr_fn(X):
            """
            Given a Series, fit a linear regression model and return its slope interpreted as a trend.
            The sequence of values in X must correspond to increasing time in order for the trend to make sense.
            """
            X_array = np.asarray(range(len(X)))
            y_array = X
            if np.isnan(y_array).any():
                nans = ~np.isnan(y_array)
                X_array = X_array[nans]
                y_array = y_array[nans]

            #X_array = X_array.reshape(-1, 1)  # Make matrix
            #model = linear_model.LinearRegression()
            #model.fit(X_array, y_array)
            #slope = model.coef_[0]

            slope, intercept, r, p, se = stats.linregress(X_array, y_array)

            return slope

        #if 'buy_score_slope' not in df.columns:
        #    w = 10  #model.get("buy_window")
        #    df['buy_score_slope'] = df['buy_score_column'].rolling(window=w, min_periods=max(1, w // 2)).apply(linear_regr_fn, raw=True)
        #    w = 10  #model.get("sell_window")
        #    df['sell_score_slope'] = df['sell_score_column'].rolling(window=w, min_periods=max(1, w // 2)).apply(linear_regr_fn, raw=True)

        # Final boolean signal using final thresholds
        df['buy_signal_column'] = df['buy_score_column'] >= model.get("buy_signal_threshold")
        df['sell_signal_column'] = df['sell_score_column'] >= model.get("sell_signal_threshold")

        # High score and low slope
        #df['buy_signal_column'] = (df['buy_score_column'] >= model.get("buy_signal_threshold")) & (df['buy_score_slope'].abs() <= model.get("buy_slope_threshold"))
        #df['sell_signal_column'] = (df['sell_score_column'] >= model.get("sell_signal_threshold")) & (df['sell_score_slope'].abs() <= model.get("sell_slope_threshold"))

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
    performances = sorted(performances, key=lambda x: x['performance']['profit_per_month'], reverse=True)
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
    out_path = (out_path / App.config.get("signal_file_name")).with_suffix(".txt").resolve()

    if out_path.is_file():
        add_header = False
    else:
        add_header = True
    with open(out_path, "a+") as f:
        if add_header:
            f.write(",".join(keys) + "\n")
        #f.writelines(lines)
        f.write("\n".join(lines) + "\n\n")

    print(f"Simulation results stored in: {out_path}. Lines: {len(lines)}.")

    elapsed = datetime.now() - now
    print(f"Finished simulation in {str(elapsed).split('.')[0]}")


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
        profit_per_month=profit / (len(df) / minutes_in_month),
        profit_per_transaction=profit / transactions if transactions else 0.0,
        profitable=profitable / transactions if transactions else 0.0,
        transactions_per_month=transactions / (len(df) / minutes_in_month),
        #transactions=transactions,
        #profit=profit,
    )

    return performance, long_performance, short_performance


if __name__ == '__main__':
    main()
