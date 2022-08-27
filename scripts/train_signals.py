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

    start_index = 0  # 200_000 for 1m btc
    end_index = None

    # True if buy and sell hyper-parameters are equal
    # Only buy parameters will be used and sell parameters will be ignored
    buy_sell_equal = False

    # Haw many best performing parameters from the grid to store
    topn_to_store = 10

#
# Specify the ranges of signal hyper-parameters
#
grid_signals = [
    {
        "buy_point_threshold": [None], # np.arange(0.01, 0.21, 0.01).tolist(),  # None means do not use
        "buy_window": [5],
        "buy_signal_threshold": np.arange(0.15, 0.30, 0.01).tolist(),
        "buy_slope_threshold": [None],

        # If two groups are equal, then these values are ignored
        "sell_point_threshold": [None],
        "sell_window": [5],
        "sell_signal_threshold": np.arange(0.15, 0.30, 0.01).tolist(),
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
    df = df.reset_index(drop=True)

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
    if set(buy_labels + sell_labels) - set(df.columns):
        missing_labels = list(set(buy_labels + sell_labels) - set(df.columns))
        print(f"ERROR: Some buy/sell labels from config are not present in the input data. Missing labels: {missing_labels}")
        return

    buy_score_column_avg = 'buy_score_column_avg'
    sell_score_column_avg = 'sell_score_column_avg'

    df[buy_score_column_avg] = df[buy_labels].mean(skipna=True, axis=1)
    df[sell_score_column_avg] = df[sell_labels].mean(skipna=True, axis=1)

    if P.buy_sell_equal:
        grid_signals[0]["sell_point_threshold"] = [None]
        grid_signals[0]["sell_window"] = [None]
        grid_signals[0]["sell_signal_threshold"] = [None]
        grid_signals[0]["sell_slope_threshold"] = [None]

    months_in_simulation = (df[time_column].iloc[-1] - df[time_column].iloc[0]) / timedelta(days=30.5)

    performances = list()
    for model in tqdm(ParameterGrid(grid_signals), desc="MODELS"):
        #
        # If equal parameters, then use the first group
        #
        if P.buy_sell_equal:
            model["sell_point_threshold"] = model["buy_point_threshold"]
            model["sell_window"] = model["buy_window"]
            model["sell_signal_threshold"] = model["buy_signal_threshold"]
            model["sell_slope_threshold"] = model["buy_slope_threshold"]

        #
        # Add two pairs of columns: buy_score_column/sell_score_column and buy_signal_column/sell_signal_column
        #
        generate_signal_columns(df, model, buy_score_column_avg, sell_score_column_avg)

        #
        # Simulate trade using close price and two boolean signals
        # Add a pair of two dicts: performance dict and model parameters dict
        #
        performance, long_performance, short_performance = \
            simulated_trade_performance(df, 'sell_signal_column', 'buy_signal_column', 'close')

        # Remove some items. Remove lists of transactions which are not needed
        long_performance.pop('transactions', None)
        short_performance.pop('transactions', None)

        # Add some metrics. Add per month metrics
        performance["profit_percent_per_month"] = performance["profit_percent"] / months_in_simulation
        performance["transaction_no_per_month"] = performance["transaction_no"] / months_in_simulation
        performance["profit_percent_per_transaction"] = performance["profit_percent"] / performance["transaction_no"] if performance["transaction_no"] else 0.0
        performance["profit_per_month"] = performance["profit"] / months_in_simulation

        long_performance["profit_percent_per_month"] = long_performance["profit_percent"] / months_in_simulation
        short_performance["profit_percent_per_month"] = short_performance["profit_percent"] / months_in_simulation

        performances.append(dict(
            model=model,
            performance={k: performance[k] for k in ['profit_percent_per_month', 'profitable', 'profit_percent_per_transaction', 'transaction_no_per_month']},
            long_performance={k: long_performance[k] for k in ['profit_percent_per_month', 'profitable']},
            short_performance={k: short_performance[k] for k in ['profit_percent_per_month', 'profitable']}
        ))

    #
    # Flatten
    #

    # Sort
    performances = sorted(performances, key=lambda x: x['performance']['profit_percent_per_month'], reverse=True)
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
        record = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in record]
        record_str = ",".join(record)
        lines.append(record_str)

    #
    # Store simulation parameters and performance
    #
    out_path = (out_path / App.config.get("signal_models_file_name")).with_suffix(".txt").resolve()

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


if __name__ == '__main__':
    main()
