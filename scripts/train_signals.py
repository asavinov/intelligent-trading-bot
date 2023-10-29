from pathlib import Path
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay)
from sklearn.model_selection import ParameterGrid

from service.App import *
from common.utils import *
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

    train_signal_config = App.config["train_signal_model"]

    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return
    out_path = Path(App.config["data_folder"]) / symbol
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

    #
    # Load data with (rolling) label point-wise predictions and signals generated
    #
    file_path = (data_path / App.config.get("signal_file_name")).with_suffix(".csv")
    if not file_path.exists():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading signals from input file: {file_path}")
    df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    print(f"Signals loaded. Length: {len(df)}. Width: {len(df.columns)}")

    #
    # Limit the source data
    #

    data_start = train_signal_config.get("data_start", 0)
    if isinstance(data_start, str):
        data_start = find_index(df, data_start)
    data_end = train_signal_config.get("data_end", None)
    if isinstance(data_end, str):
        data_end = find_index(df, data_end)

    df = df.iloc[data_start:data_end]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

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

    months_in_simulation = (df[time_column].iloc[-1] - df[time_column].iloc[0]) / timedelta(days=30.5)

    #
    # Load signal train parameters
    #
    parameter_grid = train_signal_config.get("grid")
    direction = train_signal_config.get("direction", "")
    if direction not in ['long', 'short', 'both', '']:
        raise ValueError(f"Unknown value of {direction} in signal train model. Only 'long', 'short' and 'both' are possible.")
    topn_to_store = train_signal_config.get("topn_to_store", 10)

    # Evaluate strings to produce lists
    if isinstance(parameter_grid.get("buy_signal_threshold"), str):
        parameter_grid["buy_signal_threshold"] = eval(parameter_grid.get("buy_signal_threshold"))
    if isinstance(parameter_grid.get("buy_signal_threshold_2"), str):
        parameter_grid["buy_signal_threshold_2"] = eval(parameter_grid.get("buy_signal_threshold_2"))
    if isinstance(parameter_grid.get("sell_signal_threshold"), str):
        parameter_grid["sell_signal_threshold"] = eval(parameter_grid.get("sell_signal_threshold"))
    if isinstance(parameter_grid.get("sell_signal_threshold_2"), str):
        parameter_grid["sell_signal_threshold_2"] = eval(parameter_grid.get("sell_signal_threshold_2"))

    # Disable sell parameters in grid search - they will be set from the buy parameters
    if train_signal_config.get("buy_sell_equal"):
        parameter_grid["sell_signal_threshold"] = [None]
        parameter_grid["sell_signal_threshold_2"] = [None]

    performances = list()
    for parameters in tqdm(ParameterGrid([parameter_grid]), desc="MODELS"):

        #
        # If equal parameters, then derive the sell parameter from the buy parameter
        #
        if train_signal_config.get("buy_sell_equal"):
            parameters["sell_signal_threshold"] = -parameters["buy_signal_threshold"]
            #signal_model["sell_slope_threshold"] = -signal_model["buy_slope_threshold"]
            if parameters.get("buy_signal_threshold_2") is not None:
                parameters["sell_signal_threshold_2"] = -parameters["buy_signal_threshold_2"]

        trade_model = App.config["trade_model"].copy()
        trade_model["parameters"] = parameters

        #
        # Do not aggregate but assume that we have already the aggregation results in the data
        #
        pass
        # We need only to get the column names for the scores to be used for rules
        score_aggregation_sets = App.config['score_aggregation_sets']
        score_column_names = [sa_set.get("column") for sa_set in score_aggregation_sets]

        #
        # Apply signal rule and generate binary buy_signal_column/sell_signal_column
        #
        if parameters.get('rule_name') == 'two_dim_rule':
            apply_rule_with_score_thresholds_2(df, score_column_names, trade_model)
        else:  # Default one dim rule
            apply_rule_with_score_thresholds(df, score_column_names, trade_model)

        #
        # Simulate trade and compute performance using close price and two boolean signals
        # Add a pair of two dicts: performance dict and model parameters dict
        #
        performance, long_performance, short_performance = \
            simulated_trade_performance(df, 'sell_signal_column', 'buy_signal_column', 'close')

        # Remove some items. Remove lists of transactions which are not needed
        long_performance.pop('transactions', None)
        short_performance.pop('transactions', None)

        if direction == "long":
            performance = long_performance
        elif direction == "short":
            performance = short_performance

        # Add some metrics. Add per month metrics
        performance["profit_percent_per_month"] = performance["profit_percent"] / months_in_simulation
        performance["transaction_no_per_month"] = performance["transaction_no"] / months_in_simulation
        performance["profit_percent_per_transaction"] = performance["profit_percent"] / performance["transaction_no"] if performance["transaction_no"] else 0.0
        performance["profit_per_month"] = performance["profit"] / months_in_simulation

        #long_performance["profit_percent_per_month"] = long_performance["profit_percent"] / months_in_simulation
        #short_performance["profit_percent_per_month"] = short_performance["profit_percent"] / months_in_simulation

        performances.append(dict(
            model=parameters,
            performance={k: performance[k] for k in ['profit_percent_per_month', 'profitable', 'profit_percent_per_transaction', 'transaction_no_per_month']},
            #long_performance={k: long_performance[k] for k in ['profit_percent_per_month', 'profitable']},
            #short_performance={k: short_performance[k] for k in ['profit_percent_per_month', 'profitable']}
        ))

    #
    # Flatten
    #

    # Sort
    performances = sorted(performances, key=lambda x: x['performance']['profit_percent_per_month'], reverse=True)
    performances = performances[:topn_to_store]

    # Column names (from one record)
    keys = list(performances[0]['model'].keys()) + \
           list(performances[0]['performance'].keys())
           #list(performances[0]['long_performance'].keys()) + \
           #list(performances[0]['short_performance'].keys())

    lines = []
    for p in performances:
        record = list(p['model'].values()) + \
                 list(p['performance'].values())
                 #list(p['long_performance'].values()) + \
                 #list(p['short_performance'].values())
        record = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in record]
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
