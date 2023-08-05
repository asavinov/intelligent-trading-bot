from pathlib import Path
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay)
from sklearn.model_selection import ParameterGrid

from service.App import *
from common.label_generation_topbot import *
from common.signal_generation import *

"""
Use predictions to process scores, generate signals and simulate trades over the whole period.
The results of the trade simulation with signals and performances is stored in the output file.
The results can be used to further analyze (also visually) the selected signal and trade strategy.
"""

class P:
    in_nrows = 100_000_000

    start_index = 0  # 200_000 for 1m btc
    end_index = None


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
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
    # Find maximum performance possible based on true labels only (and not predictions)
    #
    # Best parameters (just to compute for known parameters)
    #df['buy_signal_column'] = score_to_signal(df[bot_score_column], None, 5, 0.09)
    #df['sell_signal_column'] = score_to_signal(df[top_score_column], None, 10, 0.064)
    #performance_long, performance_short, long_count, short_count, long_profitable, short_profitable, longs, shorts = performance_score(df, 'sell_signal_column', 'buy_signal_column', 'close')
    # TODO: Save maximum performance in output file or print it (use as a reference)

    # Maximum possible on labels themselves
    #performance_long, performance_short, long_count, short_count, long_profitable, short_profitable, longs, shorts = performance_score(df, 'top10_2', 'bot10_2', 'close')

    #
    # Aggregate and post-process
    #
    score_aggregation_sets = App.config['score_aggregation_sets']
    # Temporary (post-processed) columns for each aggregation set
    buy_column = 'aggregated_buy_score'
    sell_column = 'aggregated_sell_score'
    for i, sa_set in enumerate(score_aggregation_sets):

        buy_labels = sa_set.get("buy_labels")
        sell_labels = sa_set.get("sell_labels")
        if set(buy_labels + sell_labels) - set(df.columns):
            missing_labels = list(set(buy_labels + sell_labels) - set(df.columns))
            print(f"ERROR: Some buy/sell labels from config are not present in the input data. Missing labels: {missing_labels}")
            return

        parameters = sa_set.get("parameters", {})
        # Aggregate predictions of different algorithms separately for buy and sell
        aggregate_scores(df, parameters, buy_column, buy_labels)  # Output is buy column
        aggregate_scores(df, parameters, sell_column, sell_labels)  # Output is sell column

        trade_score_column = sa_set.get("column")

        # Here we want to take into account relative values of buy and sell scores
        # Mutually adjust two independent scores with opposite buy/sell semantics
        combine_scores(df, parameters, buy_column, sell_column, trade_score_column)
    # Delete temporary columns
    del df[buy_column]
    del df[sell_column]

    #
    # Apply signal rule and generate binary buy_signal_column/sell_signal_column
    #
    signal_model = App.config['signal_model']
    if signal_model.get('rule_name') == 'two_dim_rule':
        apply_rule_with_score_thresholds_2(df, signal_model)
    else:  # Default one dim rule
        apply_rule_with_score_thresholds(df, signal_model)

    #
    # Simulate trade and compute performance using close price and two boolean signals
    # Add a pair of two dicts: performance dict and model parameters dict
    #
    signal_column_names = signal_model.get("signal_columns")

    performance, long_performance, short_performance = \
        simulated_trade_performance(df, signal_column_names[1], signal_column_names[0], 'close')

    #
    # Convert to columns: longs, shorts, signal, profit (both short and long)
    #
    long_df = pd.DataFrame(long_performance.get("transactions")).set_index(0, drop=True)
    short_df = pd.DataFrame(short_performance.get("transactions")).set_index(0, drop=True)
    df["buy_transaction"] = False
    df["sell_transaction"] = False
    df["transaction_type"] = None

    df.loc[long_df.index, "buy_transaction"] = True
    df.loc[long_df.index, "transaction_type"] = "BUY"
    df.loc[short_df.index, "sell_transaction"] = True
    df.loc[short_df.index, "transaction_type"] = "SELL"

    df["profit_long_percent"] = 0.0
    df["profit_short_percent"] = 0.0
    df["profit_percent"] = 0.0
    df.update(short_df[4].rename("profit_long_percent"))
    df.update(long_df[4].rename("profit_short_percent"))

    df.update(short_df[4].rename("profit_percent"))
    df.update(long_df[4].rename("profit_percent"))

    #
    # Store statistics
    #
    lines = []

    # Score statistics
    score_column_names = signal_model.get("score_columns")
    for score_col_name in score_column_names:
        lines.append(f"'{score_col_name}':\n" + df[score_col_name].describe().to_string())

    # TODO: Profit

    metrics_file_name = f"signal-metrics.txt"
    metrics_path = (data_path / metrics_file_name).resolve()
    with open(metrics_path, 'a+') as f:
        f.write("\n".join(lines) + "\n\n")

    print(f"Metrics stored in path: {metrics_path.absolute()}")

    #
    # Store data
    #
    out_columns = ["timestamp", "open", "high", "low", "close"]  # Source data
    out_columns.extend(App.config.get('labels'))  # True labels
    out_columns.extend(score_column_names)  # Aggregated post-processed scores
    out_columns.extend(signal_column_names)  # Rule results
    out_columns.extend(["buy_transaction", "sell_transaction", "transaction_type", "profit_long_percent", "profit_short_percent", "profit_percent"])  # Simulation results

    out_df = df[out_columns]

    out_path = data_path / App.config.get("signal_file_name")

    print(f"Storing output file...")
    out_df.to_csv(out_path.with_suffix(".csv"), index=False, float_format='%.4f')
    print(f"Signals stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    elapsed = datetime.now() - now
    print(f"Finished signal generation in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
