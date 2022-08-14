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
Use predictions to simulate trade over the whole period based on one signal model.
The results of the trade simulation with signals and performances is stored in the output file.
The results can be used to further analyze (also visually) the selected trade strategy.
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

    #
    # Add two pairs of columns: buy_score_column/sell_score_column and buy_signal_column/sell_signal_column
    #
    model = App.config["signal_model"]
    generate_signal_columns(df, model, buy_score_column_avg, sell_score_column_avg)

    #
    # Simulate trade using close price and two boolean signals
    # Add a pair of two dicts: performance dict and model parameters dict
    #
    performance, long_performance, short_performance = \
        simulated_trade_performance(df, 'sell_signal_column', 'buy_signal_column', 'close')

    #
    # Convert to columns: longs, shorts, signal, profit (both short and long)
    #
    long_df = pd.DataFrame(long_performance.get("transactions")).set_index(0, drop=True)
    short_df = pd.DataFrame(short_performance.get("transactions")).set_index(0, drop=True)
    df["buy_signal"] = False
    df["sell_signal"] = False
    df["signal"] = None

    df.loc[long_df.index, "buy_signal"] = True
    df.loc[long_df.index, "signal"] = "BUY"
    df.loc[short_df.index, "sell_signal"] = True
    df.loc[short_df.index, "signal"] = "SELL"

    df["profit_long_percent"] = 0.0
    df["profit_short_percent"] = 0.0
    df["profit_percent"] = 0.0
    df.update(short_df[4].rename("profit_long_percent"))
    df.update(long_df[4].rename("profit_short_percent"))

    df.update(short_df[4].rename("profit_percent"))
    df.update(long_df[4].rename("profit_percent"))

    # TODO: Include true labels and performance/signals on true labels

    #
    # Store
    #
    out_columns = [
        "timestamp", "open", "high", "low", "close",
        "buy_score_column_avg", "sell_score_column_avg",
        "buy_score_column", "sell_score_column", "buy_signal_column", "sell_signal_column",
        "buy_signal", "sell_signal", "signal", "profit_long_percent", "profit_short_percent", "profit_percent"
    ]
    out_df = df[out_columns]

    out_path = data_path / App.config.get("signal_file_name")

    print(f"Storing output file...")
    out_df.to_csv(out_path.with_suffix(".csv"), index=False, float_format='%.2f')
    print(f"Signals stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    elapsed = datetime.now() - now
    print(f"Finished signal generation in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
