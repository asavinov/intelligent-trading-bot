from pathlib import Path
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay)
from sklearn.model_selection import ParameterGrid

from service.App import *
from common.utils import *
from common.gen_signals import *
from common.classifiers import *
from common.generators import generate_feature_set

"""
The script is intended for finding best trade parameters for a certain trade algorithm
by executing trade simulation (backtesting) for all specified parameters.
It performs exhaustive search in the space of all specified parameters by computing 
trade performance and then choosing the parameters with the highest profit (or maybe
using other selection criteria like stability of the results or minimum allowed losses etc.)

Notes:
- The optimization is based on certain trade algorithm. This means that a trade algorithm
is a parameter for this script. Different trade algorithms have different trade logics and 
also have different parameters. Currently, the script works with a very simple threshold-based
trade algorithm: if some score is higher than the threshold (parameter) then buy, if it is lower
than another threshold then sell. There is also a version with two thresholds for two scores.
- The script consumes the results of signal script but it then varies parameters of one entry
responsible for generation of trade signals. It then measures performance.
"""


class P:
    in_nrows = 100_000_000


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
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
    # Load data with (rolling) label point-wise predictions and signals generated
    #
    file_path = data_path / App.config.get("signal_file_name")
    if not file_path.exists():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading signals from input file: {file_path}")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unknown extension of the 'signal_file_name' file '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Signals loaded. Length: {len(df)}. Width: {len(df.columns)}")

    #
    # Limit the source data
    #
    train_signal_config = App.config["train_signal_model"]

    data_start = train_signal_config.get("data_start", 0)
    if isinstance(data_start, str):
        data_start = find_index(df, data_start)
    data_end = train_signal_config.get("data_end", None)
    if isinstance(data_end, str):
        data_end = find_index(df, data_end)

    df = df.iloc[data_start:data_end]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    months_in_simulation = (df[time_column].iloc[-1] - df[time_column].iloc[0]) / timedelta(days=365/12)

    #
    # Load signal train parameters
    #
    parameter_grid = train_signal_config.get("grid")
    direction = train_signal_config.get("direction", "")
    if direction not in ['long', 'short', 'both', '']:
        raise ValueError(f"Unknown value of {direction} in signal train model. Only 'long', 'short' and 'both' are possible.")
    topn_to_store = train_signal_config.get("topn_to_store", 10)

    # Evaluate strings to produce lists with ranges of parameters
    if isinstance(parameter_grid.get("buy_signal_threshold"), str):
        parameter_grid["buy_signal_threshold"] = eval(parameter_grid.get("buy_signal_threshold"))
    if isinstance(parameter_grid.get("buy_signal_threshold_2"), str):
        parameter_grid["buy_signal_threshold_2"] = eval(parameter_grid.get("buy_signal_threshold_2"))
    if isinstance(parameter_grid.get("sell_signal_threshold"), str):
        parameter_grid["sell_signal_threshold"] = eval(parameter_grid.get("sell_signal_threshold"))
    if isinstance(parameter_grid.get("sell_signal_threshold_2"), str):
        parameter_grid["sell_signal_threshold_2"] = eval(parameter_grid.get("sell_signal_threshold_2"))

    # If necessary, disable sell parameters in grid search - they will be set from the buy parameters
    if train_signal_config.get("buy_sell_equal"):
        parameter_grid["sell_signal_threshold"] = [None]
        parameter_grid["sell_signal_threshold_2"] = [None]

    #
    # Find the generator, the parameters of which will be varied
    #
    generator_name = train_signal_config.get("signal_generator")
    signal_generator = next((ss for ss in App.config.get("signal_sets", []) if ss.get('generator') == generator_name), None)
    if not signal_generator:
        raise ValueError(f"Signal generator '{generator_name}' not found among all 'signal_sets'")

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

        #
        # Set new parameters of the signal generator
        #
        signal_generator["config"]["parameters"].update(parameters)

        #
        # Execute the signal generator with new parameters by producing new signal columns
        #
        df, new_features = generate_feature_set(df, signal_generator, last_rows=0)

        #
        # Simulate trade and compute performance using close price and two boolean signals
        # Add a pair of two dicts: performance dict and model parameters dict
        #

        # These boolean columns are used for performance measurement. Alternatively, they are in trade_signal_model
        buy_signal_column = signal_generator["config"]["names"][0]
        sell_signal_column = signal_generator["config"]["names"][1]

        # Perform backtesting
        performance, long_performance, short_performance = simulated_trade_performance(
            df,
            buy_signal_column, sell_signal_column,
            'close'
        )

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
