import asyncio
from pathlib import Path
import click
from tqdm import tqdm

import numpy as np
import pandas as pd

from service.App import *
from common.utils import *
from common.model_store import *
from common.generators import output_feature_set

"""
Execute outputs based on signal file.
"""


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)
    config = App.config

    App.model_store = ModelStore(config)
    App.model_store.load_models()

    time_column = config["time_column"]

    now = datetime.now()

    symbol = config["symbol"]
    data_path = Path(config["data_folder"]) / symbol

    #
    # Load data with (rolling) label point-wise predictions and signals generated
    #
    file_path = data_path / config.get("signal_file_name")
    if not file_path.exists():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading signals from input file: {file_path}")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
    else:
        print(f"ERROR: Unknown extension of the input file '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Signals loaded. Length: {len(df)}. Width: {len(df.columns)}")

    #
    # Execute all output sets. Note that they can be async
    #

    # For data processing, a data frame is supposed to have timestamp in index
    df = df.set_index(time_column, inplace=False)

    output_sets = App.config.get("output_sets", [])
    for os in output_sets:
        try:
            #await output_feature_set(df, os, App.config, App.model_store)
            asyncio.run(output_feature_set(df, os, App.config, App.model_store))
        except Exception as e:
            log.error(f"Error in output function: {e}. Generator: {os.get("generator")}. Output config: {os}")
            return

    elapsed = datetime.now() - now
    print(f"Finished executing outputs in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
