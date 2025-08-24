from pathlib import Path
from datetime import datetime, timezone, timedelta
import click
from tqdm import tqdm

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from service.App import *
from common.model_store import *
from common.classifiers import compute_scores_regression, compute_scores
from common.generators import predict_feature_set

"""
Apply models to (previously generated) features and compute prediction scores.
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

    # Determine desired data length depending on train/predict mode
    is_train = config.get("train")
    if is_train:
        window_size = config.get("train_length")
        print(f"WARNING: Train mode is specified although this script is intended for prediction and will not train models.")
    else:
        window_size = config.get("predict_length")
    features_horizon = config.get("features_horizon")
    if window_size:
        window_size += features_horizon

    #
    # Load data
    #
    file_path = data_path / config.get("matrix_file_name")
    if not file_path.is_file():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading data from source data file {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
    else:
        print(f"ERROR: Unknown extension of the input file '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Finished loading {len(df)} records with {len(df.columns)} columns from the source file {file_path}")

    # Select only the data necessary for analysis
    if window_size:
        df = df.tail(window_size)
        df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    #
    # Apply ML algorithm predictors
    #
    train_features_all = config.get("train_features")
    labels_all = config["labels"]

    # Select necessary features and label
    out_columns = [time_column, 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [x for x in out_columns if x in df.columns]
    labels_present = set(labels_all).issubset(df.columns)
    if labels_present:
        all_features = train_features_all + labels_all
    else:
        all_features = train_features_all
    df = df[out_columns + [x for x in all_features if x not in out_columns]]

    # Handle NULLs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    na_df = df[ df[train_features_all].isna().any(axis=1) ]
    if len(na_df) > 0:
        print(f"WARNING: There exist {len(na_df)} rows with NULLs in some feature columns. These rows will be removed.")
        df = df.dropna(subset=train_features_all)
        df = df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    #
    # Generate/predict train features
    #
    train_feature_sets = config.get("train_feature_sets", [])
    if not train_feature_sets:
        print(f"ERROR: no train feature sets defined. Nothing to process.")
        return

    print(f"Start generating trained features for {len(df)} input records.")

    labels_hat_df = pd.DataFrame()  # Collect predictions
    features = []

    for i, fs in enumerate(train_feature_sets):
        fs_now = datetime.now()
        print(f"Start train feature set {i}/{len(train_feature_sets)}. Generator {fs.get('generator')}...")

        fs_out_df, fs_features = predict_feature_set(df, fs, config, App.model_store)

        labels_hat_df = pd.concat([labels_hat_df, fs_out_df], axis=1)
        features.extend(fs_features)

        fs_elapsed = datetime.now() - fs_now
        print(f"Finished train feature set {i}/{len(train_feature_sets)}. Generator {fs.get('generator')}. Time: {str(fs_elapsed).split('.')[0]}")

    print(f"Finished generating trained features.")

    #
    # Store predictions
    #
    # Store only selected original data, labels, and their predictions
    out_df = labels_hat_df.join(df[out_columns + (labels_all if labels_present else [])])

    out_path = data_path / config.get("predict_file_name")

    print(f"Storing predictions with {len(out_df)} records and {len(out_df.columns)} columns in output file {out_path}...")
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format='%.6f')
    else:
        print(f"ERROR: Unknown extension of the output file '{out_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Predictions stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    #
    # Compute and store scores
    #
    score_lines = []
    # For each predicted column, find the corresponding true label column and then compare them
    for score_column_name in labels_hat_df.columns:
        label_column, _ = score_to_label_algo_pair(score_column_name)

        # Drop nans from scores
        df_scores = pd.DataFrame({"y_true": out_df[label_column], "y_predicted": out_df[score_column_name]})
        df_scores = df_scores.dropna()

        y_true = df_scores["y_true"]
        y_predicted = df_scores["y_predicted"]
        y_predicted_class = np.where(y_predicted.values > 0.5, 1, 0)

        if ptypes.is_float_dtype(y_true) and ptypes.is_float_dtype(y_predicted):
            score = compute_scores_regression(y_true, y_predicted)  # Regression stores
        else:
            score = compute_scores(y_true.astype(int), y_predicted)  # Classification stores

        score_lines.append(f"{score_column_name}: {score}")

    #
    # Store scores
    #
    score_path = out_path.with_suffix('.txt')
    with open(score_path, "a+") as f:
        f.write("\n".join([str(x) for x in score_lines]) + "\n\n")

    print(f"Prediction scores stored in path: {score_path.absolute()}")

    #
    # End
    #
    elapsed = datetime.now() - now
    print(f"Finished predicting in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
