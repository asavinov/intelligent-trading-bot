from pathlib import Path
from datetime import datetime, timezone, timedelta
import time
import click
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import multiprocessing as mp

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from service.App import *
from common.model_store import *
from common.gen_features import *
from common.utils import compute_scores_regression, compute_scores
from common.generators import train_feature_set, predict_feature_set

"""
Generate label predictions for the whole input feature matrix by iteratively training models using historic data and predicting labels for some future horizon.
The main parameter is the step of iteration, that is, the future horizon for prediction.
As usual, we can specify past history length used to train a model.
The output file will store predicted labels in addition to all input columns (generated features and true labels).
This file is intended for training signal models (by simulating trade process and computing overall performance for some long period).
The output predicted labels will cover shorter period of time because we need some relatively long history to train the very first model.
"""

#
# Main
#

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
    # Load matrix data with regular time series
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

    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    #
    # Limit the source data
    #
    rp_config = config["rolling_predict"]

    data_start = rp_config.get("data_start", None)
    data_end = rp_config.get("data_end", None)

    if data_start:
        if isinstance(data_start, str):
            df = df[ df[time_column] >= data_start ]
        elif isinstance(data_start, int):
            df = df.iloc[data_start:]

    if data_end:
        if isinstance(data_end, str):
            df = df[ df[time_column] < data_end ]
        elif isinstance(data_end, int):
            df = df.iloc[:-data_end]

    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    #
    # Determine parameters of the rolling prediction loop
    #

    prediction_start = rp_config.get("prediction_start", None)
    if isinstance(prediction_start, str):
        prediction_start = find_index(df, prediction_start)
    prediction_size = rp_config.get("prediction_size")
    prediction_steps = rp_config.get("prediction_steps")

    # Compute a missing parameter if any
    if not prediction_start:
        if not prediction_size or not prediction_steps:
            raise ValueError(f"Only one of the three rolling prediction loop parameters can be empty.")
        # Where we have to start in order to perform the specified number of steps each having the specified length
        prediction_start = len(df) - prediction_size*prediction_steps
    elif not prediction_size:
        if not prediction_start or not prediction_steps:
            raise ValueError(f"Only one of the three rolling prediction loop parameters can be empty.")
        # Size of one prediction in order to get the specified number of steps with the specified length
        prediction_size = (len(df) - prediction_start) // prediction_steps
    elif not prediction_steps:
        if not prediction_start or not prediction_size:
            raise ValueError(f"Only one of the three rolling prediction loop parameters can be empty.")
        # Number of steps with the specified length with the specified start
        prediction_steps = (len(df) - prediction_start) // prediction_size

    # Check consistency of the loop parameters
    if len(df) - prediction_start < prediction_steps * prediction_size:
        raise ValueError(f"Not enough data for {prediction_steps} steps each of size {prediction_size} starting from {prediction_start}. Available data for prediction: {len(df) - prediction_start}")

    #
    # Prepare data by selecting columns and rows
    #
    train_features_all = config.get("train_features")
    labels_all = config.get("labels")

    # Select necessary features and label
    out_columns = [time_column, 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [x for x in out_columns if x in df.columns]
    labels_present = set(labels_all).issubset(df.columns)
    if labels_present:
        all_features = train_features_all + labels_all
    else:
        all_features = train_features_all
    df = df[out_columns + [x for x in all_features if x not in out_columns]]

    for label in labels_all:
        if np.issubdtype(df[label].dtype, bool):
            df[label] = df[label].astype(int)  # For classification tasks we want to use integers

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #in_df = in_df.dropna(subset=labels)
    df = df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    print(f"Start index: {prediction_start}. Number of steps: {prediction_steps}. Step size: {prediction_size}")

    #
    # Rolling/moving train-predict sequence
    #

    train_feature_sets = config.get("train_feature_sets", [])
    if not train_feature_sets:
        print(f"ERROR: no train feature sets defined. Nothing to process.")
        return

    # TODO: It will be removed because we do not use directly a list of algorithms (it can be empty)
    label_horizon = config["label_horizon"]  # Labels are generated from future data and hence we might want to explicitly remove some tail rows
    train_length = config.get("train_length")

    labels_hat_df = pd.DataFrame()  # Result rows. Here store only rows for which we make predictions

    print(f"Starting rolling predict loop...")

    use_multiprocessing = rp_config.get("use_multiprocessing", False)
    max_workers = rp_config.get("max_workers", None)
    if use_multiprocessing:
        parallel = Parallel(n_jobs=max_workers, backend="loky", verbose=13)  # ['loky', 'multiprocessing', 'sequential', 'threading']
        #parallel = mp.Pool(processes=max_workers)
        #parallel = ProcessPoolExecutor(max_workers=max_workers)
    else:
        parallel = None

    for step in range(prediction_steps):

        # Predict data

        predict_start = prediction_start + (step * prediction_size)
        predict_end = predict_start + prediction_size

        predict_df = df.iloc[predict_start:predict_end]  # Assume iloc equal to index
        df_X_test = predict_df[train_features_all]

        # Train data

        # We exclude recent objects from training, because they do not have labels yet - the labels are in future
        # In real (stream) data, we will have null labels for recent objects. During simulation, labels are available and hence we need to ignore/exclude them manually
        train_end = predict_start - label_horizon - 1
        if train_length:
            train_start = max(0, train_end - train_length)
        else:
            train_start = 0

        train_df = df.iloc[train_start:train_end]  # We assume that iloc is equal to index
        train_df = train_df.dropna(subset=train_features_all)

        print(f"\n===>>> Start step {step}/{prediction_steps}. Train range: [{train_start}, {train_end}]={train_end-train_start}. Prediction range: [{predict_start}, {predict_end}]={predict_end-predict_start}")

        step_start_time = datetime.now()

        #
        # Real execution of one step
        #
        predict_labels_df = execute_train_predict_step(config, train_df, predict_df, parallel)

        # Append predicted rows to the end of previous predicted rows
        labels_hat_df = pd.concat([labels_hat_df, predict_labels_df])

        elapsed = datetime.now() - step_start_time
        print(f"End step {step}/{prediction_steps}. Scores predicted: {len(predict_labels_df.columns)}. Time elapsed: {str(elapsed).split('.')[0]}")

    # End of loop over prediction steps
    print("")
    print(f"Finished all {prediction_steps} prediction steps each with {prediction_size} predicted rows (stride). ")
    print(f"Size of predicted dataframe {len(labels_hat_df)}. Number of rows in all steps {prediction_steps*prediction_size} (steps * stride). ")
    print(f"Number of predicted columns {len(labels_hat_df.columns)}")

    #
    # Store data
    #
    # We do not store features. Only selected original data, labels, and their predictions
    out_df = labels_hat_df.join(df[out_columns + labels_all])

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
    print(f"Finished rolling prediction in {str(elapsed).split('.')[0]}")


def execute_train_predict_step(config: dict, train_df: pd.DataFrame, predict_df: pd.DataFrame, parallel):
    """
    This function is supposed to be used in one step of rolling prediction.
    It gets train data (which is supposed to move along larger data set) and data to be used for predictions.
    The predict data set is supposed to be selected from future data relative to the train data.

    :param config: global config
    :param train_df: data to be used for training with all features and all true labels
    :param predict_df: data to be used for prediction with all features and true labels used for computing score
    :return: predictions for all labels for objects in predict_df
    """
    rp_config = config["rolling_predict"]

    train_feature_sets = config.get("train_feature_sets", [])

    #
    # 1 Execute only training (copy from train script)
    #

    fs_now = datetime.now()
    print(f"Start train all models from {len(train_feature_sets)} feature sets {"sequentially" if not parallel else "in parallel"}. Train set size:  {len(train_df)} ")

    models = dict()
    if isinstance(parallel, Parallel):
        results = parallel(delayed(train_feature_set)(train_df, fs, config) for fs in train_feature_sets)
        for fs_models in results:
            models.update(fs_models)
    elif isinstance(parallel, mp.pool.Pool):
        #results = parallel.starmap(train_feature_set, [(train_df, fs, config) for fs in train_feature_sets])
        results = [parallel.apply(train_feature_set, args=(train_df, fs, config)) for fs in train_feature_sets]
    elif isinstance(parallel, ProcessPoolExecutor):
        # Submit all in a loop
        execution_results = dict()  # Futures for each label
        for i, fs in enumerate(train_feature_sets):
            score_column_name = f"label_{i}"
            execution_results[score_column_name] = parallel.submit(train_feature_set, train_df, fs, config)

        results = dict()
        for score_column_name, future in execution_results.items():
            results[score_column_name] = future.result()
            if future.exception():
                print(f"Exception while train-predict {score_column_name}.")
                return

    else:  # No multiprocessing - sequential execution
        for i, fs in enumerate(train_feature_sets):  #  Execute sequentially
            fs_models = train_feature_set(train_df, fs, config)
            models.update(fs_models)

    fs_elapsed = datetime.now() - fs_now
    print(f"Finished train all. Time: {str(fs_elapsed).split('.')[0]}")

    #
    # 2 Store all collected models in files
    #

    # NOTE: train generator does NOT store models in model store but only returns them
    #   yet, predict generator, expects the models to be in model store - it constructs the necessary model name and extracts it from model store
    #   so we need to put the trained models in the model store (they will be stored automatically and overwrite previously trained models which is ok)
    for score_column_name, model_pair in models.items():
        App.model_store.put_model_pair(score_column_name, model_pair)

    print(f"Models stored in path: {App.model_store.model_path.absolute()}")

    # 3 Execute only predict where df for the generator is a different predict df
    # (Copy from predict script)

    fs_now = datetime.now()
    print(f"Start predictions for {len(predict_df)} input records.")

    out_df = pd.DataFrame()  # Collect predictions
    features = []

    for i, fs in enumerate(train_feature_sets):
        fs_out_df, fs_features = predict_feature_set(predict_df, fs, config, App.model_store)
        out_df = pd.concat([out_df, fs_out_df], axis=1)
        features.extend(fs_features)

    fs_elapsed = datetime.now() - fs_now
    print(f"Finished predictions. Time: {str(fs_elapsed).split('.')[0]}")

    return out_df


if __name__ == '__main__':
    main()
