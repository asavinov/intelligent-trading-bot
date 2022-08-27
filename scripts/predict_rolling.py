from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor
import click

import numpy as np
import pandas as pd

from service.App import *
from common.utils import *
from common.classifiers import *
from common.feature_generation import *
from common.model_store import *

"""
Generate label predictions for the whole input feature matrix by iteratively training models using historic data and predicting labels for some future horizon.
The main parameter is the step of iteration, that is, the future horizon for prediction.
As usual, we can specify past history length used to train a model.
The output file will store predicted labels in addition to all input columns (generated features and true labels).
This file is intended for training signal models (by simulating trade process and computing overall performance for some long period).
The output predicted labels will cover shorter period of time because we need some relatively long history to train the very first model.
"""

#
# Parameters
#
class P:
    in_nrows = 100_000_000

    start_index = 0
    end_index = None

    # First row for starting predictions: "2020-02-01 00:00:00" - minimum start for futures
    prediction_start_str = "2017-08-01"  # For BTC: "2020-02-01 00:00:00"
    # How frequently re-train models: 1 day: 1_440 = 60 * 24, one week: 10_080
    prediction_length = 10  # For 1m (BTC): 2*7*1440 (2 weeks)
    prediction_count = 0  # How many prediction steps. If None or 0, then from prediction start till the data end. Use: https://www.timeanddate.com/date/duration.html

    use_multiprocessing = False
    max_workers = 8  # None means number of processors


#
# Main
#

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)

    time_column = App.config["time_column"]

    now = datetime.now()

    #
    # Load feature matrix
    #
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol

    file_path = (data_path / App.config.get("matrix_file_name")).with_suffix(".csv")
    if not file_path.is_file():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading data from source data file {file_path}...")
    df = pd.read_csv(file_path, parse_dates=[time_column], nrows=P.in_nrows)
    print(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[P.start_index:P.end_index]
    df = df.reset_index(drop=True)

    #
    # Prepare data by selecting columns and rows
    #
    label_horizon = App.config["label_horizon"]  # Labels are generated from future data and hence we might want to explicitly remove some tail rows
    train_length = App.config.get("train_length")
    train_features = App.config.get("train_features")
    labels = App.config["labels"]
    algorithms = App.config.get("algorithms")

    # Select necessary features and label
    out_columns = [time_column, 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [x for x in out_columns if x in df.columns]
    all_features = train_features + labels
    df = df[all_features + out_columns]

    for label in labels:
        # "category" NN does not work without this (note that we assume a classification task here)
        df[label] = df[label].astype(int)

    # Spot and futures have different available histories. If we drop nans in all of them, then we get a very short data frame (corresponding to futureus which have little data)
    # So we do not drop data here but rather when we select necessary input features
    # Nans result in constant accuracy and nan loss. MissingValues procedure does not work and produces exceptions
    pd.set_option('use_inf_as_na', True)
    #in_df = in_df.dropna(subset=labels)
    df = df.reset_index(drop=True)  # We must reset index after removing rows to remove gaps

    prediction_start = find_index(df, P.prediction_start_str)
    print(f"Start index: {prediction_start}")

    #
    # Rolling train-predict loop
    #
    stride = P.prediction_length
    steps = P.prediction_count
    if not steps:
        # Use all available rest data (from the prediction start to the dataset end)
        steps = (len(df) - prediction_start) // stride
    if len(df) - prediction_start < steps * stride:
        raise ValueError(f"Number of steps {steps} is too large (not enough data after start). Data available for prediction: {len(df) - prediction_start}. Data to be predicted: {steps * stride} ")

    print(f"Starting rolling predict loop with {steps} steps. Each step with {stride} horizon...")

    # Result rows. Here store only rows for which we make predictions
    labels_hat_df = pd.DataFrame()

    for step in range(steps):

        print(f"\n===>>> Start step {step}/{steps}")

        # Predict data

        predict_start = prediction_start + (step * stride)
        predict_end = predict_start + stride

        predict_df = df.iloc[predict_start:predict_end]  # We assume that iloc is equal to index
        # predict_df = predict_df.dropna(subset=features)  # Nans will be droped by the algorithms themselves

        # Here we will collect predicted columns
        predict_labels_df = pd.DataFrame(index=predict_df.index)

        # Predict data

        df_X_test = predict_df[train_features]
        #df_y_test = predict_df[predict_label]  # It will be set in the loop over labels

        # Train data

        # We exclude recent objects from training, because they do not have labels yet - the labels are in future
        # In real (stream) data, we will have null labels for recent objects. During simulation, labels are available and hence we need to ignore/exclude them manually
        train_end = predict_start - label_horizon - 1
        train_start = train_end - train_length
        train_start = 0 if train_start < 0 else train_start

        train_df = df.iloc[int(train_start):int(train_end)]  # We assume that iloc is equal to index
        train_df = train_df.dropna(subset=train_features)

        print(f"Train range: [{train_start}, {train_end}]={train_end-train_start}. Prediction range: [{predict_start}, {predict_end}]={predict_end-predict_start}. ")

        for label in labels:  # Train-predict different labels (and algorithms) using same X

            if P.use_multiprocessing:
                # Submit train-predict algorithms to the pool
                execution_results = dict()
                with ProcessPoolExecutor(max_workers=P.max_workers) as executor:
                    for algo_name in algorithms:
                        model_config = get_model(algo_name)
                        algo_type = model_config.get("algo")
                        algo_train_length = model_config.get("train", {}).get("length")
                        score_column_name = label + label_algo_separator + algo_name

                        # Limit length according to algorith parameters
                        if algo_train_length and algo_train_length < train_length:
                            train_df_2 = train_df.iloc[-algo_train_length:]
                        else:
                            train_df_2 = train_df
                        df_X = train_df_2[train_features]
                        df_y = train_df_2[label]
                        df_y_test = predict_df[label]

                        if algo_type == "gb":
                            execution_results[score_column_name] = executor.submit(train_predict_gb, df_X, df_y, df_X_test, model_config)
                        elif algo_type == "nn":
                            execution_results[score_column_name] = executor.submit(train_predict_nn, df_X, df_y, df_X_test, model_config)
                        elif algo_type == "lc":
                            execution_results[score_column_name] = executor.submit(train_predict_lc, df_X, df_y, df_X_test, model_config)
                        elif algo_type == "svc":
                            execution_results[score_column_name] = executor.submit(train_predict_svc, df_X, df_y, df_X_test, model_config)
                        else:
                            print(f"ERROR: Unknown algorithm type {algo_type}. Check algorithm list.")
                            return

                # Process the results as the tasks are finished
                for score_column_name, future in execution_results.items():
                    predict_labels_df[score_column_name] = future.result()
                    if future.exception():
                        print(f"Exception while train-predict {score_column_name}.")
                        return
            else:  # No multiprocessing - sequential execution
                for algo_name in algorithms:
                    model_config = get_model(algo_name)
                    algo_type = model_config.get("algo")
                    algo_train_length = model_config.get("train", {}).get("length")
                    score_column_name = label + label_algo_separator + algo_name

                    # Limit length according to algorith parameters
                    if algo_train_length and algo_train_length < train_length:
                        train_df_2 = train_df.iloc[-algo_train_length:]
                    else:
                        train_df_2 = train_df
                    df_X = train_df_2[train_features]
                    df_y = train_df_2[label]
                    df_y_test = predict_df[label]

                    if algo_type == "gb":
                        predict_labels_df[score_column_name] = train_predict_gb(df_X, df_y, df_X_test, model_config)
                    elif algo_type == "nn":
                        predict_labels_df[score_column_name] = train_predict_nn(df_X, df_y, df_X_test, model_config)
                    elif algo_type == "lc":
                        predict_labels_df[score_column_name] = train_predict_lc(df_X, df_y, df_X_test, model_config)
                    elif algo_type == "svc":
                        predict_labels_df[score_column_name] = train_predict_svc(df_X, df_y, df_X_test, model_config)
                    else:
                        print(f"ERROR: Unknown algorithm type {algo_type}. Check algorithm list.")
                        return

        #
        # Append predicted *rows* to the end of previous predicted rows
        #

        # Predictions for all labels and histories (and algorithms) have been generated for the iteration
        labels_hat_df = pd.concat([labels_hat_df, predict_labels_df])

        print(f"End step {step}/{steps}.")
        print(f"Predicted {len(predict_labels_df.columns)} labels.")


    # End of loop over prediction steps
    print("")
    print(f"Finished all {steps} prediction steps each with {stride} predicted rows (stride). ")
    print(f"Size of predicted dataframe {len(labels_hat_df)}. Number of rows in all steps {steps*stride} (steps * stride). ")
    print(f"Number of predicted columns {len(labels_hat_df.columns)}")

    #
    # Store data
    #
    # We do not store features. Only selected original data, labels, and their predictions
    out_df = labels_hat_df.join(df[out_columns + labels])

    out_path = data_path / App.config.get("predict_file_name")

    print(f"Storing output file...")
    out_df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Predictions stored in file: {out_path}. Length: {len(out_df)}. Columns: {len(out_df.columns)}")

    #
    # Compute accuracy for the whole data set (all segments)
    #

    score_lines = []
    for score_column_name in labels_hat_df.columns:
        label_column, _ = score_to_label_algo_pair(score_column_name)

        # Drop nans from scores
        df_scores = pd.DataFrame({"y_true": out_df[label_column], "y_predicted": out_df[score_column_name]})
        df_scores = df_scores.dropna()

        y_true = df_scores["y_true"].astype(int)
        y_predicted = df_scores["y_predicted"]
        y_predicted_class = np.where(y_predicted.values > 0.5, 1, 0)

        print(f"Using {len(df_scores)} non-nan rows for scoring.")

        score = compute_scores(y_true, y_predicted)

        score_lines.append(f"{score_column_name}, {score.get('auc'):.3f}, {score.get('ap'):.3f}, {score.get('f1'):.3f}, {score.get('precision'):.3f}, {score.get('recall'):.3f}")

    #
    # Store hyper-parameters and scores
    #
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write("\n".join([str(x) for x in score_lines]) + "\n\n")

    elapsed = datetime.now() - now
    print(f"Finished feature prediction in {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    main()
