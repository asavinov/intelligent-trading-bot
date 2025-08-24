from typing import Tuple

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from common.types import Venue
from common.classifiers import *
from common.model_store import *
from common.gen_features import *
from common.gen_labels_highlow import generate_labels_highlow, generate_labels_highlow2
from common.gen_labels_topbot import generate_labels_topbot, generate_labels_topbot2
from common.gen_signals import (
    generate_smoothen_scores, generate_combine_scores,
    generate_threshold_rule, generate_threshold_rule2
)

from outputs.notifier_scores import *
from outputs.notifier_diagram import *
from outputs.notifier_trades import *
from outputs import get_trader_functions


def generate_feature_set(df: pd.DataFrame, fs: dict, config: dict, model_store: ModelStore, last_rows: int) -> Tuple[pd.DataFrame, list]:
    """
    Apply the specified resolved feature generator to the input data set.
    """

    #
    # Select columns from the data set to be processed by the feature generator
    #
    cp = fs.get("column_prefix")
    if cp:
        cp = cp + "_"
        f_cols = [col for col in df if col.startswith(cp)]
        f_df = df[f_cols]  # Alternatively: f_df = df.loc[:, df.columns.str.startswith(cf)]
        # Remove prefix because feature generators are generic (a prefix will be then added to derived features before adding them back to the main frame)
        f_df = f_df.rename(columns=lambda x: x[len(cp):] if x.startswith(cp) else x)  # Alternatively: f_df.columns = f_df.columns.str.replace(cp, "")
    else:
        f_df = df[df.columns.to_list()]  # We want to have a different data frame object to add derived featuers and then join them back to the main frame with prefix

    #
    # Resolve and apply feature generator functions from the configuration
    #
    generator = fs.get("generator")
    gen_config = fs.get('config', {})
    if generator == "itblib":
        features = generate_features_itblib(f_df, gen_config, last_rows=last_rows)
    elif generator == "depth":
        features = generate_features_depth(f_df)
    elif generator == "tsfresh":
        features = generate_features_tsfresh(f_df, gen_config, last_rows=last_rows)
    elif generator == "talib":
        features = generate_features_talib(f_df, gen_config, last_rows=last_rows)
    elif generator == "itbstats":
        features = generate_features_itbstats(f_df, gen_config, last_rows=last_rows)

    # Labels
    elif generator == "highlow":
        horizon = gen_config.get("horizon")

        # Binary labels whether max has exceeded a threshold or not
        print(f"Generating 'highlow' labels with horizon {horizon}...")
        features = generate_labels_highlow(f_df, horizon=horizon)

        print(f"Finished generating 'highlow' labels. {len(features)} labels generated.")
    elif generator == "highlow2":
        print(f"Generating 'highlow2' labels...")
        f_df, features = generate_labels_highlow2(f_df, gen_config)
        print(f"Finished generating 'highlow2' labels. {len(features)} labels generated.")
    elif generator == "topbot":
        column_name = gen_config.get("columns", "close")

        top_level_fracs = [0.01, 0.02, 0.03, 0.04, 0.05]
        bot_level_fracs = [-x for x in top_level_fracs]

        f_df, features = generate_labels_topbot(f_df, column_name, top_level_fracs, bot_level_fracs)
    elif generator == "topbot2":
        f_df, features = generate_labels_topbot2(f_df, gen_config)

    # Signals
    elif generator == "smoothen":
        f_df, features = generate_smoothen_scores(f_df, gen_config)
    elif generator == "combine":
        f_df, features = generate_combine_scores(f_df, gen_config)
    elif generator == "threshold_rule":
        f_df, features = generate_threshold_rule(f_df, gen_config)
    elif generator == "threshold_rule2":
        f_df, features = generate_threshold_rule2(f_df, gen_config)

    else:
        # Resolve generator name to a function reference
        generator_fn = resolve_generator_name(generator)
        if generator_fn is None:
            raise ValueError(f"Unknown feature generator name or name cannot be resolved: {generator}")

        # Call this function
        f_df, features = generator_fn(f_df, gen_config, config, model_store)

    #
    # Add generated features to the main data frame with all other columns and features
    #
    f_df = f_df[features]
    fp = fs.get("feature_prefix")
    if fp:
        f_df = f_df.add_prefix(fp + "_")

    new_features = f_df.columns.to_list()

    # Delete new columns if they already exist
    df.drop(list(set(df.columns) & set(new_features)), axis=1, inplace=True)

    df = df.join(f_df)  # Attach all derived features to the main frame

    return df, new_features


def predict_feature_set(df, fs, config, model_store: ModelStore) -> Tuple[pd.DataFrame, list]:

    train_features, labels, algorithms = get_features_labels_algorithms(fs, config)

    train_df = df[train_features]

    features = []
    out_df = pd.DataFrame(index=train_df.index)  # Collect predictions

    for label in labels:
        for model_config in algorithms:

            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = label + label_algo_separator + algo_name

            # It is an entry from loaded model dict
            model_pair = model_store.get_model_pair(score_column_name)  # Trained model from model registry

            print(f"Predict '{score_column_name}'. Algorithm {algo_name}. Label: {label}. Train length {len(train_df)}. Train columns {len(train_df.columns)}")

            if algo_type == "gb":
                df_y_hat = predict_gb(model_pair, train_df, model_config)
            elif algo_type == "nn":
                df_y_hat = predict_nn(model_pair, train_df, model_config)
            elif algo_type == "lc":
                df_y_hat = predict_lc(model_pair, train_df, model_config)
            elif algo_type == "svc":
                df_y_hat = predict_svc(model_pair, train_df, model_config)
            else:
                raise ValueError(f"Unknown algorithm type {algo_type}. Check algorithm list.")

            out_df[score_column_name] = df_y_hat
            features.append(score_column_name)

    return out_df, features


def train_feature_set(df, fs, config) -> dict:

    train_features, labels, algorithms = get_features_labels_algorithms(fs, config)

    # Only for train mode
    df = df.dropna(subset=train_features).reset_index(drop=True)
    df = df.dropna(subset=labels).reset_index(drop=True)

    models = dict()  # Here collect the resulted trained models

    for label in labels:
        for model_config in algorithms:

            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = label + label_algo_separator + algo_name

            # Limit length according to the algorith train parameters
            algo_every_nth_row = model_config.get("params", {}).get("every_nth_row")
            if algo_every_nth_row:
                train_df = df.iloc[::algo_every_nth_row, :]
            else:
                train_df = df
            algo_train_length = model_config.get("params", {}).get("length")
            if algo_train_length:
                train_df = train_df.tail(algo_train_length)

            df_X = train_df[train_features]
            df_y = train_df[label]

            print(f"Train '{score_column_name}'. Algorithm {algo_name}. Label: {label}. Train length {len(df_X)}. Train columns {len(df_X.columns)}")

            if algo_type == "gb":
                model_pair = train_gb(df_X, df_y, model_config)
                models[score_column_name] = model_pair
            elif algo_type == "nn":
                model_pair = train_nn(df_X, df_y, model_config)
                models[score_column_name] = model_pair
            elif algo_type == "lc":
                model_pair = train_lc(df_X, df_y, model_config)
                models[score_column_name] = model_pair
            elif algo_type == "svc":
                model_pair = train_svc(df_X, df_y, model_config)
                models[score_column_name] = model_pair
            else:
                raise ValueError(f"Unknown algorithm type {algo_type}. Check algorithm list.")

    return models


def get_features_labels_algorithms(fs, config) -> Tuple[list, list, list]:
    """
    Get three lists by combining the entries from default lists in the config file
    and lists in the generator config. The function will return a list from the specific
    generator config if it is available and the default list otherwise.
    For the algorithm list, it will resolve the algorithm names into their definitions if necessary.
    """
    train_features_all = config.get("train_features", [])
    train_features = fs.get("config").get("columns", [])
    if not train_features:
        train_features = fs.get("config").get("features", [])
    if not train_features:
        train_features = train_features_all

    labels_all = config.get("labels", [])
    labels = fs.get("config").get("labels", [])
    if not labels:
        labels = labels_all

    algorithms_all = config.get("algorithms")
    algorithms_str = fs.get("config").get("functions", [])
    if not algorithms_str:
        algorithms_str = fs.get("config").get("algorithms", [])
    # The algorithms can be either strings (names) or dicts (definitions) so we resolve the names
    algorithms = []
    for alg in algorithms_str:
        if isinstance(alg, str):  # Find in the list of algorithms
            alg = find_algorithm_by_name(algorithms_all, alg)
        elif not isinstance(alg, dict):
            raise ValueError(f"Algorithm has to be either dict or name")
        algorithms.append(alg)
    if not algorithms:
        algorithms = algorithms_all

    return train_features, labels, algorithms


async def output_feature_set(df, fs: dict, config: dict, model_store: ModelStore) -> None:
    #
    # Resolve and apply feature generator functions from the configuration
    #
    generator = fs.get("generator")
    gen_config = fs.get('config', {})

    if generator == "score_notification_model":
        generator_fn = send_score_notification
    elif generator == "diagram_notification_model":
        generator_fn = send_diagram
    elif generator == "trader_simulation":
        generator_fn = trader_simulation
    elif generator == "trader_binance":
        generator_fn = get_trader_functions(Venue.BINANCE)["trader"]
    elif generator == "trader_mt5":
        generator_fn = get_trader_functions(Venue.MT5)["trader"]

    else:
        # Resolve generator name to a function reference
        generator_fn = resolve_generator_name(generator)
        if generator_fn is None:
            raise ValueError(f"Unknown feature generator name or name cannot be resolved: {generator}")

    # Call the resolved function
    if asyncio.iscoroutinefunction(generator_fn):
        if asyncio.get_running_loop():
            await generator_fn(df, gen_config, config, model_store)
        else:
            asyncio.run(generator_fn(df, gen_config, config, model_store))
    else:
        generator_fn(df, gen_config, config, model_store)
