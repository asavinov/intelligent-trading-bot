from typing import Tuple

import numpy as np
import pandas as pd

from common.classifiers import *
from common.model_store import *
from common.gen_features import *
from common.gen_labels_highlow import generate_labels_highlow, generate_labels_highlow2
from common.gen_labels_topbot import generate_labels_topbot, generate_labels_topbot2
from common.gen_signals import (
    generate_smoothen_scores, generate_combine_scores,
    generate_threshold_rule, generate_threshold_rule2
)


def generate_feature_set(df: pd.DataFrame, fs: dict, last_rows: int) -> Tuple[pd.DataFrame, list]:
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
        f_df, features = generator_fn(f_df, gen_config)

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


def predict_feature_set(df, fs, config, models: dict):

    labels = fs.get("config").get("labels")
    if not labels:
        labels = config.get("labels")

    algorithms = fs.get("config").get("functions")
    if not algorithms:
        algorithms = fs.get("config").get("algorithms")
    if not algorithms:
        algorithms = config.get("algorithms")

    train_features = fs.get("config").get("columns")
    if not train_features:
        train_features = fs.get("config").get("features")
    if not train_features:
        train_features = config.get("train_features")

    train_df = df[train_features]

    features = []
    scores = dict()
    out_df = pd.DataFrame(index=train_df.index)  # Collect predictions

    for label in labels:
        for model_config in algorithms:

            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = label + label_algo_separator + algo_name
            algo_train_length = model_config.get("train", {}).get("length")

            # It is an entry from loaded model dict
            model_pair = models.get(score_column_name)  # Trained model from model registry

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
                raise ValueError(f"Unknown algorithm type '{algo_type}'")

            out_df[score_column_name] = df_y_hat
            features.append(score_column_name)

            # For each new score, compare it with the label true values
            if label in df:
                scores[score_column_name] = compute_scores(df[label], df_y_hat)

    return out_df, features, scores


def train_feature_set(df, fs, config):

    labels = fs.get("config").get("labels")
    if not labels:
        labels = config.get("labels")

    algorithms = fs.get("config").get("functions")
    if not algorithms:
        algorithms = fs.get("config").get("algorithms")
    if not algorithms:
        algorithms = config.get("algorithms")

    train_features = fs.get("config").get("columns")
    if not train_features:
        train_features = fs.get("config").get("features")
    if not train_features:
        train_features = config.get("train_features")

    models = dict()
    scores = dict()
    out_df = pd.DataFrame()  # Collect predictions

    for label in labels:
        for model_config in algorithms:

            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = label + label_algo_separator + algo_name
            algo_train_length = model_config.get("train", {}).get("length")

            # Limit length according to the algorith parameters
            if algo_train_length:
                train_df = df.tail(algo_train_length)
            else:
                train_df = df
            df_X = train_df[train_features]
            df_y = train_df[label]

            print(f"Train '{score_column_name}'. Algorithm {algo_name}. Label: {label}. Train length {len(df_X)}. Train columns {len(df_X.columns)}")

            if algo_type == "gb":
                model_pair = train_gb(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_gb(model_pair, df_X, model_config)
            elif algo_type == "nn":
                model_pair = train_nn(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_nn(model_pair, df_X, model_config)
            elif algo_type == "lc":
                model_pair = train_lc(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_lc(model_pair, df_X, model_config)
            elif algo_type == "svc":
                model_pair = train_svc(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_svc(model_pair, df_X, model_config)
            else:
                print(f"ERROR: Unknown algorithm type {algo_type}. Check algorithm list.")
                return

            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            out_df[score_column_name] = df_y_hat

    return out_df, models, scores


def resolve_generator_name(gen_name: str):
    """
    Resolve the specified name to a function reference.
    Fully qualified name consists of module name and function name separated by a colon,
    for example:  'mod1.mod2.mod3:my_func'.

    Example: fn = resolve_generator_name("common.gen_features_topbot:generate_labels_topbot3")
    """

    mod_and_func = gen_name.split(':', 1)
    mod_name = mod_and_func[0] if len(mod_and_func) > 1 else None
    func_name = mod_and_func[-1]

    if not mod_name:
        return None

    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return None
    if mod is None:
        return None

    try:
        func = getattr(mod, func_name)
    except AttributeError as e:
        return None

    return func
