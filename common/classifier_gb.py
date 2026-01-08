import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import lightgbm as lgbm

def train_predict_gb(df_X, df_y, df_X_test, model_config: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    model_pair = train_gb(df_X, df_y, model_config)
    y_test_hat = predict_gb(model_pair, df_X_test, model_config)
    return y_test_hat

def train_gb(df_X, df_y, model_config: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    params = model_config.get("params", {})

    is_scale = params.get("is_scale", False)
    is_regression = params.get("is_regression", False)

    #
    # Scale
    #
    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    #
    # Create model
    #
    train_conf = model_config.get("train", {})

    objective = train_conf.get("objective")

    max_depth = train_conf.get("max_depth")
    learning_rate = train_conf.get("learning_rate")
    num_boost_round = train_conf.get("num_boost_round")

    lambda_l1 = train_conf.get("lambda_l1")
    lambda_l2 = train_conf.get("lambda_l2")

    lgbm_params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,  # Can be -1
        #"n_estimators": 10000,

        #"min_split_gain": params['min_split_gain'],
        "min_data_in_leaf": int(0.01*len(df_X)),  # Best: ~0.02 * len() - 2% of size
        #'subsample': 0.8,
        #'colsample_bytree': 0.8,
        'num_leaves': 32,  # or (2 * 2**max_depth)
        #"bagging_freq": 5,
        #"bagging_fraction": 0.4,
        #"feature_fraction": 0.05,

        # gamma=0.1 ???
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,

        'is_unbalance': 'true',
        # 'scale_pos_weight': scale_pos_weight,  # is_unbalance must be false

        'boosting_type': 'gbdt',  # dart (slow but best, worse than gbdt), goss, gbdt

        'objective': objective,  # binary cross_entropy cross_entropy_lambda

        'metric': {'cross_entropy'},  # auc auc_mu map (mean_average_precision) cross_entropy binary_logloss cross_entropy_lambda binary_error

        'verbose': 0,
    }

    model = lgbm.train(
        lgbm_params,
        train_set=lgbm.Dataset(X_train, y_train),
        num_boost_round=num_boost_round,
        #valid_sets=[lgbm.Dataset(X_validate, y_validate)],
        #early_stopping_rounds=int(num_boost_round / 5),
        #verbose_eval=100,
    )

    return (model, scaler)

def predict_gb(models: tuple, df_X_test, model_config: dict):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    #
    # Scale
    #
    scaler = models[1]
    is_scale = scaler is not None

    input_index = df_X_test.index
    if is_scale:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)
    else:
        df_X_test = df_X_test

    df_X_test_nonans = df_X_test.dropna()  # Drop nans, possibly create gaps in index
    nonans_index = df_X_test_nonans.index

    y_test_hat_nonans = models[0].predict(df_X_test_nonans.values)
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)  # Attach indexes with gaps

    df_ret = pd.DataFrame(index=input_index)  # Create empty dataframe with original index
    df_ret["y_hat"] = y_test_hat_nonans  # Join using indexes
    sr_ret = df_ret["y_hat"]  # This series has all original input indexes but NaNs where input is NaN

    return sr_ret
