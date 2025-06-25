from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, SVR

import lightgbm as lgbm

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import *
from keras.callbacks import *

#
# GB
#

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


#
# NN
#

def train_predict_nn(df_X, df_y, df_X_test, model_config: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    model_pair = train_nn(df_X, df_y, model_config)
    y_test_hat = predict_nn(model_pair, df_X_test, model_config)
    return y_test_hat


def train_nn(df_X, df_y, model_config: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    params = model_config.get("params", {})

    is_scale = params.get("is_scale", True)
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
    n_features = X_train.shape[1]
    layers = params.get("layers")  # List of ints
    if not layers:
        layers = [n_features // 4]  # Default
    if not isinstance(layers, list):
        layers = [layers]

    # Topology
    model = Sequential()
    # sigmoid, relu, tanh, selu, elu, exponential
    # kernel_regularizer=l2(0.001)

    reg_l2 = 0.001

    train_conf = model_config.get("train", {})
    learning_rate = train_conf.get("learning_rate")
    n_epochs = train_conf.get("n_epochs")
    batch_size = train_conf.get("bs")

    for i, out_features in enumerate(layers):
        in_features = n_features if i == 0 else layers[i-1]
        model.add(Dense(out_features, activation='sigmoid', input_dim=in_features))  # , kernel_regularizer=l2(reg_l2)
        #model.add(Dropout(rate=0.5))

    if is_regression:
        model.add(Dense(units=1))

        model.compile(
            loss='mean_squared_error',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error"),
                tf.keras.metrics.R2Score(name="r2_score"),
            ],
        )
    else:
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    #model.summary()

    # Default arguments for early stopping
    es_args = dict(
        monitor = "loss",  # val_loss loss
        min_delta = 0.00001,  # Minimum change qualified as improvement
        patience = 5,  # Number of epochs with no improvements
        verbose = 0,
        mode = 'auto',
    )
    es_args.update(train_conf.get("es", {}))  # Overwrite default values with those explicitly specified in config

    es = EarlyStopping(**es_args)

    #
    # Train
    #
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        #validation_split=0.05,
        #validation_data=(X_validate, y_validate),
        #class_weight={0: 1, 1: 20},
        callbacks=[es],
        verbose=1,
    )

    return (model, scaler)


def predict_nn(models: tuple, df_X_test, model_config: dict):
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

    # Resets all (global) state generated by Keras
    # Important if prediction is executed in a loop to avoid memory leak
    tf.keras.backend.clear_session()

    y_test_hat_nonans = models[0].predict_on_batch(df_X_test_nonans.values)  # NN returns matrix with one column as prediction
    y_test_hat_nonans = y_test_hat_nonans[:, 0]  # Or y_test_hat.flatten()
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)  # Attach indexes with gaps

    df_ret = pd.DataFrame(index=input_index)  # Create empty dataframe with original index
    df_ret["y_hat"] = y_test_hat_nonans  # Join using indexes
    sr_ret = df_ret["y_hat"]  # This series has all original input indexes but NaNs where input is NaN

    return sr_ret


#
# LC - Linear Classifier
#

def train_predict_lc(df_X, df_y, df_X_test, model_config: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    model_pair = train_lc(df_X, df_y, model_config)
    y_test_hat = predict_lc(model_pair, df_X_test, model_config)
    return y_test_hat


def train_lc(df_X, df_y, model_config: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    params = model_config.get("params", {})

    is_scale = params.get("is_scale", True)
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

    args = train_conf.copy()
    args["n_jobs"] = -1
    args["verbose"] = 0
    model = LogisticRegression(**args)

    #
    # Train
    #
    model.fit(X_train, y_train)

    return (model, scaler)


def predict_lc(models: tuple, df_X_test, model_config: dict):
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

    y_test_hat_nonans = models[0].predict_proba(df_X_test_nonans.values)  # It returns pairs or probas for 0 and 1
    y_test_hat_nonans = y_test_hat_nonans[:, 1]  # Or y_test_hat.flatten()
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)  # Attach indexes with gaps

    df_ret = pd.DataFrame(index=input_index)  # Create empty dataframe with original index
    df_ret["y_hat"] = y_test_hat_nonans  # Join using indexes
    sr_ret = df_ret["y_hat"]  # This series has all original input indexes but NaNs where input is NaN

    return sr_ret


#
# SVC - SVN Classifier
#

def train_predict_svc(df_X, df_y, df_X_test, model_config: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    model_pair = train_svc(df_X, df_y, model_config)
    y_test_hat = predict_svc(model_pair, df_X_test, model_config)
    return y_test_hat


def train_svc(df_X, df_y, model_config: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    params = model_config.get("params", {})

    is_scale = params.get("is_scale", True)

    is_regression = params.get("is_regression", False)

    #
    # Prepare data
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
    args = train_conf.copy()
    if is_regression:
        model = SVR(**args)
    else:
        args['probability'] = True  # Required if we are going to use predict_proba()
        model = SVC(**args)

    #
    # Train
    #
    model.fit(X_train, y_train)

    return (model, scaler)


def predict_svc(models: tuple, df_X_test, model_config: dict):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    is_regression = model_config.get("params", {}).get("is_regression", False)

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

    if is_regression:
        y_test_hat_nonans = models[0].predict(df_X_test_nonans.values)
    else:
        y_test_hat_nonans = models[0].predict_proba(df_X_test_nonans.values)  # It returns pairs or probas for 0 and 1
        y_test_hat_nonans = y_test_hat_nonans[:, 1]  # Or y_test_hat.flatten()
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)  # Attach indexes with gaps

    df_ret = pd.DataFrame(index=input_index)  # Create empty dataframe with original index
    df_ret["y_hat"] = y_test_hat_nonans  # Join using indexes
    sr_ret = df_ret["y_hat"]  # This series has all original input indexes but NaNs where input is NaN

    return sr_ret


#
# Utils
#

def compute_scores(y_true, y_hat):
    """Compute several scores and return them as dict."""
    y_true = y_true.astype(int)
    y_hat_class = np.where(y_hat.values > 0.5, 1, 0)

    try:
        auc = metrics.roc_auc_score(y_true, y_hat.fillna(value=0))
    except ValueError:
        auc = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

    try:
        ap = metrics.average_precision_score(y_true, y_hat.fillna(value=0))
    except ValueError:
        ap = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

    f1 = metrics.f1_score(y_true, y_hat_class)
    precision = metrics.precision_score(y_true, y_hat_class)
    recall = metrics.recall_score(y_true, y_hat_class)

    scores = dict(
        auc=round(auc, 3),
        ap=round(ap, 3),
        f1=round(f1, 3),
        precision=round(precision, 3),
        recall=round(recall, 3),
    )

    return scores


def compute_scores_regression(y_true, y_hat):
    """Compute regression scores. Input columns must have numeric data type."""

    try:
        mae = metrics.mean_absolute_error(y_true, y_hat)
    except ValueError:
        mae = np.nan

    try:
        mape = metrics.mean_absolute_percentage_error(y_true, y_hat)
    except ValueError:
        mape = np.nan

    try:
        r2 = metrics.r2_score(y_true, y_hat)
    except ValueError:
        r2 = np.nan

    #
    # How good it is in predicting the sign (increase of decrease)
    #

    y_true_class = np.where(y_true.values > 0.0, +1, -1)
    y_hat_class = np.where(y_hat.values > 0.0, +1, -1)

    try:
        auc = metrics.roc_auc_score(y_true_class, y_hat_class)
    except ValueError:
        auc = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

    try:
        ap = metrics.average_precision_score(y_true_class, y_hat_class)
    except ValueError:
        ap = 0.0  # Only one class is present (if dataset is too small, e.g,. when debugging) or Nulls in predictions

    f1 = metrics.f1_score(y_true_class, y_hat_class)
    precision = metrics.precision_score(y_true_class, y_hat_class)
    recall = metrics.recall_score(y_true_class, y_hat_class)

    scores = dict(
        mae=round(mae, 3),
        mape=round(mape, 3),
        r2=round(r2, 3),

        auc=round(auc, 3),
        ap=round(ap, 3),
        f1=round(f1, 3),
        precision=round(precision, 3),
        recall=round(recall, 3),
    )

    return scores


if __name__ == '__main__':
    pass
