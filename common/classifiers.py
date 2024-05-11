from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

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
    #
    # Double column set if required
    #
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        max_shift = max(shifts)
        df_X = double_columns(df_X, shifts)
        df_X = df_X.iloc[max_shift:]
        df_y = df_y.iloc[max_shift:]

    #
    # Scale
    #
    is_scale = model_config.get("train", {}).get("is_scale", False)
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
    params = model_config.get("params")

    objective = params.get("objective")

    max_depth = params.get("max_depth")
    learning_rate = params.get("learning_rate")
    num_boost_round = params.get("num_boost_round")

    lambda_l1 = params.get("lambda_l1")
    lambda_l2 = params.get("lambda_l2")

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
    # Double column set if required
    #
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        df_X_test = double_columns(df_X_test, shifts)

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
    #
    # Double column set if required
    #
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        max_shift = max(shifts)
        df_X = double_columns(df_X, shifts)
        df_X = df_X.iloc[max_shift:]
        df_y = df_y.iloc[max_shift:]

    #
    # Scale
    #
    is_scale = model_config.get("train", {}).get("is_scale", True)
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
    params = model_config.get("params")

    n_features = X_train.shape[1]
    layers = params.get("layers")  # List of ints
    if not layers:
        layers = [n_features // 4]  # Default
    if not isinstance(layers, list):
        layers = [layers]
    learning_rate = params.get("learning_rate")
    n_epochs = params.get("n_epochs")
    batch_size = params.get("bs")

    # Topology
    model = Sequential()
    # sigmoid, relu, tanh, selu, elu, exponential
    # kernel_regularizer=l2(0.001)

    reg_l2 = 0.001

    for i, out_features in enumerate(layers):
        in_features = n_features if i == 0 else layers[i-1]
        model.add(Dense(out_features, activation='sigmoid', input_dim=in_features))  # , kernel_regularizer=l2(reg_l2)
        #model.add(Dropout(rate=0.5))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    #model.summary()

    es = EarlyStopping(
        monitor="loss",  # val_loss loss
        min_delta=0.0001,  # Minimum change qualified as improvement
        patience=3,  # Number of epochs with no improvements
        verbose=0,
        mode='auto',
    )

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
    # Double column set if required
    #
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        df_X_test = double_columns(df_X_test, shifts)

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
    #
    # Double column set if required
    #
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        max_shift = max(shifts)
        df_X = double_columns(df_X, shifts)
        df_X = df_X.iloc[max_shift:]
        df_y = df_y.iloc[max_shift:]

    #
    # Scale
    #
    is_scale = model_config.get("train", {}).get("is_scale", True)
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
    args = model_config.get("params").copy()
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
    # Double column set if required
    #
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        df_X_test = double_columns(df_X_test, shifts)

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
    is_scale = model_config.get("train", {}).get("is_scale", True)

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
    args = model_config.get("params").copy()
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
    #
    # Double column set if required
    #
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        df_X_test = double_columns(df_X_test, shifts)

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
        auc=auc,
        ap=ap,  # it summarizes precision-recall curve, should be equivalent to auc
        f1=f1,
        precision=precision,
        recall=recall,
    )

    return scores


def double_columns(df, shifts: List[int]):
    if not shifts:
        return df
    df_list = [df.shift(shift) for shift in shifts]
    df_list.insert(0, df)
    max_shift = max(shifts)

    # Shift and add same columns
    df_out = pd.concat(df_list, axis=1)  # keys=('A', 'B')

    return df_out


if __name__ == '__main__':
    pass
