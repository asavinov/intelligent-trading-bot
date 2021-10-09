import sys
import os
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from joblib import dump, load

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

from tensorflow.keras.optimizers import *
from keras.regularizers import *
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import *

#
# GB
#

def train_predict_gb(df_X, df_y, df_X_test, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    models = train_gb(df_X, df_y, params)
    y_test_hat = predict_gb(models, df_X_test)
    return y_test_hat

def train_gb(df_X, df_y, params: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    is_scale = params.get("is_scale", False)

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

        'objective': objective, # binary cross_entropy cross_entropy_lambda

        'metric': {'cross_entropy'},  # auc auc_mu map (mean_average_precision) cross_entropy binary_logloss cross_entropy_lambda binary_error

        'verbose': 0,
    }

    model = lgbm.train(
        lgbm_params,
        train_set=lgbm.Dataset(X_train, y_train),
        num_boost_round=num_boost_round,
        #valid_sets=[lgbm.Dataset(X_validate, y_validate)],
        #early_stopping_rounds=int(num_boost_round / 5),
        verbose_eval=100,
    )

    return (model, scaler)

def predict_gb(models: tuple, df_X_test):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    scaler = models[1]
    is_scale = scaler is not None

    input_index = df_X_test.index
    if is_scale:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)
    else:
        df_X_test = df_X_test

    df_X_test_nonans = df_X_test.dropna()  # Drop nans, create gaps in index
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

def train_predict_nn(df_X, df_y, df_X_test, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    models = train_nn(df_X, df_y, params)
    y_test_hat = predict_nn(models, df_X_test)
    return y_test_hat

def train_nn(df_X, df_y, params: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    is_scale = params.get("is_scale", True)

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
    n_features = X_train.shape[1]
    layers = params.get("layers")  # List of ints
    learning_rate = params.get("learning_rate")
    n_epochs = params.get("n_epochs")
    batch_size = params.get("bs")

    # Topology
    model = Sequential()
    # sigmoid, relu, tanh, selu, elu, exponential
    # kernel_regularizer=l2(0.001)

    reg_l2 = 0.001

    model.add(
        Dense(n_features, activation='sigmoid', input_dim=n_features)  # , kernel_regularizer=l2(reg_l2)
    )

    model.add(Dense(n_features // 2, activation='sigmoid'))  # One hidden layer

    #model.add(Dense(layers[0], activation='sigmoid', input_dim=n_features, kernel_regularizer=l2(reg_l2)))
    #if len(layers) > 1:
    #    model.add(Dense(layers[1], activation='sigmoid', kernel_regularizer=l2(reg_l2)))
    #if len(layers) > 2:
    #    model.add(Dense(layers[2], activation='sigmoid', kernel_regularizer=l2(reg_l2)))

    model.add(
        Dense(1, activation='sigmoid')
    )

    # Compile model
    optimizer = Adam(lr=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    es = EarlyStopping(
        monitor="loss",  # val_loss loss
        min_delta=0.0001,  # Minimum change qualified as improvement
        patience=0,  # Number of epochs with no improvements
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
        #validation_data=(X_validate, y_validate),
        #class_weight={0: 1, 1: 20},
        callbacks=[es],
        verbose=0,
    )

    return (model, scaler)

def predict_nn(models: tuple, df_X_test):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    scaler = models[1]
    is_scale = scaler is not None

    input_index = df_X_test.index
    if is_scale:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)
    else:
        df_X_test = df_X_test

    df_X_test_nonans = df_X_test.dropna()  # Drop nans, create gaps in index
    nonans_index = df_X_test_nonans.index

    y_test_hat_nonans = models[0].predict(df_X_test_nonans.values)  # NN returns matrix with one column as prediction
    y_test_hat_nonans = y_test_hat_nonans[:, 0]  # Or y_test_hat.flatten()
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)  # Attach indexes with gaps

    df_ret = pd.DataFrame(index=input_index)  # Create empty dataframe with original index
    df_ret["y_hat"] = y_test_hat_nonans  # Join using indexes
    sr_ret = df_ret["y_hat"]  # This series has all original input indexes but NaNs where input is NaN

    return sr_ret

#
# LC - Linear Classifier
#

def train_predict_lc(df_X, df_y, df_X_test, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    models = train_lc(df_X, df_y, params)
    y_test_hat = predict_lc(models, df_X_test)
    return y_test_hat

def train_lc(df_X, df_y, params: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    is_scale = params.get("is_scale", True)

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
    args = params.copy()
    del args["is_scale"]
    args["n_jobs"] = 1
    model = LogisticRegression(**args)

    #
    # Train
    #
    model.fit(X_train, y_train)

    return (model, scaler)

def predict_lc(models: tuple, df_X_test):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    scaler = models[1]
    is_scale = scaler is not None

    input_index = df_X_test.index
    if is_scale:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)
    else:
        df_X_test = df_X_test

    df_X_test_nonans = df_X_test.dropna()  # Drop nans, create gaps in index
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

def train_predict_svc(df_X, df_y, df_X_test, params: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    models = train_lc(df_X, df_y, params)
    y_test_hat = predict_lc(models, df_X_test)
    return y_test_hat

def train_svc(df_X, df_y, params: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    is_scale = params.get("is_scale", True)

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
    args = params.copy()
    del args["is_scale"]
    args['probability'] = True  # Required to use predict_proba()
    model = SVC(**args)

    #
    # Train
    #
    model.fit(X_train, y_train)

    return (model, scaler)

def predict_svc(models: tuple, df_X_test):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    scaler = models[1]
    is_scale = scaler is not None

    input_index = df_X_test.index
    if is_scale:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)
    else:
        df_X_test = df_X_test

    df_X_test_nonans = df_X_test.dropna()  # Drop nans, create gaps in index
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

def save_model_pair(model_path, score_column_name: str, model_pair: tuple):
    """Save two models in two files with the corresponding extensions."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    model = model_pair[0]
    scaler = model_pair[1]
    # Save scaler
    scaler_file_name = model_path.joinpath(score_column_name).with_suffix(".scaler")
    dump(scaler, scaler_file_name)
    # Save prediction model
    if score_column_name.endswith("_nn"):
        model_extension = ".h5"
        model_file_name = model_path.joinpath(score_column_name).with_suffix(model_extension)
        save_model(model, model_file_name)
    else:
        model_extension = ".pickle"
        model_file_name = model_path.joinpath(score_column_name).with_suffix(model_extension)
        dump(model, model_file_name)

def load_model_pair(model_path, score_column_name: str):
    """Load a pair consisting of scaler model (possibly null) and prediction model from two files."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    # Load scaler
    scaler_file_name = model_path.joinpath(score_column_name).with_suffix(".scaler")
    scaler = load(scaler_file_name)
    # Load prediction model
    if score_column_name.endswith("_nn"):
        model_extension = ".h5"
        model_file_name = model_path.joinpath(score_column_name).with_suffix(model_extension)
        model = load_model(model_file_name)
    else:
        model_extension = ".pickle"
        model_file_name = model_path.joinpath(score_column_name).with_suffix(model_extension)
        model = load(model_file_name)

    return (model, scaler)

def load_models(model_path, labels: list, feature_sets: list, algorithms: list):
    """Load all model pairs for all combinations of the model parameters and return as a dict."""
    models = {}
    for predicted_label in itertools.product(labels, feature_sets, algorithms):
        score_column_name = predicted_label[0] + "_" + predicted_label[1][0] + "_" + predicted_label[2]
        model_pair = load_model_pair(model_path, score_column_name)
        models[score_column_name] = model_pair
    return models

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


if __name__ == '__main__':
    pass
