from __future__ import annotations  # Eliminates problem with type annotations like list[int]
import os
from datetime import datetime, timezone, timedelta
from typing import Union
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors

import lightgbm as lgbm

from trade.feature_generation import *
from trade.label_generation import *

"""
Feature prediction.
These functions allow for training transformation parameters from the historic data.
"""

def predict_labels(df, models: dict):
    """
    Generate a number of labels (features) from the models.
    This function is used only during prediction and the all necessary features must be already in the input data.

    Notes:
    - *all* columns will be used as inputs for prediction
    - input data frame cannot contain None or NaN
    - There can be different kinds of models and prediction is performed using predict() function (which therefore must exist)
    - Output features produced have the names of the labels

    :param models: A dictionary with label names as keys and model objects as values
    :return: A list of label column names added to the data frame
    """
    X = df.values
    out_df = pd.DataFrame(index=df.index)

    for label, model in models.items():
        y_hat = model.predict(X)
        out_df[label] = y_hat

    return out_df

def train_models_1(df):
    """
    Use historic data to produce (future) labels and train multiple models for all of them.
    This function uses a certain set of generated features. Precisely these features must be present when applying these models.

    :return: A dict with label names as keys and model objects as values.
    """
    features = generate_features(df)

    labels = generate_labels_thresholds(df, horizon=60)

    df.dropna(subset=features, inplace=True)
    df.dropna(subset=labels, inplace=True)

    X = df[features].values

    # Train a model for each label
    models = {}
    for label in labels:
        y = df[label].values
        y = y.reshape(-1)
        model = train_model_gb_classifier(X, y)
        models[label] = model

    return models

def train_model_gb_classifier(X, y, params={}):
    """Train gb using the features for the label and return a model for the label"""

    train_fraction = 0.90
    length = len(X)
    val_start = int(length * train_fraction)

    X_train = X[0:val_start]
    y_train = y[0:val_start]

    X_validate = X[val_start:length]
    y_validate = y[val_start:length]

    #
    # Hyper-parameters
    #
    max_depth = params.get("max_depth", 1)  # 10 (long, production)
    learning_rate = params.get("learning_rate", 0.1)
    num_boost_round = int(params.get("num_boost_round", 1_000))  # 20_000

    params_classifier = {
        'boosting_type': 'gbdt',  # dart (slow but best, worse than gbdt), goss, gbdt
        'objective': 'cross_entropy', # binary cross_entropy cross_entropy_lambda
        'metric': {'cross_entropy'},  # auc binary_logloss cross_entropy cross_entropy_lambda binary_error
        'max_depth': max_depth,
        'learning_rate': learning_rate,

        #"n_estimators": 10000,
        #"min_split_gain": params['min_split_gain'],
        #"min_data_in_leaf": params['min_data_in_leaf'],
        # TODO: [LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves
        #'num_leaves': params['num_leaves'],

        'is_unbalance': 'true',
        #'scale_pos_weight': scale_pos_weight,  # is_unbalance must be false

        'verbose': -1,  # Suppress warnings
    }

    model = lgbm.train(
        params_classifier,
        train_set=lgbm.Dataset(X_train, y_train),
        valid_sets=[lgbm.Dataset(X_validate, y_validate)],
        num_boost_round=num_boost_round,
        early_stopping_rounds=int(num_boost_round / 5),
        verbose_eval=-1,
    )

    return model

def train_model_gb_regressor(X, y):
    """Train gb using the features for the label and return a model for the label"""

    # Split
    X_train, X_validate, y_train, y_validate = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=True,
        #stratify=y,
        #random_state=1234
    )

    params_regressor = {
        'boosting_type': 'gbdt',  # dart (slow but best, worse than gbdt), goss, gbdt
        'objective': 'regression',
        'metric': 'rlsme',  # TODO: Check other metrics
        'max_depth': 1,
        #'num_leaves': 6,  # [LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves
        'learning_rate': 0.2,
        'verbose': -1,  # -1 to completely suppress warnings
    }

    model = lgbm.train(
        params_regressor,
        train_set=lgbm.Dataset(X_train, y_train),
        valid_sets=[lgbm.Dataset(X_validate, y_validate)],
        num_boost_round=100,
        #early_stopping_rounds=5_000,  # ERROR: ValueError: For early stopping, at least one dataset and eval metric is required for evaluation
        verbose_eval=100,
    )

    return model

def train_model_knn_classifier(X, y, params={}):
    """Create knn classifier"""

    #
    # Hyper-parameters
    #
    n_neighbors = int(params.get("n_neighbors", 20))
    weights = params.get("weights", "distance")  # ['uniform', 'distance']
    n_jobs = -1

    model = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, n_jobs=n_jobs)
    # RadiusNeighborsClassifier

    model.fit(X, y)

    return model

def predict_model_knn(X, model):
    """Find predictions for the input data by using the provided data as a model"""
    #y_hat = model.predict(X)  # Returns boolean values
    y_hat = model.predict_proba(X)  # Returns probabilities for 2 classes as a matrix
    y_hat = y_hat[:,1]  # Get second columns. It represents probabilities of second class
    return y_hat
