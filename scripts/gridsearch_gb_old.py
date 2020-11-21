from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgbm


#from common.utils import *
#from trade.feature_generation import *


# ===
seed = 42
np.random.seed(seed)

def main():
    #
    # Load data frame
    #
    data_file_name = r"C:\DATA2\BITCOIN\BTCUSDT-1m-data.csv"  # ETHBTC BTCUSDT IOTAUSDT
    data_df = pd.read_csv(data_file_name, parse_dates=['timestamp'])

    start = int(len(data_df) * 0.1)
    end = -1
    df = data_df.iloc[start:end]

    print(f"Data loaded. Length {len(data_df)}")

    #
    # Generate label (future window maximum)
    #
    growth = 1.0
    add_label_column(df, 60, growth, max_column_name='high', ref_column_name='close', out_column_name='label')
    # NOTE: It will remove rows with None labels (because of absent future)

    negative_samples = len(df[df['label'] < 0.5])
    positive_samples = len(df[df['label'] >= 0.5])
    scale_pos_weight = negative_samples / positive_samples

    print(f"Label generated. Growth {growth}. Positiv weight {scale_pos_weight:.2f}")

    #
    # Generate features
    #
    columns = []
    #columns += ['close']

    # TODO: Check that relative values use same scale (how?)
    # TODO: We can try to work with diffs (of close). Convert close to diffs (from previous) and then apply all further steps for analysis
    # NOTE: It will NOT remove rows with None feature values

    # If necessary, switch to differences
    #df['close'] = to_diff(df['close'])
    #df['volume'] = to_diff(df['volume'])
    #df['trades'] = to_diff(df['trades'])

    df = add_relative_aggregations(df, 'close', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    columns += ['close_1', 'close_2', 'close_5', 'close_20', 'close_60', 'close_180']

    df = add_relative_aggregations(df, 'close', [1, 2, 5, 20, 60, 180, 300], np.std, '_std')
    columns += ['close_std_1', 'close_std_2', 'close_std_5', 'close_std_20', 'close_std_60', 'close_std_180']

    df = add_relative_aggregations(df, 'volume', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    columns += ['volume_1', 'volume_2', 'volume_5', 'volume_20', 'volume_60', 'volume_180']

    df = add_relative_aggregations(df, 'trades', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    columns += ['trades_1', 'trades_2', 'trades_5', 'trades_20', 'trades_60', 'trades_180']

    # high-low difference
    df['span'] = df['high']-df['low']
    df = add_relative_aggregations(df, 'span', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    columns += ['span_1', 'span_2', 'span_5', 'span_20', 'span_60', 'span_180']

    # quote_av,tb_base_av,tb_quote_av

    # tb_base_av / volume varies around 0.5 in base currancy
    df['tb_base'] = df['tb_base_av'] / df['volume']
    df = add_relative_aggregations(df, 'tb_base', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    columns += ['tb_base_1', 'tb_base_2', 'tb_base_5', 'tb_base_20', 'tb_base_60', 'tb_base_180']

    # tb_quote_av / quote_av varies around 0.5 in quote currancy
    df['tb_quote'] = df['tb_quote_av'] / df['quote_av']
    df = add_relative_aggregations(df, 'tb_quote', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    columns += ['tb_quote_1', 'tb_quote_2', 'tb_quote_5', 'tb_quote_20', 'tb_quote_60', 'tb_quote_180']

    #df = add_relative_aggregations(df, 'quote_av', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    #columns += ['quote_av_1', 'quote_av_2', 'quote_av_5', 'quote_av_20', 'quote_av_60', 'quote_av_180']

    #df = add_relative_aggregations(df, 'tb_base_av', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    #columns += ['tb_base_av_1', 'tb_base_av_2', 'tb_base_av_5', 'tb_base_av_20', 'tb_base_av_60', 'tb_base_av_180']

    #df = add_relative_aggregations(df, 'tb_quote_av', [1, 2, 5, 20, 60, 180, 300], np.mean, '')
    #columns += ['tb_quote_av_1', 'tb_quote_av_2', 'tb_quote_av_5', 'tb_quote_av_20', 'tb_quote_av_60', 'tb_quote_av_180']

    print(f"Features generated. Number of features {len(columns)}")

    #
    # Split
    #
    df.dropna(subset=columns, inplace=True)
    df.dropna(subset=['label'], inplace=True)

    X = df[columns].values
    y = df['label'].values
    y = y.reshape(-1)

    X_train, X_validate, y_train, y_validate = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=True,
        stratify=y,
        #random_state=1234
    )


    #
    # Train and predict
    #
    #y_hat_validate = train_predict_gb(X_train, y_train, X_validate, y_validate)
    y_hat_validate = train_predict_knn(X_train, y_train, X_validate, y_validate)

    #
    # Test and score
    #
    print(f"\nScores:\n")

    aucroc = metrics.roc_auc_score(y_validate, y_hat_validate)
    #aucroc_all = metrics.roc_auc_score(y_train, y_hat_train)
    print(f"AUC: validate {aucroc:.3f}")
    #print(f"AUC: train {aucroc_all:.3f}")

    precision, recall, thresholds = metrics.precision_recall_curve(y_validate, y_hat_validate)
    N = len(thresholds)
    N_05 = int(N*0.5)
    N_06 = int(N*0.6)
    N_07 = int(N*0.7)
    N_08 = int(N*0.8)
    N_09 = int(N*0.9)
    N_095 = int(N*0.95)
    print("\n")
    print(f"Precision by threshold (validate): 0.5 = {precision[N_05]:.3f}; 0.6 = {precision[N_06]:.3f}; 0.7 = {precision[N_07]:.3f}; 0.8 = {precision[N_08]:.3f}; 0.9 = {precision[N_09]:.3f}; 0.95 = {precision[N_095]:.3f};")

    #average_prec = metrics.average_precision_score(y_validate, y_hat_validate)
    #average_prec_all = metrics.average_precision_score(y, y_hat)
    #print(f"AVE PREC: validate {average_prec:.3f}, all {average_prec_all:.3f}")

    #f1 = metrics.f1_score(y_validate, y_hat_validate_bin)
    #f1_all = metrics.f1_score(y, y_hat_validate_bin_all)
    #print(f"F1 ({threshold}): validate {f1:.3f}, all {f1_all:.3f}")

    #prec = metrics.precision_score(y_validate, y_hat_validate_bin)
    #prec_all = metrics.precision_score(y, y_hat_validate_bin_all)
    #print(f"PREC ({threshold}): validate {prec:.3f}, all {prec_all:.3f}")

    #cm = metrics.confusion_matrix(y_validate, y_hat_validate_bin)
    #cm_all = metrics.confusion_matrix(y, y_hat_validate_bin_all)
    #print(cm)
    #print(cm_all)

    threshold = 0.5
    #y_hat_train_bin = y_hat_train > threshold
    y_hat_validate_bin = y_hat_validate > threshold

    cr = metrics.classification_report(y_validate, y_hat_validate_bin)
    #cr_all = metrics.classification_report(y_train, y_hat_train_bin)
    print(cr)
    #print("\n")
    #print(cr_all)

    # pr, re, f1, su = precision_recall_fscore_support(y_validate, y_hat_validate_bin, pos_label=1, average=None)

    pass

def weighted_layered_train_predict(X_train, y_train, X_validate, y_validate, layers, train_predict_fn):
    """
    Generate predictions by using weighted average of several models trained on different histories: from long temr to short term.
    It is actually one step in training, that is, training means producing a list of columns each having some weight (weights are parameters)
    We also modify predict function which now applies all models from the list and then averaging their results using the weights
    Parameters: a list of lengths (and their weights) starting from smallest and ending with largest (very large number means all available data)
    The function will select the train data according to this length starting from now and back to the specified length.
    Then it will be used for training and the model stored in the output list

    layers is a list of tuples where a tuple stores: historic length for traing (-1 means all), weight in the result
    return scores for the whole validate set by weigting all models described in the layer parameter
    """

    y_hat_validate = None
    sum_of_weights = 0
    for layer in layers:
        # Prepare slice for training the model
        length = min(layer[0], len(X_train))
        if length > 0:
            X_layer = X_train[-length:-1]
            y_layer = y_train[-length:-1]
        else:
            X_layer = X_train
            y_layer = y_train
    
        # Train on this slice and predict by always applying to the same predict set
        layer_prediction = train_predict_fn(X_layer, y_layer, X_validate, y_validate)

        weight = layer[1]
        # Multiply by the layer weight
        if y_hat_validate is None:
            y_hat_validate = layer_prediction * weight
        else:
            y_hat_validate += layer_prediction * weight

        sum_of_weights += weight

    y_hat_validate /= sum_of_weights

    return y_hat_validate

def train_predict_gb(X_train, y_train, X_validate, y_validate):
    #
    # Train model
    #
    lgbm_params = {
        'learning_rate': 0.1,
        'max_depth': 10,
        #"n_estimators": 10000,

        #"min_split_gain": params['min_split_gain'],
        #"min_data_in_leaf": params['min_data_in_leaf'],
        #'num_leaves': params['num_leaves'],

        'boosting_type': 'gbdt',  # dart (slow but best, worse than gbdt), goss, gbdt

        'objective': 'cross_entropy', # binary cross_entropy cross_entropy_lambda
        'is_unbalance': 'true',
        #'scale_pos_weight': scale_pos_weight,  # is_unbalance must be false

        'metric': {'cross_entropy'},  # auc binary_logloss cross_entropy cross_entropy_lambda binary_error
    }

    model = lgbm.train(
        lgbm_params,
        train_set=lgbm.Dataset(X_train, y_train),
        num_boost_round=20_000,
        valid_sets=[lgbm.Dataset(X_validate, y_validate)],
        early_stopping_rounds=1_000,
        verbose_eval=100,
    )

    print(f"Model trained.")

    #
    # Predict
    #

    # Predict test
    y_hat_validate = model.predict(X_validate)

    print(f"Results predicted.")

    return y_hat_validate

def train_predict_knn(X_train, y_train, X_validate, y_validate):
    """
    IMPROVEMENTS:
    - We can perform several classifications: small latest window, larger latest window and so on by adding more historical records
      All these classifications are then aggregated using weighted average: results with older history get smaller weithers, which recent result get higher weight
    - We can perform several classifications with different k (neighbors).
      We can simply find average.
      Standard deviation can show confidence - how reliable the result is, e.g., if k=3,5,7 are approximately the same then this might be treated as high confidence
      !!! Same can be done with any other approach: we train short term and long term models, and then aggregate their results using corresonding weights, as well as estimating devision as an indicator of significance.
    """
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=21).fit(X_train, y_train)

    print(f"Model trained.")

    # Predict all (train+test)

    # Predict test
    y_hat_validate = model.predict_proba(X_validate)

    print(f"Results predicted.")

    return y_hat_validate[:,1]

def find_frequencies_of_labels():
    #
    # Load data frame
    #
    data_file_name = r"C:\DATA2\BITCOIN\BTCUSDT-1m-data.csv"  # ETHBTC BTCUSDT IOTAUSDT
    data_df = pd.read_csv(data_file_name, parse_dates=['timestamp'])

    start = int(len(data_df) * 0.0)
    end = -1
    df = data_df.iloc[start:end]

    print(f"Data loaded. Length {len(data_df)}")

    #
    # Generate label (future window maximum)
    #
    add_future_min_percent(df, 30, min_column_name='low', ref_column_name='close', out_column_name='label_1')
    add_future_min_percent(df, 60, min_column_name='low', ref_column_name='close', out_column_name='label_2')
    add_future_min_percent(df, 90, min_column_name='low', ref_column_name='close', out_column_name='label_3')
    add_future_min_percent(df, 120, min_column_name='low', ref_column_name='close', out_column_name='label_4')

    no_higher = len(df[df['label_1'] > 0.5])


def klines_to_csv():
    klines = pd.read_pickle(r"C:\DATA2\BITCOIN\ETHBTC-1m-data.pkl")  # ETHBTC BTCUSDT
    # Length: 1107636, 1163506
    # Row: [1502942400000, '4261.48000000', '4261.48000000', '4261.48000000', '4261.48000000', '1.77518300', 1502942459999, '7564.90685084', 3, '0.07518300', '320.39085084', '7960.54017996']
    df = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    df.to_csv(r"C:\DATA2\BITCOIN\ETHBTC-1m-data.csv")

def search_signal_model():
    """
    Load file with many rolling predictions and labels, and train a good model which is intended for signal generation
    """

    # Best results auc (LogisticRegression): high_60_10 -> ~0.77, high_60_15 -> ~0.81, high_60_20 -> ~0.85
    # Best results precision (LogisticRegression): high_60_10 -> ~0.77, high_60_15 -> ~0.70, high_60_20 -> ~0.68

    # Try random forest or gradient boosting
    # Next: score threshold for getting required accuracy, say, 95%?
    # Next: can we increase accuracy by using all 3 scores? Maybe again a linear model?
    # Next: check how important it is to use all 12 input scores - maybe we can get same results using only individual scores or a few scores

    # high_60_10,high_60_15,high_60_20,low_60_10,low_60_15,low_60_20
    label = "high_60_10"
    features = [
        "high_60_10_k_12", "high_60_15_k_12", "high_60_20_k_12", "low_60_10_k_12", "low_60_15_k_12", "low_60_20_k_12",
        "high_60_10_f_03", "high_60_15_f_03", "high_60_20_f_03", "low_60_10_f_03", "low_60_15_f_03", "low_60_20_f_03",
    ]

    scoring = {
        'roc_auc': None,
        'accuracy': make_scorer(accuracy_score),
        #'precision': make_scorer(precision_score),
        #'recall': make_scorer(recall_score),
    }
    #scoring = 'f1'
    #scoring='accuracy'
    scoring='precision'
    #scoring = 'roc_auc'
    #scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}


    in_df = pd.read_csv(r"C:\DATA2\BITCOIN\GENERATED\BTCUSDT-1m-features-rolling.csv", parse_dates=['timestamp'])
    pd.set_option('use_inf_as_na', True)
    in_df = in_df.dropna()
    print(f"Data loaded.")

    X = in_df[features].values
    y = in_df[label].values
    y = y.reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True, stratify=y)

    #
    # SVM
    #

    # base=10 (default), start=base^start, start=base^stop
    # Use base=2 for finer grid: C=[-5,15] ~21, gamma=[-15,3] ~19
    C_range = np.logspace(-5, 10, 2, base=2)
    gamma_range = np.logspace(-15, 1, 2, base=2)
    C_range = [1e-4]
    gamma_range = [1e-5]
    param_grid = dict(C=C_range, gamma=gamma_range)  # gamma=gamma_range
    param_grid = {'C': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, .01, .1]}

    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=42)

    grid = GridSearchCV(
        estimator=svm.LinearSVC(),  #  svm.LinearSVC() svm.SVC()
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        cv=cv,
        refit=False,
        verbose=10,
    )
    #grid.fit(X, y)
    #scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))

    #
    # LogisticRegression
    #
    param_grid = {'C': [0.001, 0.005, .01, .05, .1], 'class_weight': [None, "balanced", 0.5, 1.0, 10.0]}
    grid = GridSearchCV(
        linear_model.LogisticRegression(),
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        cv=5,
        return_train_score=True,
        verbose = 10,
    )
    grid.fit(X, y)
    #print("The best parameters are {} with a score of {:.2f}".format(grid.best_params_, grid.best_score_))

    #
    # Gradient Boosting
    #

    parameters = {
        "loss": ["deviance"],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_samples_split": np.linspace(0.1, 0.5, 12),
        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth": [5],
        #"max_features": ["log2", "sqrt"],
        #"criterion": ["friedman_mse", "mae"],
        "subsample": [0.5, 0.6, 0.7],
        "n_estimators": [10]
    }
    parameters = {
        #"loss": ["deviance"],
        "learning_rate": [0.05],
        #"min_samples_split": [0.005],  # min_samples_split': 0.01: 0.84 -> 0.86
        #"min_samples_leaf": np.linspace(0.1, 0.5, 10),
        "max_depth": [10],
        #"max_features": ["log2", "sqrt"],  # Both are bad
        #"criterion": ["friedman_mse", "mae"],  # Hangs
        #"subsample": [0.5],  # 'subsample': 0.618 0.84 -> 0.86
        "n_estimators": [20]
    }
    grid = GridSearchCV(
        GradientBoostingClassifier(),
        parameters,
        scoring=scoring,
        refit=False,
        cv=5,
        n_jobs=-1,
        verbose=10,
    )
    grid.fit(X, y)

    print("The best parameters are {} with a score of {:.2f}".format(grid.best_params_, grid.best_score_))

    # The best parameters are {'learning_rate': 0.15, 'max_depth': 3, 'n_estimators': 10} with a score of 0.73
    # The best parameters are {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 10} with a score of 0.84
    # The best parameters are {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 20} with a score of 0.84
    # The best parameters are {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 20} with a score of 0.84
    # The best parameters are {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 20, 'subsample': 0.618} with a score of 0.86
    # The best parameters are {'learning_rate': 0.05, 'max_depth': 4, 'min_samples_split': 0.008, 'n_estimators': 20, 'subsample': 0.4} with a score of 0.89
    # The best parameters are {'learning_rate': 0.05, 'max_depth': 4, 'min_samples_split': 0.006, 'n_estimators': 20, 'subsample': 0.6} with a score of 0.89

    #C_2d_range = [1e-2, 1, 1e2]
    #gamma_2d_range = [1e-1, 1, 1e1]
    #models = []
    #for C in C_2d_range:
    #    for gamma in gamma_2d_range:
    #        model = svm.SVC(C=C, gamma=gamma)
    #        model.fit(X, y)
    #        models.append((C, gamma, model))

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
    #model = svm.SVC(C=1.0, gamma=1000.0)
    #model.fit(X_train, y_train)

    #y_train_hat = model.predict(X_train)
    #auc = metrics.roc_auc_score(y_train, y_train_hat)
    #print(f"AUC train: {auc:.3f}")

    #y_test_hat = model.predict(X_test)
    #auc = metrics.roc_auc_score(y_test, y_test_hat)
    #print(f"AUC test: {auc:.3f}")

    pass

def build_roc_curve():
    # high_60_10,high_60_15,high_60_20,low_60_10,low_60_15,low_60_20
    label = "high_60_10"
    features = [
        "high_60_10_k_12", "high_60_15_k_12", "high_60_20_k_12", "low_60_10_k_12", "low_60_15_k_12", "low_60_20_k_12",
        "high_60_10_f_03", "high_60_15_f_03", "high_60_20_f_03", "low_60_10_f_03", "low_60_15_f_03", "low_60_20_f_03",
    ]

    in_df = pd.read_csv(r"C:\DATA2\BITCOIN\GENERATED\BTCUSDT-1m-features-rolling.csv", parse_dates=['timestamp'])
    pd.set_option('use_inf_as_na', True)
    in_df = in_df.dropna()
    print(f"Data loaded.")

    X = in_df[features].values
    y = in_df[label].values
    y = y.reshape(-1)

    model = linear_model.LogisticRegression(C=0.001, class_weight=None)
    model.fit(X, y)

    y_hat = model.decision_function(X)  # Threshold 0.0
    #y_hat = model.predict_proba(X)  # Threshold 0.5
    auc = metrics.roc_auc_score(y, y_hat)

    print("Results".format())

    fpr, tpr, thresholds = roc_curve(y, y_hat)
    plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (area = {auc})")

    precision, recall, thresholds = precision_recall_curve(y, y_hat)
    plt.plot(recall, precision, color='darkorange')

    disp = plot_precision_recall_curve(model, X, y)


    # Find first offset (score) with the desired tpr
    best_tpr = 0.95
    best_score = np.searchsorted(tpr, best_tpr)

    pass

if __name__=='__main__':
    #build_roc_curve()
    search_signal_model()
    pass
