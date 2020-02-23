from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

import lightgbm as lgbm

from trade.utils import *
from trade.feature_generation import *

"""
Classify time series using gradient boosting.
"""

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


if __name__=='__main__':

    # Study:
    find_frequencies_of_labels()

	# Predicted characteristics:
	# high up (at least one greater than): 0.5, 1.0, 1.5, 2.0, (2.5)
	# high dn (all highs less than): (0.05), 0.1, 0.2, 0.3, 0.4, 0.5
	# low up (all lows greater than): (0.05), 0.1, 0.2, 0.3, 0.4, 0.5 (negative)
	# low dn (at least one less than): 0.5, 1.0, 1.5, 2.0, (2.5) (negative)
	"""
	|
	| 0.5-2.5 high up - at least one greater than   big probability -> buy
	|
	| 0.1-0.5 high dn - all highs less than         big probability -> sell
	| 0
	| 0.1-0.5 low up (all lows greater than)        big probability -> buy
	|
	| 0.5-2.5 low dn (at least one less than)       big probability -> sell
	|
	"""
    # buy signal: increase outside high - big score (say, for 2.0), stay inside high - small score (say, for 0.5), stay inside low - big score (say, for -0.5), increase outside low - small score (say, 2.0)
    # sell signal: opposite
    #   s]    [b   s]    [b  - buy (growing trend)
    #   b]    [s   b]    [s  - sell (falling trend)
    # thresholds for scores can be tuned using back testing and measuring performance
    # strategy: buy when (very) high buy score (buy carefully with high threshold), sell when (relatively low) sell score (sell frequently and quicly)
    # stragety: derive one common score like trend (between -1 and +1), and then buy/sell using thresholds, say, but with high like 0.8 and sell with low like -0.2 or 0.0
    #   assymmetry reflects the assymetry of weights of assets but it is not necessarily best strategy so it is better to back test

    # STATISTICS of levels frequencies
	# OUTSIDE (at least one) (len=1107635)
    # horizon 30 min: (>0.5-2.5): 317551, 131352, 65020, 36626, 21636; 
    # horizon 30 max: (>0.5-2.5): 308848, 118790, 55469, 29778, 17997; 
    # ]317551-308848[ ]131352-118790[ ]65020-55469[ ]36626-29778[ ]21636-17997[ - number of occurances with at least one time outside the interval border

    # horizon 60 min: (>0.5-2.5): 444127, 215635, 120415, 74397, 47682; 
    # horizon 60 max: (>0.5-2.5): 435144, 200594, 106655, 62775, 39177; 

    # horizon 90 min: (>0.5-2.5): 522824, 276701, 164837, 106779, 72314; 
    # horizon 90 max: (>0.5-2.5): 515887, 262866, 149348, 92625, 60050; 

    # horizon 120 min: (>0.5-2.5): 579805, 324882, 201415, 135353, 94411; 
    # horizon 120 max: (>0.5-2.5): 574280, 313174, 185776, 119201, 80311; 

    # INSIDE (all) (len=1107635)
    # horizon 30 low: (<0.01, 0.05, 0.1-0.5): 64942, 156472, 271713, 462498, 606882, 712723, 790033
    # horizon 30 high: (<0.01, 0.05, 0.1-0.5): 57925, 151666, 270548, 468439, 614395, 720676, 798739
    # [64942-57925] [156472-151666] [271713-270548] [462498-468439] [606882-614395] [712723-720676] [790033-798739] - number of occurances with all elements within the border

    # horizon 60 low: (<0.01, 0.05, 0.1-0.5): 44796, 107663, 191312, 345033, 475932, 580733, 663429
    # horizon 60 high: (<0.01, 0.05, 0.1-0.5): 39403, 104023, 190178, 348970, 482077, 588042, 672415

    # horizon 90 low: (<0.01, 0.05, 0.1-0.5): 35986, 85943, 153709, 284680, 402750, 501855, 584698
    # horizon 90 high: (<0.01, 0.05, 0.1-0.5): 31078, 82247, 152228, 287752, 407646, 508420, 591650

    # horizon 120 low: (<0.01, 0.05, 0.1-0.5): 30813, 73062, 130838, 245630, 353362, 446662, 527692
    # horizon 120 high: (<0.01, 0.05, 0.1-0.5): 26220, 69638, 129221, 248650, 358285, 452793, 533225


    # Drift: measure how various characteristics change in time: 
	# - price variance (volatility), number of levels (histogram)
	# - predicted parameter frequencies (average count, variance etc.)


    # PLAN: Launch and iterate: implement a primitive "load-features-predict-signal-buy/sell" loop and then improve it
    # - feature generation, (for training, label generation), 
	# - predict by applying (previously trained) models to generated different labels
    # - generate buy-sell signals by (rule-based) analyzing predicted labels (scores)
    # - buy-sell by create order objects, assume that it is immediately executed, computed its execution price, update the current state and logs
    # - repeat - note that the most important function is to sell the previously bought asset 
    #   - does it make sense to train a model specifically for selling what we already have?
    #   - what would be criteria for such a "right moment for selling" model in the sense of label (loss)? if we sell now, is it good or bad?
    #   - note that we cannot measure performance (loss) in terms of buy price - this price is already history and we have new situation and more data
    #     sell quality depends on the probability to lose in short term, 
    #     in other words, we sell what is going to decrease its value (and buy what is going to increase its value)
    #   - therefore, generate sell signal if price is going down (and/or is not going to grow): decrease outside high, stay inside low
    #     similarly, we buy if price is going up (and/or is not going to fall): increase outside low high, stay inside high low.

    #main()

    #df = pd.read_pickle("_15__1_0__5.pkl")
    #main_gb(df)

    # TODO:
    # - Develop pipeline for validation where model is trained on (large) historic data, and prediciton is done on (small) next data
    #   Then we move forward by including the just predicted data into the training set, and repeat the procedure
    #   The results are collected and then accuracy is computed as usual
    #   - Parameters: train (history) size, predict (future) size, start iteration, end iteration
    #   - It would be interesting to compare precision with the standard approach
    
    pass
