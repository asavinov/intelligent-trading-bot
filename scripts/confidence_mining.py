from pathlib import Path

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from service.App import App
from common.signal_generation import *


#
# Parameters
#
class P:
    feature_sets = ["kline", ]  # "futur"

    labels = App.config["labels"]
    features_kline = App.config["features_kline"]
    features_futur = App.config["features_futur"]
    features_depth = App.config["features_depth"]

    in_path_name = r"C:\DATA2\BITCOIN\GENERATED"
    # in_file_name = r"_BTCUSDT-1m-rolling-predictions-no-weights.csv"
    # in_file_name = r"_BTCUSDT-1m-rolling-predictions-with-weights.csv"
    in_file_name = r"BTCUSDT-1m-features-rolling.csv"
    in_nrows = 1_500_000

    out_path_name = r"_TEMP_FEATURES"
    out_file_name = r"_BTCUSDT-1m-signals"

    simulation_start = 263519  # Good start is 2019-06-01 - after it we have stable movement
    simulation_end = -0  # After 2020-11-01 there is sharp growth which we might want to exclude


def partition_column(sr: pd.Series):
    """Find conditions for partitioning the specified column and return binary columns with true
    values with rows included in one partition.
    """
    # Find mean value
    # Return two boolean series
    pass


def confidence_mining():
    """
    """
    min_rank = 3
    max_rank = 3

    features = P.features_kline  # Input features to be mined
    feature_true = P.labels[0]  # Label (binary)
    feature_score = 'score'  # Prediction score

    #
    # Load data, partition columns,
    #
    in_path = Path(P.in_path_name).joinpath(P.in_file_name)
    in_df = pd.read_csv(in_path, parse_dates=['timestamp'], nrows=P.in_nrows)

    in_df = generate_score_high_low(in_df, P.feature_sets)  # "score" columns is added
    # TODO: Our predictions are scores, therefore generate binary prediction (with some threshold, maybe derived from average score)
    feature_pred = ''  # Prediction (binary)

    # in_df = in_df.iloc[100_000:200_000]

    # Load feature file (because it is not stored in rolling predictions)
    f_df = pd.read_csv(r"C:\DATA2\BITCOIN\GENERATED\BTCUSDT-1m-features.csv", parse_dates=['timestamp'],
                       nrows=500_000_000)
    f_df = f_df[['timestamp'] + P.features_kline]

    df = in_df.merge(f_df, left_on='timestamp', right_on='timestamp')
    pd.set_option('use_inf_as_na', True)
    df = df.dropna(subset=features + [feature_true, feature_score])
    N = len(df)
    print(f"Size of the data set {N}. Features: {len(features)}")

    auc_mean = roc_auc_score(df[feature_true], df[feature_score])
    print(f"Mean AUC: {auc_mean}")

    #
    # Partition individual features
    #
    feature_partitions = dict()  # Key is feature, value is a list of binary maps
    for f in features:
        sr = df[f]
        f_mean = np.nanmean(sr)
        f_std = np.nanstd(sr)

        p1 = norm.ppf(0.3333333333, loc=f_mean, scale=f_std)
        p2 = norm.ppf(0.6666666666, loc=f_mean, scale=f_std)

        p1, p2 = list(np.quantile(sr, [0.3333333333, 0.6666666666]))

        # Two partitions
        #f_0 = (sr < f_mean)
        #f_1 = (sr >= f_mean)

        # Three partitions
        f_0 = (sr <= p1)
        f_1 = ((p1 < sr) & (sr < p2))
        f_2 = (sr >= p2)

        feature_partitions[f] = [f_0, f_1, f_2]

    #
    # For each combination of variable values, find the subset (using AND)
    #
    start_dt = datetime.now()

    scores = list()
    for r in range(min_rank, max_rank + 1):
        feature_combinations = list(itertools.combinations(features, r))
        print(f"Rank {r}. Feature combinations: {len(feature_combinations)}")
        itemsets = list(itertools.product([0, 1, 2], repeat=r))
        for fset_no, fset in enumerate(feature_combinations):
            if fset_no % 100 == 0:
                print(f"Feature set no: {fset_no}/{len(feature_combinations)}")

            # For this itemset, find all combinations of partitions
            for iset in itemsets:
                subset = np.ones(N, dtype=bool)  # All 1s
                for i in range(r):
                    f = fset[i]
                    if iset[i] == 0:
                        np.logical_and(subset, feature_partitions[f][0], out=subset)
                    elif iset[i] == 1:
                        np.logical_and(subset, feature_partitions[f][1], out=subset)
                    elif iset[i] == 2:
                        np.logical_and(subset, feature_partitions[f][2], out=subset)
                    else:
                        raise ValueError()
                # Select data using the subset
                subset_df = df[subset]

                auc = 0.0
                try:
                    auc = roc_auc_score(subset_df[feature_true], subset_df[feature_score])
                except Exception as ve:
                    pass

                precision = 0.0
                try:
                    precision = average_precision_score(subset_df[feature_true], subset_df[feature_score])
                except Exception as e:
                    pass
                scores.append([auc, precision, fset, iset, len(subset_df)])

                # Compute metrics: false positives, false negatives
                # tn, fp, fn, tp = confusion_matrix(subset_df[], subset_df[]).ravel()

                # high precision -> low false positive rate, high recall -> low false negative rate
                # We can use: average precision, average recall etc.

                # For the subset, compute metrics of labels: false negative, false positives

    elapsed = datetime.now() - start_dt
    print(f"Finished feature prediction in {str(elapsed)}.")

    # Store them and report intervals (combinations) with minimum false negative/positives
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    print(scores[:20])
    textfile = open("confidence_mining_by_auc.txt", "w")
    for element in scores[:100]:
        textfile.write(str(element) + "\n")
    textfile.close()

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(scores[:20])
    textfile = open("confidence_mining_by_precision.txt", "w")
    for element in scores[:100]:
        textfile.write(str(element) + "\n")
    textfile.close()

    pass


if __name__ == '__main__':
    confidence_mining()
