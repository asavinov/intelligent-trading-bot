import pytest
import numpy.testing as npt

from common.utils import *
from common.gen_signals import *
from common.gen_labels_topbot import *


def test_extremum_labels():
    data = [10, 30, 50, 70, 90, 70, 50, 30, 9]
    data = [10, 40, 30, 70, 90, 50, 60, 30, 9]
    sr = pd.Series(data)
    sr = pd.Series(data * 2)
    level_frac = 0.5
    tolerance_frac = 0.1

    maximums = find_all_extremums(sr, True, level_frac, tolerance_frac)
    minimums = find_all_extremums(sr, False, level_frac, tolerance_frac)

    # Merge into a sequence of interleaving minimums and maximums
    all = list()
    all.extend(maximums)
    all.extend(minimums)

    all.sort(key=lambda x: x[1])

    # We have indexes (x coordinates) and need to find their y values
    extr_x = [x[1] for x in all]
    extr_y = [sr[x] for x in extr_x]
    extr_df = pd.DataFrame({'x': extr_x, 'y': extr_y})

    import seaborn as sns
    # https://matplotlib.org/stable/api/markers_api.html
    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    sns.lineplot(data=sr)
    sns.lineplot(data=extr_df, x="x", y="y", marker="o", markersize=10, linestyle='dotted')  # "^" 'v'

    pass


def test_interval_and_aggregation():
    data = [10, 40, 30, 70, 90, 50, 60, 30, 9]
    sr = pd.Series(data * 2)

    df = pd.DataFrame(data={'close': sr, 'score': sr / 10})

    level_frac = 0.5
    tolerance_frac = 0.1

    # For debugging
    maximums = find_all_extremums(sr, True, level_frac, tolerance_frac)

    # Add label
    df, _ = add_extremum_features(df, column_name='close', level_fracs=[level_frac], tolerance_frac=tolerance_frac, out_names=['is_close_top'])

    # Aggregate score with chosen parameters
    aggregate_scores(df, 'score_agg', ['score'], None, 2)

    threshold = 6
    interval_df = find_interval_precision(df, label_column='is_close_top', score_column='score_agg', threshold=threshold)

    pass
