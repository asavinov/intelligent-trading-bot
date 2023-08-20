from pathlib import Path
from typing import Union
import pandas as pd

"""
Generate top and bottom label columns with the specified parameters.
A top or bottom label has two parameters:
* level - height of minimum jump form minimum or maximum 
* tolerance - within this distance from minimum or maximum the label is true

Terminology and concepts:

* Level (fraction). This parameter determines the minimum necessary "jump" up or down for 
the potential extremum to be selected as extremum. The jump height must be present from
both left and right. For example, for a maximum, there must be a point on the left and on
the right which are lower then this maximum by this amount. 
It is specified as a fraction (percentage) relative to the maximum or minimum.
Once the extremums were selected, this parameter is not used anymore so it is a constraint
for selecting extremums only.

* Tolerance (fraction). Once an extremum was found, this parameter is used to select the width
of its interval, that is, the width on the left and right. We select left points and rights points
which are smaller/greater than the extremum by this fraction.

Store the labels, scores and some source columns in an output file.
"""

def generate_labels_topbot2(df, config: dict):
    """Find and label top points."""
    init_column_number = len(df.columns)

    column_name = config.get('columns')
    if not column_name:
        raise ValueError(f"The 'columns' parameter must be a non-empty string. {type(column_name)}")
    elif not isinstance(column_name, str):
        raise ValueError(f"Wrong type of the 'columns' parameter: {type(column_name)}")
    elif column_name not in df.columns:
        raise ValueError(f"{column_name} does not exist  in the input data. Existing columns: {df.columns.to_list()}")

    function = config.get('function')
    if not isinstance(function, str):
        raise ValueError(f"Wrong type of the 'function' parameter: {type(function)}")
    if function not in ['top', 'bot']:
        raise ValueError(f"Unknown function name {function}. Only 'top' or 'bot' are possible")

    tolerances = config.get('tolerances')  # For example, 0.0025 for 0.25% tolerance
    if not isinstance(tolerances, list):
        tolerances = [tolerances]

    level = config.get('level')  # For example, 0.01 for 1% (positive or negative)
    if function == 'top':
        level = abs(level)
    elif function == 'bot':
        level = -abs(level)

    names = config.get('names')  # For example, ['top1_025', 'top1_01'] for two tolerances
    if len(names) != len(tolerances):
        raise ValueError(f"'topbot2' Label generator: for each tolerance value one name has to be provided.")

    labels = []
    for i, tolerance in enumerate(tolerances):
        df, new_labels = add_extremum_features(df, column_name=column_name, level_fracs=[level], tolerance_frac=abs(level)*tolerance, out_names=names[i:i+1])
        labels.extend(new_labels)

    print(f"{len(names)} topbot2 labels computed: {names}")

    labels = df.columns.to_list()[init_column_number:]

    return df, labels


def generate_labels_topbot(df, column_name: str, top_level_fracs: list, bot_level_fracs: list):
    """For the specified levels, generate extremum labels with different pre-defined tolerances."""
    init_column_number = len(df.columns)

    # Tolerance 0.0025
    tolerance_frac = 0.0025
    top_labels = ['top1_025', 'top2_025', 'top3_025', 'top4_025', 'top5_025']
    bot_labels = ['bot1_025', 'bot2_025', 'bot3_025', 'bot4_025', 'bot5_025']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.005
    tolerance_frac = 0.005
    top_labels = ['top1_05', 'top2_05', 'top3_05', 'top4_05', 'top5_05']
    bot_labels = ['bot1_05', 'bot2_05', 'bot3_05', 'bot4_05', 'bot5_05']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.0075
    tolerance_frac = 0.0075
    top_labels = ['top1_075', 'top2_075', 'top3_075', 'top4_075', 'top5_075']
    bot_labels = ['bot1_075', 'bot2_075', 'bot3_075', 'bot4_075', 'bot5_075']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.01
    tolerance_frac = 0.01
    top_labels = ['top1_1', 'top2_1', 'top3_1', 'top4_1', 'top5_1']
    bot_labels = ['bot1_1', 'bot2_1', 'bot3_1', 'bot4_1', 'bot5_1']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.0125
    tolerance_frac = 0.0125
    top_labels = ['top1_125', 'top2_125', 'top3_125', 'top4_125', 'top5_125']
    bot_labels = ['bot1_125', 'bot2_125', 'bot3_125', 'bot4_125', 'bot5_125']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.015
    tolerance_frac = 0.015
    top_labels = ['top1_15', 'top2_15', 'top3_15', 'top4_15', 'top5_15']
    bot_labels = ['bot1_15', 'bot2_15', 'bot3_15', 'bot4_15', 'bot5_15']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.0175
    tolerance_frac = 0.0175
    top_labels = ['top1_175', 'top2_175', 'top3_175', 'top4_175', 'top5_175']
    bot_labels = ['bot1_175', 'bot2_175', 'bot3_175', 'bot4_175', 'bot5_175']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.02
    tolerance_frac = 0.02
    top_labels = ['top1_2', 'top2_2', 'top3_2', 'top4_2', 'top5_2']
    bot_labels = ['bot1_2', 'bot2_2', 'bot3_2', 'bot4_2', 'bot5_2']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.025
    tolerance_frac = 0.025
    top_labels = ['top1_25', 'top2_25', 'top3_25', 'top4_25', 'top5_25']
    bot_labels = ['bot1_25', 'bot2_25', 'bot3_25', 'bot4_25', 'bot5_25']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    # Tolerance 0.03
    tolerance_frac = 0.03
    top_labels = ['top1_3', 'top2_3', 'top3_3', 'top4_3', 'top5_3']
    bot_labels = ['bot1_3', 'bot2_3', 'bot3_3', 'bot4_3', 'bot5_3']

    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
    print(f"Top labels computed: {top_labels}")
    df, labels = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)
    print(f"Bottom labels computed: {bot_labels}")

    labels = df.columns.to_list()[init_column_number:]

    return df, labels


def add_extremum_features(df, column_name: str, level_fracs: list, tolerance_frac: float, out_names: list):
    """
    For each specified level fraction, compute and attach an output boolean label
    column to the specified data frame (with the specified name), which is true if the
    value is within the tolerance interval and false otherwise. In other words, this
    label column says if the current point is on top/bottom or not.

    The number of extremums (contiguous top/bottom intervals with true value) is determined
    by the level fraction (minimum necessary jump height). The greater the level, the fewer
    true intervals we get.

    The width of the contiguous top/bottom intervals with true value is determined by the
    tolerance fraction. The greater the fraction, the wider true intervals we get.
    """
    column = df[column_name]
    out_columns = []
    for i, level_frac in enumerate(level_fracs):
        if level_frac > 0.0:  # Max
            extrems = find_all_extremums(column, True, level_frac, tolerance_frac)
        else:  # Min
            extrems = find_all_extremums(column, False, -level_frac, tolerance_frac)

        out_name = out_names[i]
        out_column = pd.Series(data=False, index=df.index, dtype=bool, name=out_name)

        # Convert a list of extremums to a boolean (label) column
        # (left_level, left_tolerance, extremum, right_tolerance, right_level)
        for extr in extrems:
            out_column.loc[extr[1]: extr[3]] = True  # Assign value to slice

        out_columns.append(out_column)

    # Attach all generated label columns to the input data frame
    df = pd.concat([df] + out_columns, axis=1)

    return df, out_names


def find_all_extremums(sr: pd.Series, is_max: bool, level_frac: float, tolerance_frac: float) -> list:
    """
    Find all extremums in the input series along with their level/tolerance intervals.
    Return a (sorted) list of tuples each representing one extremum.

    The recursive algorithm is based on the function, which finds one absolute maximum
    for the selected sub-interval. First, it is applied to the whole series
    length. After that, it is applied to the left tails and right tails. After each call,
    we split the interval into two left/right sub-intervals and then find their extremums.
    If two equal maximums are found, then they are both investigated. This means that one call
    can return one or more maximums (but not all) which split the interval into parts.

    :param sr:
    :param is_max: either maximum or minimum
    :param level_frac: Minimum height (percentage of the extremum) required for a maximum
        or minimum to be selected (qualify)
    :param tolerance_frac: If selected, then it is the level for 0 values of the output
        (percentage of the extremum)
    :return: List of tuples representing extremum tuples
    """
    extremums = list()

    # ALl intervals that need to be analyzed by finding one minimum and one maximum
    intervals = [(sr.index[0], sr.index[-1] + 1)]
    while True:
        # Get next interval. If no, then break
        if not intervals:
            break
        interval = intervals.pop()

        # Find extremum within the selected sub-intervals (if any)
        extremum = find_one_extremum(sr.loc[interval[0]: interval[1]], is_max, level_frac, tolerance_frac)
        # If found store for return
        if extremum[0] and extremum[-1]:
            extremums.append(extremum)

        # Split and add two intervals for processing during next iteration
        if extremum[0] and interval[0] < extremum[0]:
            intervals.append((interval[0], extremum[0]))
        if extremum[-1] and extremum[-1] < interval[1]:
            intervals.append((extremum[-1], interval[1]))

    return sorted(extremums, key=lambda x: x[2])


def find_one_extremum(sr: pd.Series, is_max: bool, level_frac: float, tolerance_frac: float) -> tuple:
    """
    For the specified series, find its extremum along with level and tolerance intervals
    if they within this series. If the level/tolerance intervals are not within this
    series, then the corresponding output indexes are null. The function is supposed to
    be used in the recursive algorithm where it is applied to various sub-series.

    Return a tuple with the extremum index, and left/right indexes (if any) of the level
    and tolerance intervals.

    Algorithm:
    - Find one absolute maximum in the interval
    - Check if the tails of this maximum satisfy the constraint

    Links:
    - https://stackoverflow.com/questions/48023982/pandas-finding-local-max-and-min
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html
    """
    #
    # Find the first maximum in the specified interval
    #
    if is_max:
        extr_idx = sr.idxmax()
        extr_val = sr.loc[extr_idx]
        level_val = extr_val * (1 - level_frac)
        tolerance_val = extr_val * (1 - tolerance_frac)
    else:
        extr_idx = sr.idxmin()
        extr_val = sr.loc[extr_idx]
        level_val = extr_val / (1 - level_frac)  # extr_val * (1 + level_frac)
        tolerance_val = extr_val / (1 - tolerance_frac)  # extr_val * (1 + tolerance_frac)

    # Split into two sub-intervals in order to find the left and right ends separately
    sr_left = sr.loc[:extr_idx]
    sr_right = sr.loc[extr_idx:]

    # Check the height condition, that is, if we reach the necessary height on the left and right
    left_level_idx = _left_level_idx(sr_left, is_max, level_val)
    right_level_idx = _right_level_idx(sr_right, is_max, level_val)
    # Index is None if the height condition is not satisfied

    # Find tolerance interval
    left_tol_idx = _left_level_idx(sr_left, is_max, tolerance_val)
    right_tol_idx = _right_level_idx(sr_right, is_max, tolerance_val)

    return (left_level_idx, left_tol_idx, extr_idx, right_tol_idx, right_level_idx)


def _left_level_idx(sr_left: pd.Series, is_max: bool, level_val: float):
    """Find index of the first element starting from the right edge which is supposed to be an extremum."""
    # Approach 1 based on selection (filter) and getting very first element
    if is_max:
        sr_left_level = sr_left[sr_left < level_val]
    else:
        sr_left_level = sr_left[sr_left > level_val]

    if len(sr_left_level) > 0:
        left_idx = sr_left_level.index[-1]
    else:
        left_idx = None  # Not found. Maximum is bad.Bad height

    # Approach 2: based on mask and finding first true element
    # left_idx2 = sr_left[sr_left < level_val].loc[:extr_idx].last_valid_index()

    return left_idx


def _right_level_idx(sr_right: pd.Series, is_max: bool, level_val: float):
    """Find index of the first element starting from the left edge which is supposed to be an extremum."""
    # Approach 1 based on selection (filter) and getting very first element
    if is_max:
        sr_right_level = sr_right[sr_right < level_val]
    else:
        sr_right_level = sr_right[sr_right > level_val]

    if len(sr_right_level) > 0:
        right_idx = sr_right_level.index[0]
    else:
        right_idx = None  # Not found. Maximum is bad. Bad height

    # Approach 2: based on mask and finding first true element
    # right_idx2 = sr_right[sr_right < level_val].first_valid_index()

    return right_idx
