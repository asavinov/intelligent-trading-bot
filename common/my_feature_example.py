from pathlib import Path
from typing import Union
import pandas as pd

"""
Example of a feature
"""

def my_feature_example(df, config: dict):
    """
    Add a parameter to the column or multiply by this parameter
    """

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
    if function not in ['add', 'mul']:
        raise ValueError(f"Unknown function name {function}. Only 'add' or 'mul' are possible")

    parameter = config.get('parameter')  # Numeric parameter
    if not isinstance(parameter, (float, int)):
        raise ValueError(f"Wrong 'parameter' type {type(parameter)}. Only numbers are supported")

    names = config.get('names')  # Output feature name
    if not names:
        names = f"{column_name}_{function}"

    if function == 'add':
        df[names] = df[column_name] + parameter
    elif function == 'mul':
        df[names] = df[column_name] * parameter

    print(f"Finished computing feature '{names}'")

    return df, [names]
