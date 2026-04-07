# Features

## Column-oriented design

A *feature* in a column-oriented design can be viewed as one column.
In this project, it is assumed that all data is stored in one table and hence all features exist as columns of this main table.
This main data table is represented as a `pandas` `DataFrame` object and all features are columns of this data frame while rows correspond to timestamps in a time raster of certain frequency.

Most of the logic of data analysis of the system is represented in how its features are computed.
The main idea is that for each new data rows appended to the main dataframe, it is necessary to compute all (initially empty) feature values.
Once these new data values are computed, it is possible to make trade decisions.

## Feature dependencies

How a feature computes its value depending on already existing (current and past) feature values is specified in a `feature definition`.
In general, any feature definition uses and hence depends on some other features. Accordingly, its value will be used in other features.
In this sense, all features define a graph of computations.

A feature also depends on a certain number of previous rows. In the simplest case, a feature depends on only the current row.
For example, we could define a feature equal to the difference between high and low price for this current day.
It depends on only this row and only two other features (high and low) which must be set before this difference can be computed.
A feature which computes moving average will depend on one feature (like close price) and certain number of previous values.

## Defining features

All features are represented as a list where one item is a dictionary with one feature definition:
```jsonc
"feature_sets": [
  {...}, // First feature
  {...}, // Second feature
  {...} // Third feature
]
```

One feature definition is a dictionary with the following attributes:
```jsonc
{
  "generator": "talib", // Name of the generator (pluggable) which will process and execute this feature
  "column_prefix": "", // This prefix will be removed from column names before the data is passed to the generator
  "feature_prefix": "", // Appended to new features after they have been generated
  "config":  {} // Parameters of this feature
 }
```

The attributes of the feature definition have the following interpretations:
- `generator`: This string is either a pre-defined name of a built-in generator or a user-defined Python function which implements a custom feature generator
- `column_prefix`: Before the data is passed to the generator, this prefix will be removed from its columns.
For example, if we analyze data of BTC and ETH together then there will be columns like `btc_close` and `eth_close`.
Yet, we want to have a generator which processes only `open` column. This can be done by providing `column_prefix: btc` or `column_prefix: eth` for two features
with the same configuration parameters. Alternatively, it is possible to specify input column in the `config` parameters of the generator.
- `feature_prefix`: This prefix (with underscore symbol) will be automatically appended to all features returned by the generator.
It could be the same prefix as in `column_prefix`. Here again, it is possible to use the `config` section to provide a desired output name of the generated feature.
- `config`: It is a dictionary with the feature configuration parameters which are specific to each generator

## Currently available generators and their features

ITB provides a collection of ready to use feature generators which allow for defining standard technical indicators as well as some original features.A

### Features based on TA-Lib

[TA-Lib](https://ta-lib.org/) is a native library written in C/C++ which implements about 200 indicators for financial analysis and trading applications.
It has a [Python wrapper](https://github.com/ta-lib/ta-lib-python) which is used by ITB to expose these indicators as feature definitions.

Note that to use this feature generator, you need already TA-Lib native library (binary) already installed.
This can be done in several ways:
- Install TA-Lib as a Linux package
- Build and install TA-Lib from source code and make sure that the library is accessible from Python
- Install TA-Lib native library via some Python package manager. For example: `$ conda install -c conda-forge libta-lib`
- Install Python wrapper via a Python package manager which supports platform-specific libraries.
For example, this can be done by Conda for some platforms and Python versions: `conda install -c conda-forge ta-lib`
- In some cases it might be also possible to simply find somewhere the library and copy it to the location where it is accessible from Python

In order to use TA-Lib it is necessary to set the generator name to "talib: `"generator": "talib"`.
The generator will map attributes of the `config` to arguments of TA-Lib indicators, call TA-Lib functions,
and return the result as one or more `pandas` columns attached to the main dataframe.

Here is how attributes of the generator `config` are interpreted in terms of TA-Lib (not all are needed for all features):
- `columns`: A list of column (feature) names to be passed to the TA-Lib function.
For example, `columns: ["close","high","low"]` means that these three columns with high, low and close prices will be used in computing this feature
- `functions`: A list of functions as defined in TA-Lib here: https://ta-lib.org/functions/. For each function name, one feature will be generated.
For example, `functions: [SMA]` means Simple Moving Average, that is, each feature value will be computed as average of several previous values.
- `windows`: A list of integers which are interpreted as the number of previous rows or window size (including the current time).
For each window value one feature will be generated.
For example, `windows: [20]` in case of `SMA` function means computing 20 rows simple moving average. In case of daily data, it is 20 days moving average.
- `parameters`: A dictionary of parameters for post-processing a series of indicators returned by TA-Lib (not specific for TA-Lib).
They are applied to a series of time-series resulted from a series of different `windows` parameter. The idea is that we want to compute relative values in this sequence.
  - `rel_base`: These values are possible:
    - `next`: relative to the next element in the sequence
    - `last`: relative to the last element in the sequence
    - `prev`: relative to the previous element in the sequence
    - `first`: relative to the first element in the sequence
  - `rel_func`: How a new value is computed from the original value and the reference value (next, previous, last or first):
    - `diff`: difference between the original and the reference value: value-reference_value
    - `rel`: ratio between the original and the reference value: value/reference_value
    - `rel_diff`: relative difference (value-reference_value)/reference_value

In this example we compute 3 output features which are 1, 10, 20 moving averages of the close price:
```jsonc
{
  "generator": "talib",
  "column_prefix": "", "feature_prefix": "",
  "config":  {
    "columns": ["close"],
    "functions": ["SMA"],
    "windows": [1, 10, 20],
    "parameters": {"rel_base": "next", "rel_func":  "diff"}
  }
}
```
The first output feature is the absolute difference between the close price (1-MA) and its 10-rows moving average.
The second output feature is the absolute difference between the 10-MA of the close price and 20-MA.
The last feature is 20-MA without computing relative value just because there is no next element in the series of MAs.

Note that some functions in TA-Lib are characterized as unstable or having unstable period.
Such functions may return wrong results and therefore should be used very cautiously.
Such functions have a note like this in the documentation: `The ADX function has an unstable period.`

## Defining custom (your own) features

Custom feature generators can be specified as a user-defined Python function.
In this case, the value of the `generator` attribute of the feature definition is a fully qualified Python function name.
This name consists of the module name and function name separated by colon.

Here is an example of a custom feature definition:
```jsonc
{
  "generator": "common.my_feature_example:my_feature_example",
  "column_prefix": "", "feature_prefix": "",
  "config":  {"columns": "close", "function": "add", "parameter": 2.0, "names": "close_add"}
}
```

This function can be implemented as follows:
```python
def my_feature_example(df, config: dict, global_config: dict, model_store: ModelStore):
    # Parse feature config. See source code for details
    if function == 'add':
        df[names] = df[column_name] + parameter
    elif function == 'mul':
        df[names] = df[column_name] * parameter
    return df, [names]
```
This function gets input column name and operation type from config.
Then depending on the operation name it either adds the constant parameter to the input column or applies multiplication.
The result column is appended to the input dataframe and returned along with the new feature name.

Note that the configuration dictionary may have arbitrary format and attributes.
However, the feature generator must have this signature.
