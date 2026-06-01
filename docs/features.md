# Features

## Column-oriented design

In a column-oriented design, a *feature* can be viewed as a single column.
In this project, all data is assumed to be stored in one table; therefore, all features exist as columns within this main table.
This primary data table is represented as a `pandas` `DataFrame`, where features correspond to columns and rows correspond to timestamps at a specific frequency.

Most of the system's data analysis logic is embodied in how its features are computed.
The core concept is that for each new row appended to the main DataFrame, all (initially empty) feature values must be computed.
Once these new values are calculated, trade decisions can be made.

## Feature dependencies

How a feature computes its value based on existing (current and past) feature values is specified in a `feature definition`.
Generally, any feature definition uses—and thus depends on—other features, and its value may subsequently be used by other features.
In this sense, all features collectively define a computation graph.

A feature also depends on a specific number of previous rows. In the simplest case, a feature depends only on the current row.
For example, we could define a feature equal to the difference between the high and low prices for the current day.
This depends only on the current row and two other features (high and low), which must be set before this difference can be computed.
Conversely, a feature that computes a moving average will depend on one feature (such as close price) and a certain number of previous values.

## Defining features

All features are represented as a list in which each item is a dictionary containing a single feature definition:
```jsonc
"feature_sets": [
  {...}, // First feature
  {...}, // Second feature
  {...} // Third feature
]
```

Features can also be defined in the `signal_sets` section, which is evaluated after trainable features.

A single feature definition is a dictionary with the following attributes:
```jsonc
{
  "generator": "talib", // Name of the generator (pluggable) that processes and executes this feature
  "column_prefix": "", // Prefix removed from column names before data is passed to the generator
  "feature_prefix": "", // Appended to new features after generation
  "config":  {} // Parameters specific to this feature
 }
```

The attributes of the feature definition have the following interpretations:
- `generator`: A string representing either a predefined built-in generator name or a user-defined Python function implementing a custom feature generator.
- `column_prefix`: This prefix is removed from column names before data is passed to the generator.
For example, if analyzing BTC and ETH data simultaneously, columns might include `btc_close` and `eth_close`.
However, if a generator should process only the `open` column, this can be achieved by specifying `column_prefix: btc` or `column_prefix: eth` for two features sharing the same configuration. Alternatively, the input column can be specified directly in the generator's `config` parameters.
- `feature_prefix`: This prefix (followed by an underscore) is automatically appended to all features returned by the generator.
It may match the `column_prefix`. Again, the `config` section can be used to provide a desired output name for the generated feature.
- `config`: A dictionary containing feature configuration parameters specific to each generator.

## Currently available generators and their features

ITB provides a collection of ready-to-use feature generators that allow for the definition of standard technical indicators as well as some original features.

### Features based on TA-Lib

[TA-Lib](https://ta-lib.org/) is a native C/C++ library implementing approximately 200 indicators for financial analysis and trading applications.
It includes a [Python wrapper](https://github.com/ta-lib/ta-lib-python), which ITB uses to expose these indicators as feature definitions.

Note that using this feature generator requires the native TA-Lib binary library to be installed.
This can be accomplished in several ways:
- Install TA-Lib as a Linux package.
- Build and install TA-Lib from source, ensuring the library is accessible from Python.
- Install the native TA-Lib library via a Python package manager. For example: `$ conda install -c conda-forge libta-lib`.
- Install the Python wrapper via a package manager that supports platform-specific libraries.
For example, Conda supports this for certain platforms and Python versions: `conda install -c conda-forge ta-lib`.
- In some cases, it may be possible to locate the library manually and copy it to a location accessible from Python.

To use TA-Lib, set the generator name to `"talib"`: `"generator": "talib"`.
The generator maps `config` attributes to TA-Lib indicator arguments, calls the appropriate TA-Lib functions, and returns the results as one or more `pandas` columns attached to the main DataFrame.

The generator `config` attributes are interpreted as follows (not all are required for every feature):
- `columns`: A list of column (feature) names to pass to the TA-Lib function.
For example, `columns: ["close","high","low"]` indicates that these three price columns will be used to compute the feature.
- `functions`: A list of functions as defined in the [TA-Lib documentation](https://ta-lib.org/functions/). One feature is generated for each function name.
For example, `functions: ["SMA"]` specifies Simple Moving Average, meaning each feature value is computed as the average of several previous values.
- `windows`: A list of integers interpreted as the window size (number of previous rows, including the current time).
One feature is generated for each window value.
For example, `windows: [20]` with the `SMA` function computes a 20-row simple moving average. For daily data, this represents a 20-day moving average.
- `parameters`: A dictionary of parameters for post-processing the series of indicators returned by TA-Lib (not specific to TA-Lib itself).
These are applied to the time series resulting from different `windows` values, typically to compute relative values within the sequence.
  - `rel_base`: Possible values include:
    - `next`: Relative to the next element in the sequence.
    - `last`: Relative to the last element in the sequence.
    - `prev`: Relative to the previous element in the sequence.
    - `first`: Relative to the first element in the sequence.
  - `rel_func`: Defines how the new value is derived from the original and reference values:
    - `diff`: Difference between the original and reference value (`value - reference_value`).
    - `rel`: Ratio of the original to the reference value (`value / reference_value`).
    - `rel_diff`: Relative difference (`(value - reference_value) / reference_value`).

In this example, three output features are computed: the 1-, 10-, and 20-period moving averages of the close price:
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
The first output feature is the absolute difference between the close price (1-period MA) and its 10-period moving average.
The second output feature is the absolute difference between the 10-period and 20-period MAs of the close price.
The final feature is the 20-period MA without a relative value calculation, as there is no subsequent element in the MA series.

Note that some TA-Lib functions are characterized as having an unstable period.
Such functions may return incorrect results during this period and should therefore be used with caution.
These functions include a note in the documentation such as: `The ADX function has an unstable period.`

## Defining custom (your own) features

Custom feature generators can be specified as user-defined Python functions.
In this case, the `generator` attribute in the feature definition must be a fully qualified Python function name consisting of the module name and function name separated by a colon.

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
This function retrieves the input column name and operation type from the config.
Depending on the operation name, it either adds a constant parameter to the input column or performs multiplication.
The resulting column is appended to the input DataFrame and returned alongside the new feature name.

Note that while the configuration dictionary may have an arbitrary format and attributes, the feature generator must adhere to this specific signature.
