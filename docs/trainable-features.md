# Trainable features

## Why train?

A conventional feature produces an output column from an input column using manually specified parameters. For example, a feature computing the moving average of the close price has one parameter: the window size. This parameter is specified in the feature configuration and does not depend on any external factors. Its value must be known in advance and is typically derived from experience. If we want to produce a feature representing the deviation of the current price from the average price, we need to know this average value. We can manually compute the average price by loading historic prices and then setting it in the feature configuration. This configured value is then used during the evaluation process to compute the deviation.

In contrast to conventional features, trainable features learn their parameters from historic data. The complete set of these parameters is referred to as a *model*. A model can be as simple as a single number or as complex as a deep neural network. Crucially, it is derived by analyzing historic data; this process is referred to as *training*. Therefore, features that use trained models are referred to as trainable features or ML features.

One example of a trainable feature is one where the goal is to determine the average price from historic data. This model stores only one number: the average price value. Feature evaluation relies on this model to compute the deviation from the average. Unlike the previous example, where the average value had to be known in advance, the average value is now part of the automated procedure. The feature itself determines how to derive its model parameters from historic data before using those parameters to generate output.

Trainable features operate in two modes by implementing two types of logic:
-   How to derive feature parameters by analyzing historic data and generating a model
-   How to use feature parameters (the model) to generate output values (predictions)

Advantages of trainable features:
-   They identify optimal parameters rather than relying on intuition or educated guesses
-   They can be regularly retrained so their values reflect the latest history and adapt to drift
-   They can learn significantly more complex dependencies than manually defined features

Drawbacks of trainable features:
-   They require a substantial amount of historic data, which may be unavailable or insufficient for training complex models without overfitting
-   Manually defined features frequently represent the perspective of traders and domain experts. They constitute compressed knowledge with highly informative representations, making them difficult to replicate through automatic learning

Our approach combines both worlds: it provides a rich mechanism for defining manually parameterized features while also enabling the definition of features that automatically learn their parameters by applying statistical or machine learning algorithms.

## Defining trainable features

Trainable features are defined separately from other features as a list where each item is a dictionary containing a single feature definition:
```jsonc
"train_feature_sets": [
  {...}, // First trainable feature
  {...}, // Second trainable feature
  {...} // Third trainable feature
]
```

A single feature definition is a dictionary with the following attributes:
```jsonc
{
  "generator": "train_features", // This generator handles the trainable nature of the feature
  "columns": [], // Columns used for training and prediction. If empty, the 'train_features' list is used
  "labels": ["bot_2", "top_2"], // Columns used as labels for training
  "functions": [
    {
      "name": "mysvc",
      "algo": "svc", // Arbitrary name and predefined algorithm type
      "params": {"is_scale": true, "length": 0}, // Preprocessing parameters
      "train": {"C": 1.0, "gamma": 0.005} // Algorithm arguments
    }
  ]
}
```

The purposes of these attributes are as follows:
-   `generator`: A built-in generator function that evaluates trainable features
-   `columns`: A list of column names selected for training. If the list is empty, columns from `train_features` are used. This attribute enumerates all input columns (excluding labels)
-   `labels`: Column names used as ground truth values during training. If the list is empty, columns from `labels` are used. This attribute enumerates all columns serving as labels
-   `functions`: A list of algorithm descriptions, each being a dictionary with the following attributes:
    -   `name`: An arbitrary unique identifier for this algorithm entry. It serves as a suffix in the predicted column names (alongside the label name), indicating which algorithm (and label) generated a specific output column
    -   `algo`: The algorithm type, which resolves to a specific Python function. Currently supported types include:
        -   `svc`: Support Vector Machines
        -   `nn`: Neural Network
        -   `lc`: Linear Classifier
        -   `gb`: Gradient Boosting
    -   `params`: Parameters for the generator used to prepare the training dataset:
        -   `is_scale`: If true, all columns will be normalized
        -   `length`: The number of records to use for training. If 0, `train_length` is used. If omitted, all available data is used
        -   `every_nth_row`: Enables selection of a smaller subset for training via downsampling
        -   `is_regression`: If true, the label is treated as a numeric value and a regression model is trained
    -   `train`: A dictionary of parameters passed directly to the algorithm

For each entry in `train_feature_sets`, the number of generated features equals the number of labels. Each output feature follows the naming schema `label_algo`, where `label` is a value from the labels list and `algo` is the name specified in the `name` attribute. In the example above, two features and two output columns are generated: `bot_2_mysvc` and `top_2_mysvc`. These same names are used for the trained models stored as files.

## Custom trainable features

An alternative method for defining trainable features is compatible with standard [feature](features.md) definitions. Such feature definitions are placed in the `feature_sets` or `train_features` sections, but not in the `train_feature_sets` section of the configuration.

This approach requires a custom generator function that must handle two modes: train and predict. It must check whether it is running in train mode via the global boolean `train` attribute. If true, it must train its model using available data and then perform feature evaluation using this newly trained model. If running in predict mode (when the `train` attribute is false), it reuses the previously trained model.

Below is an example of how to define such a trainable feature to train a model that finds an average value, which is then used to calculate the deviation of current values from that average.
```jsonc
{
  "generator": "myextensions.stats:deviation_feature",
  "config": { // Passed to the generator function
    "columns": "close", // Column for which to find the average
    "function": "mean", // Other functions could be supported, e.g., median
    "names": "deviation", // Predicted column name
    "parameters": {} // Additional parameters if needed
  }
}
```

This generator function can be implemented as follows:
```python
def deviation_feature(df, config: dict, global_config: dict, model_store: ModelStore):
    column_name = config.get('columns')
    function = config.get('function')
    names = config.get('names')  # Output feature name
    if not names:
        names = f"{column_name}_{function}"

    # Load model
    model_name = config.get('model_name', names)
    model = model_store.get_model(model_name)

    # Determine if training is required before prediction
    is_train = global_config.get('train')

    # If in train mode, derive scale parameters from data and store in the model for subsequent prediction
    if is_train:
        mean = df[column_name].mean()  # Calculate mean value
        model = dict(mean=mean)  # Create model as a dictionary
        model_store.put_model(model_name, model)  # Persistently store the model

    # Perform evaluation using either the loaded or newly trained model
    if function == 'mean':
        out_column = df[column_name] - model.get("mean")

    df[names] = out_column

    return df, [names]
```

The distinguishing aspect of this code is the training block, which executes only when train mode is detected. In train mode, data from the specified column is used to calculate the mean value, which constitutes the sole parameter of the model. Subsequently, the column is transformed by subtracting this mean value, and the resulting column is returned as a new generated feature.

When executed in predict mode, the feature behaves like any standard feature, except that its model (the mean value) is loaded from the model store.
