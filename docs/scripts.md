# Scripts

Batch scripts are designed for analyzing data in files, as opposed to performing incremental in-memory analysis.
Each script serves a single purpose.
Typically, a script loads data from an input file, processes it, and stores the result in an output file.
Using multiple scripts simplifies the development of complex, multi-step data processing pipelines.

Scripts serve the following major purposes:
-   Data retrieval (data collectors) and merging of different data sources
-   Feature evaluation and generation of new derived columns
-   Machine learning model training
-   Output adapter execution
-   Simulation and backtesting

File names can be specified using the following configuration attributes:
-   `merge_file_name`: "data.csv"
-   `feature_file_name`: "features.csv"
-   `matrix_file_name`: "matrix.csv"
-   `predict_file_name`: "predictions.csv"
-   `signal_file_name`: "signals.csv"
-   `signal_models_file_name`: "signal_models"

The file extension determines the format (CSV or Parquet).

## Download and merge

The `download` script retrieves data from the sources listed in the `data_sources` section:
```console
python -m scripts.download -c config.json
```

The quote name (which also serves as the subfolder name) and file name ("klines" by default) are specified in each data source entry.
If a source file already exists, the script retrieves only the latest missing data and appends these rows to the file.
Otherwise, all available data is retrieved and stored in a new file.

The `merge` script combines all source files specified in the `data_sources` section and stores the result in a single output file:
```console
python -m scripts.merge -c config.json
```
The output file name and format are specified in the `merge_file_name` attribute.
The file is stored in the `symbol` subfolder.
Columns receive a prefix (if specified in the data source entry), and all rows are aligned according to the data frequency.
Accordingly, gaps may appear if data is missing from certain sources.

## Feature engineering

These scripts apply feature definitions and generate new columns.
Currently, features can be defined in several ways, resulting in multiple groups of feature definitions.
A dedicated script exists for each feature group.

### Features

Features defined in the `feature_sets` section are evaluated using the `features` script:
```console
python -m scripts.features -c config.json
```
It reads data from the input file specified in the `feature_sets` parameter ("data" by default), generates the features defined in the `feature_sets` section, and stores the result in the output file specified in the `feature_file_name` parameter ("features" by default).

### Labels

Features defined in the `label_sets` section are evaluated using the `labels` script:
```console
python -m scripts.labels -c config.json
```
It reads input data from the output of the `features` script.
The resulting table, with appended label columns, is stored in the `matrix_file_name` file ("matrix" by default).

### Predict

The `predict` script evaluates features defined in the `train_feature_sets` section:
```console
python -m scripts.predict -c config.json
```
It evaluates these trainable features in prediction mode using existing models previously trained by the `train` script.
Data is loaded from the output of the `labels` script.
The result is stored in the `predict_file_name` file ("predictions" by default).

### Signals

This script evaluates features defined in the `signal_sets` section:
```console
python -m scripts.signals -c config.json
```
Input data is loaded from the `predict_file_name` file (the output of the `predict` script).
The result is stored in the `signal_file_name` file.

## Train machine learning models

The `train` script runs feature definitions from the `train_feature_sets` section in training mode:
```console
python -m scripts.train -c config.json
```
Running in training mode means that these features use the data to determine optimal parameters but do not apply them.
The script outputs model files stored in the `MODELS` subfolder.
These model files are subsequently used by the `predict` script to generate feature output columns.

## Outputs

The `output` script executes the definitions from the `output_sets` section:
```console
python -m scripts.output -c config.json
```
These definitions do not produce new features; rather, they use existing data to send notifications, generate visualizations, execute trades, etc.

## Other scripts

### Generate rolling predictions

In real trading, machine learning models trained on historical data are used to predict future behavior.
The `predict_rolling` script simulates walk-forward prediction:
-   Train models using available historical data (e.g., all data up to 2026-01-01).
-   Apply these models to predict future behavior starting from the end of the training period for a relatively short interval (e.g., from 2026-01-01 to 2026-02-01).
-   Append a new data chunk and repeat the train-predict cycle (e.g., using all data up to 2026-02-01).

The result is a table of predictions generated in small chunks using models trained on progressively growing historical data.
This simulates the train-predict workflow over time, mimicking a real-world environment.
For example, with daily data, monthly chunks could be used to keep models up to date.
The script might start from an initial date (e.g., one year ago) and perform 12 iterations, training and predicting for each subsequent month.
The final output contains one year of predictions generated in a "fair" manner that closely resembles actual deployment.

The script is executed like other scripts:
```console
python -m scripts.predict_rolling -c config.json
```
It loads data from the `matrix_file_name` file ("matrix" by default).
The result is stored in the `predict_file_name` file ("predictions" by default).

Iteration parameters are specified in the `rolling_predict` section:
-   `data_start` (default: 0): Truncate input data from the start.
-   `data_end` (default: null): Truncate input data from the end.
-   `prediction_start`: First row for starting iterations (end of the first training dataset), e.g., "2020-01-02".
-   `prediction_size`: Size of each new data chunk and prediction in rows.
-   `prediction_steps`: Number of iterations to perform (e.g., 12 for 12 train-predict cycles). If None or 0, iterations continue from `prediction_start` to the end of the data.
-   `use_multiprocessing`: Whether to enable multiprocessing (true or false).
-   `max_workers`: Number of workers to use when multiprocessing is enabled.

### Simulation and backtesting

Data analysis in ITB focuses on generating new features, including those based on machine learning algorithms.
This provides insights into future behavior, such as the probability that the price will increase by 2% within the next week.
However, this knowledge cannot be used *directly* for trading and reveals little about trade performance.
The primary reason is that many possible trading strategies can be derived from knowledge of future behavior.
These strategies generate buy and sell signals based on current market conditions and asset states. The simulation script bridges this gap by evaluating trade performance based on a chosen strategy and its parameters.

Simulation (backtesting) parameters are specified in the `simulate_model` section of the configuration file:
```console
python -m scripts.simulate -c config.json
```

The script consumes predictions from the `predict_file_name` file.
This file must include all derived features and should be produced by the `predict_rolling` script.

Note that the script does not evaluate performance for a single parameter set.
Instead, it evaluates a large set of parameters to identify the best-performing trade model.

The `simulate_model` section describes a set of trade models and includes the following attributes:
-   `data_start`: Start simulation from this date (e.g., "1998-01-05").
-   `data_end`: End simulation on this date. If empty, all available data is used.
-   `direction`: "long" or "short".
-   `topn_to_store`: Number of top-performing trade models (and their parameters) to store in the output file.
-   `signal_generator`: Name of the generator that produces the column used for buy/sell decisions.
-   `buy_sell_equal`: true or false. Indicates whether threshold parameters for buy and sell signals are symmetric.
-   `grid`: Dictionary containing lists. Each combination of values from these lists constitutes one model. Attributes are lists or strings evaluated to lists
    -   `buy_signal_threshold`: List of values interpreted as buy thresholds to test (e.g., [0.1, 0.2, 0.3]). If provided as a string, it is evaluated as Python code and must return a list (e.g., "np.arange(0.0, 0.5, 0.1).tolist()").
    -   `sell_signal_threshold`: Same as above, but used to identify sell signals.
