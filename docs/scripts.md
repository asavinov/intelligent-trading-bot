# Scripts

Batch scripts are intended for analyzing data in files as opposed to incremental analysis of data in-memory.
One script has one purpose.
It typically loads data from one input file, processes the data, and stores the result in an output file.
Having multiple scripts simplifies the development of complex data processing pipelines consisting of several steps.

There are the following major purposes of scripts:
- Data retrieval (data collectors) and merge of different data sources
- Evaluation of features and producing new derived columns
- Training machine learning models
- Executing output adapters
- Simulation and backtesting

File names can be specified in these attributes of the configuration:
- `merge_file_name`: "data.csv",
- `feature_file_name`: "features.csv",
- `matrix_file_name`: "matrix.csv",
- `predict_file_name`: "predictions.csv",
- `signal_file_name`: "signals.csv",
- `signal_models_file_name`: "signal_models",
The file extension is used to determine the format: CSV or parquet.

## Download and merge

The `download` script retrievs data the data sources listed in the `data_sources` section:
```console
python -m scripts.download -c config.json
```

Quote name (also subfolder) and file name ("klines" by default) are specified in the entries of the data source.
If a source file already exists, then the script will retrieve only latest missing data and append these rows to the file.
Otherwise, all available data will be retrieved and stored in a new file.

The `merge` script is used to combine all source files specified in the `data_sources` section and store the result in one output file:
```console
python -m scripts.merge -c config.json
```
The output file name and format is specified in the `merge_file_name` attribute.
The file is stored in the `symbol` subfolder.
Columns will get a prefix (if specified in the data source entry) and all rows will be aligned according to the data frequency.
Accordintly, gaps can appear in case of missing data in some data source.

## Feature engineering

These scripts are intended for applying feature definitions and generating new columns.
Currently features can be defined in several ways and hence there are several groups of feature definitions.
There is one script for each such group of features.

### Features

Features defined in the `feature_sets` section are evaluated using the `features` script:
```console
python -m scripts.features -c config.json
```
It reads data from its input file specified in the `feature_sets` parameter ("data" by default),
generates features defined in the `feature_sets` section, and stores the result in the output file
specified in the `feature_file_name` parameter ("features" by default).

### Labels

Features defined in the `label_sets` section are evaluated using the `labels` script:
```console
python -m scripts.labels -c config.json
```
It reads its input data from the output of the `features` script.
The result table with appended label columns is stored in the `matrix_file_name` file ("matrix" by default).

### Predict

The `predict` script evaluates features defined in the `train_feature_sets` section:
```console
python -m scripts.predict -c config.json
```
It evaluates these trainable features in predict mode by using existing models (previously trained by the `train` script).
The data is loaded from the output of the `labels` script.
The result is stored in the `predict_file_name` file ("predictions" by default).

### Signals

This script evaluates features defined in the `signal_sets` section:
```console
python -m scripts.signals -c config.json
```
Input data is loaded from the `predict_file_name` file (output of the `predict` script).
The result is stored in the `signal_file_name` file.

## Train machine learning models

The `train` script runs feature definitions from `train_feature_sets` section in train mode
```console
python -m scripts.train -c config.json
```
Running in train mode means that these features use the data to find optimal parameters of the features (models) but do not apply these parameters.
The results of the script are model files stored in the `MODELS` subfoler.
These model files will be used by the `predict` script to generate feature output columns.

## Outputs

The `output` script runs the definitions from the `output_sets` section:
```console
python -m scripts.output -c config.json
```
They do not produce new features but rather use the existing data to send notifications, visualizations, do trding etc.

## Other scripts

### Generate rolling predictions

In real trading, we use machine learning models trained on previous data in order to predict future behavior.
The `predict_rolling` is intended for repeating this sequence:
- train models using available historic data
- apply these models to predict future behavior
- append (small) new data chunk and repeat the train-predict analysis
The result is a table with predictions which are generated in small chunks using the models trained on the previous data (which also gradueally grows).
Thus we simulate the train-predict steps by moving in time as if we ware doing it in real context.
For example, for daily data, we could use monthly data chunks in order to keep our models up-to-date.
The script will start from some initial date, say, 1 year ago, and then do 12 iterations by training and predicting for each new month.
The result has 1 year of predictions but these predictions are done in a "fair" manner closely to real environment.

The script is executed like other scripts:
```console
python -m scripts.predict_rolling -c config.json
```
It loads data from the `matrix_file_name` file ("matrix" by default).
The result is stored in the `predict_file_name` file ("predictions" by default).

The parameters for iterations are specified in the `rolling_predict` section:
- `data_start` (default 0) Trancate input data from start
- `data_end` (default null) Trancate input data from end
- `prediction_start` First row for starting iterations (end of the first training data), for example, "2020-01-02"
- `prediction_size` New data chunk size in rows and the size of each prediction
- `prediction_steps` How many iterations to perform, for example, 12 if we want to do 12 train-predict iterations.
If None or 0, then from prediction start till the data end
- `use_multiprocessing` Whether to use multiprocessing (true or false)
- `max_workers` How many workers in case of multiprocessing

### Simulation and backtesting

Data analysis in ITB is based on generating new features including features based on machine learning algorithms.
This allows us to know more about the future behavior like probablity that the price will be 2% higher during the next week.
However, this knowledge by itself cannot be used *directly* for trading and says almost nothing about trade performance.
The main reason is that there can be many possible trade strategies based on the knowledge of future behavior.
These trade strategies produce buy and sell signals depending on the current situation on the market as well as
the current state of the available assets. The simulation script is intended for filling this gap by evaluating
trade performance based on the chosen trade strategy and its parameters.

The simulation (backtesting) parameters are specified in the `simulate_model` section of the configuration file:
```console
python -m scripts.simulate -c config.json
```

The script consumes predictions from the `predict_file_name` file.
This file includes all derived features and should be produced by the `predict_rolling` script.

Note that the script does not find performance for one set of parameters.
Rather, it evaluates a (large) set of parameters with the goal to find best performing trade model.

The `simulate_model` section describes a set of trade models and has the following attributes:
- `data_start` Start simulation from this date, for example, "1998-01-05"
- `data_end` End simulation with this date. If empty then all data will be used
- `direction` "long" or "short"
- `topn_to_store` How many best performing trade models (their parameters) to store in the output file
- `signal_generator`: Name of the generator which produces the column used to make buy/sell decisions
- `buy_sell_equal` true or false. Whether the threshold parameters for buy and sell signals are symmetric
- `grid` dictionary with lists. Each combination of values from these lists is considered one model
// Attributes are ists or evaluated to lists if string
  - `buy_signal_threshold` It is a list of values interpreted as buy thresholds to test like [0.1, 0.2, 0.3]
If it is a string then it is evaluated as Python code and the result is expected to be a list of values, for example, "np.arange(0.0, 0.5, 0.1).tolist()"
  - `sell_signal_threshold` Same as the previous parameter but used to find sell signals
