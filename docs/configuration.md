# Configuration and parameterization

## Parameterization

All parameters for scripts, server and some high-level functions are provided as JSON object.
It is represented as a JSON file with comments or as a Python dictionary object at run time.
Not all parameters and parameter sections are used by all scripts and functions.

## Global parameters

Here the most important parameters:

- General parameters:
  - `symbol` In particular, it is used as a subfolder name for all generated files (within `data_folder`)
  - `description` Any textual desciprition of this configuraiton. It can be then used by various plugable components, for example, for text or visual output.
  - `freq` Frequency of data in `pandas` format, for example, `1h` for hourly data or `1min` for minutely data.
  - `train` Boolean attribute specifying if the analysis has to run in train (if true) or predict (if false) mode.
If it is true, then all trainable features will train their models based on historic data.
If false, then the existing models will be used for prediction.
- Persistence:
  - `data_folder` Location of all the data files for this analysis
- Data providers:
  - `venue` Name of the data provider and the corresponding connector. Currently these values are supported: `binance`, `yahoo`, `mt5`.
  - `api_key` and `api_secret` Credentials for the selected venue (data provider). They will be passed to the connector.
  - `client_args` Dictionary with arbitrary arguments passed to the data connector
- Output:
  - `telegram_bot_token` and `telegram_chat_id` Used to send notifications to Telegram bot by the corresponding output adapters

## Analaysis table parameters

All data during analysis are represented as a dataframe which has these parameters.
This dataframe has to have certain shape which is specified via the following parameters which are used mainly by the server:
- `label_horizon` The minimum number of future rows required to compute a label. It is our prediction horizon.
It is taken into account in training by ignoring rows which do not have enough future data.
- `features_horizon` The minimum number of past rows for a feature to be valid. For example, if we want to computing
moving average for 10 days then the feature requires 10 previous rows.
Essentially, the very first `features_horizon` rows will be considered invalid and ignored in analysis.
This value should be taken from the feature defintions.
- `train_length` Default limit for the train data set size.
It is a maximum value for all ML-features but individual features can set their own values.
0 means all available data.
- `predict_length` This minimum number of rows will be kept up-to-date and valid in online mode.
Since their values must be valid, these rows must have at least `features_horizon` before.
- `append_overlap_records` In online mode, the server will request more records than strictly required (missing).
This additional number is specified in this parameter.
The received new records will be (again) evaluated and overwrite previuos values.
It is desirable in case of connection errors or in case last rows have small deviations and differ from what is provided later.

## Parameter sections

Model registry is a list `model_registry` with entries consisting of `name` and `file` attributes.

Feature definitions are specified in four sections:
- `feature_sets`
- `label_sets` labels needed in train mode
- `train_feature_sets` trainable features
- `signal_sets` features evaluated after ML-features
Features use these global parameters:
- `train_features` a list of all column names which will be by default selected to pass data to train algorithms (in both train and predict modes)
Algorithms (trainable features) can set their own lists of features they want to use for learning and prediction.
- `labels` A list of all labels (if not overwritten by individual trainable algorithms)
- `algorithms` Obsolete. Use either `train_feature_sets` or normal `feature_sets` to define trainable features

Outputs are defined in the `output_sets` which is a list of dictionaries passed to the output adapters after analysis.
For example, here a trading or notification adapter can be specified.

Other sections can describe utilities, for example, `rolling_predict` or `simulate_model`.
