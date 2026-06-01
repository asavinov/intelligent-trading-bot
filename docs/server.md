# Server

The server provided in this project is just one possible implementation of a column-oriented approach that applies feature engineering and machine learning to time series analysis and trading.
The server adheres to the following design goals and assumptions:
- The primary role of the server is to orchestrate various activities.
- All activities are triggered by the scheduler at regular intervals.
- All data is managed in memory.
- Only one symbol is currently supported.
- One iteration consists of the following steps:
  - Data retrieval using input adapters.
  - Merging and appending data.
  - Data analysis by evaluating all features.
  - Executing output adapters by notifying external consumers (such as a Telegram channel) or interacting with an exchange.

The server is started as follows:
```console
python -m service.server -c config.jsonc
```

It configures its scheduler using the `freq` parameter.
Then, it performs initial data retrieval and initial data analysis.
Afterward, it wakes up regularly to retrieve missing data, evaluate features, and output data.

## Custom servers

The standard activities (input, analysis, output) can also be executed using a custom server.
One approach is to regularly execute the available scripts in the appropriate order using a scheduler:
- The `download` script retrieves missing data and updates the source files.
- The `features`, `predict`, and `signals` scripts, executed sequentially, generate all derived features using available ML models.
- The `output` script executes output adapters, which can perform real trades or notify other services.

Note that the scripts expect certain input and output file names, which must be configured correctly so that the output of the previous script is consumed as the input of the next script.

This sequence assumes that ML models have been trained and are running in predict mode (the `train` attribute is false).
If you want to train models each time new data is appended, the `train` attribute must be set to true, and the analysis sequence (between data input and output) must include label generation and training: `features`, `labels`, `train`, `predict`, `signals`.
Again, note that the input and output file names for the scripts should be set correctly.
This scenario will be time-consuming due to the training process, but it may be desirable in some cases.
