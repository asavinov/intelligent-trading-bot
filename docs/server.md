# Server

The server provided in the project is only one possible implementation of the column-oriented approach with feature engineering and machine learning to time series analysis and trading.
The server has the following design goals and assumptions:
- The major role of the server is to orchestrate different activities
- All activities are triggered by the scheduler at equal time intervals
- All the data is managed in-memory
- Only one symbol is currently supported
- One iteration consists of the following steps:
  - data retrieval using input adapters
  - merge and append data
  - data analysis by evaluating all features
  - executing output adapaters by notifying external consumers (like Telegram channel) or interacting wtih an exchange

The server is started as follows:
```console
python -m service.server -c config.jsonc
```

It will configure its scheduler using the `freq` parameter.
Then it performs initial data retrieval and initial data analysis.
After that it will wake up regularly and do missing data retrieval, evaluation of features and data output.

## Custom servers

The standard activities (input, analysis, output) can be also executed using a custom server.
One approach is to simply regularly execute the available scripts in appropriate order using a scheduler:
- `downlaod` script will retrieve missing data and update the source files
- `features`, `predict`, `signals` scripts executed sequentially will generate all derived features using available ML-models
- `output` script will execute output adapters which can do real trade or notify other services

Note that the scripts expecte certain input and output file names which have to be configured correctly so that
output of the previous script is consumed by input of the next script.

This sequence assumes that ML-models have been trained and they run in predict mode (`train` attribute is false).
If we want to train models each time new data is appended, then `train` attribute has to be true
and the sequence of analysis (between data input and output) has to include label generation and training:
- `features`, `labels`, `train`, `predict`, `signals`
Note again that input-ouput file names for the scripts should be set correctly.
Such a scenario will be time consuming because of training but in some cases it might be desirable.
