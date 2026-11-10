# Outputs and trading

## Define output functions

At a high level, operation in online (stream) mode consists of regularly executing three steps:

- Fetch new data by executing the specified collector (input) functions
- Analyze the data by evaluating all derived features and updating/appending new data, including predicted values and indicators
- Use the new values by sending them to external consumers

This section describes the third step. Examples of such output adapters include:

- Sending notifications to external consumers when certain conditions are met. For example, a notification can be sent to a Telegram channel or an API endpoint
- Storing selected data persistently in a log file or database
- Interacting with an exchange, for example by executing buy/sell orders

The functions that are executed after analysis are specified in the `output_sets` section of the configuration file. This is a list of dictionaries, where each dictionary defines one function to be executed. Output functions do not modify the data state; their purpose is to act as a bridge or adapter between this service and external services.

```jsonc
"output_sets": [
  {...}, // First output adapter
  {...}, // Second output adapter
  {...}  // Third output adapter
]

The (Python) function executed by an output adapter is defined in the `generator` attribute. It is either a predefined name that the framework resolves to a Python function, or the explicit name of a custom Python function defined in an additional module.

For example, a diagram with a line chart can be sent regularly to a Telegram channel using the following output adapter:

```jsonc
{
  "generator": "diagram_notification_model", "config": {
    "diagram_notification": true,
    "notification_freq": "1D",

    "score_column_names": "trade_score",
    "score_thresholds":[-0.08, 0.08],

    "resampling_freq": "1h", "nrows": 168
  }
}
```

A custom output adapter can be defined by implementing a Python function that receives the dataframe containing the analysis results together with configuration parameters. This is done in a manner similar to the definition of [custom features](features.md).

## Trader

A trader is not a dedicated service but an output adapter. Like other output adapters, it receives the latest data with intelligent indicators and then decides what to do (buy, sell, etc.). Importantly, it maintains intermediate state containing the current assets, orders, and other data required for trading. It also knows how to interact with the exchange by executing the necessary API requests.

Many trading strategies can be built on the same signals or (intelligent) indicators. The project currently implements one very simple strategy: buy when an intelligent indicator exceeds a certain threshold and sell when the indicator falls below a (parameterized) threshold. This trading strategy can be simulated without interacting with any exchange by using the following output adapter:

```jsonc
{
  "generator": "trader_simulation", "config": {
    "buy_signal_column": "buy_signal_column",
    "sell_signal_column": "sell_signal_column"
}}
```

It relies on two columns with binary values: `buy_signal_column` and `sell_signal_column`. These columns must be defined as features and computed during analysis. The column definitions include a threshold that determines whether the value is true or false. The output adapter then simulates transactions by buying or selling accordingly.

For real trading, the `trader_binance` adapter is available. It executes trades against the Binance exchange. It performs various checks (current balance, available assets, previously submitted orders, latest price) and monitors the execution of any open orders before submitting new ones.

Note, however, that this adapter is not thoroughly tested and should be used with caution.
