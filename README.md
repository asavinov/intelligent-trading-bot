```
 ___       _       _ _ _                  _     _____              _ _               ____        _ 
|_ _|_ __ | |_ ___| | (_) __ _  ___ _ __ | |_  |_   _| __ __ _  __| (_)_ __   __ _  | __ )  ___ | |_
 | || '_ \| __/ _ \ | | |/ _` |/ _ \ '_ \| __|   | || '__/ _` |/ _` | | '_ \ / _` | |  _ \ / _ \| __|
 | || | | | ||  __/ | | | (_| |  __/ | | | |_    | || | | (_| | (_| | | | | | (_| | | |_) | (_) | |_ 
|___|_| |_|\__\___|_|_|_|\__, |\___|_| |_|\__|   |_||_|  \__,_|\__,_|_|_| |_|\__, | |____/ \___/ \__|
                         |___/                                               |___/                   
₿   Ξ   ₳   ₮   ✕   ◎   ●   Ð   Ł   Ƀ   Ⱥ   ∞   ξ   ◈   ꜩ   ɱ   ε   ɨ   Ɓ   Μ   Đ  ⓩ  Ο   Ӿ   Ɍ  ȿ
```

> [![https://t.me/intelligent_trading_signals](https://img.shields.io/badge/Telegram-2CA5E0?logo=telegram&style=for-the-badge&logoColor=white)](https://t.me/intelligent_trading_signals) 📈 **<span style="font-size:1.5em;">[Intelligent Trading Signals](https://t.me/intelligent_trading_signals)</span>** 📉 **<https://t.me/intelligent_trading_signals>**

# Intelligent trading bot

The project is aimed at developing an intelligent trading bot for automatic trading cryptocurrencies using state-of-the-art machine learning approaches to data processing and analysis. The project provides the following major functions:
* Analyzing historic data and training machine learning models as well as finding their best hyper-parameters. It is performed in batch off-line mode.
* Signaling service which is regularly requests new data from the exchange and generates buy-sell signals by applying the previously trained models
* Trading service which does real trading by buying or selling the assets according to the generated signals

Note that is an experimental project aimed at studying how various machine learning and feature engineering methods can be applied to cryptocurrency trading. 

# Intelligent trading channel

This software is running in a cloud and sends its signals to this Telegram channel:

📈 **[Intelligent Trading Signals](https://t.me/intelligent_trading_signals)** 📉 **<https://t.me/intelligent_trading_signals>**

Everybody can subscribe to the channel to get the impression about the signals this bot generates.

Currently, the bot is configured using the following parameters:
* Exchange: Binance
* Cryptocurrency: ₿ Bitcoin
* Frequency: 1 minute
* Score between -1 and +1. <0 means decrease, and >0 means increase
* Filter: notifications are sent only if score is greater than ±0.15
* One increase/decrease sign is added for each step of 0.5 (after the filter threshold) 
* Prediction horizon 3 hours ahead. For example, if the score is +0.25 then the price is likely to increase 1-2% during next 3 hours
* History taken into account for forecasts: 12-24 hours

There are silent periods when the score in lower than the threshold (currently 0.15) and no notifications are sent to the channel. If the score is greater than the threshold, then every minute a notification is sent which looks like 

> ₿ 60,518 📉📉📉 Score: -0.26

The first number is the latest close price. The score -0.26 means that it is very likely to see the price 1-2% lower than the current close price during next few hours. The three decrease signs mean three 0.5 steps after the threshold 0.15.

# Signaler

The signaler performs the following steps to make a decision about buying or selling an asset:
* Retrieve the latest data from the server
* Compute derived features based on the latest history
* Apply several (previously trained) ML models by forecasting some future values (not necessarily prices) which are also treated as (more complex) derived features
* Aggregate the results of forecasting produced by different ML models by computing the final score which reflects the strength of the upward or downward trend. Positive score means growth and negative score means fall
* Apply a previously trained signal model to make a decision about buying or selling. The signal models are trained separately and are relatively simple in comparison to the forecasting ML models

The final result of the signaler is the score (between -1 and +1) and the signal (BUY or SELL). The score can be used for further decisions while signal is supposed to be used for executing real transactions.

Starting the service: `python3 -m service.server -c config.json`

# Training machine learning models

The following batch scripts are used to train the models needed by the signaler:
* Download the latest historic data: `python -m scripts.download_data -c config.json`
* Merge several historic datasets into one dataset: `python -m scripts.merge_data -c config.json`
* Generate feature matrix: `python -m scripts.generate_features -c config.json`
* Train prediction models: `python -m scripts.train_predict_models -c config.json`

There exist also batch scripts for hyper-parameter tuning and signal model generation based on backtesting:
* Generate rolling predictions which simulate what we do by regularly re-training the models and using them for prediction: `python -m scripts.generate_rolling_predictions -c config.json`
* Train signal models for choosing best thresholds for sell-buy signals which produce best performance on historic data: `python -m scripts.train_signal_models -c config.json` 

# Configuration parameters

The configuration parameters are specified in two files:
* `service.App.py` in the `config` field of the `App` class
* `-c config.jsom` argument to the services and scripts. The values from this config file will overwrite those in the `App.config` when this file is loaded into a script or service

Here are some most important fields (in both `App.py` and `config.json`):
* `symbol` it is a trading pair like `BTCUSDT` - it is important for almost all cases
* `data_folder` - location of data files which are needed only for batch scripts and not for services
* `model_folder` - location of trained ML models which are stored by batch scripts and then are loaded by the services
* Analyzer parameters. These mainly columns names.
  * `labels` List of column names which are treated as labels. If you define a new label used for training and then for prediction then you need to specify its name here. Note that we use multiple target variables (e.g., with different prediction horizons) and multiple prediction algorithms.
  * `class_labels_all` It is not used by the system and is created for convenience by listing *all* labels we compute so that it is easier to choose labels we want to experiment with during hyper-parameter tuning.
  * `features_kline` List of all column names used as input features for training and prediction.
  * `features_futur` Experimental. Currently, not used. Features based on future prices.
  * `features_depth` Experimental. Currently, not used. Features based on market depth (order book data).
* `signaler` is a section for signaler parameters
  * `notification_threshold` It is an integer like 0, 1, 3, 4 etc., which specifies the score threshold for sending notifications. Instead of using an absolute continuous threshold like 0.123, we specify the number of steps each step being (currently) equal 0.05. If you want to receive *all* notifications every minute, then set this parameter to 0. If you want to receive messages if score is greater than 0.10, then set it to 2. The notifier will also add one or more icons to each message and this number of icons is also equal to the number of 0.05 intervals exceeding the current threshold.
  * `analysis.features_horizon` This parameter specifies maximum history length used for training and prediction. The unit is the number of rows (not time). The system must know how much previous quotes is needed in order to be able to compute derived features and make predictions. For example, if we use rolling mean with window size 60 (1 hour in the case of minute data), then this parameter has to be equal 60. We suggest a higher value like 70 to guarantee that all 60 measurements are available. This parameter needs to be changed if you change your feature definitions, particularly, in `common.feature_generation.py`.
* `trader` is a section for trader parameters. Currently, not thoroughly tested.
* `collector` These parameter section is intended for data collection services. There are two types of data collection services: synchronous with regular requests to the data provider and asynchronous streaming service which subscribes to the data provider and gets notifications as soon as new data is available. They are working but not thoroughly tested and integrated into the main service. The current main usage pattern relies on manual batch data updates, feature generation and model training. One reason for having these data collection services is 1) to have faster updates 2) to have data not available in normal API like order book (there exist some features which use this data but they are not integrated into the main workflow).

Here is a sample `config.json` file:
```json
{
  "api_key": "<binance-key-only-for-trading>",
  "api_secret": "<binance-secret-only-for-trading>",

  "telegram_bot_token": "<source-chat-id>",
  "telegram_chat_id": "<destination-chat-id>",

  "symbol": "BTCUSDT",
  "base_asset": "BTC",
  "quote_asset": "USDT",

  "data_folder": "C:/DATA2/BITCOIN/GENERATED/BTCUSDT",
  "model_folder": "C:/DATA2/BITCOIN/MODELS/BTCUSDT"
}
```

# Trader

The trader is working but not thoroughly debugged, particularly, not tested for stability and reliability. Therefore, it should be considered a prototype with basic functionality. It is currently integrated with the Signaler but in a better design should be a separate service.
