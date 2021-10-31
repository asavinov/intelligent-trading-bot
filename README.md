```
 ___       _       _ _ _                  _     _____              _ _               ____        _ 
|_ _|_ __ | |_ ___| | (_) __ _  ___ _ __ | |_  |_   _| __ __ _  __| (_)_ __   __ _  | __ )  ___ | |_
 | || '_ \| __/ _ \ | | |/ _` |/ _ \ '_ \| __|   | || '__/ _` |/ _` | | '_ \ / _` | |  _ \ / _ \| __|
 | || | | | ||  __/ | | | (_| |  __/ | | | |_    | || | | (_| | (_| | | | | | (_| | | |_) | (_) | |_  
|___|_| |_|\__\___|_|_|_|\__, |\___|_| |_|\__|   |_||_|  \__,_|\__,_|_|_| |_|\__, | |____/ \___/ \__|
                         |___/                                               |___/                   
₿   Ξ   ₳   ₮   ✕   ◎   ●   Ð   Ł   Ƀ   Ⱥ   ∞   ξ   ◈   ꜩ   ɱ   ε   ɨ   Ɓ   Μ   Đ   ⓩ   Ο   Ӿ   Ɍ   ȿ
```

> [![https://t.me/intelligent_trading_signals](https://img.shields.io/badge/Telegram-2CA5E0?logo=telegram&style=for-the-badge&logoColor=white)](https://t.me/intelligent_trading_signals) 📈 <span style="font-size:1.5em;">Intelligent Trading Signals</span> 📉 <https://t.me/intelligent_trading_signals>

# Intelligent trading bot

The project is aimed at developing an intelligent trading bot for automatic trading cryptocurrencies using state-of-the-art machine learning approaches to data processing and analysis. The project provides the following major functions:
* Analyzing historic data and training machine learning models as well as finding their best hyper-parameters. It is performed in batch off-line mode.
* Signaling service which is regularly requests new data from the exchange and generates buy-sell signals by applying the previously trained models
* Trading service which does real trading by buying or selling the assets according to the generated signals

Note that is an experimental project aimed at studying how various machine learning and feature engineering methods can be applied to cryptocurrency trading. 

# Intelligent trading channel

This software is running in a cloud and sends its signals to this Telegram channel:

📈 **Intelligent Trading Signals** 📉 <https://t.me/intelligent_trading_signals>

Everybody can subscribe to the channel to get the impression about the signals this bot generates.

Currently, the bot is configured using the following parameters:
* Exchange: Binance
* Cryptocurrency: ₿ Bitcoin
* Frequency: 1 minute
* Score is between -1 and +1. -1 means decrease, and +1 means increase.
* Filter: notification is sent to the channel only if score is greater than ±0.15
* One increase/decrease sign is added for each step of 0.5 (after the filter threshold) 
* Prediction horizon 3 hours ahead. For example, if the score is +0.25 then the price is likely to increase during next 3 hours
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
* `-c config.jsom` argument to the services and scripts. The values from this config file will overwrite those in the `App.config` 

Here are some most important fields:
* `symbol` it is a trading pair like `BTCUSDT` - it is important for almost all cases
* `data_folder` - location of data files which are needed only for batch scripts and not for services
* `model_folder` - location of trained ML models which are stored by batch scripts and then are loaded by the services
* `signaler` is a section for signaler parameters
* `trader` is a section for trader parameters

# Trader

The trader is working but not thoroughly debugged, particularly, not tested for stability and reliability. Therefore, it should be considered a prototype with basic functionality. It is currently integrated with the Signaler.
