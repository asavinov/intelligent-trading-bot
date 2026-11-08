# Intelligent Trading Bot

## About the project: goals and main properties

The aim of the project is to develop a generic forecasting framework by combining machine learning (ML) and feature engineering (FE). The project provides the following major functionalities:

- Clear and consistent separation between *two modes*:

    - *offline* (or batch) mode for extracting knowledge or parameters from historic data, particularly, training ML models and determining statistical parameters

    - *online* (or stream) mode for making forecasts based on the knowledge or parameters extracted from historic data and relying on (expert) knowledge encoded in domain-specific feaures using feature engineering

- *Extensible design* providing an easy way to define and use *derived features* using (Python) functions including standard technical indicators as well as arbitrary [custom features](features.md). The features can use encode expert knowledge or knowledge extracted from historic data. The feature-engineering part relies on the [column-oriented approach](column-oriented-design.md) where new features are defined in terms of previously defined features so that all features exist in a direct-acyclic graph (DAG) of definitions

- Possibility to work with different *trade frequencies* (time rasters or sample rates), for example, 1 minute, 1 hour or 1 day

- Integration with the external environment by means of input and output adapters:
    - *Input* via custom data collectors for fetching data from data source in online mode or downloading historic data in offline mode
    - Outputs via custom functions for sending the results of analysis like signals or predictions (in online mode) to external consumers lie Telegram channels, API end-point, databases or executing real transactions

- [Scripts](scripts.md) for *backtesting* and measuring trade performance on historic data. This requires periodic re-train of the intermediate ML models and perform walk-forward predictions

- *[Trading service](server.md)* for online mode which uses a [configuration file](configuration.md) to regtularly retrieve data updates, do analysis and send signals or execute trade transactions

## Introduction: Combining feature engineering and machine learining

### Feature engineering: defining features manually

Time series analysis in general and forecasting in particular heavily rely on [derived features](features.md) which are used instead of the original attributes. For example, it is known that comparing the original stock price with its moving average (derived features) or comparing different moving averages is a very good indicator for how the price will behave in future.

One major reason why we frequently use custom features is that we want to encode already existing precious expert knowledge. It is especially important in case we do not have enough data in order to explore it automatically or manually. An expert could simply say that for this analysis moving average is a good indicator so we can immediately encode it as a feature and use for forecasting.

Another reason is that even if we do not know which feature is good we might want to define many features and explore how they influence the results. We know that the original data is a bad predictor and we know that some kind of generic processing might help. But we do not know the best parameters for these custom indicators. Therefore, we define a series of features with different features and measure their performance.

Therefore having a good design for defining and using custom domain-specific features is a highly important aspect in time series analysis and forecasting. This project provides such a framework where new features can be defined based on already existing features. Our design relies on the [column-oriented approach](column-oriented-design.md) where columns of one table are defined in terms of other columns of this table.

### Machine learning: defining features automatically

In addition to manually defining new features, it is very important to have a mechanism for defining new features automatically based on the analysis of historic data. Instead of asking an expert what is a good indicator, we provide large amount of historic data and ask an algorithm to find a good predictor. Typically it is a statistical algorith or a machine learning algorithm. The result is a feature which processes [inputs](data-inputs.md) and computes outputs. The outputs can be immediately interpreted as predictions but it is not necessary and they can be well intermediate values (similar to hidden layers of neural networks). An advantage of having machine learning is that we can find complex features with complex definitions (like Support Vector Machines) which are almost impossible to define manually. Yet, applying ML algorithms requires large amounts of data and higher expertise when defining and evaluating the results. It also can be computationally expensive.

For training ML algorithms and finding best predictors we normally need explicitly defined [labels](labels.md). Labels in our approach are normal features which compute values from other features. One difference of labels is that they are typcially computed from future values while normal predicting features are computed from past values. Another difference is that labels are not needed and will not be computed in online mode where they are not needed and typically cannot be computed just because we do not know future values.

## Unifying feature eengineering and machine learning

In this project, we unify the usage of the methods of features engineering (manually defining features) and machine learning (automatically defined features) by providing these functions:

- A feature is defined as a generic function which converts input values to output value(s) at evaluation time by hiding the way it was defined - manually encoded as a Python function or automatically derived from historic data
- Features can use already existing features so that they compose a direct acyclic graph (DAG) of definitions
- A feature can be parameterized either by static (manually defined) parameters or dynamic (automatically derived) parameters. It is where we introduce machine learning. Intelligent indicators (features) use parameters derived from historic data which are referred to as a machine learning model. For example, a feature could use some regression model or a neuron network model
- Features which are able to derive their parameters from historic data are called [trainable features](trainable-features.md). They provide a function which takes historic data as input and produce a model as output. This model is then used at prediction time. When to use this training function is determined by the mode: offline (batch) or online (stream). All features work in online (stream) mode by means of their evaluation function. Some features can also work in offline (batch) mode by providing a function which produces a machine learning model (and this model is then consumed by this same feature in online mode).
- In online model, the framework evaluates all features by executing their prediction functions according to their dependencies. In offline mode, the framework will evaluate the features by executing their train functions before the prediction function is evaluated (so that the model is created or updated).
