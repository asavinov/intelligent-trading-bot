# Intelligent Trading Bot

## Goals and main properties

The aim of the project is to develop a generic forecasting framework that combines machine learning (ML) and feature engineering (FE). The project provides the following major functionalities:

- Clear and consistent separation between *two modes*:

    - *Offline* (or batch) mode for extracting knowledge or parameters from historic data, particularly for training ML models and determining statistical parameters

    - *Online* (or stream) mode for making forecasts based on the knowledge or parameters extracted from historic data, relying on (expert) knowledge encoded in domain-specific features via feature engineering

- *Extensible design* that provides an easy way to define and use *derived features* via (Python) functions. This includes both standard technical indicators and arbitrary [custom features](features.md). Features can encode expert knowledge or knowledge extracted from historic data. The feature-engineering layer relies on the [column-oriented approach](column-oriented-design.md), where new features are defined in terms of previously defined features, so that all features form a directed acyclic graph (DAG) of definitions

- Support for different *trade frequencies* (time rasters or sample rates), for example 1 minute, 1 hour, or 1 day

- Integration with the external environment through input and output adapters:
    - *Input* via custom data collectors that fetch data from a data source in online mode or download historic data in offline mode
    - *Output* via custom functions that send analysis results (such as signals or predictions in online mode) to external consumers, for example Telegram channels, API endpoints, databases, or systems that execute real transactions

- [Scripts](scripts.md) for *backtesting* and measuring trade performance on historic data. This requires periodic re-training of intermediate ML models and walk-forward prediction

- A *[trading service](server.md)* for online mode that uses a [configuration file](configuration.md) to regularly retrieve data updates, perform analysis, and send signals or execute trade transactions

## Feature engineering: defining features manually

Time-series analysis in general, and forecasting in particular, relies heavily on [derived features](features.md) that are used in place of the original attributes. For example, it is well known that comparing the original stock price with its moving average (a derived feature), or comparing different moving averages, is a strong indicator of how the price is likely to behave in the future.

One major reason for using custom features is the desire to encode already existing expert knowledge. This is especially important when there is insufficient data to discover patterns automatically or manually. An expert can simply state that a moving average is a useful indicator for a given analysis; the feature can then be encoded immediately and used for forecasting.

Another reason is that, even when the best features are unknown, it is often useful to define many candidate features and explore how they influence the results. The original data is typically a weak predictor, and some form of generic processing is known to help, but the optimal parameters of the custom indicators are not known in advance. Therefore a series of features with different parameterizations can be defined and their performance measured.

A well-designed mechanism for defining and using custom domain-specific features is therefore a critical aspect of time-series analysis and forecasting. This project provides such a framework, in which new features can be defined in terms of already existing features. The design follows the [column-oriented approach](column-oriented-design.md), where columns of a table are defined in terms of other columns of the same table.

## Machine learning: defining features automatically

In addition to manually defining new features, it is important to have a mechanism that can [define features automatically](trainable-features.md) from the analysis of historic data. Instead of asking an expert for a good indicator, a large amount of historic data is supplied and an algorithm is asked to find a useful predictor. Typically this is a statistical or machine-learning algorithm. The result is a feature that processes [inputs](data-inputs.md) and produces outputs. These outputs can be interpreted directly as predictions, but they need not be; they may also serve as intermediate values (analogous to the hidden layers of a neural network).

An advantage of machine learning is that it can discover complex features with complex definitions (for example Support Vector Machines) that would be almost impossible to specify by hand. However, applying ML algorithms requires large amounts of data, greater expertise in defining and evaluating the results, and can be computationally expensive.

Training ML algorithms and finding good predictors normally requires explicitly defined [labels](labels.md). In this approach labels are ordinary features that compute values from other features. One difference is that labels are typically computed from future values, whereas ordinary predictive features are computed from past values. Another difference is that labels are not needed—and will not be computed—in online mode, both because they are unnecessary and because future values are unavailable.

## Unifying feature engineering and machine learning

In this project the methods of feature engineering (manually defined features) and machine learning (automatically defined features) are unified by the following principles:

- A feature is defined as a generic function that converts input values into output value(s) at evaluation time, hiding the way the feature was created—whether it was encoded manually as a Python function or derived automatically from historic data

- Features may depend on already existing features, thereby forming a directed acyclic graph (DAG) of definitions

- A feature can be parameterized either by static (manually defined) parameters or by dynamic (automatically derived) parameters. This is the point at which machine learning is introduced. Intelligent indicators (features) use parameters derived from historic data; these parameters constitute a machine-learning model. For example, a feature may rely on a regression model or a neural-network model

- Features that are able to derive their parameters from historic data are called [trainable features](trainable-features.md). They provide a function that takes historic data as input and produces a model as output. The model is then used at prediction time. When the training function is invoked is determined by the operating mode: offline (batch) or online (stream). All features operate in online (stream) mode via their evaluation function. Some features can also operate in offline (batch) mode by supplying a training function that produces a machine-learning model; that model is subsequently consumed by the same feature in online mode

- In online mode the framework evaluates all features by executing their prediction functions according to their dependency order. In offline mode the framework first executes the training functions of the relevant features (so that models are created or updated) before the prediction functions are evaluated
