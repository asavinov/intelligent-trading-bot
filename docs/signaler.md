# Signaler: Signal server

## Principles

The task of this server is to *monitor* the state of the market and generate signals for the trader server which has to execute them. Signals are recommended actions in the context of the current market which however do not take into account the current actor state like available resources etc. Essentially, the signaler describes the future of the market and what to do (in general) in order to benefit from this future. Whether you can really use these opportunities is already what the trader does.

## Architecture

Main components:
- Singleton object representing:
  - the common state
  - parameters loaded at start up
- Analysis trigger:
  - Synchronous scheduler which will trigger analysis with some fixed period
  - Asynchronous trigger which will trigger analysis depending on incoming events like subscriptions or websocket events
- Analysis event loop. Each task means that new data about the market is available and new analysis has to be done. Each new task (triggered by a scheduler) performs the following steps:
  - collect latest data about the market
  - analyze the current (just updated) state of the market (if the state has really changed since the last time)
  - generate signals (if any) and notify trade server (or any other subscriber including logging), for example, by sending a message or putting the message into a queue of the trade server

## Analysis

### Predicted features

* Add RF algorithm to predictions.
* Add Linear SVM or logistic regression to predictions.
* Think about adding more sophisticated algorithms like KernelPCA or SVM.
Maybe for short histories in order to take into account drift or local dependencies.

### Feature generation

* Add trade-related features from https://github.com/bukosabino/ta
  * compute and use weighted average price
* Add long-term features (moving averages), for example, for 1 month or more.
* Add ARIMA-like features
