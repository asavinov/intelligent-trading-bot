-----
# Conventions

## Input data

Column names are defined by the `download_data` functions. Note that future and kline data have the same column names when loaded from API. Future columns get prefix "f_".

## Derived features

The columns added to df are hard-coded by functions in module `common.feature_generation.py`

* klines columns: function `generate_features()`. They are listed in `App.config.features_kline`
* futur columns: function `generate_features_futur()`. They are listed in `App.config.features_futur`
* depth columns: function `generate_features_depth()`. They are listed in `App.config.features_depth`

## Labels

The columns added to df are hard-coded by functions in module `common.label_generation.py`. These columns are similar to derived features but describe future values. Note that labels are derived from kline data only (at least if we trade on spot market).

Currently we generate labels using `generate_labels_thresholds()` which is a long list in `App.config.class_labels_all`. Only a subset of these labels is really used listed in `App.config.labels`. 

## Rolling predictions and predictions (label scores)

Predicting columns are generated using the following dimensions:
* target label (like `high_15`): we generate predictions for each chosen label. Note that labels themselves may have structure like threshold level (10, 15, 20 etc.) but here we ignore this and assume that all labels are independent
* input features (`k`, `f`): currently we use either klines or future
* algorithm (`gb`, `nn`, `lc`): we do the same predictions using different algorithms. We assume that algorithm name includes also its hyper-parameters and maybe even history length.
* history length: how much historic data was used to train the model. Currently we do not use this parameter. We could also assume that it is part of the algorithm or its hyper-parameters.

Accordingly, the generated (predicted) column names have the following structure:

    <target_label>_<input_data>_<algorithm>

For example: `high_15_k_nn` means label `high_15` for kline derived features and `nn` prediction model.

-----
# Data/knowledge processing pipeline

## Main steps

Note: see also start.py

#### 1. Download historic (klines) data

* Download historic data. Currently we use 2 separate sources stored in 2 separate files:
  * klines (spot market), its history is quite long
  * futures (also klines but from futures market), its history is relative short
  * Script: scripts.download_data.py or "python start.py download_data"
  * Script can be started from local folder and will store result in this folder.

Notes:
* Edit main in binance-data.py by setting necessary symbol
* Run script binance-data.py which will directly call get_klines_all()
* Get klines for futures: Use the same function get_klines_all() but uncomment the section at the beginning.

#### 2. Merge historic data into one dataset

* Merge historic data into one dataset. We analyse data using one common raster and different column names. Also, we fix problems with gaps by producing a uniform time raster. Note that columns from different source files will have different history length so short file will have Nones in the long file.
  * Script: scripts.merge_data.py
  * If necessary, edit input data file locations as absolute paths

#### 3. Generate feature matrix

* Generate features. Here we compute derived features (also using window-functions) and produce a final feature matrix. We also compute target features (labels). Note that we compute many possible features and labels but not all of them have to be used. In parameters, we define past history length (windows) and future horizon for labels. Currently, we generate 3 kinds of features independently: klines features (source 1), future features (source 2), and label features (our possible predictions targets).
  * Script: scripts.generate_features.py or "python start.py generate_features

Notes:
* The goal here is to load source (kline) data, generate derived features and labels, and store the result in output file. The output is supposed to be used for other procedures like training prediction models.
* Ensure that latest source data has been downloaded from binance server (previous step)
* Max past window and max future horizon are currently not used (None will be stored)
* Future horizon for labels is hard-coded (currently 300). Change if necessary
* If necessary, uncomment line with storing to parquet (install the packages)
* Output file will store features and labels as they are implemented in the trade module. Copy the header line to get the list.
* Same number of lines in output as in input file
* Approximate time: ~20-30 minutes (on YOGA)

#### 4. Generate rolling predictions

* Generate rolling predictions. Here we train a model using previous data less frequently, say, once per day or week, but use much more previous data than in typical window-based features. We apply then one constant model to predict values for the future time until it is re-trained again using newest data. (If the re-train frequency is equal to sample rate, that is, we do it for each new row, then we get normal window-based derived feature with large window sizes.) Each feature is based on some algorithm with some hyper-parameters and some history length. This procedure does not choose best hyper-parameters - for that purpose we need some other procedure, which will optimize the predicted values with the real ones. Normally, the target values of these features are directly related to what we really want to predict, that is, to some label. Output of this procedure is same file (feature matrix) with additional predicted features (scores). This file however will be much shorter because we need some quite long history for some features (say, 1 year). Note that for applying rolling predictions, we have to know hyper-parameters which can be found by a simpler procedure.
  * Script: scripts.generate_rolling_predictions.py or "python start.py generate_rolling_predictions.py"

Notes:
* Prerequisite: We already have to know the best prediction model(s) and its best parameters
* There can be several models used for rolling predictions
* Essentially, the predicting models are treated here as (more complex) feature definitions
* Choose the best model and its parameters using grid search (below)
* The results of this step are consumed by signal generator

#### 5. (Grid) search for best parameters of and/or best prediction models

The goal is to find best prediction models and their best parameters using hyper-parameter optimization. The results of this step are certain model (like nn, gradient boosting etc.) and, importantly, its best hyper-parameters.

Notes:
* The results are consumed by the rolling prediction step
* There can be many algorithms and many historic horizons or input feature set

* Grid search.
  * Script: grid_search.py

#### 6. Train prediction models

Here we regularly train prediciton models to be used in the production service as parameters of the corresponding predicted feature generation procedures.

Notes:
* There can be many predicted features and models, for example, for spot and future markets or based on different prediction algorithms or historic horizons

Script: train_predict_models.py 

#### 7. Train signal models

Here we find best parameters for signal generation like thresholds.

* Train signal models. The input is a feature matrix with all scores (predicted features). Our goal is to define a feature the output of which will be directly used for buy/sell decisions. We need search for the best hyper-parameters starting from simple score threshold and ending with some data mining algorithm.
  * Script: scripts.train_signal_models.py or "python start.py train_signal_models.py"

Notes:
* We consume the results of rolling predictions
* We assume that rolling prediction produce many highly informative features
* The grid search (brute force) of this step has to test our trading strategy using back testing as (direct) metric. In other words, trading performance on historic data is our metric for brute force or simple ML 
* Normally the result is some thresholds or some simple ML model
* Important: The results of this step are consumed in the production service to generate signals 

-----
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

-----
# Trader: trade server

The trader receives signals and then executes them depending on the current situation which includes its own state as well as the market state.

## Architecture

* An incoming queue contains signals in the order of their generation. We use a queue because we want to ensure that signals are processed sequentially by one procedure rather than concurrently
* A trading main procedure retrieves all signals and processes the last one as the most up-to-date
* Executing a signal involves various checks before the real execution:
  * Check the current balance in our local state as well as on the server
  * Check whether we have already an order submitted (which means some resources are frozen)
* Execution
  * Retrieve the latest price
  * Retrieve the current order book and get the latest available offer (price and volume)
  * Determine parameters of the order: price, volume, time to live etc.
  * Submit order and set the 
* Regularly check the order status:
  * Synchronous. We do it either by using a synchronizer (say, once per second) which will do nothing if we have no orders, or by a procedure which will be activated only on the case of submitted orders
  * Asynchronous. Here we are waiting for a confirmation event
  * In any case, this procedure changes our local state depending on the result of execution. So submitting an order is considered an operation or action which however has a special asynchronous mechanism of return (finishing). The return/finish is needed because this event will continue the main submission procedure. 

Problem of order status monitoring and order execution confirmation (rejection, execution etc.) For example, we submit an order which means that some independent (remote) process has been started. After that, we suspend the current procedure and want to resume it after getting the result of order execution. There are the following possibilities:
* We really suspend this task and resume it (somehow) when a confirmation is received. This approach has some difficulties. For example, it is not clear how to suspend and then resume a task using removing interactions. Second, a task might be suspended for a quite a long time, possibly, because of some problems with the remote server or this client. So we need to be able to resume processing even after rebooting our server or repairing the connection with the remote server. In any case, we need to mark somewhere in our local state that there is a remote process (order execution) with some parameters in order for other procedurs to know (say, if next signal is received and its processing started)
* We assume that any submitted order means creating a remote procedure which must be somehow explicitly represented in our local state. Note that this local state is restored/initialized after each start of the local server. In other words, when the server starts, we read the remote list of orders and store it locally so that all our procedures know the current state. In this case, we may have a special procedure which only processes returns/results of order executions, that is, its task is to update our local state each time the remote order list changes as well as notify other procedures about this change. We can implement a special data structure which represents the remote order list along with procedures for submitting orders and updating order status and notifying others about the changes. Thus submitting orders and processing order status changes are decoupled.

## Order state and balance management

We need a component which will be responsible for the representation of the remote orders and their life-cycle as well as the corresponding balances. Essentially, we need to maintain a local list of orders. When we submit an order, we add an entry into this list (if it was created successfully). What is more difficult, if an order changes its remote state, then we need to update the state of the entry in our local list. Typically, a remote order can be rejected, expire, be (partially) executed or cancelled in some other way. In addition to submission, we ourselves could cancel an order.

It is necessary to distinguish:
* Asynchronous like execution, cancellation, expiration. These changes can happen at any time. We get the result at any time either independently or by sending a special request.  
* Synchronous operations with orders like creation, cancellation, getting status. We get the result as a return value of the request. For the synchronous approach, we need to implement a regular procedure which will request latest status of orders, update local state, and notify other procedures which wait for this status change (say, after order execution). This same procedure might also do some other tasks, for example, notify other procedures about time-out which is not stored in the remote state but is part of our own logic of order processing. Important is that we need a dedicated (frequent) scheduler which will request remote order state (if there are local orders to be checked) as well as do some other tasks like requesting balance status and maybe market latest information. All these tasks are performed with the purpose to update our local state (orders status, balances, market prices) and notifying/triggering other procedures, e.g., creating a task when an order has been executed or cancelled.

One problem is distribution of tasks between the (frequent) remote state synchronization scheduler which can do some processing, and tasks in event loop which are triggered by this scheduler and also can do some tasks including changing the state of the order list. We also need to understand that the same order list is represented remotely and locally, and at least two independent processes can change it: the remote server (execution, cancellation, expiration etc.) and our local trader server (creation, cancellation, modify amount etc.) We need to develop a consistent design for such changes of the common order list. If balances and market state are passive, that is, we only read its state from the server, then order list can be changed also locally. We probably need to specify exactly what opreations can be done locally: 
* Order creation/submission (this is never done remotely). We send a submission request to the server by asking it to create an order. We can create a (empty) record with this order if we get an immediate response. And then the standard procedure will synchronize the local state with the remote state. In the case something happens, a new local task is created or we can process this change immediately.
* Order cancellation. We send this request to the server, it deletes the order, and we update the local state.

One approach is that we introduce a high level function with the semantics of changing the side by moving assets like "move from USD to BTC some amount with max price X". Once we call this function, the state of assets has some intermediate status because some amount is locked and it will be in this intermediate state till the process returns (either successfully or with error). In any case, we need to distinguish this intermediate state and we need to be able to wait for the end of this procedure by being notified. In some cases, we might want to break this intermediate state by cancelling an order. So we should talk about intermediate state of some asset transfer process rather than individual orders. At the level of the logic, we work with asset transfers and orders are simply a way to implement these asset movements. Essentially, we need a class with such methods like get_current_assets, can_buy/sell, move_assets (trade), is_in_move, cancel move etc. Internally, it reflects the latest state of the orders/assets but will periodically (if necessary, e.g., if there are open orders) synchronize the state. It will also notfy about state changes (order executed, i.e., movement finished etc.)

## Simple 1 minute trading strategy

Modes:

* [not in market - buy mode] Ready and try to enter the market.
  * If there is a buy signal, then generate market order or limit buy order.
  * Check for execution of the order. For market order immediately. For limit order regularly.
  * After buy order executed, create limit sell order and store its id or otherwise remember that we are are in sell mode.

* [in market - sell mode] Check the execution status of the existing limit order
  * If not executed then
    * do nothing (continue waiting)
    * possibly update its limit price
    * if time out, then force sell by converting it to market order or makring limit price very low and confirm its execution
    * if (strong) sell signal, the force sell
  * If executed:
    * Log transaction
    * Switch to buy mode and execute the logic of buy mode

## Principles of our strategy

* Buy order is market order, sell order is a limit order
* Only 3 cases: buy order running, sell order running, no orders running (buy and sell orders running is impossible)
  * if running orders,
      * buy order is running: no-op (wait for the result by checking order status regularly maybe more frequently), maybe price adjustment
      * sell order running: check if adjustment is needed, time out with (market price *adjustment*), or cancel
  * if no running orders then two situations:
    * we have money (and can buy), if buy signal, then immediately new *buy* order (for market price), otherwise no-op
    * we have coins (and can sell), immediately new limit *sell* order ideally right after buy order executed (everything else is controlled via its limit price)

Define independent functions executed synchronously or awaited:
* buy order immediately executed as either market order or with very small limit (which means that there is probability that it will not be executed immediatelly and hence has to be cancelled)
* check the state of an existing order (still active, filled or cancelled)
* cancel and order
* update limit price of an order

Functions can return None which is an indication of some problem and we have to process this result (cancelling current processing step or maybe delay next processing step by raising some "problem flag")
Functions have some timeout (say, 5-10 seconds) and number of retries.

Check possibility of using OCO and other special orders and options.

## Problems

#### Problem 1: Modify/adjust limit order price

Currently main problem is modifying/adjusting existing order price - either to force-sell or to adjust-sell.
Approach 1: Kill existing order, modify its parameters and submit again.
  1. In one cycle. Kill synchronously, check funds request (if coins are still available for sale), submit new sell order.
  2. In two cycles. Kill, continue cycle, check orders as usual on the next cycle: if exists then kill-continue, if does not exist (killed) then create new sell order as usual.
Approach 2: Submit order (price) modification request.
Solution. We assume that price cannot be modified. Therefore, we must kill the current order and create a new (modified) order.
Solution. We assume that price cannot be modified. Therefore, we must kill the current order and create a new (modified) order.
  Kill synchronously using cancel request.
  Check response "status" field. If "CANCELED" then continue. If not then send check status request in some time until it status is cancelled.
  Create new sell request as usual with new parameters.

#### Problem 2: Convert limit order to market order

This can be done by simply updating limit price lower than the market. But we need to be able to retrieve market price.

#### Problem 3: Sync response for some requests like kill order

We do not want to wait one cycle or even worse regularly check the status.
newOrderRespType parameter: ACK, RESULT, or FULL
- MARKET and LIMIT order types default to FULL,
- all other orders default to ACK

#### Problem 4 (low priority, future): Async processing triggered by incoming web-socket stream

We work synchronous to our local time every 1 minute.
It is theoretically possible that when we request data (klines etc.) right after 1 local minute, the server is not yet ready.
An alternative solution is to subscribe to the service and listen to its update.
In this case, we trigger processing precisely when a new 1m-kline is received (independent of our local clocks).
Our system is then synchronized with the service and is driven by the service data updates being them 1m-klines or order executions.
Java WebSocket: client.onCandlestickEvent("ethbtc", CandlestickInterval.ONE_MINUTE, response -> System.out.println(response));
Python WebSocket: conn_key = bm.start_kline_socket('BNBBTC', process_message, interval=KLINE_INTERVAL_30MINUTE)
Our processing logic should remain the same but now it is triggered not by the local scheduler but rather by external events.

-----
# General system

#### General system functions:

* Ping the server: https://python-binance.readthedocs.io/en/latest/general.html#id1
* Get system status: https://python-binance.readthedocs.io/en/latest/general.html#id3
* Check server time and compare with local time: https://python-binance.readthedocs.io/en/latest/general.html#id2

#### Time synchronization in OS and time zones

https://www.digitalocean.com/community/tutorials/how-to-set-up-time-synchronization-on-ubuntu-16-04

Check synchronization status:
```
$ timedatectl
```

"Network time on: yes" means synchronization is enabled. "NTP synchronized: yes" means time has been synchronized.

If timesyncd isn’t enabled, turn it on with timedatectl:
```
$ sudo timedatectl set-ntp on
```

Using chrony:

https://www.fosslinux.com/4059/how-to-sync-date-and-time-from-the-command-line-in-ubuntu.htm

```
$ sudo apt install chrony
chronyd  # One-shot time check without setting the time
chronyd -q  # One-shot sync
```

#### Create virtual environment

```
$ python3.7 -m pip install --user pip --upgrade
$ python3.7 -m pip install --user virtualenv --upgrade
```

```
$ python3.7 -m virtualenv --version
virtualenv 20.0.13
$ python3.7 -m virtualenv venv
```

#### Start from Linux

Modify start.py by entering data collection command. Alternatively, pass the desired command as an argument.
See the file for additional comments. For example:
* `collect_data` is used to collect depth data by making the corresponding requests.
  * It is possible to specify frequency 1m, 5s etc.
  * It is possible to specify depth (high depth will decrease weight of the request)
* `collect_data_ws` is used to collect stream data like klines 1m and depth.
  * klines will get update every 1 or 2 seconds for the current 1m kline
  * Depth stream will send new depth information (limited depth) every 1 second
  * Other streams could be added to the app configuration

```
switch to the project root dir
$ source venv/bin/activate OR source ../trade/venv/bin/activate
(venv) $ python3.7 --version
Python 3.7.3
(venv) $ nohup python3.7 start.py &
(venv) $ nohup python3.7 start.py collect_data_ws &
<Enter>
$ logout
```
End:
```
login
ps -ef | grep python3.7
kill pid_no
```
#### Compress and download collected data files

Zip into multiple files with low priority one file:
```
nice -n 20 zip -s 10m -7 dest.zip source.txt
```

Information about zip and nice (priority):
```
zip -s 100m archivename.zip filename1.txt
zip -s 100m -r archivename.zip my_folder  # folder and all files recursively
nice -10 perl test.pl - run with niceness 10 (lower priority) (- is hyphen - not negative).
nice --10 perl test.pl - start with high priority (negative niceness)
nice -n -5 perl test.pl - increase priority
nice -n 5 perl test.pl - decrease priority
nice -n 10 apt-get upgrade - start with lower priority (lower values of niceness mean higher priority, so we need higher values)
```

#### Sudden reboots

Information about last reboot:
```
last reboot

tail /var/log/syslog or less /var/log/syslog
```

System wide logger:
```
tail /var/log/syslog
less /var/log/syslog
```
Kernel log:
```
tail /var/log/kern.log
```

Example automatic reboot:

```
last reboot
reboot   system boot  4.15.0           Thu Apr 30 08:55   still running
reboot   system boot  4.15.0           Thu Apr 30 08:21   still running
```

```
syslog
Apr 30 06:03:01 linux CRON[23790]: (root) CMD (cd / && run-parts --report /etc/cron.hourly)
Apr 30 06:40:33 linux systemd[1]: Starting Daily apt upgrade and clean activities...
Apr 30 06:40:34 linux systemd[1]: Started Daily apt upgrade and clean activities.
Apr 30 06:55:06 linux systemd[1]: getty@tty2.service: Service has no hold-off time, scheduling restart.
Apr 30 06:55:06 linux systemd[1]: getty@tty2.service: Scheduled restart job, restart counter is at 876.
```

Check available timers, particularly, daily upgrade timer:
```
sudo systemctl list-timers
Fri 2020-05-01 06:14:53 UTC  19h left      Thu 2020-04-30 06:40:33 UTC  3h 58min ago apt-daily-upgrade.timer      apt
```
Solutions (https://superuser.com/questions/1327884/how-to-disable-daily-upgrade-and-clean-on-ubuntu-16-04):
* simply remove package unattended-upgrades: apt-get remove unattended-upgrades (but it might be insufficient)
* disable:
```
systemctl stop apt-daily-upgrade.timer
systemctl disable apt-daily-upgrade.timer
(systemctl disable apt-daily.service) - not clear if necessary
systemctl daemon-reload
```

-----
# Additional information

#### Order types

MARKET: taker order executed immediately at the best price

LIMIT: exists in order book and can be filled at any time

* STOP_LOSS and TAKE_PROFIT have a trigger and hence do not exist in order book, they are inserted in order book only using trigger.
  *after* trigger works, it is inserted in order book either as a market order or as a limit order.
* STOP_LOSS and TAKE_PROFIT will execute a MARKET order when the stopPrice is reached.
* Trigger rules:
  * Price above market price: STOP_LOSS BUY, TAKE_PROFIT SELL
  * Price below market price: STOP_LOSS SELL, TAKE_PROFIT BUY
* We can specify timeInForce (so that the order is automatically killed after time out)

* Is used when price gets worse:
  * STOP_LOSS: quantity, stopPrice (trigger), execution price is market price
  * STOP_LOSS_LIMIT: timeInForce, quantity, price, stopPrice (trigger)

* Probably is used when price gets better:
  * TAKE_PROFIT: quantity, stopPrice (trigger), execution price is market price
  * TAKE_PROFIT_LIMIT: timeInForce, quantity, price, stopPrice

* LIMIT_MAKER are LIMIT orders that will be rejected if they would immediately match and trade as a taker.
  * LIMIT_MAKER: quantity, price

What is OCO:
* One-Cancels-the-Other Order - (OCO)

#### Order API for getting order status

Link: https://python-binance.readthedocs.io/en/latest/account.html#orders

* Get all orders: orders = client.get_all_orders(symbol='BNBBTC', limit=10)
* Get all open orders: orders = client.get_open_orders(symbol='BNBBTC')
* Check order status: order = client.get_order(symbol='BNBBTC', orderId='orderId')

#### Account API for getting funds

Link: https://python-binance.readthedocs.io/en/latest/account.html#account

* Get asset balance: balance = client.get_asset_balance(asset='BTC')

#### Software

* binance official API: https://github.com/binance-exchange/binance-official-api-docs
* python-binance: https://github.com/sammchardy/python-binance
* LiveDataFrame = Python + Pandas + Streaming: https://docs.livedataframe.com/
