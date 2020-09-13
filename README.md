
# Data processing pipeline

Note: see also start.py

* Download historic data. Currently we use 2 separate sources stored in 2 separate files:
  * klines (spot market), its history is quite long
  * futures (also klines but from futures market), its history is relative short
  * Script: scripts.download_data.py

* Merge historic data into one dataset. We analyse data using one common raster and different column names. Also, we fix problems with gaps by producing a uniform time raster. Note that columns from different source files will have different history length so short file will have Nones in the long file.
  * Script: scripts.merge_data.py

* Generate features. Here we compute derived features (also using window-functions) and produce a final feature matrix. We also compute target features (labels). Note that we compute many possible features and labels but not all of them have to be used. In parameters, we define past history length (windows) and future horizon for labels. Currently, we generate 3 kinds of features independently: klines features (source 1), future features (source 2), and label features (our possible predictions targets).
  * Script: scripts.generate_features.py

* Generate rolling predictions. Here we train a model using previous data less frequently, say, once per day or week, but use much more previous data than in typical window-based features. We apply then one constant model to predict values for the future time until it is re-trained again using newest data. (If the re-train frequency is equal to sample rate, that is, we do it for each new row, then we get normal window-based derived feature with large window sizes.) Each feature is based on some algorithm with some hyper-parameters and some history length. This procedure does not choose best hyper-parameters - for that purpose we need some other procedure, which will optimize the predicted values with the real ones. Normally, the target values of these features are directly related to what we really want to predict, that is, to some label. Output of this procedure is same file (feature matrix) with additional predicted features (scores). This file however will be much shorter because we need some quite long history for some features (say, 1 year). Note that for applying rolling predictions, we have to know hyper-parameters which can be found by a simpler procedure.
  * Script: scripts.generate_rolling_predictions.py

* Train signal models. The input is a feature matrix with all scores (predicted features). Our goal is to define a feature the output of which will be directly used for buy/sell decisions. We need search for the best hyper-parameters starting from simple score threshold and ending with some data mining algorithm.
  * Script: scripts.train_signal_models.py

* Grid search.
  * Script: classification_nn.py
  * Script: classification_gb.py

# TODO

#### GENERAL

Do not focus (spend time) on tuning algorithm hyper-parameters - choose something simple and reasonable but in such a way that it can be extended later.
Instead, try and generate more predictions with *independent* algorithms: svm (maybe even linear), rf, ...
Also, add new *independent* data like bitcoin future prices or cross-prices (bcn-eth etc.) even by reduce btc derived features (do we really need so many moving windows)
Conceptually, focus on concept drift and adaptation to nearest trends and behavioral patterns, e.g., using short-middle-long windows.

#### Simple 1 minute trading strategy
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

Principles of our strategy:

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

#### Separate the logic of data state management and update from the logic of (latest) data processing

This means that the processor should be unaware how it is triggered - when it starts it loads latest batch and (if it is really latest data, that is, not too old),
and follows the logic of processing: generate features, generate signals, send orders etc.
On the other hand, the data updater is a function which requests data, sends it to the data manager and notifies the processor (maybe data manager notifies the processor).
Scheduler triggers/notifies the updater, and updater or data manager triggers/notifiers the processor.

#### Generate several prediction files with different past horizon for training: unlimited, 12 months, 6 months

Store them in some common folder.
Maybe update source files before from the service.
Use these files for comparing same signal strategies, e.g., how they influence stability/volatility.

#### Different signal generations

* Simple algorithm to evaluate precision of signals
We do not do trades but rather compute how many buy signals are false.
The algorithm should be fast and we should be able to quickly find signal hyper-parameters with best precision.

* Flexible exit (sell) strategy using price adjustment or early exit depending on the situation.
Lower the sell price - either fixed adjustments or (better) depending on the current buy/sell signal values.
Do this after time out or earlier. For example, in a couple of minutes the situation can gets worse so stop loss.
Check if we can leverage OCO feature with automatic sell order (what we use now) plus a stop-loss order which will be executed at the same time if the situation (price) gets worse.
Do grid search over various sell adjustment parameters (exit or stop loss parameters) with fixed buy (enter) signal parameters.

#### Predicted features
* Add RF algorithm to predictions.
* Add Linear SVM or logistic regression to predictions.
* Think about adding more sophisticated algorithms like KernelPCA or SVM.
Maybe for short histories in order to take into account drift or local dependencies.

#### Feature generation:
* Add trade-related features from https://github.com/bukosabino/ta
  * compute and use weighted average price
* Add long-term features (moving averages), for example, for 1 month or more.
* Add ARIMA-like features

#### General system functions:
* Ping the server: https://python-binance.readthedocs.io/en/latest/general.html#id1
* Get system status: https://python-binance.readthedocs.io/en/latest/general.html#id3
* Check server time and compare with local time: https://python-binance.readthedocs.io/en/latest/general.html#id2

## How to

#### Structure of functions

General sequence:
* Update klines data set from the binance server: "python start.py download_data"
* Compute features and labels (label lists are hard-coded but not all of them have to be used): "python start.py generate_features.py
* Generate rolling predictions (specify hyper-model parameters as well as features to use and labels to predict): "python start.py generate_rolling_predictions.py"
* Train signal (trade) models: "python start.py train_signal_models.py"

Regular updates of the trade server:
* Load (update) source data (currently klines but in future other data sources could be used like futures)
* Generate features and labels for the new data set (it is needed for model training)
* Re-train label prediction models using new data and fixed (previously optimized) hyper-parametes
  * Upload these models to the trade server

Optimizing hyper-parameters of the prediction models:
* Update data set
* Compute features and labels
* TODO: In grid search for possible gb hyper-parameters execute:
  * Compute rolling predictions with the current grid parameters
  * Store average accuracy in a file
  * (In future, instead of computing average accuracy for all rolling segments, we can give higher weight to last rolling segments.)
* Choose hyper-parameters with best mean accuracy and use them for further model training

Optimizing hyper-parameters of the trade models:
* Update data set
* Compute features and labels
* Compute rolling predictions using the chosen (best) hyper-parameters for prediction models
* In grid search for possible threshold parameters execute:
  * Compute overall performance by simulating trades for the whole time interval
  * Store average performance in a file
  * (In future, instead of computing average performance for all rolling segments, we can give higher weight to last rolling segments. As segments, we can use any interval like month.)
* Choose hyper-parameters of the signal model with best mean performance and use them for further signal generation.

#### Load new historic (klines) data

Script: scripts/download_data.py

Get symbol klines:
* Edit main in binance-data.py by setting necessary symbol
* Run script binance-data.py which will directly call get_klines_all()

Get klines for futures:
* Use the same function get_klines_all() but uncomment the section at the beginning.

#### Generate feature matrix

The goal here is to load source (kline) data, generate derived features and labels, and store the result in output file.
The output is supposed to be used for other procedures like training prediction models.

Execute from project root:
```
$ python start.py generate_features
```

* Ensure that latest source data has been downloaded from binance server
* Max past window and max future horizon are currently not used (None will be stored)
* Future horizon for labels is hard-coded (currently 60). Change if necessary
* If necessary, uncomment line with storing to parquet (install the packages)
* Output file will store features and labels as they are implemented in the trade module. Copy the header line to get the list.
* Same number of lines in output as in input file
* Approximate time: ~10 minutes (on YOGA)

#### Train predict models

See start.py

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

## Additional information

#### Order types

MARKET: taker order executed immediately at the best price

LIMIT: exists in order book and can be filled at any time

# STOP_LOSS* and TAKE_PROFIT* have a trigger and hence do not exist in order book, they are inserted in order book only using trigger.
  *after* trigger works, it is inserted in order book either as a market order or as a limit order.
# STOP_LOSS and TAKE_PROFIT will execute a MARKET order when the stopPrice is reached.
# Trigger rules:
- Price above market price: STOP_LOSS BUY, TAKE_PROFIT SELL
- Price below market price: STOP_LOSS SELL, TAKE_PROFIT BUY
# We can specify timeInForce (so that the order is automatically killed after time out)

# Is used when price gets worse:
STOP_LOSS: quantity, stopPrice (trigger), execution price is market price
STOP_LOSS_LIMIT: timeInForce, quantity, price, stopPrice (trigger)

# Probably is used when price gets better:
TAKE_PROFIT: quantity, stopPrice (trigger), execution price is market price
TAKE_PROFIT_LIMIT: timeInForce, quantity, price, stopPrice

# LIMIT_MAKER are LIMIT orders that will be rejected if they would immediately match and trade as a taker.
LIMIT_MAKER: quantity, price

What is OCO:
- One-Cancels-the-Other Order - (OCO)

#### Order API for getting order status

Link: https://python-binance.readthedocs.io/en/latest/account.html#orders
* Get all orders: orders = client.get_all_orders(symbol='BNBBTC', limit=10)
* Get all open orders: orders = client.get_open_orders(symbol='BNBBTC')
* Check order status: order = client.get_order(symbol='BNBBTC', orderId='orderId')

#### Account API for getting funds

Link: https://python-binance.readthedocs.io/en/latest/account.html#account
* Get asset balance: balance = client.get_asset_balance(asset='BTC')

#### Software

* binance offician API: https://github.com/binance-exchange/binance-official-api-docs
* python-binance: https://github.com/sammchardy/python-binance
* LiveDataFrame = Python + Pandas + Streaming: https://docs.livedataframe.com/

## Problems

#### Problem 1: Modify/adjust limit order price

Currently main problem is modifying/adjusting existing order price - either to force-sell or to adjust-sell.
Approach 1: Kill existing order, modify its parameters and submit again.
  1. In one cycle. Kill synchronously, check funds request (if coins are still available for sale), submit new sell order.
  2. In two cycles. Kill, continue cycle, check orders as usual on the next cycle: if exists then kill-continue, if does not exist (killed) then create new sell order as usual.
Approach 2: Submit order (price) modification request.
Solution. We assume that price cannot be modified. Therefore, we must kill the current order and create a new (modifed) order.
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
