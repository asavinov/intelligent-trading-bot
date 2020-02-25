# TODO

#### GENERAL

Do not focus (spend time) on tuning algorithm hyper-parameters - choose something simple and reasonable but in such a way that it can be extended later.
Instead, try and generate more predictions with *independent* algorithms: svm (maybe even linear), rf, ...
Also, add new *independent* data like bitcoin future prices or cross-prices (bcn-eth etc.) even by reduce btc derived features (do we really need so many moving windows)
Conceptually, focus on concept drift and adaptation to nearest trends and behavioral patterns, e.g., using short-middle-long windows.

#### Simple 1 minute trading strategy
Principles
* [buy mode] entering market (buy market order executed immediately) immediately followed by a limit sell order
* [sell mode] regularly checking the execution state of the limit order
  * possibly update its limit price
  * force cell because of (strong) sell signal (essentially means that the limit price is equal or lower than the market price)
  * force sell after time out by converting into a market order

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

#### Start from Linux

Modify start.py by entering data collection command. Alternatively, pass the desired command as an argument.
See the file for additional comments. For example:
* `collect_data` is used to collect depth data by making the corresponding requests.
  * It is possible to specify frequency 1m, 5s etc.
  * It is possible to specifiy depth (high depth will decrease weight of the request)
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
nice -10 perl test.pl - run with niceness 10 (lower priority) (- is hyphen - not negative).
nice --10 perl test.pl - start with high priority (negative niceness)
nice -n -5 perl test.pl - increase priority
nice -n 5 perl test.pl - decrease priority
nice -n 10 apt-get upgrade - start with lower priority (lower values of niceness mean higher priority, so we need higher values)
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
Our system is then sychronized with the service and is driven by the service data updates being them 1m-klines or order executions.
Java WebSocket: client.onCandlestickEvent("ethbtc", CandlestickInterval.ONE_MINUTE, response -> System.out.println(response));
Python WebSocket: conn_key = bm.start_kline_socket('BNBBTC', process_message, interval=KLINE_INTERVAL_30MINUTE)
Our processing logic should remain the same but now it is triggered not by the local scheduler but rather by external events.
