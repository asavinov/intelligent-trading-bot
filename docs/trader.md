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

## Additional information

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
