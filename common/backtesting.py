import numpy as np
import pandas as pd

"""
Backtesting and trade performance using trade simulation
"""

def simulated_trade_performance(df, buy_signal_column, sell_signal_column, price_column):
    """
    The function simulates trades over the time by buying and selling the asset
    according to the specified buy/sell signals and price. Essentially, it assumes
    the existence of some initial amount, then it moves forward in time by finding
    next buy/sell signal and accordingly buying/selling the asset using the current
    price. At the end, it finds how much it earned by comparing with the initial amount.

    It returns short and long performance as a number of metrics collected during
    one simulation pass.
    """
    is_buy_mode = True

    long_profit = 0
    long_profit_percent = 0
    long_transactions = 0
    long_profitable = 0
    longs = list()  # Where we buy

    short_profit = 0
    short_profit_percent = 0
    short_transactions = 0
    short_profitable = 0
    shorts = list()  # Where we sell

    # The order of columns is important for itertuples
    df = df[[sell_signal_column, buy_signal_column, price_column]]
    for (index, sell_signal, buy_signal, price) in df.itertuples(name=None):
        if not price or pd.isnull(price):
            continue
        if is_buy_mode:
            # Check if minimum price
            if buy_signal:
                previous_price = shorts[-1][2] if len(shorts) > 0 else 0.0
                profit = (previous_price - price) if previous_price > 0 else 0.0
                profit_percent = 100.0 * profit / previous_price if previous_price > 0 else 0.0
                short_profit += profit
                short_profit_percent += profit_percent
                short_transactions += 1
                if profit > 0:
                    short_profitable += 1
                shorts.append((index, previous_price, price, profit, profit_percent))  # Bought
                is_buy_mode = False
        else:
            # Check if maximum price
            if sell_signal:
                previous_price = longs[-1][2] if len(longs) > 0 else 0.0
                profit = (price - previous_price) if previous_price > 0 else 0.0
                profit_percent = 100.0 * profit / previous_price if previous_price > 0 else 0.0
                long_profit += profit
                long_profit_percent += profit_percent
                long_transactions += 1
                if profit > 0:
                    long_profitable += 1
                longs.append((index, previous_price, price, profit, profit_percent))  # Sold
                is_buy_mode = True

    # Performance of buy at low price and sell at high price
    long_performance = {
        "#transactions": long_transactions,
        "profit": round(long_profit, 2),
        "%profit": round(long_profit_percent, 1),

        "#profitable": long_profitable,
        "%profitable": round(100.0 * long_profitable / long_transactions, 1) if long_transactions else 0.0,

        "profit/T": round(long_profit / long_transactions, 2),
        "%profit/T": round(long_profit_percent / long_transactions, 1),

        #"transactions": longs,  # Sell transactions
    }

    # Performance of sell at high price and buy at low price
    short_performance = {
        "#transactions": short_transactions,
        "profit": round(short_profit, 2),
        "%profit": round(short_profit_percent, 1),

        "#profitable": short_profitable,
        "%profitable": round(100.0 * short_profitable / short_transactions, 1) if short_transactions else 0.0,

        "profit/T": round(short_profit / short_transactions, 2),
        "%profit/T": round(short_profit_percent / short_transactions, 1),

        #"transactions": shorts,  # Buy transactions
    }

    profit = long_profit + short_profit
    profit_percent = long_profit_percent + short_profit_percent
    transaction_no = long_transactions + short_transactions
    profitable = (long_profitable + short_profitable) / transaction_no if transaction_no else 0.0
    #minutes_in_month = 1440 * 30.5
    performance = {
        "#transactions": transaction_no,
        "profit": profit,
        "%profit": profit_percent,

        "profitable": profitable,
        "profitable_percent": round(100.0 * profitable / transaction_no, 1) if transaction_no else 0.0,

        "profit/T": round(profit / transaction_no, 2) if transaction_no else 0.0,
        "%profit/T": profit_percent / transaction_no if transaction_no else 0.0,

        #"transactions": transactions,

        #"profit_per_month": profit / (len(df) / minutes_in_month),
        #"transactions_per_month": transaction_no / (len(df) / minutes_in_month),
    }

    return performance, long_performance, short_performance
