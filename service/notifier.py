import os
import sys
from datetime import timedelta, datetime

import asyncio

import pandas as pd
import requests

from service.App import *
from common.utils import *

import logging
log = logging.getLogger('notifier')

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

transaction_file = Path("transactions.txt")


async def send_score_notification():
    symbol = App.config["symbol"]

    signal = App.signal

    close_price = signal.get('close_price')
    close_time = signal.get('close_time')
    trade_scores = signal.get('trade_score')
    trade_score_primary = trade_scores[0]
    trade_score_secondary = trade_scores[1] if len(trade_scores) > 1 else None

    model = App.config["score_notification_model"]
    # Determine the band for the current score
    if trade_score_primary > 0:
        bands = model.get("positive_bands", [])
        band_no, band = next(((i, x) for i, x in enumerate(bands) if trade_score_primary <= x.get("edge")), (len(bands), None))
    else:
        bands = model.get("negative_bands", [])
        band_no, band = next(((i, x) for i, x in enumerate(bands) if trade_score_primary >= x.get("edge")), (len(bands), None))

    if not band:
        log.error(f"Notification band for the score {trade_score_primary} not found. Check the list of bands in config. Ignore")
        return

    #
    # To message or not to message depending on score value and time
    #

    # Determine if the band was changed since the last time
    prev_band_no = model.get("prev_band_no")
    band_up = prev_band_no is not None and prev_band_no < band_no
    band_dn = prev_band_no is not None and prev_band_no > band_no
    model["prev_band_no"] = band_no  # Store for the next time in the config section

    if band.get("frequency"):
        new_to_time_interval = close_time.minute % band.get("frequency") == 0
    else:
        new_to_time_interval = False

    # Send only if one of these conditions is true  or entered new time interval (current time)
    notification_is_needed = (
        (model.get("notify_band_up") and band_up) or  # entered a higher band (absolute score increased)
        (model.get("notify_band_dn") and band_dn) or  # returned to a lower band (absolute score decreased)
        new_to_time_interval  # new time interval is started like 10 minutes
    )

    if not notification_is_needed:
        return  # Nothing important happened: within the same band and same time interval

    #
    # Build a message with parameters from the current band
    #

    # Crypto Currency Symbols: https://github.com/yonilevy/crypto-currency-symbols
    if symbol == "BTCUSDT":
        symbol_char = "₿"
    elif symbol == "ETHUSDT":
        symbol_char = "Ξ"
    else:
        symbol_char = symbol

    if band_up:
        band_change_char = "↑"
    elif band_dn:
        band_change_char = "↓"
    else:
        band_change_char = ""

    primary_score_str = f"{trade_score_primary:+.2f} {band_change_char} "
    secondary_score_str = f"{trade_score_secondary:+.2f}" if trade_score_secondary is not None else ''

    message = f"{band.get('sign', '')} {symbol_char} {int(close_price):,} Score: {primary_score_str} {secondary_score_str} {band.get('text', '')}"
    if band.get("bold"):
        message = "*" + message + "*"

    message = message.replace("+", "%2B")  # For Telegram to display plus sign

    #
    # Send notification
    #
    bot_token = App.config["telegram_bot_token"]
    chat_id = App.config["telegram_chat_id"]

    try:
        url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message
        response = requests.get(url)
        response_json = response.json()
        if not response_json.get('ok'):
            log.error(f"Error sending notification.")
    except Exception as e:
        log.error(f"Error sending notification: {e}")


async def send_transaction_message(transaction):

    profit, profit_percent, profit_descr, profit_percent_descr = await generate_transaction_stats()

    if transaction.get("status") == "SELL":
        message = "⚡💰 *SOLD: "
    elif transaction.get("status") == "BUY":
        message = "⚡💰 *BOUGHT: "
    else:
        log.error(f"ERROR: Should not happen")

    message += f" Profit: {profit_percent:.2f}% {profit:.2f}₮*"

    bot_token = App.config["telegram_bot_token"]
    chat_id = App.config["telegram_chat_id"]
    try:
        url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message
        response = requests.get(url)
        response_json = response.json()
        if not response_json.get('ok'):
            log.error(f"Error sending notification.")
    except Exception as e:
        log.error(f"Error sending notification: {e}")

    #
    # Send stats about previous transactions (including this one)
    #
    if transaction.get("status") == "SELL":
        message = "↗ *LONG transactions stats (4 weeks)*\n"
    elif transaction.get("status") == "BUY":
        message = "↘ *SHORT transactions stats (4 weeks)*\n"
    else:
        log.error(f"ERROR: Should not happen")

    message += f"🔸sum={profit_percent_descr['count'] * profit_percent_descr['mean']:.2f}% 🔸count={int(profit_percent_descr['count'])}\n"
    message += f"🔸mean={profit_percent_descr['mean']:.2f}% 🔸std={profit_percent_descr['std']:.2f}%\n"
    message += f"🔸min={profit_percent_descr['min']:.2f}% 🔸median={profit_percent_descr['50%']:.2f}% 🔸max={profit_percent_descr['max']:.2f}%\n"

    try:
        url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=markdown&text=' + message
        response = requests.get(url)
        response_json = response.json()
        if not response_json.get('ok'):
            log.error(f"Error sending notification.")
    except Exception as e:
        log.error(f"Error sending notification: {e}")


async def simulate_trade():
    """
    Very simple trade strategy where we only buy and sell using the whole available amount
    """
    symbol = App.config["symbol"]

    status = App.status
    signal = App.signal
    signal_side = signal.get("side")
    close_price = signal.get('close_price')
    close_time = signal.get('close_time')

    # Previous transaction: BUY (we are currently selling) or SELL (we are currently buying)
    t_status = App.transaction.get("status")
    t_price = App.transaction.get("price")
    if signal_side == "BUY" and (not t_status or t_status == "SELL"):
        profit = t_price - close_price if t_price else 0.0
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="BUY")
    elif signal_side == "SELL" and (not t_status or t_status == "BUY"):
        profit = close_price - t_price if t_price else 0.0
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="SELL")
    else:
        return None

    # Save this transaction
    App.transaction = t_dict
    with open(transaction_file, 'a+') as f:
        f.write(",".join([f"{v:.2f}" if isinstance(v, float) else str(v) for v in t_dict.values()]) + "\n")

    return t_dict


async def generate_transaction_stats():
    """Here we assume that the latest transaction is saved in the file and this function computes various properties."""

    df = pd.read_csv(transaction_file, parse_dates=[0], header=None, names=["timestamp", "close", "profit", "status"], date_format="ISO8601")

    mask = (df['timestamp'] >= (datetime.now() - timedelta(weeks=4)))
    df = df[max(mask.idxmax()-1, 0):]  # We add one previous row to use the previous close

    df["prev_close"] = df["close"].shift()
    df["profit_percent"] = df.apply(lambda x: 100.0*x["profit"]/x["prev_close"], axis=1)

    df = df.iloc[1:]  # Remove the first row which was added to compute relative profit

    long_df = df[df["status"] == "SELL"]
    short_df = df[df["status"] == "BUY"]

    #
    # Determine properties of the latest transaction
    #

    # Sample output:
    # BTC, LONG or SHORT
    # sell price 24,000 (now), buy price (datetime) 23,000
    # profit abs: 1,000.00,
    # profit rel: 3.21%

    last_transaction = df.iloc[-1]
    transaction_dt = last_transaction["timestamp"]
    transaction_type = last_transaction["status"]
    profit = last_transaction["profit"]
    profit_percent = last_transaction["profit_percent"]

    #
    # Properties of last period of trade
    #

    if transaction_type == "SELL":
        df2 = long_df
    elif transaction_type == "BUY":
        df2 = short_df

    # Sample output for abs profit
    # sum 1,200.00, mean 400.00, median 450.00, std 250.00, min -300.0, max 1200.00

    profit_sum = df2["profit"].sum()
    profit_descr = df2["profit"].describe()  # count, mean, std, min, 50% max

    profit_percent_sum = df2["profit_percent"].sum()
    profit_percent_descr = df2["profit_percent"].describe()  # count, mean, std, min, 50% max

    return profit, profit_percent, profit_descr, profit_percent_descr


async def send_diagram(freq, nrows):
    """
    Produce a line chart based on latest data and send it to the channel.

    :param freq: Aggregation interval 'H' - hour.
    :param nrows: Time range (x axis) of the diagram, for example, 1 week 168 hours, 2 weeks 336 hours
    """
    model = App.config["trade_signal_model"]

    buy_signal_threshold = model.get("parameters", {}).get("buy_signal_threshold", 0)
    sell_signal_threshold = model.get("parameters", {}).get("sell_signal_threshold", 0)

    #
    # Prepare data to be visualized
    #
    # Get main df with high, low, close for the symbol.
    df_ohlc = App.feature_df[['open', 'high', 'low', 'close']]
    df_ohlc = resample_ohlc_data(df_ohlc.reset_index(), freq, nrows, buy_signal_column=None, sell_signal_column=None)

    # Get transaction data.
    df_t = load_all_transactions()  # timestamp,price,profit,status
    df_t['buy_long'] = df_t['status'].apply(lambda x: True if isinstance(x, str) and x == 'BUY' else False)
    df_t['sell_long'] = df_t['status'].apply(lambda x: True if isinstance(x, str) and x == 'SELL' else False)
    df_t = df_t[df_t.timestamp >= df_ohlc.timestamp.min()]  # select only transactions for the last time
    transactions_exist = len(df_t) > 0

    if transactions_exist:
        df_t = resample_transaction_data(df_t, freq, 0, 'buy_long', 'sell_long')
    else:
        df_t = None

    # Merge because we need signals along with close price in one df
    if transactions_exist:
        df = df_ohlc.merge(df_t, how='left', left_on='timestamp', right_on='timestamp')
    else:
        df = df_ohlc

    # Load score
    score_exists = False

    symbol = App.config["symbol"]
    title = f"$\\bf{{{symbol}}}$"

    description = App.config.get("description", "")
    if description:
        title += ": " + description

    fig = generate_chart(
        df, title,
        buy_signal_column="buy_long" if transactions_exist else None,
        sell_signal_column="sell_long" if transactions_exist else None,
        score_column="score" if score_exists else None,
        thresholds=[buy_signal_threshold, sell_signal_threshold]
    )

    import io
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')  # Convert and save in buffer
        im_bytes = buf.getvalue()  # Get complete content (while read returns from current position)
    img_data = im_bytes

    #
    # Send image
    #
    bot_token = App.config["telegram_bot_token"]
    chat_id = App.config["telegram_chat_id"]

    files = {'photo': img_data}
    payload = {
        'chat_id': chat_id,
        'caption': f"",  # Currently no text
        'parse_mode': 'markdown'
    }

    try:
        url = 'https://api.telegram.org/bot' + bot_token + '/sendPhoto'
        req = requests.post(url=url, data=payload, files=files)
        response = req.json()
    except Exception as e:
        log.error(f"Error sending notification: {e}")


def resample_ohlc_data(df, freq, nrows, buy_signal_column, sell_signal_column):
    """
    Resample ohlc data to lower frequency. Assumption: time in 'timestamp' column.
    """
    # Aggregation functions
    ohlc = {
        'timestamp': 'first',  # It will be in index
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }

    # These score columns are optional
    if buy_signal_column:
        # Buy signal if at least one buy signal was during this time interval
        ohlc[buy_signal_column]: lambda x: True if not all(x == False) else False
        # Alternatively, 0 if no buy signals, 1 if only 1 buy signal, 2 or -1 if more than 1 any signals (mixed interval)
    if sell_signal_column:
        # Sell signal if at least one sell signal was during this time interval
        ohlc[sell_signal_column]: lambda x: True if not all(x == False) else False

    df_out = df.resample(freq, on='timestamp').apply(ohlc)
    del df_out['timestamp']
    df_out.reset_index(inplace=True)

    if nrows:
        df_out = df_out.tail(nrows)

    return df_out


def resample_transaction_data(df, freq, nrows, buy_signal_column, sell_signal_column):
    """
    Given a list of transactions with arbitrary timestamps,
    return a regular time series with True or False for the rows with transactions
    Assumption: time in 'timestamp' column

    PROBLEM: only one transaction per interval (1 hour) is possible so if we buy and then sell within one hour then we cannot represent this
      Solution 1: remove
      Solution 2: introduce a special symbol (like dot instead of arrows) which denotes one or more transactions - essentially error or inability to visualize
      1 week 7*1440=10080 points, 5 min - 2016 points, 10 mins - 1008 points
    """
    # Aggregation functions
    transactions = {
        'timestamp': 'first',  # It will be in index
        buy_signal_column: lambda x: True if not all(x == False) else False,
        sell_signal_column: lambda x: True if not all(x == False) else False,
    }

    df_out = df.resample(freq, on='timestamp').apply(transactions)
    del df_out['timestamp']
    df_out.reset_index(inplace=True)

    if nrows:
        df_out = df_out.tail(nrows)

    return df_out


def generate_chart(df, title, buy_signal_column, sell_signal_column, score_column, thresholds: list):
    """
    All columns in one input df with desired length and desired freq
    Visualize columns 1 (pre-defined): high, low, close
    Visualize columns 1 (via parameters): buy_signal_column, sell_signal_column
    Visualize columns 2: score_column (optional) - in [-1, +1]
    Visualize columns 2: Threshold lines (as many as there are values in the list)
    """
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns

    # List of colors: https://matplotlib.org/stable/gallery/color/named_colors.html
    sns.set_style('white')  # darkgrid, whitegrid, dark, white, ticks
    #sns.color_palette("rocket")
    #sns.set(rc={'axes.facecolor': 'gold', 'figure.facecolor': 'white'})
    #sns.set(rc={'figure.facecolor': 'gold'})

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # plt.tight_layout()

    # === High, Low, Close

    # Fill area between high and low
    plt.fill_between(df.timestamp, df.low, df.high, step="mid", lw=0.0, facecolor='skyblue', alpha=.4)  # edgecolor='red',

    # Draw close price
    sns.lineplot(data=df, x="timestamp", y="close", drawstyle='steps-mid', lw=.5, color='darkblue', ax=ax1)

    # Buy/sell markters (list of timestamps)
    # buy_df = df[df.buy_transaction]
    # sell_df = df[df.sell_transaction]

    # === Transactions

    triangle_adj = 15
    df["close_buy_adj"] = df["close"] - triangle_adj
    df["close_sell_adj"] = df["close"] + triangle_adj

    # markersize=6, markerfacecolor='blue'
    sns.lineplot(data=df[df[buy_signal_column] == True], x="timestamp", y="close_buy_adj", lw=0, markerfacecolor="green", markersize=10, marker="^", alpha=0.6, ax=ax1)
    sns.lineplot(data=df[df[sell_signal_column] == True], x="timestamp", y="close_sell_adj", lw=0, markerfacecolor="red", markersize=10, marker="v", alpha=0.6, ax=ax1)

    # g2.set(yticklabels=[])
    # g2.set(title='Penguins: Body Mass by Species for Gender')
    ax1.set(xlabel=None)  # remove the x-axis label
    # g2.set(ylabel=None)  # remove the y-axis label
    ax1.set_ylabel('Close price', color='darkblue')
    # g2.tick_params(left=False)  # remove the ticks
    min = df['low'].min()
    max = df['high'].max()
    ax1.set(ylim=(min - (max - min) * 0.05, max + (max - min) * 0.005))

    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d"))  # "%H:%M:%S" "%d %b"
    ax1.tick_params(axis="x", rotation=0)
    #ax1.xaxis.grid(True)

    # === Score

    if score_column and score_column in df.columns:
        ax2 = ax1.twinx()
        # ax2.plot(x, y1, 'o-', color="red" )
        sns.lineplot(data=df, x="timestamp", y=score_column, drawstyle='steps-mid', lw=.2, color="red", ax=ax2)  # marker="v" "^" , markersize=12
        ax2.set_ylabel('Score', color='r')
        ax2.set_ylabel('Score', color='b')
        ax2.set(ylim=(-0.5, +3.0))

        ax2.axhline(0.0, lw=.1, color="black")

        for threshold in thresholds:
            ax2.axhline(threshold, lw=.1, color="red")
            ax2.axhline(threshold, lw=.1, color="red")

    # fig.suptitle("My figtitle", fontsize=14)  # Positioned higher
    # plt.title('Weekly: $\\bf{S&P 500}$', fontsize=16)  # , weight='bold' or MathText
    plt.title(title, fontsize=14)
    # ax1.set_title('My Title')

    # plt.show()

    return fig


if __name__ == '__main__':
    bands = [
        {"edge": 0.01, "frequency": None},
        {"edge": 0.02, "frequency": 10, "up_sign": "〉", "dn_sign": "<"},
        {"edge": 0.03, "frequency": 5, "up_sign": "🟢〉〉", "emphasize": True, "up_text": "buy zone"}
    ]

    #score = 0.01
    bi, band = next(((i, x) for i, x in enumerate(bands) if score <= x.get("edge")), (len(bands), None))
    pass
