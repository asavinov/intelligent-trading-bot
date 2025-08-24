import os
import sys
import io
from datetime import timedelta, datetime

import asyncio

import pandas as pd
import pandas.api.types as ptypes

import requests

from service.App import *
from common.utils import *
from common.model_store import *
from outputs.notifier_trades import load_all_transactions

import logging
log = logging.getLogger('notifier')

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


async def send_diagram(df, model: dict, config: dict, model_store: ModelStore):
    """
    Produce a line chart based on latest data and send it to the channel.
    """
    freq = config.get("freq")
    time_column = config["time_column"]

    # Ensure that timestamp is in index. It is needed for visualization
    if not ptypes.is_datetime64_dtype(df.index):  # Alternatively df.index.inferred_type == "datetime64"
        if time_column in df.columns:
            df = df.set_index('timestamp', inplace=False)
        else:
            raise ValueError(f"Neither index nor time columns '{time_column}' are of datetime type")

    notification_freq = model.get("notification_freq")
    if notification_freq:
        # Continue only if system interval start is equal to the (longer) diagram interval start
        if pandas_get_interval(notification_freq)[0] != pandas_get_interval(freq)[0]:
            return

    score_column_names = model.get("score_column_names")
    if isinstance(score_column_names, list) and len(score_column_names) > 0:
        score_column_names = score_column_names[0]

    score_thresholds = model.get("score_thresholds")

    resampling_freq = model.get("resampling_freq")  # Resampling (aggregation) frequency
    nrows = model.get("nrows")  # Time range (x axis) of the diagram, for example, 1 week 168 hours, 2 weeks 336 hours

    row = df.iloc[-1]  # Last row stores the latest values we need

    #
    # Prepare data to be visualized
    #
    # Get main df with high, low, close for the symbol.
    vis_columns = ['open', 'high', 'low', 'close']
    if score_column_names:
        vis_columns.append(score_column_names)
    df_ohlc = df[vis_columns]
    df_ohlc = resample_ohlc_data(df_ohlc.reset_index(), resampling_freq, nrows, score_column=score_column_names, buy_signal_column=None, sell_signal_column=None)

    # Get transaction data
    df_t = load_all_transactions()  # timestamp,price,profit,status
    df_t['buy_long'] = df_t['status'].apply(lambda x: True if isinstance(x, str) and x == 'BUY' else False)
    df_t['sell_long'] = df_t['status'].apply(lambda x: True if isinstance(x, str) and x == 'SELL' else False)
    df_t = df_t[df_t.timestamp >= df_ohlc.timestamp.min()]  # select only transactions for the last time
    transactions_exist = len(df_t) > 0

    if transactions_exist:
        df_t = resample_transaction_data(df_t, resampling_freq, 0, 'buy_long', 'sell_long')
    else:
        df_t = None

    # Merge because we need signals along with close price in one df
    if transactions_exist:
        df = df_ohlc.merge(df_t, how='left', left_on='timestamp', right_on='timestamp')
    else:
        df = df_ohlc

    symbol = config["symbol"]
    title = f"$\\bf{{{symbol}}}$"

    description = config.get("description", "")
    if description:
        title += ": " + description

    fig = generate_chart(
        df, title,
        buy_signal_column="buy_long" if transactions_exist else None,
        sell_signal_column="sell_long" if transactions_exist else None,
        score_column=score_column_names if score_column_names else None,
        thresholds=score_thresholds
    )

    with io.BytesIO() as buf:
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)  # Convert and save in buffer
        im_bytes = buf.getvalue()  # Get complete content (while read returns from current position)
    img_data = im_bytes

    #
    # Send image
    #
    bot_token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]

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


def resample_ohlc_data(df, freq, nrows, score_column, buy_signal_column, sell_signal_column):
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

    # Optional columns
    if score_column:
        ohlc[score_column] = lambda x: max(x) if len(x) > 0 and all(x > 0.0) else min(x) if len(x) > 0 and all(x < 0.0) else np.mean(x)

    if buy_signal_column:
        # Buy signal if at least one buy signal was during this time interval
        ohlc[buy_signal_column] = lambda x: True if not all(x == False) else False
        # Alternatively, 0 if no buy signals, 1 if only 1 buy signal, 2 or -1 if more than 1 any signals (mixed interval)
    if sell_signal_column:
        # Sell signal if at least one sell signal was during this time interval
        ohlc[sell_signal_column] = lambda x: True if not all(x == False) else False

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
    sns.set_style('white', {'axes.grid': False, 'axes.spines.top': True, 'axes.edgecolor': 'lightgrey'})  # white, darkgrid, whitegrid, dark, ticks
    # sns.set_style('white', {'axes.facecolor':'yellow', 'grid.color': '.8', 'font.family':'Times New Roman'})
    sns.set_context("notebook")  # notebook (default), talk, paper, poster

    # sns.despine(left=True, offset=10, trim=True)
    # sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    # sns.despine(bottom = True, left = False)

    # sns.color_palette("rocket")
    # sns.set(rc={'axes.facecolor': 'gold', 'figure.facecolor': 'white'})
    # sns.set(rc={'figure.facecolor': 'gold'})

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # plt.tight_layout()

    # plt.grid(axis="x")

    #
    # Price: High, Low, Close
    #

    # Fill area between high and low
    plt.fill_between(df.timestamp, df.low, df.high, step="mid", lw=0.0, facecolor='skyblue', alpha=.4)  # edgecolor='red',

    # Close price
    sns.lineplot(data=df, x="timestamp", y="close", drawstyle='steps-mid', lw=.5, color='blue', ax=ax1)
    # sns.pointplot(data=df, x="timestamp", y="close", color='darkblue', ax=ax1)

    #
    # Transactions (optional): buy or sell triangles over price curve
    #

    triangle_adj = 15
    df["close_buy_adj"] = df["close"] - triangle_adj
    df["close_sell_adj"] = df["close"] + triangle_adj

    if buy_signal_column:
        sns.lineplot(data=df[df[buy_signal_column] == True], x="timestamp", y="close_buy_adj", lw=0, markerfacecolor="green", markersize=10, marker="^", alpha=0.6, ax=ax1)
    if sell_signal_column:
        sns.lineplot(data=df[df[sell_signal_column] == True], x="timestamp", y="close_sell_adj", lw=0, markerfacecolor="red", markersize=10, marker="v", alpha=0.6, ax=ax1)

    # g2.set(yticklabels=[])
    # g2.set(title='Penguins: Body Mass by Species for Gender')
    ax1.set(xlabel=None)  # remove the x-axis label
    # g2.set(ylabel=None)  # remove the y-axis label
    ax1.set_ylabel('Close price', color='blue', fontsize=16)
    # g2.tick_params(left=False)  # remove the ticks
    ymin = df['low'].min()
    ymax = df['high'].max()
    ax1.set(ylim=(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.005))

    # ax1.set_frame_on(False)

    # ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # "%H:%M:%S" "%d %b"
    ax1.tick_params(axis="x", rotation=90)
    ax1.xaxis.grid(True)

    #
    # Indicator line
    #

    if score_column and score_column in df.columns:
        ax2 = ax1.twinx()

        ymax = max(df[score_column].abs().max(), max(thresholds) if thresholds else 0.0)
        ax2.set(ylim=(-ymax * 1.2, +ymax * 1.2))

        # ax2.set_frame_on(False)
        ax2.xaxis.grid(True)

        # ax2.axhline(0.0, lw=.1, color="black")

        for threshold in thresholds:
            ax2.axhline(threshold, lw=3.0, color="lightgray")

        # ax2.plot(x, y1, 'o-', color="red" )
        sns.lineplot(data=df, x="timestamp", y=score_column, drawstyle='steps-mid', lw=1.0, color="red", ax=ax2)  # marker="v" "^" , markersize=12
        ax2.set_ylabel('Intelligent Indicator', color='r', fontsize=16)
        # ax2.set_ylabel('Score', color='b')

    # fig.suptitle("My figtitle", fontsize=14)  # Positioned higher
    # plt.title('Weekly: $\\bf{S&P 500}$', fontsize=16)  # , weight='bold' or MathText
    plt.title(title, fontsize=14)
    # ax1.set_title('My Title')

    # plt.show()

    return fig
