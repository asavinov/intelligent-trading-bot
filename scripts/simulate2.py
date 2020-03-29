from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

import lightgbm as lgbm

from trade.utils import *

"""
OBSOLETE:

GOAL:
- Simulate by choose best buy moment and then virtually create sell order with some sell price.
  So instead of generating sell signals, we need to generate sell price.
  Sell price can be fixed like 1% higher than the current close price. Or it can be proportional to prediction.
  We detect if the sell is fulfilled or not by checking future max price without having the moment of execution.
  Alternatively, we can scan the high price forward and find moment when the order is executed.
  We need to have a time out when we force sell the order: either after time out, or using sell signals.
  So sell signals are really needed. Indeed, why to wait for auto-sell, if prices are known to fell.
"""

def report(price):
    # Final amount quote asset and total performance in percent to initial amount
    print(f"Signal counts. Buy: {buy_signal_count}, Sell: {sell_signal_count}, Buysell: {buysell_signal_count}")

    print(f"Trade counts. Buy: {buy_count}, Sell: {sell_count}")

    total_performance = 100.0 * (total_assets(price) - initial_amount) / initial_amount
    print(f"Total performance: {total_performance:.2f}")

def total_assets(price):
    return quote_amount + (base_amount * price)

#
# Parameters
#

data_path_name = r"C:\DATA2\BITCOIN"
data_file_name = r"BTCUSDT-1m-data.csv"
historic_df = None

temp_data_path_name = r"_TEMP_FEATURES"
temp_path = Path(temp_data_path_name)
temp_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists
temp_data_file_name = r"_BTCUSDT-1m-data-features_0-predictions.csv"
out_data_file_name = r""

simulation_start = 0
simulation_end = 201_500

initial_amount = 1_000.0
base_amount = 0.0
quote_amount = initial_amount

buy_signal_count = 0
sell_signal_count = 0
buy_count = 0
sell_count = 0
buysell_signal_count = 0

#
# Load data
#
print(f"Loading data...")
path = Path(temp_data_path_name).joinpath(temp_data_file_name)
predict_df = pd.read_csv(path)

#
# Main loop over trade sessions
#
print(f"Starting simulation loop...")
for i in range(simulation_start, simulation_end):
    row = predict_df.iloc[i]
    price = row["close"]

    signal = ""
    diff = abs(row["high_60_max"]) - abs(row["low_60_min"])

    if row["high_60_max"] > 0.25 and row["high_60_15"] > 0.15:
        signal = "buy"
        buy_signal_count += 1

    if row["high_60_max"] < 0.15 or row["low_60_15"] < 0.025:
        signal = "sell"
        sell_signal_count += 1

    #if diff < -0.0 and row["low_60_15"] > 0.05 and row["high_60_05"] > 0.01: # Sell when price is going to drop
    #    signal = "sell"
    #    sell_signal_count += 1

    # Execute signal by doing trade
    if signal == "buy":
        if quote_amount > 0.0:  # Check available quote asset like USDT (we will spend it)
            # Increase base using price, decrease quote to 0
            base_amount += quote_amount / price
            quote_amount = 0.0
            buy_count += 1
    elif signal == "sell":
        if base_amount > 0.0:  # Check available base asset like BTC (we will spend it)
            # Increase quote using price, decrease base to 0
            quote_amount += base_amount * price
            base_amount = 0.0
            sell_count += 1

#
# Close. Sell rest base asset if available for the last price
#
if base_amount > 0.0:  # Check base asset like BTC (we will spend it)
    # Increase quote using price, decrease base to 0
    quote_amount += base_amount * price
    base_amount = 0.0
    sell_count += 1

#
# Performance report
#
print(f"")
print(f"Simulation finished:")
report(price)
