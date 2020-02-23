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
from trade.feature_generation import *
from trade.feature_prediction import *

"""
GOAL: Store a (source) file with attached predicted labels, so that this file can be then used for testing trade strategies.
- the file has to container all labels which are used for signaling, making buy/sell decisions and measuring performance.
- ideally, simply add new columns to the source file so that all source columns and timestamps are retained.
- we can generate more labels so that different other strategies can be tested
- trained models can be stored but only if we want to reuse them to generate predicted labels.
- TODO: 
  - Do not dismiss predicted lables after each prediction, but rather append them to the collected labels
  - Finally, attach the labels to the source df (using indexes for matching). Initial and maybe some end intervals will be None.
  - Store the file. Feature columns can be deleted.

GOAL: Separate simulation script which will assume the existence of a source file with predicted labels
- the script loads the file, iterates through it, and uses labels to trade
- the label generation file is not trading (but can do so)
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

labels_horizon = 60  # Labels are generated using this number of steps ahead
features_horizon = 300  # Features are generated using this past window length

train_start = 0  # Note that it is used *after* removing all nans
train_end = 718_170  # 718_170 "2019-01-01 00:00:00" <--- PARAMS
train_period = 60 * 24  # Re-train once per day
models = None  # Dict of last trained models for each label
train_count = 0
model_path_name = r"MODELS"  # MODELS_TEST <--- PARAMS
model_path = Path(model_path_name)
model_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

simulation_start = train_end
simulation_end = 1_106_909  # 1_106_909 "2019-09-29 00:00:00" <--- PARAMS

initial_amount = 1_000.0
base_amount = 0.0
quote_amount = initial_amount

buy_signal_count = 0
sell_signal_count = 0
buy_count = 0
sell_count = 0
buysell_signal_count = 0

#
# Load historic data
#
print(f"Loading data...")
path = Path(data_path_name).joinpath(data_file_name)
historic_df = pd.read_csv(path, parse_dates=['timestamp'], nrows=simulation_end+10_000)

#
# Generate features and labels
#
print(f"Generating features...")
features = generate_features(historic_df)
print(f"Generating labels...")
labels = generate_labels_sim(historic_df, horizon=60)

historic_df.dropna(subset=features, inplace=True)
historic_df.dropna(subset=labels, inplace=True)  # Note that if labels are True/False, then nan will NOT be gnerated by comparison and hence rows are NOT dropped

#
# Main loop over trade sessions
#
print(f"Starting simulation loop...")
for i in range(simulation_start, simulation_end):
    price = historic_df.iloc[i]["close"]

    #
    # Re-train models
    #
    if (i - simulation_start) % train_period == 0:
        retrain_start = train_start  # Always from the very beginning
        retrain_end = i - labels_horizon - 1  # Ignore some recent data for which there is no future data to generate true labels (1 is to ensure that there is no future leak in the model)
        train_df = historic_df.iloc[retrain_start:retrain_end]

        #
        # Train model for each label separately
        #
        print(f"Retrain using {len(train_df)} records...")
        X = train_df[features].values
        models = {}
        for label in labels:
            # Load a model using the timestamp (retrain_end) as a tag in the name
            model_file_name = str(int(train_df["timestamp"].iloc[-1].timestamp())) + "-" + label  # Model name is tagged by a timestamp and label
            model_file = model_path / (model_file_name + ".pkl")

            if model_file.exists():  # Load an earlier trained model
                with model_file.open('rb') as f:
                    model = pickle.load(f)
            else:  # Train model if not found
                y = train_df[label].values
                y = y.reshape(-1)
                model = train_model_gb_classifier(X, y)
                with model_file.open('wb+') as f:
                    pickle.dump(model, f)  # Store model for future use

            models[label] = model

        # Report
        print(f"")
        print(f"Trained or loaded {len(labels)} models. Step {i}. Retrain count: {train_count}")
        report(price)

        train_count += 1

    #
    # Predict
    #

    # Select prediction data
    # Normally, we will have to include feature (past) horizon to compute features
    # But now features are pre-computed, so we essentially predict one row
    predict_start = i
    predict_end = i + 1  # Add 1 because end of slice of df is exclusive
    predict_df = historic_df.iloc[predict_start:predict_end]

    # Predict labels by applying corresponding models
    predict_labels_df = predict_labels(predict_df[features], models)

    # Generate signals by analyzing predicted labels in the last interval
    row = predict_labels_df.iloc[-1]
    signal = ""
    if row["high_60_20"] > 0.5 and row["low_60_02"] > 0.5:  # High will be over threshold and low will be over (negative) threshold
        signal = "buy"
        buy_signal_count += 1
    if row["low_60_20"] > 0.5 and row["high_60_02"] > 0.5:  # Low will be under (negative) threshold and high will be under threshold
        if signal == "buy":  # Two signals simultaneously
            signal = "buysell"
            buysell_signal_count += 1
            print(f"WARNING: Two signals simultaniously.")
        else:
            signal = "sell"
            sell_signal_count += 1

    # Execute signal by doing trade
    price = predict_df.iloc[-1]["close"]
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
