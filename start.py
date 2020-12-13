import os
import sys

#from common.utils import *
#from trade.analysis import *
#from trade.main import *

script_name = "merge_data"  # Default value

if len(sys.argv) > 1:
    script_name = sys.argv[1]

# Generate predictions params
params = {
    'max_depth': 4,  # [1, 2, 3, 4, 5]
    'learning_rate': 0.05,  # [0.01, 0.05, 0.1]
    'num_boost_round': 1_000,  # [500, 1_000, 2_000, 5_000]

    'n_neighbors': 1000,
    'weights': 'uniform',
}
# Set parameters in environment variables
for key, value in params.items():
    os.environ[key] = str(value)

# === BATCH SCRIPTS ===
#
if script_name == "download_data":
    # download data
    # NOTE: Must be started from Anaconda console since some special (encription, certificate) libs are not otherwise found
    # NOTE: Uncomment 3 lines to retrieve futures
    # NOTE: Currently we download data from 2 sources (klines and futures) which will have to be immediately merged on the next step
    import scripts.download_data
    exitcode = scripts.download_data.main(sys.argv[1:])

    #from scripts.download_data import main
    #exitcode = main(sys.argv[1:])

if script_name == "merge_data":
    # Merge two or three source files (klines, futur, depth) into one source file with different column names
    # It not only merges data, but also generates a regular correct 1m raster (in order to remove possible gaps)
    # Currently we do it for klines and futures only
    # The result file is 1m file with original klines and future klines with column names prefixed with "f_"
    import scripts.merge_data
    exitcode = scripts.merge_data.main(sys.argv[1:])

if script_name == "generate_features":
    # Generate feature matrix file from source data: Read input file, generate features, generate labels, store the result
    # NOTE: The source file is a merged file with source features from: klines, futures
    import scripts.generate_features
    exitcode = scripts.generate_features.main(sys.argv[1:])

    #from scripts.generate_features import main
    #exitcode = main(sys.argv[1:])

if script_name == "generate_rolling_predictions":
    # In a loop with increasing history (we add more and more new data):
    # - train model using the current historic data
    # - apply this model to next horizon and store prediction
    # - repeat by adding the horizon to history
    # Finally, store the sequentially generated predictions along with the data and features in a file (to train signal models)
    # Output: file with source source features, labels, and (generated) predicted label scores
    import scripts.generate_rolling_predictions
    exitcode = scripts.generate_rolling_predictions.main(sys.argv[1:])

    #from scripts.generate_rolling_predictions import main
    #exitcode = main(sys.argv[1:])

if script_name == "train_signal_models":
    import scripts.train_signal_models
    exitcode = scripts.train_signal_models.main(sys.argv[1:])

    #from scripts.generate_rolling_predictions import main
    #exitcode = main(sys.argv[1:])

if script_name == "train_predict_models":
    # Use feature matrix to train several predict models one for each label and algorithm.
    # This procedure is used to periodically update models and upload them to the server
    # If models are needed for different data ranges (say, shorter latest data) then new run is needed with other parameters
    # Output: model files (one per label and algorithm)
    # NOTE: many NaN rows will be dropped because we include features with take/maker which have no data at the beginning of history
    #   Check quality of predictions and overall performance if we exclude initial history (with no take/maker data)
    import scripts.train_predict_models
    exitcode = scripts.train_predict_models.main(sys.argv[1:])

    #from scripts.train_predict_models import main
    #exitcode = main(sys.argv[1:])

# === DATA COLLECTOR SERVER ===
#
if script_name == "collect_data":
    # Regularly (1m) request new data: depth
    # - start command: python start.py collect_data &
    # CHECK:
    # - In "collector"-"depth": symbols, depth (how may items in order book), freq (1m)
    from trade.main import *
    exitcode = main(sys.argv[1:])

if script_name == "collect_data_ws":
    # Subscribing to WebSocket data feed and receiving two types of streams: klines and depth
    # - start command: python start.py collect_data_ws &
    # CHECK:
    # - in Database.py::store_queue, rotate_suffix has to store in monthly files (not daily or hourly or whatever was used for debugging)
    # - in App.py, check list of symbols and depth of order book (20 and not 5 or whatever was used for debugging)
    # - in App.py, check frequency of flushes and set it to 300 seconds or 60 seconds.
    from trade.collector_ws import *
    exitcode = start_collector_ws()
    #import trade.collector_ws
    #exitcode = trade.collector_ws.start_collector_ws()

# === TRADER SERVER ===
#
if script_name == "trade_server":
    # Regularly (1m) request data, analyze (feature, predictions, signals), trade (buy or check sold)
    # - start command: python start.py trade_server &
    # CHECK:
    # In "config"|"trader"
    # - "analysis", "kline_window" must be >300 (base window), e.g., 400 or 600
    # - "analysis", "features" copy all derived feature names
    # - "analysis", "labels" copy all predicted labels (one for each trained model used)
    # - "parameters": check all parameters of trade logic
    from trade.trader import *
    exitcode = start_trader()
    #import trade.trade_server
    #exitcode = trade.trade_server.start_trader()
