import os
import sys

#from trade.utils import *
#from trade.analysis import *
#from trade.main import *

script_name = "collect_data_ws"  # Default value

if len(sys.argv) > 1:
    script_name = sys.argv[1]

# Generate predictions params
params = {
    'train_max_length': 525_600,  # [43_800, 131_400, 262_800, 525_600, 10_000_000]

    'max_depth': 5,  # [1, 2, 3, 4, 5]
    'learning_rate': 0.05,  # [0.01, 0.05, 0.1]
    'num_boost_round': 1_000,  # [500, 1_000, 2_000, 5_000]

    'n_neighbors': 1000,
    'weights': 'uniform',
}
# Set parameters in environment variables
for key, value in params.items():
    os.environ[key] = str(value)

if script_name == "collect_data":
    from trade.main import *
    exitcode = main(sys.argv[1:])

if script_name == "collect_data_ws":
    # CHECK:
    # - in Database.py::store_queue, rotate_suffix has to store in monthly files (not daily or hourly or whatever was used for debugging)
    # - in App.py, check list of symbols and depth of order book (20 and not 5 or whatever was used for debugging)
    # - in App.py, check frequency of flushes and set it to 300 seconds or 60 seconds.
    # - start command: python start.py collect_data_ws &
    from trade.collect_ws import *
    exitcode = start_collect_ws()
    #import trade.collect_ws
    #exitcode = trade.collect_ws.start_collect_ws()

if script_name == "generate_features":
    import scripts.generate_features
    exitcode = scripts.generate_features.main(sys.argv[1:])

    #from scripts.generate_features import main
    #exitcode = main(sys.argv[1:])

elif script_name == "generate_predictions":
    import scripts.generate_predictions
    exitcode = scripts.generate_predictions.main(sys.argv[1:])

    #from scripts.generate_predictions import main
    #exitcode = main(sys.argv[1:])

else:
    print(f"ERROR: Unknown script name {script_name}")
