import os
import sys
from sklearn.model_selection import ParameterGrid

grid_predictions = [
    {
        'train_max_length': [262_800],
        'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'num_boost_round': [500],
        'n_neighbors': [5, 10, 20], 'weights': ['uniform', 'distance'],
    },
    # Debug
    #{
    #    'train_max_length': [262_800],
    #    'max_depth': [3], 'learning_rate': [0.05], 'num_boost_round': [500],
    #    'n_neighbors': [1000, 2000], 'weights': ['uniform', 'distance'],
    #},
]

grid_signals = [
    {
        'threshold_buy_10': [0.25, 0.26, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.34, 0.35], 'threshold_buy_20': [0.0],
        'knn_threshold_buy_10': [0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.15], 'knn_threshold_buy_20': [0.0],
        'threshold_sell': [1.017, 1.018, 1.019, 1.02],
        'forced_sell_length': [80, 90, 100],
    },
    # Debug
    #{
    #    'threshold_buy_10': [0.29, 0.31],
    #    'threshold_buy_20': [0.0],
    #    'knn_threshold_buy_10': [0.09, 0.1, 0.11],
    #    'knn_threshold_buy_20': [0.0],
    #    'threshold_sell': [1.015, 1.016, 1.017, 1.018, 1.019, 1.020],
    #    'forced_sell_length': [90]
    #},
]


grid = ParameterGrid(grid_signals)

for params in grid:
    # Set parameters in environment variables
    for key, value in params.items():
        os.environ[key] = str(value)

    #
    # Generate predictions
    #
    #import scripts.generate_predictions
    #exitcode = scripts.generate_predictions.main([sys.argv[0], "generate_predictions"])

    #
    # Generate signals
    #
    import scripts.generate_signals
    exitcode = scripts.generate_signals.main([sys.argv[0], "generate_signals"])
