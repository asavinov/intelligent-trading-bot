
params_gb = {
    "objective": "cross_entropy",
    "max_depth": 1,
    "learning_rate": 0.01,
    "num_boost_round": 1_500,

    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "shifts": [],
}

params_nn = {
    #"is_scale": True,  # by default True
    "layers": [29],  # It is equal to the number of input features (different for spot and futur). Currently not used
    "learning_rate": 0.001,
    "n_epochs": 5,
    "bs": 128,
    "shifts": [],
}

params_lc = {
    "is_scale": True,
    "penalty": "l2",
    "C": 1.0,
    "class_weight": None,
    "solver": "sag",  # liblinear, lbfgs, sag/saga (stochastic gradient descent for large datasets, should be scaled)
    "max_iter": 200,
    "shifts": [],
    #"tol": 0.1,  # Tolerance for performance (check how it influences precision)
}
