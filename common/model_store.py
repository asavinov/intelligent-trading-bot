import itertools
from pathlib import Path

from joblib import dump, load

from keras.models import save_model, load_model

label_algo_separator = "_"


"""
It is a model stored implemented as a Python module.
"""

def get_model(name: str):
    """Given model name, return its JSON object"""
    return next(x for x in models if x.get("name") == name)


def get_algorithm(algorithms: list, name: str):
    """Given a list of algorithms (from config), find an entry for the algorithm with the specified model name"""
    return next(x for x in algorithms if x.get("name") == name)


def load_models_from_file(file):
    """Load model store from file to memory"""
    pass


def save_model_pair(model_path, score_column_name: str, model_pair: tuple):
    """Save two models in two files with the corresponding extensions."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    model_path = model_path.absolute()

    model = model_pair[0]
    scaler = model_pair[1]
    # Save scaler
    scaler_file_name = (model_path / score_column_name).with_suffix(".scaler")
    dump(scaler, scaler_file_name)
    # Save prediction model
    if score_column_name.endswith("_nn"):
        model_extension = ".h5"
        model_file_name = (model_path / score_column_name).with_suffix(model_extension)
        save_model(model, model_file_name)
    else:
        model_extension = ".pickle"
        model_file_name = (model_path / score_column_name).with_suffix(model_extension)
        dump(model, model_file_name)


def load_model_pair(model_path, score_column_name: str):
    """Load a pair consisting of scaler model (possibly null) and prediction model from two files."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    model_path = model_path.absolute()
    # Load scaler
    scaler_file_name = (model_path / score_column_name).with_suffix(".scaler")
    scaler = load(scaler_file_name)
    # Load prediction model
    if score_column_name.endswith("_nn"):
        model_extension = ".h5"
        model_file_name = (model_path / score_column_name).with_suffix(model_extension)
        model = load_model(model_file_name)
    else:
        model_extension = ".pickle"
        model_file_name = (model_path / score_column_name).with_suffix(model_extension)
        model = load(model_file_name)

    return (model, scaler)


def load_models(model_path, labels: list, algorithms: list):
    """Load all model pairs for all combinations of labels and algorithms and return as a dict."""
    models = {}
    for label_algorithm in itertools.product(labels, algorithms):
        score_column_name = label_algorithm[0] + label_algo_separator + label_algorithm[1]["name"]
        try:
            model_pair = load_model_pair(model_path, score_column_name)
        except Exception as e:
            print(f"ERROR: Cannot load model {score_column_name} from path {model_path}. Skip.")
            continue
        models[score_column_name] = model_pair
    return models


def load_models_for_generators(config: dict, model_path):
    """Load all model pairs which are really used according to the algorithm section."""

    labels_default = config.get("labels", [])
    algorithms_default = config.get("algorithms")

    # For each entry, a list of labels and a list of algorithms is retrieved, and then all their models are loaded
    train_feature_sets = config.get("train_feature_sets", [])
    models = {}
    for i, fs in enumerate(train_feature_sets):

        labels = fs.get("config").get("labels", [])
        if not labels:
            labels = labels_default

        algorithms_default = config.get("algorithms")
        algorithm_names = fs.get("config").get("functions", [])
        if not algorithm_names:
            algorithm_names = fs.get("config").get("algorithms", [])
        algorithms = resolve_algorithms_for_generator(algorithm_names, algorithms_default)

        # Load models for all combinations of labels and algorithms
        fs_models = load_models(model_path, labels, algorithms)

        models.update(fs_models)

    return models


def resolve_algorithms_for_generator(algorithm_names: list, algorithms_default: list):
    """Get all algorithm configs for a list of algorithm names."""

    # The algorithms can be either strings (names) or dicts (definitions) so we resolve the names
    algorithms = []
    for alg in algorithm_names:
        if isinstance(alg, str):  # Find in the list of algorithms
            alg = next(a for a in algorithms_default if a['name'] == alg)
        elif not isinstance(alg, dict):
            raise ValueError(f"Algorithm has to be either dict or name")
        algorithms.append(alg)
    if not algorithms:
        algorithms = algorithms_default

    return algorithms


def score_to_label_algo_pair(score_column_name: str):
    """
    Parse a score column name and return its two constituents: label column name and algorithm name.
    """
    # Return split from right, because underscore occurs also in label names
    label_name, algo_name = score_column_name.rsplit(label_algo_separator, 1)
    return label_name, algo_name

# Deprecated. Use them for reference and include in "algorithms" in config instead.
models = [
    {
        "name": "nn",
        "algo": "nn",
        "params": {
            "layers": [29], # It is equal to the number of input features (different for spot and futur). Currently not used
            "learning_rate": 0.001,
            "n_epochs": 50,  # 5 for quick analysis, 20 or 30 for production
            "bs": 1024,
        },
        "train": {"is_scale": True, "length": int(3.0 * 525_600), "shifts": []},
        "predict": {"length": "1w"}
    },
    {
        "name": "lc",
        "algo": "lc",
        "params": {
            "penalty": "l2",
            "C": 1.0,
            "class_weight": None,
            "solver": "sag", # liblinear, lbfgs, sag/saga (stochastic gradient descent for large datasets, should be scaled)
            "max_iter": 200,
            # "tol": 0.1,  # Tolerance for performance (check how it influences precision)
        },
        "train": {"is_scale": True, "length": int(3.0 * 525_600), "shifts": []},
        "predict": {"length": 1440}
    },
    {
        "name": "gb",
        "algo": "gb",
        "params": {
            "objective": "cross_entropy",
            "max_depth": 1,
            "learning_rate": 0.01,
            "num_boost_round": 1_500,

            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
        },
        "train": {"is_scale": False, "length": int(3.0 * 525_600), "shifts": []},
        "predict": {"length": 1440}
    },

    {
        "name": "gb-y",
        "algo": "gb",
        "params": {
            #"boosting_type": 'gbdt',
            "objective": "cross_entropy",  # binary cross_entropy cross_entropy_lambda
            "max_depth": 1,
            "learning_rate": 0.05,
            "num_boost_round": 1_500,  # 10_000

            #"is_unbalance": True,
            #"metric": 'auc',  # auc binary_logloss cross_entropy cross_entropy_lambda binary_error
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
        },
        "train": {"is_scale": True, "length": None, "shifts": []},
        "predict": {"length": 1440}
    },
    {
        "name": "svc-y",
        "algo": "svc",
        "params": {"C": 10.0},  # , "gamma": 0.02
        "train": {"is_scale": True, "length": None, "shifts": []},
        "predict": {"length": 1440}
    },

    {
        "name": "nn_long",
        "algo": "nn",
        "params": {"layers": [29], "learning_rate": 0.001, "n_epochs": 20, "bs": 128, },
        "train": {"is_scale": True, "length": int(2.0 * 525_600), "shifts": []},
        "predict": {"length": 0}
    },
    {
        "name": "nn_middle",
        "algo": "nn",
        "params": {"layers": [29], "learning_rate": 0.001, "n_epochs": 20, "bs": 128, },
        "train": {"is_scale": True, "length": int(1.5 * 525_600), "shifts": []},
        "predict": {"length": 0}
    },
    {
        "name": "nn_short",
        "algo": "nn",
        "params": {"layers": [29], "learning_rate": 0.001, "n_epochs": 20, "bs": 128, },
        "train": {"is_scale": True, "length": int(1.0 * 525_600), "shifts": []},
        "predict": {"length": 0}
    },
]
