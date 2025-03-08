import itertools
from pathlib import Path

from joblib import dump, load

from keras.models import save_model, load_model

label_algo_separator = "_"


"""
It is a model stored implemented as a Python module.
"""

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
            alg = find_algorithm_by_name(algorithms_default, alg)
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


def find_algorithm_by_name(algorithms: list, name: str):
    """Given a list of algorithms (from config), find an entry for the algorithm with the specified model name"""
    return next(x for x in algorithms if x.get("name") == name)
