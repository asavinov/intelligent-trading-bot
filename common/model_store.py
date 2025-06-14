import itertools
from pathlib import Path

from joblib import dump, load

from keras.models import save_model, load_model

import logging
log = logging.getLogger('model_store')


label_algo_separator = "_"


class ModelStore:
    """
    Model store which stores named models used (mainly) by feature generators

    Models are objects stored persistently in files and representing parameters for feature generators.
    There are two ways how parameters of generators can be represented:
    - In the config file
    - In model files

    Why do we need model files for representing feature generation parameters?
    Because these parameters are supposed to be generated automatically and are not known in advance.
    We run certain analysis algorithms to find these parameters, and the results might depend on
    the historic data (as well as on external data). In contrast, parameters in config files are more
    stable and do not change too frequently.
    """

    def __init__(self, config):
        """
        Create a new a new model store.

        Model objects are supposed to be stored persistently at some location (currently only local files
        but in future also databases and remote locations).
        The models are loaded and stored in-memory so that they can be easily accessed at run-time.

        The models are stored persistently if their in-memory object is updated/written.
        """
        self.config = config

        symbol = config["symbol"]

        data_path = Path(config["data_folder"]) / symbol
        model_path = Path(config["model_folder"])
        if not model_path.is_absolute():
            model_path = data_path / model_path
        #model_path = model_path.absolute()

        self.model_path = model_path.resolve()

        self.model_registry = config.get("model_registry", [])

        self.models = []  # List of dicts with model attributes

        #
        # Load models
        #

        # For compatibility with the previous version where models are identified by the label-algo pair in the generator config
        # It returns a dict with (model name, model object) pairs by getting information from generator (previous version) configs
        #self.ml_models = self.load_models_for_generators()

    #
    # Old approach where models are identified by label-algo pairs
    #

    def save_model_pair(self, score_column_name: str, model_pair: tuple):
        """Save two models in two files with the corresponding extensions."""
        self.model_path.mkdir(parents=True, exist_ok=True)  # Ensure that folder exists

        model = model_pair[0]
        scaler = model_pair[1]
        # Save scaler
        scaler_file_name = (self.model_path / score_column_name).with_suffix(".scaler")
        dump(scaler, scaler_file_name)
        # Save prediction model
        model_extension = ".pickle"
        model_file_name = (self.model_path / score_column_name).with_suffix(model_extension)
        dump(model, model_file_name)

    def load_model_pair(self, score_column_name: str):
        """Load a pair consisting of scaler model (possibly null) and prediction model from two files."""
        # Load scaler
        scaler_file_name = (self.model_path / score_column_name).with_suffix(".scaler")
        scaler = load(scaler_file_name)
        # Load prediction model
        model_extension = ".pickle"
        model_file_name = (self.model_path / score_column_name).with_suffix(model_extension)
        model = load(model_file_name)

        return (model, scaler)

    def load_models(self, labels: list, algorithms: list):
        """Load all model pairs for all combinations of labels and algorithms and return as a dict."""
        models = {}
        for label_algorithm in itertools.product(labels, algorithms):
            score_column_name = label_algorithm[0] + label_algo_separator + label_algorithm[1]["name"]
            try:
                model_pair = self.load_model_pair(score_column_name)
            except Exception as e:
                log.error(f"ERROR: Cannot load model {score_column_name} from path {self.model_path}. Skip.")
                continue
            models[score_column_name] = model_pair
        return models

    def load_models_for_generators(self):
        """Load all model pairs which are really used according to the algorithm section."""

        labels_default = self.config.get("labels", [])
        algorithms_default = self.config.get("algorithms")

        # For each entry, a list of labels and a list of algorithms is retrieved, and then all their models are loaded
        train_feature_sets = self.config.get("train_feature_sets", [])
        models = {}
        for i, fs in enumerate(train_feature_sets):

            labels = fs.get("config").get("labels", [])
            if not labels:
                labels = labels_default

            algorithms_default = self.config.get("algorithms")
            algorithm_names = fs.get("config").get("functions", [])
            if not algorithm_names:
                algorithm_names = fs.get("config").get("algorithms", [])
            algorithms = resolve_algorithms_for_generator(algorithm_names, algorithms_default)

            # Load models for all combinations of labels and algorithms
            fs_models = self.load_models(labels, algorithms)

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


def find_algorithm_by_name(algorithms: list, name: str):
    """Given a list of algorithms (from config), find an entry for the algorithm with the specified model name"""
    return next(x for x in algorithms if x.get("name") == name)


def score_to_label_algo_pair(score_column_name: str):
    """
    Parse a score column name and return its two constituents: label column name and algorithm name.
    """
    # Return split from right, because underscore occurs also in label names
    label_name, algo_name = score_column_name.rsplit(label_algo_separator, 1)
    return label_name, algo_name
