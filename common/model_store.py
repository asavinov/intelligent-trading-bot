import itertools
from pathlib import Path

from joblib import dump, load

from keras.models import Sequential, save_model, load_model

"""
Next:
- rework the logic of saving into and reading from model files with pattern <label-model>. Here probably again we need to have a list of labels and a list of algorithms.
  Alternatively, we might have a list of predict score labels which will have to be split

- currently we use modifier to append to file names. Instead, we could create a subfolder with this modifier.
- Same for model files which should be stored to MODELS and then modifier
- note that modifier is used only for derived files and not for source files

- currently: source files -> merge with prefix like "f_" -> generate features (yet, cannot use f_ columns)
  what we need: source file -> merge with prefix k_ and d_ and f_ -> generate features for each of them so that the procedure recoginizes prefix for selecting columns
  actually, our goal is to have btc and eth in one file, and then generate features for both of them with btc_ and eth_ prefix
  - What about feature set names? we use feature sets as named lists of input features. they might be also prefixed by symbol and channel: "kline_featrues" etc.

- formalize (in config) feature lists, i.e., named lists (do we really need this?=

- algo config has "predict" length - where should we use it? In rolling predictions?

PLAN:
criteria, use cases and alternatives
- use case: conventional stocks and indexes like DJ-index, DAX etc. each in its folder and daily klines - how to process
- use case: forex with EURUSD etc.
- assumption: symbol folder (and column prefix) with kline/futures.csv data source files merged to data.csv files
- alternatives: prefix is attached to columns: 1) during download 2) during merge 3) during feature generation 
- criterion: the matrix has contain derived features from different symbols like "btc_close_60", "eth_close_60"
  we then use these features in one feature list for training
- criterion: the matrix has to contain labels with prefix like "btc_top4_1" and "eth_top4_1"
  however, we use for training only one label origin like "btc_top4_1" (all with "btc_*")
- principles: we need to apply feature generation several times with parameter source-symbol/column-prefix
  - either to separate source files - bad because they are not synced
  - to prefixed features in merged file by adding/overwriting features to any existing file (similar to labels)
    - fg procedure itself is generic and assumes generic column names with certain *semantics* like "close"
      therefore, its functions will search for these generic names and produce generic output names like "close_60"
    - fg procedure needs to recognized which columns to process: columns prefix or label or whatever: "btc_k_".
      it loads data in df, selects columns using given prefix and produces an input df, 
      apples feature generation by producing output columns, adds prefix to these output columns, and attaches them to the original data (by possibly overwriting existing columns)
    - same with labels.
    - we need to reimplement features generation so that it processes any input file based on the specified column prefix
    the prefix is used to select input columns to feed to generic feature generation, and then add this prefix
    to output columns, and finally *update* the input data and store in output file.

- now we have input data file with attribute sets each having certain prefix (defined by data_sources)
- we need to apply several feature/label generators: one generator to several attribute sets, several generators to one attribute set (like features, high-low labels, top-bot labels)
- this could be described as a list:
 (column-selector-prefix, feature/label-generator name, out_prefix)
 currently (with one source):
 "", "kline-features", ""
 "", "high-low-labels", ""
 "", "top-bot-labels", ""
we can also split some features and columns
 in case of several sources with main and secondary sources like btc and eth
 "", "kline-features", ""
 "etc", "kline-features", "etc" <- additional features
 "", "top-bot-labels", ""
now we have a longer list of features like "close_60" and "etc_close_60" (we can reduce secondary features)
they need to be listed in config
"feature_sets": [
  {"column_prefix": "btc", "generator": "klines or labels etc.", "out_prefix": ""}, <- do we need out prefix?
]

- feature generation (preprocessing) operations "use_differences" if true then close/volume/trades are transformed to differences.
 it is similar to is_scale in trainining.
 - there could be also other options like apply log to certain columns or outputs 

"""

def get_model(name: str):
    """Given model name, return its JSON object"""
    return next(x for x in models if x.get("name") == name)

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


def load_models(model_path, labels: list, train_features: list, algorithms: list):
    """Load all model pairs for all combinations of the model parameters and return as a dict."""
    models = {}
    for predicted_label in itertools.product(labels, train_features, algorithms):
        score_column_name = predicted_label[0] + "_" + predicted_label[1][0] + "_" + predicted_label[2]
        model_pair = load_model_pair(model_path, score_column_name)
        models[score_column_name] = model_pair
    return models


models = [
    {
        "name": "nn",
        "algo": "nn",
        "params": {
            "layers": [29], # It is equal to the number of input features (different for spot and futur). Currently not used
            "learning_rate": 0.001,
            "n_epochs": 5,  # 5 for quick analysis, 20 for production
            "bs": 128,
        },
        "train": {"is_scale": True, "length": int(1.5 * 525_600), "shifts": []},
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
        "train": {"is_scale": True, "length": int(1.5 * 525_600), "shifts": []},
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
        "train": {"is_scale": False, "length": int(1.5 * 525_600), "shifts": []},
        "predict": {"length": 1440}
    },

    {
        "name": "nn_long",
        "algo": "nn",
        "params": {"layers": [29], "learning_rate": 0.001, "n_epochs": 20, "bs": 128, },
        "train": {"is_scale": True, "length": int(1.5 * 525_600), "shifts": []},
        "predict": {"length": 0}
    },
    {
        "name": "nn_middle",
        "algo": "nn",
        "params": {"layers": [29], "learning_rate": 0.001, "n_epochs": 20, "bs": 128, },
        "train": {"is_scale": True, "length": int(1.0 * 525_600), "shifts": []},
        "predict": {"length": 0}
    },
    {
        "name": "nn_short",
        "algo": "nn",
        "params": {"layers": [29], "learning_rate": 0.001, "n_epochs": 20, "bs": 128, },
        "train": {"is_scale": True, "length": int(0.5 * 525_600), "shifts": []},
        "predict": {"length": 0}
    },
]
