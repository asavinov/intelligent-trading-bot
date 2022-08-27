# Conventions

## Input data

Column names are defined by the `download` script. Note that future and kline data have the same column names when loaded from API. Future columns get prefix "f_".

## Derived features

The columns added to df are hard-coded by functions in module `common.feature_generation.py`

* klines columns: function `generate_features()`. They are listed in `App.config.features_kline`
* futur columns: function `generate_features_futur()`. They are listed in `App.config.features_futur`
* depth columns: function `generate_features_depth()`. They are listed in `App.config.features_depth`

## Labels

The columns added to df are hard-coded by functions in module `common.label_generation_highlow.py`. These columns are similar to derived features but describe future values. Note that labels are derived from kline data only (at least if we trade on spot market).

Currently, we generate labels using `generate_labels_thresholds()` which is a long list in `App.config.class_labels_all`. Only a subset of these labels is really used listed in `App.config.labels`. 

## Rolling predictions and predictions (label scores)

Predicting columns are generated using the following dimensions:
* target label (like `high_15`): we generate predictions for each chosen label. Note that labels themselves may have structure like threshold level (10, 15, 20 etc.) but here we ignore this and assume that all labels are independent
* input features (`k`, `f`): currently we use either klines or future
* algorithm (`gb`, `nn`, `lc`): we do the same predictions using different algorithms. We assume that algorithm name includes also its hyper-parameters and maybe even history length.
* history length: how much historic data was used to train the model. Currently we do not use this parameter. We could also assume that it is part of the algorithm or its hyper-parameters.

Accordingly, the generated (predicted) column names have the following structure:

    <target_label>_<input_data>_<algorithm>

For example: `high_15_k_nn` means label `high_15` for kline derived features and `nn` prediction model.
