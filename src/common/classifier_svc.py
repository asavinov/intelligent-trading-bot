import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC, SVR

def train_predict_svc(df_X, df_y, df_X_test, model_config: dict):
    """
    Train model with the specified hyper-parameters and return its predictions for the test data.
    """
    model_pair = train_svc(df_X, df_y, model_config)
    y_test_hat = predict_svc(model_pair, df_X_test, model_config)
    return y_test_hat

def train_svc(df_X, df_y, model_config: dict):
    """
    Train model with the specified hyper-parameters and return this model (and scaler if any).
    """
    params = model_config.get("params", {})

    is_scale = params.get("is_scale", True)

    is_regression = params.get("is_regression", False)

    #
    # Prepare data
    #
    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    #
    # Create model
    #
    train_conf = model_config.get("train", {})
    args = train_conf.copy()
    if is_regression:
        model = SVR(**args)
    else:
        args['probability'] = True  # Required if we are going to use predict_proba()
        model = SVC(**args)

    #
    # Train
    #
    model.fit(X_train, y_train)

    return (model, scaler)

def predict_svc(models: tuple, df_X_test, model_config: dict):
    """
    Use the model(s) to make predictions for the test data.
    The first model is a prediction model and the second model (optional) is a scaler.
    """
    is_regression = model_config.get("params", {}).get("is_regression", False)

    #
    # Scale
    #
    scaler = models[1]
    is_scale = scaler is not None

    input_index = df_X_test.index
    if is_scale:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)
    else:
        df_X_test = df_X_test

    df_X_test_nonans = df_X_test.dropna()  # Drop nans, possibly create gaps in index
    nonans_index = df_X_test_nonans.index

    if is_regression:
        y_test_hat_nonans = models[0].predict(df_X_test_nonans.values)
    else:
        y_test_hat_nonans = models[0].predict_proba(df_X_test_nonans.values)  # It returns pairs or probas for 0 and 1
        y_test_hat_nonans = y_test_hat_nonans[:, 1]  # Or y_test_hat.flatten()
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)  # Attach indexes with gaps

    df_ret = pd.DataFrame(index=input_index)  # Create empty dataframe with original index
    df_ret["y_hat"] = y_test_hat_nonans  # Join using indexes
    sr_ret = df_ret["y_hat"]  # This series has all original input indexes but NaNs where input is NaN

    return sr_ret
