import pytest

from common.utils import *
from common.classifiers import *


def test_nan_handling_predict():
	"""Predicted input has nans. These nans rows have to be removed before prediction but the output has to contain all rows including these nan rows."""

	is_scale = True  # Try with both False and True (explicitly)

	df_X = pd.DataFrame({"x": [1, 2, 3, 2, 1], "y": [0, 1, 0, 1, 0]})  # Input has no nans
	df_X_test = pd.DataFrame({"x": [1, 2, None, 2, np.nan], "y": [0, 1, 0, 1, 0]})  # Has nans

	test_hat = train_predict_gb(
		df_X[["x"]], df_X["y"], df_X_test[["x"]],
		model_config=dict(is_scale=is_scale, objective="cross_entropy", max_depth=1, learning_rate=0.1, num_boost_round=2),
	)
	assert 5 == len(test_hat)
	assert 2 == test_hat.isnull().sum()

	test_hat = train_predict_nn(
		df_X[["x"]], df_X["y"], df_X_test[["x"]],
		model_config=dict(is_scale=is_scale, learning_rate=0.5, n_epochs=1, bs=2),
	)
	assert 5 == len(test_hat)
	assert 2 == test_hat.isnull().sum()

	test_hat = train_predict_lc(
		df_X[["x"]], df_X["y"], df_X_test[["x"]],
		model_config=dict(is_scale=is_scale),
	)
	assert 5 == len(test_hat)
	assert 2 == test_hat.isnull().sum()

	pass

