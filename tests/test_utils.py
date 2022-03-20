import pytest
import numpy.testing as npt

from common.utils import *
from common.utils import add_area_ratio
from common.signal_generation import *


def test_decimal():
	val = "4.1E-7"
	dec = to_decimal(val)
	assert round_down_str(dec, 8) == "0.00000041"

	assert round_down_str("4.1E-7", 8) == "0.00000041"

	assert round_down_str("10.000000001", 8) == "10.00000000"  # 8 zeros and then 1

	val = "10.000000009"  # 8 zeros and then 9
	assert round_str(val, 8) == "10.00000001"

	assert round_down_str(val, 8) == "10.00000000"

	to_sell = Decimal("0.01185454")  # What we can get from the server with 8 digits
	assert round_down_str(to_sell, 6) == "0.011854"  # We need to round but so smaller value (otherwise exception with not enough funds)

	pass


def test_signal_generation():
	data = [
		(222, 1, 2),
		(333, 2, 1),
		(444, 0, 1),
	]
	df = pd.DataFrame(data, columns=["aaa", "high_60_20", "low_60_04"])

	models = {
		"buy": {"high_60_20": 1, "low_60_04": 1},  # All higher
		"sell": {"high_60_20": 1, "low_60_04": 1},  # All lower
	}

	signals = generate_signals(df, models)

	# Check existence
	assert "buy" in df.columns.to_list()
	assert "sell" in df.columns.to_list()

	# Check values of signal columns
	assert [1,1,0] == list(df["buy"])
	assert [0,0,1] == list(df["sell"])


def test_depth_density():
	# Example 1
	depth = [
		[1, 1],
		[3, 1],
		[4, 1],
		[5, 1], # Bin border
		[6, 1],
		[7, 1],
	]

	bins_ask = discretize_ask(depth=depth, bin_size=4.0, start=None)
	bins = discretize("ask", depth, bin_size=4.0, start=None)

	assert bins_ask == [1, 1]
	assert bins == [1, 1]

	depth = [[-x[0], x[1]] for x in depth]  # Invert price so that it decreases

	bins = discretize("bid", depth, bin_size=4.0, start=None)

	assert bins == [1, 1]

	# Example 2
	depth = [
		[1, 1],
		[3, 1],
		[4, 20],  # Will be split between two bins
		# [5, 1], # Bin border
		[6, 1],
		[7, 1],
	]

	bins_ask = discretize_ask(depth=depth, bin_size=4.0, start=None)
	bins = discretize("ask", depth=depth, bin_size=4.0, start=None)

	assert bins_ask == [5.75, 5.75]
	assert bins == [5.75, 5.75]

	depth = [[-x[0], x[1]] for x in depth]  # Invert price so that it decreases

	bins = discretize("bid", depth=depth, bin_size=4.0, start=None)

	assert bins == [5.75, 5.75]

	# Example 3
	depth = [
		# 0 Start (previous point volume assumed to be 0)
		[1, 1],
		# 2 Bin border
		[3, 1],
		[4, 1],  # Bin border
		[5, 2],
	]

	bins_ask = discretize_ask(depth=depth, bin_size=2.0, start=0.0)
	bins = discretize("ask", depth=depth, bin_size=2.0, start=0.0)

	assert bins_ask == [0.5, 1.0, 1.5]
	assert bins == [0.5, 1.0, 1.5]

	depth = [[-x[0], x[1]] for x in depth]  # Invert price so that it decreases

	bins = discretize("bid", depth=depth, bin_size=2.0, start=0.0)

	assert bins == [0.5, 1.0, 1.5]

	# Example 4 - empty bin
	depth = [
		# 0 Start (previous point volume assumed to be 0)
		[1, 1],
		# 2 Bin border
		# Empty bin
		# 4 Bin border
		[5, 2],
	]

	bins = discretize("ask", depth=depth, bin_size=2.0, start=0.0)

	pass


def test_area_ratio():
	price = [10, 20, 30, 20, 10, 20, 30]
	df = pd.DataFrame(data={"price": price})

	features = add_area_ratio(df, is_future=False, column_name="price", windows=4)
	# Last element has to be computed from previous 3 elements
	assert df[df.columns[1]].iloc[-1] == -1  # all elements less than this one
	assert df[df.columns[1]].iloc[-2] == 0  # 1 is less and 1 is greater than this one

	features = add_area_ratio(df, is_future=True, column_name="price", windows=4)
	# First element has to be computed from next 3 elements
	assert df[df.columns[1]].iloc[0] == 1  # all elements greater than this one
	assert df[df.columns[1]].iloc[1] == 0  # 1 is less and 1 is greater than this one

	pass


def test_linear_trends():
	price = [10, 20, 40, 40, 30, 10]
	df = pd.DataFrame(data={"price": price})

	features = add_linear_trends(df, is_future=False, column_name="price", windows=2)
	npt.assert_almost_equal(df["price_trend_2"].values, np.array([0, 10, 20, 0, -10, -20]))

	features = add_linear_trends(df, is_future=True, column_name="price", windows=2)
	npt.assert_almost_equal(df["price_trend_2"].values, np.array([10, 20, 0, -10, -20, np.nan]))

	features = add_linear_trends(df, is_future=False, column_name="price", windows=6)
	npt.assert_almost_equal(df["price_trend_6"].values, np.array([0, 10, 15, 11, 6, 0.857143]))

	pass
