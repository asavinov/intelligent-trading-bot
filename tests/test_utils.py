import pytest

from common.utils import *
from common.utils import add_area_ratio
from common.signal_generation import *

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
