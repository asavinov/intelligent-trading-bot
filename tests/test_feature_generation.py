import sys

import numpy as np
import pandas as pd
import pytest

from itb.common.gen_features import generate_features_talib


def _make_df():
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    df = pd.DataFrame({"close": close}, index=pd.date_range("2024-01-01", periods=300, freq="h"))
    df.iloc[:3, 0] = np.nan  # Leading NaNs are skipped like in TA-Lib
    return df


def _slope(x):
    return np.polyfit(np.arange(len(x)), x, 1)[0]


@pytest.fixture
def no_talib(monkeypatch):
    # Simulate an environment without the native TA-Lib library
    for name in ["talib", "talib.stream", "talib.abstract"]:
        monkeypatch.setitem(sys.modules, name, None)


def test_talib_generator_without_talib(no_talib):
    df = _make_df()
    config = {"columns": ["close"], "functions": ["SMA", "STDDEV", "LINEARREG_SLOPE"], "windows": [1, 5, 24]}

    features = generate_features_talib(df, config)

    assert features == [
        "close_SMA_1", "close_SMA_5", "close_SMA_24",
        "close_STDDEV_1", "close_STDDEV_5", "close_STDDEV_24",
        "close_LINEARREG_SLOPE_1", "close_LINEARREG_SLOPE_5", "close_LINEARREG_SLOPE_24",
    ]

    close = df["close"]
    for w in [5, 24]:
        pd.testing.assert_series_equal(df[f"close_SMA_{w}"], close.rolling(w).mean(), check_names=False)
        pd.testing.assert_series_equal(df[f"close_STDDEV_{w}"], close.rolling(w).std(ddof=0), check_names=False)
        pd.testing.assert_series_equal(df[f"close_LINEARREG_SLOPE_{w}"], close.rolling(w).apply(_slope, raw=True), check_names=False)
        assert df[f"close_SMA_{w}"].isna().sum() == 3 + w - 1


def test_talib_generator_without_talib_unsupported_function(no_talib):
    df = _make_df()
    with pytest.raises(ValueError, match="requires TA-Lib"):
        generate_features_talib(df, {"columns": ["close"], "functions": ["CCI"], "windows": [5]})


@pytest.mark.parametrize("func_name", ["SMA", "EMA", "WMA", "DEMA", "TEMA", "RSI", "MOM", "ROC", "STDDEV", "LINEARREG", "LINEARREG_SLOPE"])
def test_fin_primitives_matches_talib(func_name, monkeypatch):
    talib = pytest.importorskip("talib")
    df = _make_df()
    config = {"columns": ["close"], "functions": [func_name], "windows": [14]}

    generate_features_talib(df, config)
    expected = df[f"close_{func_name}_14"].copy()

    for name in ["talib", "talib.stream", "talib.abstract"]:
        monkeypatch.setitem(sys.modules, name, None)
    generate_features_talib(df, config)

    pd.testing.assert_series_equal(df[f"close_{func_name}_14"], expected, rtol=1e-9)
