"""Tests for src/features/engineer.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    build_feature_matrix,
    compute_log_returns,
    compute_realized_volatility,
    compute_volume_ratio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WINDOW = 21


def _make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with a random-walk Close and integer Volume."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n, freq="B")
    close = 1_000.0 * np.exp(rng.normal(0, 0.01, n).cumsum())
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.997,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv(n=120)


@pytest.fixture(scope="module")
def config() -> dict:
    return {"features": {"vol_window": _WINDOW, "volume_window": _WINDOW}}


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------


class TestComputeLogReturns:
    def test_length_preserved(self, ohlcv):
        assert len(compute_log_returns(ohlcv)) == len(ohlcv)

    def test_first_value_is_nan(self, ohlcv):
        ret = compute_log_returns(ohlcv)
        assert np.isnan(ret.iloc[0])

    def test_no_nan_after_first(self, ohlcv):
        ret = compute_log_returns(ohlcv).iloc[1:]
        assert ret.notna().all()

    def test_finite_values(self, ohlcv):
        ret = compute_log_returns(ohlcv).iloc[1:]
        assert np.isfinite(ret.values).all()

    def test_column_name(self, ohlcv):
        assert compute_log_returns(ohlcv).name == "log_return"

    def test_reasonable_magnitude(self, ohlcv):
        # Daily log returns of S&P 500 are rarely outside ±10 %
        ret = compute_log_returns(ohlcv).iloc[1:]
        assert (ret.abs() < 0.10).all()


# ---------------------------------------------------------------------------
# compute_realized_volatility
# ---------------------------------------------------------------------------


class TestComputeRealizedVolatility:
    @pytest.fixture(autouse=True)
    def _ret(self, ohlcv):
        self.ret = compute_log_returns(ohlcv)

    def test_first_window_all_nan(self):
        vol = compute_realized_volatility(self.ret, window=_WINDOW)
        # First window values (indices 0 … window-1) should all be NaN
        assert vol.iloc[:_WINDOW].isna().all()

    def test_values_after_warmup_are_positive(self):
        vol = compute_realized_volatility(self.ret, window=_WINDOW).dropna()
        assert (vol > 0).all()

    def test_annualised_scale_plausible(self):
        # Annualised vol of a typical equity index: 5 % – 100 %
        vol = compute_realized_volatility(self.ret, window=_WINDOW).dropna()
        assert (vol > 0.05).all() and (vol < 1.0).all()

    def test_column_name(self):
        vol = compute_realized_volatility(self.ret, window=_WINDOW)
        assert vol.name == "realized_vol"


# ---------------------------------------------------------------------------
# compute_volume_ratio
# ---------------------------------------------------------------------------


class TestComputeVolumeRatio:
    def test_first_window_minus_one_all_nan(self, ohlcv):
        vr = compute_volume_ratio(ohlcv["Volume"], window=_WINDOW)
        assert vr.iloc[: _WINDOW - 1].isna().all()

    def test_positive_after_warmup(self, ohlcv):
        vr = compute_volume_ratio(ohlcv["Volume"], window=_WINDOW).dropna()
        assert (vr > 0).all()

    def test_mean_near_one(self, ohlcv):
        vr = compute_volume_ratio(ohlcv["Volume"], window=_WINDOW).dropna()
        assert abs(vr.mean() - 1.0) < 0.2

    def test_column_name(self, ohlcv):
        assert compute_volume_ratio(ohlcv["Volume"], window=_WINDOW).name == "volume_ratio"


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    def test_correct_columns(self, ohlcv, config):
        X = build_feature_matrix(ohlcv, config)
        assert list(X.columns) == ["log_return", "realized_vol", "volume_ratio"]

    def test_no_nans(self, ohlcv, config):
        X = build_feature_matrix(ohlcv, config)
        assert not X.isnull().any().any()

    def test_shape(self, ohlcv, config):
        X = build_feature_matrix(ohlcv, config)
        # warm-up rows = max(vol_window, volume_window) = 21, plus 1 for log return
        expected_rows = len(ohlcv) - _WINDOW
        assert X.shape == (expected_rows, 3)

    def test_numpy_ready(self, ohlcv, config):
        arr = build_feature_matrix(ohlcv, config).values
        assert arr.ndim == 2
        assert arr.shape[1] == 3
        assert np.isfinite(arr).all()

    def test_index_is_datetime(self, ohlcv, config):
        X = build_feature_matrix(ohlcv, config)
        assert isinstance(X.index, pd.DatetimeIndex)

    def test_respects_config_windows(self, ohlcv):
        cfg = {"features": {"vol_window": 10, "volume_window": 10}}
        X = build_feature_matrix(ohlcv, cfg)
        expected_rows = len(ohlcv) - 10
        assert X.shape == (expected_rows, 3)
