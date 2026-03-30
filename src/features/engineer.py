"""Feature engineering for market regime detection."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Daily log returns from adjusted Close prices.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a ``Close`` column.

    Returns
    -------
    pd.Series
        ``ln(P_t / P_{t-1})``, named ``"log_return"``.
        First value is NaN (no prior close).
    """
    return np.log(df["Close"] / df["Close"].shift(1)).rename("log_return")


def compute_realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualised realised volatility.

    Parameters
    ----------
    log_returns:
        Daily log return series.
    window:
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        ``std(log_returns, window) × √252``, named ``"realized_vol"``.
        First ``window`` values are NaN (warm-up).
    """
    return (log_returns.rolling(window).std() * np.sqrt(252)).rename("realized_vol")


def compute_volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    """Volume relative to its rolling mean.

    A value of 1.0 means today's volume equals the rolling average;
    values > 1 indicate elevated activity.

    Parameters
    ----------
    volume:
        Daily traded volume series.
    window:
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        ``volume / rolling_mean(volume, window)``, named ``"volume_ratio"``.
        First ``window`` values are NaN (warm-up).
    """
    return (volume / volume.rolling(window).mean()).rename("volume_ratio")


def build_feature_matrix(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assemble the feature matrix used for HMM training.

    Computes the three features, concatenates them, then drops any rows
    that contain NaN (arising from the rolling-window warm-up period at the
    start of the series).

    Parameters
    ----------
    df:
        OHLCV DataFrame as returned by :func:`~src.data.fetch.fetch_ohlcv`.
    config:
        Config dict.  Uses ``config["features"]["vol_window"]`` and
        ``config["features"]["volume_window"]``.

    Returns
    -------
    pd.DataFrame
        Columns ``["log_return", "realized_vol", "volume_ratio"]``,
        no NaN values, shape ``(n_samples, 3)``.
        Call ``.values`` to get the numpy array expected by *hmmlearn*.
    """
    vol_window = config["features"]["vol_window"]
    volume_window = config["features"]["volume_window"]

    ret = compute_log_returns(df)
    vol = compute_realized_volatility(ret, window=vol_window)
    vr = compute_volume_ratio(df["Volume"], window=volume_window)

    features = pd.concat([ret, vol, vr], axis=1)

    n_before = len(features)
    features = features.dropna()
    n_dropped = n_before - len(features)
    logger.info(
        f"Feature matrix built: {len(features)} rows "
        f"({n_dropped} dropped — rolling warm-up, "
        f"window={max(vol_window, volume_window)})"
    )

    return features
