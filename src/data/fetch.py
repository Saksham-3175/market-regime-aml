"""yfinance data fetching with parquet disk cache."""

import logging
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests

logger = logging.getLogger(__name__)

# Yahoo Finance blocks plain Python/requests TLS fingerprints; curl_cffi
# impersonates Chrome so the connection succeeds.
_SESSION = curl_requests.Session(impersonate="chrome110")

_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "").replace("/", "_")
    return _RAW_DIR / f"{safe}_{start}_{end}.parquet"


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance, cached as parquet under data/raw/.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol (e.g. ``"^GSPC"``).
    start:
        Start date, ``"YYYY-MM-DD"``.
    end:
        End date, ``"YYYY-MM-DD"``, or ``None`` for today.
    use_cache:
        Return cached parquet when available; write on first download.

    Returns
    -------
    pd.DataFrame
        Columns ``[Open, High, Low, Close, Volume]``, ``DatetimeIndex``
        named ``"Date"`` (timezone-naive, UTC).  No NaN in ``Close``;
        ``Volume`` NaN filled with ``0.0``.

    Raises
    ------
    ValueError
        If yfinance returns an empty result for the requested range.
    """
    if end is None:
        end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    cache = _cache_path(ticker, start, end)

    if use_cache and cache.exists():
        logger.info(f"Cache hit: loading {cache.name}")
        return pd.read_parquet(cache)

    logger.info(f"Downloading {ticker} [{start} → {end}]")
    raw = yf.download(
        ticker, start=start, end=end,
        auto_adjust=True, progress=False,
        session=_SESSION,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker} [{start}, {end}]")

    # yfinance sometimes returns MultiIndex columns even for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Timezone-aware index → tz-naive UTC
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "Date"

    # Drop rows where Close is NaN
    n_before = len(df)
    df = df.dropna(subset=["Close"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} rows with NaN Close")

    # Fill NaN Volume with 0.0 (common for indices like ^GSPC on some dates)
    n_vol_nan = int(df["Volume"].isna().sum())
    if n_vol_nan:
        logger.warning(f"Filling {n_vol_nan} NaN Volume values with 0.0")
        df["Volume"] = df["Volume"].fillna(0.0)

    _check_gaps(df, ticker)

    if use_cache:
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        logger.info(f"Cached → {cache}")

    return df


def _check_gaps(df: pd.DataFrame, ticker: str, max_gap: int = 5) -> None:
    """Emit a UserWarning for any gap > *max_gap* consecutive business days."""
    if len(df) < 2:
        return

    idx_set = set(df.index.normalize())
    all_bdays = pd.bdate_range(df.index.min(), df.index.max())
    missing = sorted(set(all_bdays) - idx_set)

    if not missing:
        return

    # Walk the sorted missing days, group consecutive runs
    run_start = missing[0]
    run_len = 1
    for prev, curr in zip(missing, missing[1:]):
        if (curr - prev).days <= 3:  # business days are 1 or 3 calendar days apart
            run_len += 1
        else:
            if run_len > max_gap:
                warnings.warn(
                    f"{ticker}: gap of {run_len} business days starting {run_start.date()}",
                    UserWarning,
                    stacklevel=4,
                )
            run_start = curr
            run_len = 1

    if run_len > max_gap:
        warnings.warn(
            f"{ticker}: gap of {run_len} business days starting {run_start.date()}",
            UserWarning,
            stacklevel=4,
        )
