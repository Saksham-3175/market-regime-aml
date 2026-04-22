"""Pydantic request and response schemas for the Market Regime HMM API."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class OHLCVRow(BaseModel):
    """A single day of OHLCV data supplied by the caller."""

    date: date = Field(..., description="Trading date (YYYY-MM-DD).")
    open: float = Field(..., gt=0, description="Opening price.")
    high: float = Field(..., gt=0, description="Daily high price.")
    low: float = Field(..., gt=0, description="Daily low price.")
    close: float = Field(..., gt=0, description="Closing / adjusted price.")
    volume: float = Field(..., ge=0, description="Traded volume (0 for indices).")


# ---------------------------------------------------------------------------
# /regimes
# ---------------------------------------------------------------------------


class RegimesRequest(BaseModel):
    """Payload for POST /regimes."""

    rows: list[OHLCVRow] = Field(
        ...,
        min_length=30,
        description=(
            "Chronologically ordered OHLCV rows.  "
            "Minimum 30 rows required to compute rolling features."
        ),
    )

    @field_validator("rows")
    @classmethod
    def rows_must_be_sorted(cls, v: list[OHLCVRow]) -> list[OHLCVRow]:
        dates = [r.date for r in v]
        if dates != sorted(dates):
            raise ValueError("rows must be in ascending chronological order.")
        return v


class RegimeDay(BaseModel):
    """Regime classification for a single trading day."""

    date: date
    regime: str = Field(..., description="Bull | Bear | Sideways")
    state_id: int = Field(..., description="Raw HMM state index (0-based).")
    prob_bull: float = Field(..., ge=0.0, le=1.0)
    prob_bear: float = Field(..., ge=0.0, le=1.0)
    prob_sideways: float = Field(..., ge=0.0, le=1.0)


class RegimesResponse(BaseModel):
    """Response from POST /regimes."""

    ticker: str
    n_samples: int
    best_n_states: int
    regime_names: dict[int, str] = Field(
        ..., description="Mapping of state index → regime label."
    )
    regimes: list[RegimeDay]


# ---------------------------------------------------------------------------
# /transition-matrix
# ---------------------------------------------------------------------------


class TransitionMatrixResponse(BaseModel):
    """Response from GET /transition-matrix."""

    best_n_states: int
    regime_names: dict[int, str]
    matrix: list[list[float]] = Field(
        ...,
        description=(
            "Row-stochastic transition matrix.  "
            "matrix[i][j] = P(next state = j | current state = i)."
        ),
    )
    row_labels: list[str] = Field(..., description="From-state labels (rows).")
    col_labels: list[str] = Field(..., description="To-state labels (columns).")


# ---------------------------------------------------------------------------
# /model-info
# ---------------------------------------------------------------------------


class EmissionState(BaseModel):
    """Emission parameters for a single hidden state."""

    state_id: int
    regime_name: str
    mean_log_return: float
    mean_realized_vol: float
    mean_volume_ratio: float
    annualised_return_pct: float = Field(
        ..., description="mean_log_return × 252 × 100 — annualised (%)."
    )
    annualised_vol_pct: float = Field(
        ..., description="mean_realized_vol × 100 — already annualised (%)."
    )


class ModelInfoResponse(BaseModel):
    """Response from GET /model-info."""

    api_version: str
    model_path: str
    n_states: int
    covariance_type: str
    feature_columns: list[str]
    regime_names: dict[int, str]
    emission_states: list[EmissionState]
    startprob: list[float] = Field(..., description="Initial state distribution.")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = Field(..., description="'ok' when the service is healthy.")
    model_loaded: bool
    n_states: int | None
    api_version: str
