"""FastAPI application — Market Regime HMM inference API.

Endpoints
---------
GET  /health              — liveness check
GET  /model-info          — model metadata, emission params
GET  /transition-matrix   — labelled transition probability matrix
POST /regimes             — classify regimes for a supplied OHLCV series
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.api.dependencies import (
    get_config,
    get_feature_columns,
    get_model,
    get_regime_names,
)
from src.api.schemas import (
    EmissionState,
    HealthResponse,
    ModelInfoResponse,
    RegimeDay,
    RegimesRequest,
    RegimesResponse,
    TransitionMatrixResponse,
)
from src.features.engineer import build_feature_matrix
from src.model.predict import decode_regimes, predict_probabilities

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_cfg = get_config()

app = FastAPI(
    title=_cfg["api"]["title"],
    version=_cfg["api"]["version"],
    description=(
        "Unsupervised market regime classification using a Gaussian HMM. "
        "Classifies S&P 500 (or any equity index) daily OHLCV data into "
        "Bull, Bear, and Sideways regimes."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Startup — eager model load so the first request is not slow
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _startup() -> None:
    """Pre-load model and config into the LRU caches at startup."""
    try:
        get_model()
        get_regime_names()
        logger.info("Startup: model pre-loaded successfully.")
    except FileNotFoundError as exc:
        # Log but don't crash — /health will report model_loaded=False
        logger.error(f"Startup: model not found — {exc}")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Monitoring"],
)
def health() -> HealthResponse:
    """Return service health status and whether the model is loaded."""
    try:
        model = get_model()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            n_states=model.n_components,
            api_version=_cfg["api"]["version"],
        )
    except FileNotFoundError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            n_states=None,
            api_version=_cfg["api"]["version"],
        )


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata and emission parameters",
    tags=["Model"],
)
def model_info() -> ModelInfoResponse:
    """Return model metadata: n_states, emission means, regime names, startprob."""
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    regime_names = get_regime_names()
    feat_cols = get_feature_columns()
    cfg = get_config()

    emission_states = []
    for i in range(model.n_components):
        mean_ret = float(model.means_[i, 0])
        mean_vol = float(model.means_[i, 1])
        mean_vr = float(model.means_[i, 2])
        emission_states.append(
            EmissionState(
                state_id=i,
                regime_name=regime_names[i],
                mean_log_return=mean_ret,
                mean_realized_vol=mean_vol,
                mean_volume_ratio=mean_vr,
                annualised_return_pct=round(mean_ret * 252 * 100, 4),
                annualised_vol_pct=round(mean_vol * 100, 4),
            )
        )

    return ModelInfoResponse(
        api_version=cfg["api"]["version"],
        model_path=cfg["api"]["model_path"],
        n_states=model.n_components,
        covariance_type=model.covariance_type,
        feature_columns=feat_cols,
        regime_names=regime_names,
        emission_states=emission_states,
        startprob=model.startprob_.tolist(),
    )


# ---------------------------------------------------------------------------
# GET /transition-matrix
# ---------------------------------------------------------------------------


@app.get(
    "/transition-matrix",
    response_model=TransitionMatrixResponse,
    summary="Labelled transition probability matrix",
    tags=["Model"],
)
def transition_matrix() -> TransitionMatrixResponse:
    """Return the row-stochastic transition matrix with regime name labels."""
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    regime_names = get_regime_names()
    labels = [regime_names[i] for i in range(model.n_components)]

    return TransitionMatrixResponse(
        best_n_states=model.n_components,
        regime_names=regime_names,
        matrix=np.round(model.transmat_, 6).tolist(),
        row_labels=labels,
        col_labels=labels,
    )


# ---------------------------------------------------------------------------
# POST /regimes
# ---------------------------------------------------------------------------


@app.post(
    "/regimes",
    response_model=RegimesResponse,
    summary="Classify market regimes for a supplied OHLCV series",
    tags=["Inference"],
)
def regimes(request: RegimesRequest) -> RegimesResponse:
    """Classify each trading day in the supplied OHLCV series into a regime.

    The caller supplies a chronologically ordered list of OHLCV rows.
    The API computes features (log_return, realized_vol, volume_ratio),
    runs Viterbi decoding, and returns per-day regime labels with
    posterior probabilities.

    **Minimum 30 rows required** to compute the 21-day rolling features
    and have at least a few decoded days.
    """
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    regime_names = get_regime_names()
    cfg = get_config()

    # ── Build DataFrame from request rows ──────────────────────────────────
    records = [
        {
            "Date": pd.Timestamp(r.date),
            "Open": r.open,
            "High": r.high,
            "Low": r.low,
            "Close": r.close,
            "Volume": r.volume,
        }
        for r in request.rows
    ]
    df = pd.DataFrame(records).set_index("Date")
    df.index = pd.DatetimeIndex(df.index)

    # ── Feature engineering ────────────────────────────────────────────────
    try:
        features = build_feature_matrix(df, cfg)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature engineering failed: {exc}",
        ) from exc

    if len(features) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Feature matrix is empty after rolling warm-up. "
                "Supply at least 30 rows."
            ),
        )

    # ── Decode regimes ─────────────────────────────────────────────────────
    labels = decode_regimes(model, features)
    probs_df = predict_probabilities(model, features)

    # Build a name → state_id reverse map for probability lookup
    name_to_id = {v: k for k, v in regime_names.items()}
    bull_col = f"state_{name_to_id.get('Bull', 0)}"
    bear_col = f"state_{name_to_id.get('Bear', 1)}"
    side_col = f"state_{name_to_id.get('Sideways', 2)}"

    # Gracefully handle 2-state models (no Sideways)
    def _prob(col: str, idx: int) -> float:
        return float(probs_df[col].iloc[idx]) if col in probs_df.columns else 0.0

    regime_days = []
    for i, (ts, state_id) in enumerate(labels.items()):
        regime_days.append(
            RegimeDay(
                date=ts.date(),
                regime=regime_names[int(state_id)],
                state_id=int(state_id),
                prob_bull=_prob(bull_col, i),
                prob_bear=_prob(bear_col, i),
                prob_sideways=_prob(side_col, i),
            )
        )

    return RegimesResponse(
        ticker=cfg["data"]["ticker"],
        n_samples=len(features),
        best_n_states=model.n_components,
        regime_names=regime_names,
        regimes=regime_days,
    )


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=_cfg["api"]["host"],
        port=_cfg["api"]["port"],
        reload=False,
        log_level="info",
    )
