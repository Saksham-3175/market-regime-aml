"""MLflow experiment tracking for HMM model selection."""

import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.model.evaluate import regime_statistics, transition_matrix_display
from src.model.predict import decode_regimes, label_regimes
from src.model.train import (
    compute_aic,
    compute_bic,
    load_model,
    model_selection,
    save_model,
)

logger = logging.getLogger(__name__)


def _setup_mlflow(config: dict) -> str:
    """Configure MLflow tracking URI and return the experiment name.

    Parameters
    ----------
    config:
        Full config dict.  Uses ``config["mlflow"]["tracking_uri"]``
        and ``config["mlflow"]["experiment_name"]``.

    Returns
    -------
    str
        The experiment name.
    """
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI : {tracking_uri}")
    logger.info(f"MLflow experiment   : {experiment_name}")

    return experiment_name


def _log_config_params(config: dict) -> None:
    """Log all config values as MLflow params (flat key=value strings).

    Parameters
    ----------
    config:
        Full config dict.
    """
    mlflow.log_param("ticker",        config["data"]["ticker"])
    mlflow.log_param("start_date",    config["data"]["start_date"])
    mlflow.log_param("end_date",      config["data"]["end_date"] or "today")
    mlflow.log_param("vol_window",    config["features"]["vol_window"])
    mlflow.log_param("volume_window", config["features"]["volume_window"])
    mlflow.log_param("covariance_type", config["model"]["covariance_type"])
    mlflow.log_param("n_iter",        config["model"]["n_iter"])
    mlflow.log_param("n_seeds",       config["model"]["n_seeds"])
    mlflow.log_param("random_state",  config["model"]["random_state"])
    mlflow.log_param("n_states_range", str(config["model"]["n_states_range"]))


def _log_per_k_metrics(
    bic_scores: dict[int, float],
    aic_scores: dict[int, float],
) -> None:
    """Log BIC and AIC for every candidate k as MLflow metrics.

    Uses ``step=k`` so the MLflow UI renders a curve over k.

    Parameters
    ----------
    bic_scores:
        ``{n_states: bic_value}`` from ``model_selection``.
    aic_scores:
        ``{n_states: aic_value}`` from ``model_selection``.
    """
    for k in sorted(bic_scores):
        mlflow.log_metric("bic", bic_scores[k], step=k)
        mlflow.log_metric("aic", aic_scores[k], step=k)
        logger.info(f"  k={k}  BIC={bic_scores[k]:,.2f}  AIC={aic_scores[k]:,.2f}")


def _log_best_model_metrics(
    best_model: GaussianHMM,
    best_k: int,
    features: pd.DataFrame,
    bic_scores: dict[int, float],
    aic_scores: dict[int, float],
) -> None:
    """Log scalar summary metrics for the winning model.

    Parameters
    ----------
    best_model:
        Fitted GaussianHMM with the lowest BIC.
    best_k:
        Number of states in the best model.
    features:
        Feature matrix used for training.
    bic_scores:
        Full BIC sweep results.
    aic_scores:
        Full AIC sweep results.
    """
    X = features.values
    n = len(X)
    total_ll = best_model.score(X) * n

    mlflow.log_metric("best_n_states",           best_k)
    mlflow.log_metric("best_bic",                bic_scores[best_k])
    mlflow.log_metric("best_aic",                aic_scores[best_k])
    mlflow.log_metric("best_total_log_likelihood", total_ll)
    mlflow.log_metric("n_training_samples",      n)
    mlflow.log_metric("n_features",              features.shape[1])


def _log_emission_params(
    best_model: GaussianHMM,
    regime_names: dict[int, str],
    feature_cols: list[str],
) -> None:
    """Log per-state emission means as MLflow metrics.

    Parameters
    ----------
    best_model:
        Fitted GaussianHMM.
    regime_names:
        ``{state_id: name}`` from ``label_regimes``.
    feature_cols:
        List of feature column names (e.g. ``["log_return", ...]``).
    """
    for state_id in range(best_model.n_components):
        name = regime_names[state_id]
        for feat_idx, feat_name in enumerate(feature_cols):
            mean_val = float(best_model.means_[state_id, feat_idx])
            mlflow.log_metric(
                f"mean_{feat_name}_{name.lower()}",
                mean_val,
            )


def _log_regime_stats(
    labels: pd.Series,
    regime_names: dict[int, str],
) -> None:
    """Log per-regime duration statistics as MLflow metrics.

    Parameters
    ----------
    labels:
        Integer state label series from ``decode_regimes``.
    regime_names:
        ``{state_id: name}`` from ``label_regimes``.
    """
    stats = regime_statistics(labels, regime_names)
    for regime_name, row in stats.iterrows():
        prefix = str(regime_name).lower()
        mlflow.log_metric(f"pct_time_{prefix}",     float(row["pct_time"]))
        mlflow.log_metric(f"avg_duration_{prefix}", float(row["avg_duration_days"]))
        mlflow.log_metric(f"n_episodes_{prefix}",   float(row["n_episodes"]))
        mlflow.log_metric(f"total_days_{prefix}",   float(row["total_days"]))


def _log_transition_matrix(
    best_model: GaussianHMM,
    regime_names: dict[int, str],
) -> None:
    """Log the transition matrix as a CSV artifact.

    Parameters
    ----------
    best_model:
        Fitted GaussianHMM.
    regime_names:
        ``{state_id: name}`` from ``label_regimes``.
    """
    trans = transition_matrix_display(best_model, regime_names)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "transition_matrix.csv"
        trans.to_csv(csv_path)
        mlflow.log_artifact(str(csv_path), artifact_path="model_info")
    logger.info("Logged transition matrix as artifact.")


def _log_model_artifact(
    best_model: GaussianHMM,
    best_k: int,
) -> str:
    """Save the best model to a temp file and log it as an MLflow artifact.

    Parameters
    ----------
    best_model:
        Fitted GaussianHMM.
    best_k:
        Number of states — used to name the artifact file.

    Returns
    -------
    str
        The artifact URI where the model was logged.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / f"hmm_{best_k}state.pkl"
        save_model(best_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
    uri = mlflow.get_artifact_uri("model")
    logger.info(f"Model artifact logged → {uri}")
    return uri


def _log_feature_metadata(feature_cols: list[str]) -> None:
    """Log feature column names as a plain-text artifact.

    Parameters
    ----------
    feature_cols:
        List of feature column names.
    """
    with tempfile.TemporaryDirectory() as tmp:
        feat_path = Path(tmp) / "features.txt"
        feat_path.write_text("\n".join(feature_cols))
        mlflow.log_artifact(str(feat_path), artifact_path="model_info")


def run_experiment(
    features: pd.DataFrame,
    config: dict,
    run_name: str | None = None,
) -> dict:
    """Run the full BIC sweep and log everything to MLflow.

    This is the single entry-point for Phase 4.  It:

    1. Configures MLflow (tracking URI + experiment).
    2. Starts a new run (or resumes one if called inside an active run).
    3. Logs all config params.
    4. Runs ``model_selection`` (BIC sweep over ``n_states_range``).
    5. Logs per-k BIC/AIC metrics (step = k → renders as a curve).
    6. Logs best-model scalar metrics (total LL, best k, best BIC/AIC).
    7. Logs per-state emission means.
    8. Logs per-regime duration statistics.
    9. Logs the transition matrix as a CSV artifact.
    10. Logs the best model as a joblib artifact.
    11. Logs feature column names as a text artifact.

    Parameters
    ----------
    features:
        Feature matrix, shape ``(n_samples, n_features)``.  No NaNs.
    config:
        Full config dict (data + features + model + mlflow sections).
    run_name:
        Optional human-readable name for the MLflow run.
        Defaults to ``"hmm_sweep_k{min}_{max}"``.

    Returns
    -------
    dict
        Same structure as ``model_selection``:
        ``{best_model, best_n_states, bic_scores, aic_scores, all_models}``,
        plus two extra keys:

        * ``"run_id"``       — MLflow run ID string
        * ``"artifact_uri"`` — URI of the logged model artifact
    """
    _setup_mlflow(config)

    n_states_range: list[int] = config["model"]["n_states_range"]
    if run_name is None:
        run_name = f"hmm_sweep_k{min(n_states_range)}_{max(n_states_range)}"

    feature_cols = features.columns.tolist()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}  ({run_name})")

        # ── 1. params ──────────────────────────────────────────────────────
        _log_config_params(config)

        # ── 2. model selection sweep ───────────────────────────────────────
        logger.info("Starting model selection sweep …")
        results = model_selection(features, config)

        best_model: GaussianHMM = results["best_model"]
        best_k: int             = results["best_n_states"]
        bic_scores: dict[int, float] = results["bic_scores"]
        aic_scores: dict[int, float] = results["aic_scores"]

        # ── 3. per-k metrics ───────────────────────────────────────────────
        _log_per_k_metrics(bic_scores, aic_scores)

        # ── 4. best model scalar metrics ───────────────────────────────────
        _log_best_model_metrics(best_model, best_k, features, bic_scores, aic_scores)

        # ── 5. emission means ──────────────────────────────────────────────
        regime_names = label_regimes(best_model)
        _log_emission_params(best_model, regime_names, feature_cols)

        # ── 6. regime duration stats ───────────────────────────────────────
        labels = decode_regimes(best_model, features)
        _log_regime_stats(labels, regime_names)

        # ── 7. transition matrix artifact ──────────────────────────────────
        _log_transition_matrix(best_model, regime_names)

        # ── 8. model artifact ──────────────────────────────────────────────
        artifact_uri = _log_model_artifact(best_model, best_k)

        # ── 9. feature metadata artifact ───────────────────────────────────
        _log_feature_metadata(feature_cols)

        logger.info(
            f"MLflow run complete: run_id={run_id}  best_k={best_k}  "
            f"BIC={bic_scores[best_k]:,.2f}"
        )

    return {
        **results,
        "run_id":        run_id,
        "artifact_uri":  artifact_uri,
    }


def load_model_from_run(run_id: str, config: dict) -> GaussianHMM:
    """Load the best model artifact from a completed MLflow run.

    Parameters
    ----------
    run_id:
        MLflow run ID string (visible in the UI or returned by
        ``run_experiment``).
    config:
        Full config dict — used to set the tracking URI.

    Returns
    -------
    GaussianHMM
        The fitted model that was logged during that run.

    Raises
    ------
    FileNotFoundError
        If no model artifact or .pkl file is found for the given run.
    """
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, path="model")

    if not artifacts:
        raise FileNotFoundError(
            f"No model artifact found in run {run_id} under path 'model'."
        )

    with tempfile.TemporaryDirectory() as tmp:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=tmp,
        )
        pkl_files = list(Path(local_dir).glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(
                f"No .pkl file found in artifact 'model' for run {run_id}."
            )
        model = load_model(pkl_files[0])

    logger.info(f"Loaded model from MLflow run {run_id}")
    return model
