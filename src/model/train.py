"""HMM training with multi-seed restarts and BIC-based model selection."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


def _n_params(n_states: int, n_features: int) -> int:
    """Free parameters in a GaussianHMM with full covariance.

    Breakdown:
      initial distribution : k - 1
      transition matrix    : k * (k - 1)
      means                : k * d
      full covariances     : k * d * (d + 1) / 2
    Total = (k - 1) + k*(k - 1) + k*d + k*d*(d+1)/2
          = k² - 1 + k*d + k*d*(d+1)/2
    """
    k, d = n_states, n_features
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def compute_bic(model: GaussianHMM, features: pd.DataFrame) -> float:
    """BIC = -2 * log_likelihood + n_params * ln(n_samples).

    hmmlearn's ``score`` returns the per-sample average log-likelihood,
    so total log-likelihood = score * n_samples.
    """
    X = features.values
    n = len(X)
    ll = model.score(X) * n
    p = _n_params(model.n_components, X.shape[1])
    return -2.0 * ll + p * np.log(n)


def compute_aic(model: GaussianHMM, features: pd.DataFrame) -> float:
    """AIC = -2 * log_likelihood + 2 * n_params."""
    X = features.values
    n = len(X)
    ll = model.score(X) * n
    p = _n_params(model.n_components, X.shape[1])
    return -2.0 * ll + 2.0 * p


def train_hmm(
    features: pd.DataFrame,
    n_states: int,
    config: dict,
) -> GaussianHMM:
    """Train a GaussianHMM with multiple random seeds; return best fit.

    Runs ``config["model"]["n_seeds"]`` independent EM initialisations
    and keeps the model with the highest log-likelihood to avoid local
    optima.

    Parameters
    ----------
    features:
        Feature matrix, shape ``(n_samples, n_features)``.  No NaNs.
    n_states:
        Number of hidden states.
    config:
        Full config dict.  Uses ``config["model"]["n_iter"]``,
        ``["n_seeds"]``, ``["random_state"]``, ``["covariance_type"]``.

    Returns
    -------
    GaussianHMM
        Best-fitting model across all random seeds.

    Raises
    ------
    RuntimeError
        If every seed fails to converge.
    """
    X = features.values
    n_iter = config["model"]["n_iter"]
    n_seeds = config["model"]["n_seeds"]
    base_seed = config["model"]["random_state"]
    cov_type = config["model"]["covariance_type"]

    best_model: GaussianHMM | None = None
    best_ll = -np.inf

    for i in range(n_seeds):
        seed = base_seed + i
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_iter=n_iter,
            random_state=seed,
            verbose=False,
        )
        try:
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll = ll
                best_model = model
                logger.debug(f"  seed={seed}: ll={ll:.4f}  ← best")
            else:
                logger.debug(f"  seed={seed}: ll={ll:.4f}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  seed={seed} failed: {exc}")

    if best_model is None:
        raise RuntimeError(
            f"All {n_seeds} seeds failed for n_states={n_states}"
        )

    logger.info(
        f"n_states={n_states}: best ll={best_ll:.4f}  ({n_seeds} seeds)"
    )
    return best_model


def model_selection(
    features: pd.DataFrame,
    config: dict,
) -> dict:
    """Sweep n_states_range; return best model by BIC.

    Parameters
    ----------
    features:
        Feature matrix, shape ``(n_samples, n_features)``.
    config:
        Full config dict.  Uses ``config["model"]["n_states_range"]``.

    Returns
    -------
    dict
        Keys: ``best_model``, ``best_n_states``, ``bic_scores``,
        ``aic_scores``, ``all_models``.
    """
    n_states_range: list[int] = config["model"]["n_states_range"]

    bic_scores: dict[int, float] = {}
    aic_scores: dict[int, float] = {}
    all_models: dict[int, GaussianHMM] = {}

    for k in n_states_range:
        logger.info(f"Model selection — k={k} ...")
        model = train_hmm(features, n_states=k, config=config)
        bic_scores[k] = compute_bic(model, features)
        aic_scores[k] = compute_aic(model, features)
        all_models[k] = model
        logger.info(
            f"  k={k}  BIC={bic_scores[k]:,.2f}  AIC={aic_scores[k]:,.2f}"
        )

    best_k = min(bic_scores, key=bic_scores.__getitem__)
    logger.info(f"Best model: k={best_k} (BIC={bic_scores[best_k]:,.2f})")

    return {
        "best_model": all_models[best_k],
        "best_n_states": best_k,
        "bic_scores": bic_scores,
        "aic_scores": aic_scores,
        "all_models": all_models,
    }


def save_model(model: GaussianHMM, path: str | Path) -> None:
    """Persist a fitted GaussianHMM to disk with joblib.

    Parameters
    ----------
    model:
        Fitted GaussianHMM instance.
    path:
        Destination file path (e.g. ``"models/hmm_3state.pkl"``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str | Path) -> GaussianHMM:
    """Load a GaussianHMM from a joblib file.

    Parameters
    ----------
    path:
        Path to the saved model file.

    Returns
    -------
    GaussianHMM
        The loaded model.
    """
    model: GaussianHMM = joblib.load(Path(path))
    logger.info(f"Model loaded ← {path}")
    return model
