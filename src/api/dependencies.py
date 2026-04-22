"""Shared FastAPI dependencies — model and config loaded once at startup."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from hmmlearn.hmm import GaussianHMM

from src.model.predict import label_regimes
from src.model.train import load_model

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load and cache the YAML config (called once per process)."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {_CONFIG_PATH}")
    return cfg


@lru_cache(maxsize=1)
def get_model() -> GaussianHMM:
    """Load and cache the fitted HMM from disk (called once per process).

    The model path is read from ``config["api"]["model_path"]``.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the configured path.
    """
    cfg = get_config()
    model_path = Path(cfg["api"]["model_path"])

    # Resolve relative paths from the project root
    if not model_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run notebooks/02_model_selection.ipynb or "
            "notebooks/03_mlflow_experiment.ipynb first to train and save a model."
        )

    model = load_model(model_path)
    logger.info(
        f"Model loaded: {model_path.name}  "
        f"(n_states={model.n_components})"
    )
    return model


@lru_cache(maxsize=1)
def get_regime_names() -> dict[int, str]:
    """Return cached regime name mapping for the loaded model."""
    return label_regimes(get_model())


def get_feature_columns() -> list[str]:
    """Feature columns in the order expected by the model."""
    return ["log_return", "realized_vol", "volume_ratio"]
