"""Regime decoding and labeling for a fitted GaussianHMM."""

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


def decode_regimes(
    model: GaussianHMM,
    features: pd.DataFrame,
) -> pd.Series:
    """Viterbi decoding — globally optimal hidden state sequence.

    Parameters
    ----------
    model:
        Fitted GaussianHMM.
    features:
        Feature matrix with the same columns used for training.

    Returns
    -------
    pd.Series
        Integer state labels (0 … k-1), indexed by ``features.index``,
        named ``"regime"``.
    """
    labels = model.predict(features.values)
    return pd.Series(labels, index=features.index, name="regime")


def predict_probabilities(
    model: GaussianHMM,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Forward-backward posterior state probabilities.

    Parameters
    ----------
    model:
        Fitted GaussianHMM.
    features:
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_samples, n_states)``, columns ``["state_0", "state_1", …]``.
        Rows sum to 1.0.
    """
    _, posteriors = model.score_samples(features.values)
    cols = [f"state_{i}" for i in range(model.n_components)]
    return pd.DataFrame(posteriors, index=features.index, columns=cols)


def label_regimes(model: GaussianHMM) -> dict[int, str]:
    """Map state indices → Bull / Bear / Sideways by mean log_return.

    The first feature column is assumed to be ``log_return`` (as produced
    by ``build_feature_matrix``).  States are ranked by their emission mean
    on that feature:

    * highest mean → **Bull**
    * lowest mean  → **Bear**
    * middle       → **Sideways** (k = 3 only)

    For k = 2 the middle label is absent.
    For k > 3 extra middle states get ``"Sideways_1"``, ``"Sideways_2"``, …

    Parameters
    ----------
    model:
        Fitted GaussianHMM.

    Returns
    -------
    dict[int, str]
        ``{state_index: regime_name}`` for all states.
    """
    means = model.means_[:, 0]        # log_return emission mean per state
    ranked = np.argsort(means)        # ascending → ranked[0] = lowest mean

    k = model.n_components
    labels: dict[int, str] = {}

    if k == 1:
        labels[ranked[0]] = "Bull"
    elif k == 2:
        labels[ranked[0]] = "Bear"
        labels[ranked[1]] = "Bull"
    elif k == 3:
        labels[ranked[0]] = "Bear"
        labels[ranked[1]] = "Sideways"
        labels[ranked[2]] = "Bull"
    else:
        labels[ranked[0]] = "Bear"
        labels[ranked[-1]] = "Bull"
        for i, idx in enumerate(ranked[1:-1], start=1):
            labels[idx] = f"Sideways_{i}"

    return labels
