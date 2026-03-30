"""Regime statistics and labelled transition matrix display."""

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


def regime_statistics(
    labels: pd.Series,
    regime_names: dict[int, str],
) -> pd.DataFrame:
    """Per-regime summary statistics.

    Parameters
    ----------
    labels:
        Integer state label series from ``decode_regimes``.
    regime_names:
        ``{state_id: name}`` mapping from ``label_regimes``.

    Returns
    -------
    pd.DataFrame
        Indexed by regime name, columns:
        ``[total_days, pct_time, n_episodes, avg_duration_days]``.
    """
    n_total = len(labels)
    rows = []

    for state_id, name in sorted(regime_names.items()):
        mask = (labels == state_id).values

        total_days = int(mask.sum())
        pct_time = round(total_days / n_total * 100, 1)

        # Compute run lengths via diff on the boolean array
        durations: list[int] = []
        run = 0
        for v in mask:
            if v:
                run += 1
            elif run > 0:
                durations.append(run)
                run = 0
        if run > 0:
            durations.append(run)

        n_episodes = len(durations)
        avg_duration = round(float(np.mean(durations)) if durations else 0.0, 1)

        rows.append(
            {
                "regime": name,
                "total_days": total_days,
                "pct_time": pct_time,
                "n_episodes": n_episodes,
                "avg_duration_days": avg_duration,
            }
        )

    return pd.DataFrame(rows).set_index("regime")


def transition_matrix_display(
    model: GaussianHMM,
    regime_names: dict[int, str],
) -> pd.DataFrame:
    """Transition probability matrix as a labelled DataFrame.

    Parameters
    ----------
    model:
        Fitted GaussianHMM.
    regime_names:
        ``{state_id: name}`` mapping from ``label_regimes``.

    Returns
    -------
    pd.DataFrame
        Rows = from-state, columns = to-state.  Values are probabilities
        rounded to 4 decimal places.  Rows sum to 1.0.
    """
    names = [regime_names[i] for i in range(model.n_components)]
    return pd.DataFrame(
        np.round(model.transmat_, 4),
        index=pd.Index(names, name="from \\ to"),
        columns=names,
    )
