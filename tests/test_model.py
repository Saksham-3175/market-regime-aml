"""Tests for src/model/train.py, predict.py, evaluate.py."""

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from src.model.evaluate import regime_statistics, transition_matrix_display
from src.model.predict import decode_regimes, label_regimes, predict_probabilities
from src.model.train import (
    compute_aic,
    compute_bic,
    load_model,
    model_selection,
    save_model,
    train_hmm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Two-cluster synthetic features — helps EM converge fast."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n)
    state = rng.integers(0, 2, n)
    log_ret = np.where(state == 0,
                       rng.normal(0.001, 0.005, n),
                       rng.normal(-0.002, 0.015, n))
    vol = np.where(state == 0,
                   rng.normal(0.10, 0.01, n),
                   rng.normal(0.25, 0.02, n))
    vr = np.abs(rng.normal(1.0, 0.2, n))
    return pd.DataFrame(
        {"log_return": log_ret, "realized_vol": vol, "volume_ratio": vr},
        index=dates,
    )


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return _make_features()


@pytest.fixture(scope="module")
def config() -> dict:
    return {
        "model": {
            "n_states_range": [2, 3],
            "covariance_type": "full",
            "n_iter": 50,
            "n_seeds": 3,
            "random_state": 42,
        }
    }


@pytest.fixture(scope="module")
def model_3(features, config) -> GaussianHMM:
    return train_hmm(features, n_states=3, config=config)


@pytest.fixture(scope="module")
def regime_names_3(model_3) -> dict[int, str]:
    return label_regimes(model_3)


# ---------------------------------------------------------------------------
# train_hmm
# ---------------------------------------------------------------------------

class TestTrainHmm:
    def test_returns_gaussianhmm(self, features, config):
        m = train_hmm(features, n_states=2, config=config)
        assert isinstance(m, GaussianHMM)

    def test_n_components(self, model_3):
        assert model_3.n_components == 3

    def test_means_shape(self, model_3, features):
        assert model_3.means_.shape == (3, features.shape[1])

    def test_transmat_rows_sum_to_one(self, model_3):
        np.testing.assert_allclose(model_3.transmat_.sum(axis=1), 1.0, atol=1e-6)

    def test_startprob_sums_to_one(self, model_3):
        np.testing.assert_allclose(model_3.startprob_.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_bic / compute_aic
# ---------------------------------------------------------------------------

class TestInfoCriteria:
    def test_bic_is_finite_float(self, model_3, features):
        val = compute_bic(model_3, features)
        assert isinstance(val, float) and np.isfinite(val)

    def test_aic_is_finite_float(self, model_3, features):
        val = compute_aic(model_3, features)
        assert isinstance(val, float) and np.isfinite(val)

    def test_bic_finite(self, model_3, features):
        # BIC = -2*LL + penalty; can be negative for continuous densities
        assert np.isfinite(compute_bic(model_3, features))

    def test_bic_all_k_finite(self, features, config):
        for k in [2, 3]:
            m = train_hmm(features, n_states=k, config=config)
            assert np.isfinite(compute_bic(m, features))


# ---------------------------------------------------------------------------
# model_selection
# ---------------------------------------------------------------------------

class TestModelSelection:
    @pytest.fixture(scope="class")
    def result(self, features, config):
        return model_selection(features, config)

    def test_required_keys(self, result):
        for key in ("best_model", "best_n_states", "bic_scores", "aic_scores", "all_models"):
            assert key in result

    def test_best_n_states_in_range(self, result, config):
        assert result["best_n_states"] in config["model"]["n_states_range"]

    def test_best_is_minimum_bic(self, result):
        best = result["bic_scores"][result["best_n_states"]]
        assert all(best <= v for v in result["bic_scores"].values())

    def test_all_models_trained(self, result, config):
        assert set(result["all_models"].keys()) == set(config["model"]["n_states_range"])


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, model_3, tmp_path):
        path = tmp_path / "hmm.pkl"
        save_model(model_3, path)
        loaded = load_model(path)
        assert isinstance(loaded, GaussianHMM)
        assert loaded.n_components == model_3.n_components
        np.testing.assert_allclose(loaded.means_, model_3.means_, rtol=1e-5)
        np.testing.assert_allclose(loaded.transmat_, model_3.transmat_, rtol=1e-5)


# ---------------------------------------------------------------------------
# decode_regimes
# ---------------------------------------------------------------------------

class TestDecodeRegimes:
    def test_length_matches_input(self, model_3, features):
        assert len(decode_regimes(model_3, features)) == len(features)

    def test_index_matches_features(self, model_3, features):
        labels = decode_regimes(model_3, features)
        assert (labels.index == features.index).all()

    def test_values_within_valid_range(self, model_3, features):
        labels = decode_regimes(model_3, features)
        assert set(labels.unique()).issubset(set(range(model_3.n_components)))

    def test_series_name(self, model_3, features):
        assert decode_regimes(model_3, features).name == "regime"


# ---------------------------------------------------------------------------
# predict_probabilities
# ---------------------------------------------------------------------------

class TestPredictProbabilities:
    def test_shape(self, model_3, features):
        probs = predict_probabilities(model_3, features)
        assert probs.shape == (len(features), model_3.n_components)

    def test_rows_sum_to_one(self, model_3, features):
        probs = predict_probabilities(model_3, features)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_all_non_negative(self, model_3, features):
        probs = predict_probabilities(model_3, features)
        assert (probs.values >= 0).all()

    def test_column_names(self, model_3, features):
        cols = predict_probabilities(model_3, features).columns.tolist()
        assert cols == [f"state_{i}" for i in range(model_3.n_components)]


# ---------------------------------------------------------------------------
# label_regimes
# ---------------------------------------------------------------------------

class TestLabelRegimes:
    def test_three_state_has_three_labels(self, model_3):
        assert len(label_regimes(model_3)) == 3

    def test_three_state_contains_bull_bear_sideways(self, model_3):
        assert set(label_regimes(model_3).values()) == {"Bull", "Bear", "Sideways"}

    def test_bull_highest_mean_return(self, model_3):
        names = label_regimes(model_3)
        bull = [k for k, v in names.items() if v == "Bull"][0]
        bear = [k for k, v in names.items() if v == "Bear"][0]
        assert model_3.means_[bull, 0] > model_3.means_[bear, 0]

    def test_two_state_model(self, features, config):
        m = train_hmm(features, n_states=2, config=config)
        names = label_regimes(m)
        assert set(names.values()) == {"Bull", "Bear"}

    def test_all_states_labelled(self, model_3):
        names = label_regimes(model_3)
        assert set(names.keys()) == set(range(model_3.n_components))


# ---------------------------------------------------------------------------
# regime_statistics
# ---------------------------------------------------------------------------

class TestRegimeStatistics:
    @pytest.fixture(autouse=True)
    def _setup(self, model_3, features, regime_names_3):
        self.labels = decode_regimes(model_3, features)
        self.stats = regime_statistics(self.labels, regime_names_3)

    def test_n_rows_equals_n_states(self, model_3):
        assert len(self.stats) == model_3.n_components

    def test_total_days_sum_equals_n_samples(self, features):
        assert self.stats["total_days"].sum() == len(features)

    def test_pct_time_sums_to_100(self):
        assert abs(self.stats["pct_time"].sum() - 100.0) < 0.5

    def test_avg_duration_positive(self):
        assert (self.stats["avg_duration_days"] > 0).all()

    def test_n_episodes_positive(self):
        assert (self.stats["n_episodes"] > 0).all()


# ---------------------------------------------------------------------------
# transition_matrix_display
# ---------------------------------------------------------------------------

class TestTransitionMatrixDisplay:
    def test_shape(self, model_3, regime_names_3):
        k = model_3.n_components
        assert transition_matrix_display(model_3, regime_names_3).shape == (k, k)

    def test_rows_sum_to_one(self, model_3, regime_names_3):
        df = transition_matrix_display(model_3, regime_names_3)
        np.testing.assert_allclose(df.values.sum(axis=1), 1.0, atol=1e-3)

    def test_column_labels_match_regime_names(self, model_3, regime_names_3):
        df = transition_matrix_display(model_3, regime_names_3)
        expected = {regime_names_3[i] for i in range(model_3.n_components)}
        assert set(df.columns) == expected
