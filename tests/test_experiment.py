"""Tests for src/model/experiment.py — MLflow experiment tracking."""

import numpy as np
import pandas as pd
import pytest
import mlflow

from src.model.experiment import (
    _log_config_params,
    _log_per_k_metrics,
    _log_best_model_metrics,
    _log_emission_params,
    _log_regime_stats,
    _log_transition_matrix,
    _log_model_artifact,
    _log_feature_metadata,
    _setup_mlflow,
    run_experiment,
    load_model_from_run,
)
from src.model.train import train_hmm
from src.model.predict import decode_regimes, label_regimes


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Two-cluster synthetic features — fast EM convergence."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n)
    state = rng.integers(0, 2, n)
    log_ret = np.where(
        state == 0,
        rng.normal(0.001, 0.005, n),
        rng.normal(-0.002, 0.015, n),
    )
    vol = np.where(
        state == 0,
        rng.normal(0.10, 0.01, n),
        rng.normal(0.25, 0.02, n),
    )
    vr = np.abs(rng.normal(1.0, 0.2, n))
    return pd.DataFrame(
        {"log_return": log_ret, "realized_vol": vol, "volume_ratio": vr},
        index=dates,
    )


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return _make_features()


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> dict:
    tracking_dir = tmp_path_factory.mktemp("mlruns")
    return {
        "data": {
            "ticker":     "^GSPC",
            "start_date": "2000-01-01",
            "end_date":   None,
        },
        "features": {
            "vol_window":    21,
            "volume_window": 21,
        },
        "model": {
            "n_states_range":  [2, 3],
            "covariance_type": "full",
            "n_iter":          50,
            "n_seeds":         3,
            "random_state":    42,
        },
        "mlflow": {
            "tracking_uri":    str(tracking_dir),
            "experiment_name": "test-market-regime-hmm",
        },
    }


@pytest.fixture(scope="module")
def trained_model(features, config):
    return train_hmm(features, n_states=3, config=config)


@pytest.fixture(scope="module")
def regime_names(trained_model):
    return label_regimes(trained_model)


@pytest.fixture(scope="module")
def labels(trained_model, features):
    return decode_regimes(trained_model, features)


@pytest.fixture(scope="module")
def experiment_result(features, config):
    """Run the full experiment once; reuse across tests in this module."""
    return run_experiment(features, config, run_name="test_run")


# ---------------------------------------------------------------------------
# _setup_mlflow
# ---------------------------------------------------------------------------


class TestSetupMlflow:
    def test_returns_experiment_name(self, config):
        name = _setup_mlflow(config)
        assert name == config["mlflow"]["experiment_name"]

    def test_tracking_uri_set(self, config):
        _setup_mlflow(config)
        assert mlflow.get_tracking_uri() == config["mlflow"]["tracking_uri"]


# ---------------------------------------------------------------------------
# _log_config_params
# ---------------------------------------------------------------------------


class TestLogConfigParams:
    def test_logs_without_error(self, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run():
            _log_config_params(config)   # should not raise

    def test_ticker_param_logged(self, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run() as run:
            _log_config_params(config)
        client = mlflow.tracking.MlflowClient()
        params = client.get_run(run.info.run_id).data.params
        assert params["ticker"] == config["data"]["ticker"]

    def test_vol_window_param_logged(self, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run() as run:
            _log_config_params(config)
        client = mlflow.tracking.MlflowClient()
        params = client.get_run(run.info.run_id).data.params
        assert params["vol_window"] == str(config["features"]["vol_window"])


# ---------------------------------------------------------------------------
# _log_per_k_metrics
# ---------------------------------------------------------------------------


class TestLogPerKMetrics:
    def test_logs_without_error(self, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        bic = {2: -1000.0, 3: -1200.0}
        aic = {2: -900.0,  3: -1100.0}
        with mlflow.start_run():
            _log_per_k_metrics(bic, aic)

    def test_metrics_recorded_for_each_k(self, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        bic = {2: -1000.0, 3: -1200.0}
        aic = {2: -900.0,  3: -1100.0}
        with mlflow.start_run() as run:
            _log_per_k_metrics(bic, aic)
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(run.info.run_id, "bic")
        assert len(history) == 2


# ---------------------------------------------------------------------------
# _log_best_model_metrics
# ---------------------------------------------------------------------------


class TestLogBestModelMetrics:
    def test_logs_without_error(self, trained_model, features, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        bic = {3: -1200.0}
        aic = {3: -1100.0}
        with mlflow.start_run():
            _log_best_model_metrics(trained_model, 3, features, bic, aic)

    def test_best_n_states_metric_logged(self, trained_model, features, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        bic = {3: -1200.0}
        aic = {3: -1100.0}
        with mlflow.start_run() as run:
            _log_best_model_metrics(trained_model, 3, features, bic, aic)
        client  = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run.info.run_id).data.metrics
        assert metrics["best_n_states"] == 3.0

    def test_n_training_samples_metric_logged(self, trained_model, features, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        bic = {3: -1200.0}
        aic = {3: -1100.0}
        with mlflow.start_run() as run:
            _log_best_model_metrics(trained_model, 3, features, bic, aic)
        client  = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run.info.run_id).data.metrics
        assert metrics["n_training_samples"] == float(len(features))


# ---------------------------------------------------------------------------
# _log_emission_params
# ---------------------------------------------------------------------------


class TestLogEmissionParams:
    def test_logs_without_error(self, trained_model, regime_names, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        feat_cols = ["log_return", "realized_vol", "volume_ratio"]
        with mlflow.start_run():
            _log_emission_params(trained_model, regime_names, feat_cols)

    def test_correct_number_of_metrics(self, trained_model, regime_names, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        feat_cols = ["log_return", "realized_vol", "volume_ratio"]
        with mlflow.start_run() as run:
            _log_emission_params(trained_model, regime_names, feat_cols)
        client  = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run.info.run_id).data.metrics
        expected = trained_model.n_components * len(feat_cols)
        assert len(metrics) == expected


# ---------------------------------------------------------------------------
# _log_regime_stats
# ---------------------------------------------------------------------------


class TestLogRegimeStats:
    def test_logs_without_error(self, labels, regime_names, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run():
            _log_regime_stats(labels, regime_names)

    def test_pct_time_metrics_present(self, labels, regime_names, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run() as run:
            _log_regime_stats(labels, regime_names)
        client  = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run.info.run_id).data.metrics
        pct_keys = [k for k in metrics if k.startswith("pct_time_")]
        assert len(pct_keys) == len(regime_names)


# ---------------------------------------------------------------------------
# _log_transition_matrix
# ---------------------------------------------------------------------------


class TestLogTransitionMatrix:
    def test_logs_without_error(self, trained_model, regime_names, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run():
            _log_transition_matrix(trained_model, regime_names)

    def test_artifact_exists(self, trained_model, regime_names, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run() as run:
            _log_transition_matrix(trained_model, regime_names)
        client    = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id, path="model_info")
        names     = [a.path for a in artifacts]
        assert any("transition_matrix.csv" in n for n in names)


# ---------------------------------------------------------------------------
# _log_model_artifact
# ---------------------------------------------------------------------------


class TestLogModelArtifact:
    def test_returns_uri_string(self, trained_model, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run():
            uri = _log_model_artifact(trained_model, 3)
        assert isinstance(uri, str) and len(uri) > 0

    def test_pkl_artifact_exists(self, trained_model, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run() as run:
            _log_model_artifact(trained_model, 3)
        client    = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id, path="model")
        names     = [a.path for a in artifacts]
        assert any(".pkl" in n for n in names)


# ---------------------------------------------------------------------------
# _log_feature_metadata
# ---------------------------------------------------------------------------


class TestLogFeatureMetadata:
    def test_logs_without_error(self, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run():
            _log_feature_metadata(["log_return", "realized_vol", "volume_ratio"])

    def test_features_txt_artifact_exists(self, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run() as run:
            _log_feature_metadata(["log_return", "realized_vol", "volume_ratio"])
        client    = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id, path="model_info")
        names     = [a.path for a in artifacts]
        assert any("features.txt" in n for n in names)


# ---------------------------------------------------------------------------
# run_experiment  (integration)
# ---------------------------------------------------------------------------


class TestRunExperiment:
    def test_returns_required_keys(self, experiment_result):
        for key in ("best_model", "best_n_states", "bic_scores",
                    "aic_scores", "all_models", "run_id", "artifact_uri"):
            assert key in experiment_result

    def test_run_id_is_non_empty_string(self, experiment_result):
        assert isinstance(experiment_result["run_id"], str)
        assert len(experiment_result["run_id"]) > 0

    def test_best_n_states_in_range(self, experiment_result, config):
        assert experiment_result["best_n_states"] in config["model"]["n_states_range"]

    def test_bic_scores_all_finite(self, experiment_result):
        for v in experiment_result["bic_scores"].values():
            assert np.isfinite(v)

    def test_aic_scores_all_finite(self, experiment_result):
        for v in experiment_result["aic_scores"].values():
            assert np.isfinite(v)

    def test_run_is_finished(self, experiment_result, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        run    = client.get_run(experiment_result["run_id"])
        assert run.info.status == "FINISHED"

    def test_params_logged(self, experiment_result, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        params = client.get_run(experiment_result["run_id"]).data.params
        assert params["ticker"] == config["data"]["ticker"]

    def test_best_bic_metric_logged(self, experiment_result, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client  = mlflow.tracking.MlflowClient()
        metrics = client.get_run(experiment_result["run_id"]).data.metrics
        assert "best_bic" in metrics

    def test_best_is_minimum_bic(self, experiment_result):
        best_k   = experiment_result["best_n_states"]
        best_bic = experiment_result["bic_scores"][best_k]
        assert all(best_bic <= v for v in experiment_result["bic_scores"].values())

    def test_custom_run_name(self, features, config):
        result = run_experiment(features, config, run_name="custom_name_test")
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        run    = client.get_run(result["run_id"])
        assert run.info.run_name == "custom_name_test"


# ---------------------------------------------------------------------------
# load_model_from_run
# ---------------------------------------------------------------------------


class TestLoadModelFromRun:
    def test_loads_gaussianhmm(self, experiment_result, config):
        from hmmlearn.hmm import GaussianHMM
        model = load_model_from_run(experiment_result["run_id"], config)
        assert isinstance(model, GaussianHMM)

    def test_n_components_matches(self, experiment_result, config):
        model = load_model_from_run(experiment_result["run_id"], config)
        assert model.n_components == experiment_result["best_n_states"]

    def test_means_finite(self, experiment_result, config):
        model = load_model_from_run(experiment_result["run_id"], config)
        assert np.isfinite(model.means_).all()

    def test_invalid_run_id_raises(self, config):
        with pytest.raises(Exception):
            load_model_from_run("nonexistent_run_id_xyz", config)
