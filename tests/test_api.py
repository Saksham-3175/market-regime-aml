"""Tests for src/api/main.py — FastAPI inference API."""

from __future__ import annotations

import importlib
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
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


def _make_ohlcv_rows(n: int = 80, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 3000.0 * np.exp(rng.normal(0, 0.01, n).cumsum())
    rows = []
    for dt, c in zip(dates, close):
        rows.append({
            "date":   str(dt.date()),
            "open":   round(float(c * 0.999), 2),
            "high":   round(float(c * 1.002), 2),
            "low":    round(float(c * 0.997), 2),
            "close":  round(float(c), 2),
            "volume": float(rng.integers(1_000_000, 10_000_000)),
        })
    return rows


def _make_test_config(model_path: str, mlruns_path: str) -> dict:
    return {
        "data": {
            "ticker":     "^GSPC",
            "start_date": "2000-01-01",
            "end_date":   None,
        },
        "features": {"vol_window": 21, "volume_window": 21},
        "model": {
            "n_states_range":  [2, 3],
            "covariance_type": "full",
            "n_iter":          50,
            "n_seeds":         3,
            "random_state":    42,
        },
        "mlflow": {
            "tracking_uri":    mlruns_path,
            "experiment_name": "test",
        },
        "api": {
            "host":       "0.0.0.0",
            "port":       8000,
            "model_path": model_path,
            "title":      "Test API",
            "version":    "1.0.0",
        },
    }


# ---------------------------------------------------------------------------
# Module-scoped fixture — train model once, save to disk, build TestClient
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    Strategy: train a tiny model, save it to a temp path, then patch
    get_config so dependencies.get_model() loads from that temp path.
    This avoids fighting with lru_cache and module-level _cfg.
    """
    from fastapi.testclient import TestClient
    from src.model.train import train_hmm, save_model
    from src.model.predict import label_regimes
    import src.api.dependencies as deps

    tmp = tmp_path_factory.mktemp("api_test")
    model_pkl = str(tmp / "hmm_3state.pkl")
    mlruns    = str(tmp / "mlruns")
    cfg       = _make_test_config(model_pkl, mlruns)

    # Train and save model to disk so deps.get_model() can load it
    features      = _make_synthetic_features()
    trained_model = train_hmm(features, n_states=3, config=cfg)
    save_model(trained_model, model_pkl)
    names = label_regimes(trained_model)

    # Clear all LRU caches before patching
    deps.get_config.cache_clear()
    deps.get_model.cache_clear()
    deps.get_regime_names.cache_clear()

    # Patch get_config to return our test config.
    # get_model and get_regime_names will work naturally because
    # get_config now returns the right model_path.
    with mock.patch("src.api.dependencies.get_config", return_value=cfg), \
         mock.patch("src.api.main.get_config",         return_value=cfg):

        # Reload main so _cfg = get_config() picks up the patched version
        import src.api.main as main_module
        importlib.reload(main_module)

        # Also patch the dependency functions used inside the endpoints
        with mock.patch("src.api.main.get_config",      return_value=cfg), \
             mock.patch("src.api.main.get_model",        return_value=trained_model), \
             mock.patch("src.api.main.get_regime_names", return_value=names):

            with TestClient(main_module.app) as c:
                c._test_model = trained_model
                c._test_names = names
                c._test_cfg   = cfg
                yield c

    # Cleanup
    deps.get_config.cache_clear()
    deps.get_model.cache_clear()
    deps.get_regime_names.cache_clear()


@pytest.fixture(scope="module")
def ohlcv_rows() -> list[dict]:
    return _make_ohlcv_rows(n=80)


@pytest.fixture(scope="module")
def short_ohlcv_rows() -> list[dict]:
    return _make_ohlcv_rows(n=10)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_status_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_ok_or_degraded(self, client):
        assert client.get("/health").json()["status"] in ("ok", "degraded")

    def test_model_loaded_field_present(self, client):
        assert "model_loaded" in client.get("/health").json()

    def test_api_version_present(self, client):
        assert "api_version" in client.get("/health").json()

    def test_response_has_required_keys(self, client):
        data = client.get("/health").json()
        assert {"status", "model_loaded", "api_version"}.issubset(data.keys())


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_status_200(self, client):
        assert client.get("/model-info").status_code == 200

    def test_n_states_present_and_positive(self, client):
        data = client.get("/model-info").json()
        assert "n_states" in data
        assert data["n_states"] > 0

    def test_feature_columns_correct(self, client):
        data = client.get("/model-info").json()
        assert data["feature_columns"] == [
            "log_return", "realized_vol", "volume_ratio"
        ]

    def test_regime_names_present(self, client):
        assert "regime_names" in client.get("/model-info").json()

    def test_emission_states_count_matches_n_states(self, client):
        data = client.get("/model-info").json()
        assert len(data["emission_states"]) == data["n_states"]

    def test_emission_state_has_all_fields(self, client):
        state = client.get("/model-info").json()["emission_states"][0]
        for field in (
            "state_id", "regime_name", "mean_log_return",
            "mean_realized_vol", "mean_volume_ratio",
            "annualised_return_pct", "annualised_vol_pct",
        ):
            assert field in state, f"Missing field: {field}"

    def test_startprob_sums_to_one(self, client):
        sp = client.get("/model-info").json()["startprob"]
        assert abs(sum(sp) - 1.0) < 1e-5

    def test_startprob_length_matches_n_states(self, client):
        data = client.get("/model-info").json()
        assert len(data["startprob"]) == data["n_states"]

    def test_covariance_type_present(self, client):
        assert "covariance_type" in client.get("/model-info").json()

    def test_api_version_present(self, client):
        assert "api_version" in client.get("/model-info").json()


# ---------------------------------------------------------------------------
# GET /transition-matrix
# ---------------------------------------------------------------------------


class TestTransitionMatrix:
    def test_status_200(self, client):
        assert client.get("/transition-matrix").status_code == 200

    def test_matrix_is_square(self, client):
        data = client.get("/transition-matrix").json()
        n = data["best_n_states"]
        assert len(data["matrix"]) == n
        assert all(len(row) == n for row in data["matrix"])

    def test_rows_sum_to_one(self, client):
        for row in client.get("/transition-matrix").json()["matrix"]:
            assert abs(sum(row) - 1.0) < 1e-4

    def test_all_values_non_negative(self, client):
        for row in client.get("/transition-matrix").json()["matrix"]:
            assert all(v >= 0 for v in row)

    def test_row_labels_length(self, client):
        data = client.get("/transition-matrix").json()
        assert len(data["row_labels"]) == data["best_n_states"]

    def test_col_labels_length(self, client):
        data = client.get("/transition-matrix").json()
        assert len(data["col_labels"]) == data["best_n_states"]

    def test_labels_match_regime_names(self, client):
        data = client.get("/transition-matrix").json()
        regime_values = set(data["regime_names"].values())
        assert set(data["row_labels"]) == regime_values
        assert set(data["col_labels"]) == regime_values


# ---------------------------------------------------------------------------
# POST /regimes
# ---------------------------------------------------------------------------


class TestRegimes:
    def test_status_200(self, client, ohlcv_rows):
        r = client.post("/regimes", json={"rows": ohlcv_rows})
        assert r.status_code == 200

    def test_n_samples_positive(self, client, ohlcv_rows):
        assert client.post("/regimes", json={"rows": ohlcv_rows}).json()["n_samples"] > 0

    def test_regimes_list_length_matches_n_samples(self, client, ohlcv_rows):
        data = client.post("/regimes", json={"rows": ohlcv_rows}).json()
        assert len(data["regimes"]) == data["n_samples"]

    def test_regime_field_valid_values(self, client, ohlcv_rows):
        valid = {"Bull", "Bear", "Sideways"}
        for day in client.post("/regimes", json={"rows": ohlcv_rows}).json()["regimes"]:
            assert day["regime"] in valid

    def test_probabilities_sum_to_one(self, client, ohlcv_rows):
        for day in client.post("/regimes", json={"rows": ohlcv_rows}).json()["regimes"]:
            total = day["prob_bull"] + day["prob_bear"] + day["prob_sideways"]
            assert abs(total - 1.0) < 1e-4

    def test_probabilities_non_negative(self, client, ohlcv_rows):
        for day in client.post("/regimes", json={"rows": ohlcv_rows}).json()["regimes"]:
            assert day["prob_bull"]     >= 0
            assert day["prob_bear"]     >= 0
            assert day["prob_sideways"] >= 0

    def test_date_field_present(self, client, ohlcv_rows):
        for day in client.post("/regimes", json={"rows": ohlcv_rows}).json()["regimes"]:
            assert "date" in day

    def test_state_id_in_valid_range(self, client, ohlcv_rows):
        data = client.post("/regimes", json={"rows": ohlcv_rows}).json()
        n = data["best_n_states"]
        for day in data["regimes"]:
            assert 0 <= day["state_id"] < n

    def test_regime_names_in_response(self, client, ohlcv_rows):
        assert "regime_names" in client.post("/regimes", json={"rows": ohlcv_rows}).json()

    def test_best_n_states_positive(self, client, ohlcv_rows):
        assert client.post("/regimes", json={"rows": ohlcv_rows}).json()["best_n_states"] > 0

    def test_ticker_in_response(self, client, ohlcv_rows):
        assert "ticker" in client.post("/regimes", json={"rows": ohlcv_rows}).json()

    def test_too_few_rows_returns_422(self, client, short_ohlcv_rows):
        assert client.post("/regimes", json={"rows": short_ohlcv_rows}).status_code == 422

    def test_empty_rows_returns_422(self, client):
        assert client.post("/regimes", json={"rows": []}).status_code == 422

    def test_unsorted_rows_returns_422(self, client, ohlcv_rows):
        shuffled = list(reversed(ohlcv_rows))
        assert client.post("/regimes", json={"rows": shuffled}).status_code == 422

    def test_negative_close_returns_422(self, client, ohlcv_rows):
        bad = [dict(row) for row in ohlcv_rows]
        bad[5]["close"] = -100.0
        assert client.post("/regimes", json={"rows": bad}).status_code == 422


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


class TestDocs:
    def test_openapi_json_accessible(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_docs_accessible(self, client):
        assert client.get("/docs").status_code == 200

    def test_redoc_accessible(self, client):
        assert client.get("/redoc").status_code == 200
