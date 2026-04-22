# Market Regime Detection — HMM

## Project Overview
Unsupervised market regime classification using Hidden Markov Models with Gaussian emissions.
Training on S&P 500 daily data. Deployed via FastAPI + Streamlit + Docker.

## Tech Stack
- Python 3.12 (venv at ~/Code/ml-venv)
- hmmlearn 0.3.3 (GaussianHMM)
- yfinance 0.2.54 + curl-cffi 0.14.0 (data — see Important Notes)
- pandas 2.3.3, numpy 2.4.3, scipy 1.17.1 (features + analysis)
- matplotlib 3.10.8, seaborn 0.13.2, plotly 6.6.0 (visualization)
- statsmodels 0.14.6 (ACF plots)
- FastAPI 0.135.1 + uvicorn 0.42.0 (API)
- Streamlit 1.55.0 (dashboard)
- MLflow 3.10.1 (experiment tracking)
- Docker + docker-compose (containerization)
- joblib 1.5.3 (model serialization)
- pyarrow 23.0.1 (parquet cache)
- pytest 9.0.2 (testing)
- httpx 0.28.1 (FastAPI TestClient)

## Repo Structure
```
market-regime-hmm/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── fetch.py          # fetch_ohlcv() — yfinance + parquet cache + gap validation
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py       # log returns, realized vol, volume ratio, build_feature_matrix()
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py          # train_hmm(), model_selection(), compute_bic/aic(), save/load
│   │   ├── predict.py        # decode_regimes() Viterbi, predict_probabilities(), label_regimes()
│   │   ├── evaluate.py       # regime_statistics(), transition_matrix_display()
│   │   └── experiment.py     # run_experiment(), load_model_from_run() — MLflow tracking
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py        # Pydantic request/response models
│   │   ├── dependencies.py   # get_model(), get_config(), get_regime_names() — LRU cached
│   │   └── main.py           # FastAPI: /health, /model-info, /transition-matrix, /regimes
│   └── dashboard/
│       ├── __init__.py
│       └── app.py            # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb              # price history, return dist, rolling vol, ACF, scatter
│   ├── 02_model_selection.ipynb  # BIC sweep, best model inspection, regime chart
│   └── 03_mlflow_experiment.ipynb  # MLflow run, metric inspection, artifact round-trip
├── configs/
│   └── config.yaml           # ticker, date range, feature windows, model hyperparams, mlflow, api
├── tests/
│   ├── conftest.py
│   ├── test_features.py      # 20 tests — shape, no NaNs, columns, finite values
│   ├── test_model.py         # 35 tests — train, BIC, decode, posteriors, label, stats
│   ├── test_experiment.py    # 30 tests — MLflow params, metrics, artifacts, round-trip
│   └── test_api.py           # 35 tests — health, model-info, transition-matrix, regimes
├── data/
│   └── raw/                  # parquet cache (gitignored)
├── models/                   # saved model artifacts (gitignored)
├── mlflow/                   # MLflow tracking store (gitignored)
├── outputs/                  # plot outputs
├── scripts/
│   └── e2e_test.sh           # end-to-end test script (real data + Docker)
├── conftest.py               # sys.path fix for pytest
├── requirements.txt          # pinned versions
├── Dockerfile
├── docker-compose.yml
├── CLAUDE.md
└── README.md
```

## Key Design Decisions
- 3 hidden states (Bull/Bear/Sideways) — justified via BIC sweep, not assumed
- Features: log_return, realized_vol (21d), volume_ratio (21d) — minimal, interpretable
- GaussianHMM with full covariance — captures cross-feature correlations
- S&P 500 (^GSPC) as market proxy — standard quant convention
- Model selection: sweep n_states in [2,3,4,5], pick lowest BIC
- Viterbi for decoding — globally optimal state sequence, not just filtering
- Multi-seed training (n_seeds=10) — EM is non-convex, best ll wins
- MLflow tracking URI is a local directory (`mlflow/`) — no server needed for dev
- FastAPI dependencies use `lru_cache` — model loaded once at startup, not per-request
- Pydantic v2 schemas with field validators — input validation before any ML code runs
- Streamlit dashboard uses `@st.cache_data` / `@st.cache_resource` — no redundant retraining

## Conventions
- All source in src/ with proper __init__.py files
- Config-driven: all hyperparams in configs/config.yaml, never hardcoded
- Type hints on all public functions
- Docstrings on all public functions
- f-strings for formatting
- logging module, not print statements
- pytest for testing, synthetic data in fixtures (no live network calls)
- Black for formatting, isort for imports

## Current Phase
**Phases 1–7 complete.** Ready for end-to-end verification and commit.

Completed:
- Phase 1: Data pipeline + feature engineering ✓
- Phase 2: EDA notebook (01_eda.ipynb) ✓
- Phase 3: HMM training, Viterbi decoding, BIC model selection, model evaluation ✓
- Phase 4: MLflow experiment tracking ✓
- Phase 5: FastAPI inference API ✓
- Phase 6: Streamlit dashboard ✓
- Phase 7: Docker + docker-compose ✓

Next:
- Run scripts/e2e_test.sh — verify all sections green
- Push Docker image to DockerHub
- Verify Streamlit dashboard manually at http://localhost:8501
- Final commit + tag v1.0.0

## Important Notes

### Running the full stack
```bash
# Train model first (required before API/dashboard start)
# Run notebooks/02_model_selection.ipynb  OR:
python - <<'EOF'
import yaml
from src.data.fetch import fetch_ohlcv
from src.features.engineer import build_feature_matrix
from src.model.train import model_selection, save_model

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

df       = fetch_ohlcv(config["data"]["ticker"], start=config["data"]["start_date"])
features = build_feature_matrix(df, config)
results  = model_selection(features, config)
save_model(results["best_model"], f"models/hmm_{results['best_n_states']}state.pkl")
EOF

# Start everything with Docker Compose
docker-compose up -d

# Or run services individually:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
streamlit run src/dashboard/app.py --server.port 8501
mlflow ui --backend-store-uri mlflow --port 5000
```

### Service URLs
| Service | URL |
|---------|-----|
| FastAPI docs | http://localhost:8000/docs |
| FastAPI health | http://localhost:8000/health |
| Streamlit dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

### Running tests
```bash
pytest tests/ -v                    # all 120 tests, synthetic data only
pytest tests/test_api.py -v         # API tests only
pytest tests/test_experiment.py -v  # MLflow tests only
```

### End-to-end test (real data + Docker)
```bash
chmod +x scripts/e2e_test.sh
./scripts/e2e_test.sh
```

### API endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Liveness check — model_loaded, n_states, version |
| GET | /model-info | Emission means, regime names, startprob, covariance type |
| GET | /transition-matrix | Row-stochastic matrix with regime name labels |
| POST | /regimes | Classify OHLCV rows → per-day regime + posterior probs |

### Model must exist before starting API
The API loads the model from `configs/config.yaml → api.model_path`.
Run `notebooks/02_model_selection.ipynb` or `notebooks/03_mlflow_experiment.ipynb`
first to train and save the model to `models/hmm_3state.pkl`.
If the model file is missing, `/health` returns `model_loaded=false` and
`/regimes`, `/model-info`, `/transition-matrix` return HTTP 503.

### yfinance + curl-cffi (critical)
Yahoo Finance performs TLS fingerprint detection and stalls connections from Python/urllib3
(the request hangs indefinitely — no error, no timeout). Fix: inject a curl-cffi Chrome
session into yfinance:
```python
from curl_cffi import requests as curl_requests
session = curl_requests.Session(impersonate="chrome110")
yf.download(..., session=session)
```
This is already wired into `src/data/fetch.py` via the module-level `_SESSION`. Do not
remove it or revert to plain requests.

### hmmlearn GaussianHMM
- Expects input shape (n_samples, n_features) — always verify before `.fit()`
- `model.score(X)` returns **per-sample average** log-likelihood (LL / n_samples),
  NOT total LL. Multiply by n to get total LL for BIC/AIC.
- EM is non-convex — run multiple seeds, keep best log-likelihood

### Log returns vs raw returns
Log returns are additive across time and closer to Gaussian — use `ln(P_t / P_{t-1})`.

### BIC formula
```
BIC = -2 * total_LL + n_params * ln(n_samples)
    where total_LL = model.score(X) * n_samples
    and   n_params = (k-1) + k*(k-1) + k*d + k*d*(d+1)//2
```
BIC can be **negative** for continuous Gaussian emissions — the LL term can dominate.
Lower BIC is always better regardless of sign.

### Regime label assignment
State indices (0, 1, 2) from hmmlearn are arbitrary. `label_regimes()` assigns
Bull/Bear/Sideways post-hoc by ranking states on their mean log_return emission:
- highest mean → Bull
- lowest mean  → Bear
- middle        → Sideways

### MLflow tracking
- Tracking URI is a local directory (`mlflow/`) — add to `.gitignore`
- `model.score(X)` returns per-sample LL; multiply by n before logging total LL
- BIC/AIC logged with `step=k` so the MLflow UI renders a curve over k
- Model artifact stored under `model/hmm_Nstate.pkl` inside the run
- `load_model_from_run()` downloads the artifact to a temp dir and deserializes it
- To open the UI: `mlflow ui --backend-store-uri mlflow --port 5000`

### Docker
- Single image serves both API (port 8000) and Dashboard (port 8501)
- Models and data are mounted as volumes — not baked into the image
- docker-compose brings up API + Dashboard + MLflow UI in one command
- Health check on API container: `curl -f http://localhost:8000/health`
- Dashboard depends_on API with condition: service_healthy

### Data cache
Real ^GSPC data (2000-01-01 → today) is cached at `data/raw/GSPC_2000-01-01_<date>.parquet`.
Once cached, all subsequent runs skip yfinance entirely. Do not delete this file during
development — re-downloading risks another TLS rate-limit.

### yfinance NaN quirks
- NaN Volume: filled with 0.0 (common for index tickers on some dates)
- NaN Close: row dropped, count logged as warning
- MultiIndex columns: flattened via `raw.columns.get_level_values(0)`
- Timezone-aware index: converted to tz-naive UTC
