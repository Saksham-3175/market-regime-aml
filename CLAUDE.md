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
│   │   └── evaluate.py       # regime_statistics(), transition_matrix_display()
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py           # FastAPI: /regimes, /transition-matrix, /model-info, /health
│   └── dashboard/
│       ├── __init__.py
│       └── app.py            # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb          # price history, return dist, rolling vol, ACF, scatter
│   └── 02_model_selection.ipynb  # BIC sweep, best model inspection, regime chart
├── configs/
│   └── config.yaml           # ticker, date range, feature windows, model hyperparams
├── tests/
│   ├── conftest.py
│   ├── test_features.py      # 20 tests — shape, no NaNs, columns, finite values
│   └── test_model.py         # 35 tests — train, BIC, decode, posteriors, label, stats
├── data/
│   └── raw/                  # parquet cache (gitignored)
├── models/                   # saved model artifacts (gitignored)
├── outputs/                  # plot outputs
├── mlflow/
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
**Phase 3 complete.** Model training pipeline (train, predict, evaluate) built and tested.

Completed:
- Phase 1: Data pipeline + feature engineering ✓
- Phase 2: EDA notebook (01_eda.ipynb) ✓
- Phase 3: HMM training, Viterbi decoding, BIC model selection, model evaluation ✓

Next:
- Phase 4: MLflow experiment tracking
- Phase 5: FastAPI inference API
- Phase 6: Streamlit dashboard
- Phase 7: Docker + compose
- Phase 8: README + polish

## Important Notes

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

### Data cache
Real ^GSPC data (2000-01-01 → today) is cached at `data/raw/GSPC_2000-01-01_<date>.parquet`.
Once cached, all subsequent runs skip yfinance entirely. Do not delete this file during
development — re-downloading risks another TLS rate-limit.

### yfinance NaN quirks
- NaN Volume: filled with 0.0 (common for index tickers on some dates)
- NaN Close: row dropped, count logged as warning
- MultiIndex columns: flattened via `raw.columns.get_level_values(0)`
- Timezone-aware index: converted to tz-naive UTC
