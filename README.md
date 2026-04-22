# Market Regime Detection using Hidden Markov Models

Unsupervised probabilistic classification of S&P 500 market states using a Gaussian Hidden Markov Model (HMM). The model learns three latent regimes — **Bull**, **Bear**, and **Sideways** — directly from price action, without any manual labelling.

> **This is a regime classifier, not a price predictor.**  
> It answers: *"What kind of market environment are we in right now?"*

---

## Results

![Regime Overlay](assets/image_ea5209.png)
*S&P 500 price history segmented by inferred hidden states. Green = Bull, Red = Bear, Amber = Sideways.*

![Observation Distributions](assets/image_ea560b.png)
*Feature distributions across the three learned regimes.*

![Model Evaluation](assets/image_ea5663.png)
*BIC sweep confirming k=3 as the optimal number of hidden states.*

---

## Methodology

### Hidden States
Three latent states learned **unsupervised** — labels assigned post-hoc by ranking emission means:

| State | Label | Characteristics |
|-------|-------|-----------------|
| Highest mean log return | **Bull** | Low volatility, normal volume, positive drift |
| Lowest mean log return | **Bear** | High volatility, elevated volume, negative drift |
| Middle mean log return | **Sideways** | Intermediate vol, near-zero return — consolidation |

### Observable Features (Daily)
| Feature | Formula | Window |
|---------|---------|--------|
| `log_return` | `ln(P_t / P_{t-1})` | — |
| `realized_vol` | `std(log_returns) × √252` | 21 trading days |
| `volume_ratio` | `volume / rolling_mean(volume)` | 21 trading days |

### Training
- **Algorithm:** Baum-Welch (Expectation-Maximisation)
- **Decoding:** Viterbi — globally optimal state sequence
- **Model selection:** BIC sweep over k ∈ {2, 3, 4, 5}; lowest BIC wins
- **Robustness:** 10 random EM initialisations per k; best log-likelihood kept
- **Covariance:** Full — captures cross-feature correlations

### Why HMM?
| Evidence | Implication |
|----------|-------------|
| Volatility clusters for months at a time | Temporal persistence → hidden states |
| ACF of squared returns significant through lag 50+ | ARCH effect → variance is serially dependent |
| Fat tails (excess kurtosis >> 0) | Non-Gaussian returns → regime mixing |
| Distinct scatter clusters in feature space | Separable states exist |

---

## Project Structure

```
market-regime-hmm/
├── src/
│   ├── data/
│   │   └── fetch.py              # fetch_ohlcv() — yfinance + parquet cache
│   ├── features/
│   │   └── engineer.py           # log returns, realized vol, volume ratio
│   ├── model/
│   │   ├── train.py              # train_hmm(), model_selection(), BIC/AIC, save/load
│   │   ├── predict.py            # decode_regimes(), predict_probabilities(), label_regimes()
│   │   ├── evaluate.py           # regime_statistics(), transition_matrix_display()
│   │   └── experiment.py         # run_experiment(), load_model_from_run() — MLflow
│   ├── api/
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── dependencies.py       # get_model(), get_config() — LRU cached
│   │   └── main.py               # FastAPI: /health, /model-info, /transition-matrix, /regimes
│   └── dashboard/
│       └── app.py                # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb              # Price history, return dist, rolling vol, ACF, scatter
│   ├── 02_model_selection.ipynb  # BIC sweep, best model inspection, regime chart
│   └── 03_mlflow_experiment.ipynb # MLflow run, metric inspection, artifact round-trip
├── configs/
│   └── config.yaml               # All hyperparameters — never hardcoded in source
├── tests/
│   ├── test_features.py          # 20 tests — feature shape, NaNs, values, columns
│   ├── test_model.py             # 35 tests — train, BIC, decode, posteriors, labels, stats
│   ├── test_experiment.py        # 30 tests — MLflow params, metrics, artifacts, round-trip
│   └── test_api.py               # 35 tests — health, model-info, transition-matrix, regimes
├── scripts/
│   └── e2e_test.sh               # End-to-end test script (real data + Docker)
├── data/raw/                     # Parquet cache (gitignored — auto-created on first run)
├── models/                       # Saved model artifacts (gitignored)
├── mlflow/                       # MLflow tracking store (gitignored)
├── outputs/                      # Plot outputs
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/market-regime-hmm.git
cd market-regime-hmm

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

Run the model selection notebook, or use this one-liner:

```bash
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
print(f"Saved: models/hmm_{results['best_n_states']}state.pkl")
EOF
```

Data is fetched automatically from Yahoo Finance and cached as parquet under `data/raw/`.

> **Note:** `curl-cffi` is required to bypass Yahoo Finance TLS fingerprint detection.

### 3. Start the full stack

```bash
docker-compose up -d
```

| Service | URL |
|---------|-----|
| FastAPI docs | http://localhost:8000/docs |
| FastAPI health | http://localhost:8000/health |
| Streamlit dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

Or run services individually without Docker:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
streamlit run src/dashboard/app.py --server.port 8501
mlflow ui --backend-store-uri mlflow --port 5000
```

### 4. Run the notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Exploratory analysis — confirms regimes exist before modelling |
| `02_model_selection.ipynb` | BIC sweep, best model inspection, regime chart |
| `03_mlflow_experiment.ipynb` | Full tracked experiment with MLflow |

### 5. Run the tests

```bash
pytest tests/ -v
```

120 tests — all use synthetic data, no network calls, no yfinance dependency.

### 6. End-to-end test (real data + Docker)

```bash
chmod +x scripts/e2e_test.sh
./scripts/e2e_test.sh
```

---

## API Reference

### `GET /health`
```json
{"status": "ok", "model_loaded": true, "n_states": 3, "api_version": "1.0.0"}
```

### `GET /model-info`
Returns emission means, regime names, covariance type, initial state distribution.

### `GET /transition-matrix`
```json
{
  "best_n_states": 3,
  "regime_names": {"0": "Bear", "1": "Sideways", "2": "Bull"},
  "matrix": [[0.97, 0.02, 0.01], [0.01, 0.96, 0.03], [0.01, 0.02, 0.97]],
  "row_labels": ["Bear", "Sideways", "Bull"],
  "col_labels": ["Bear", "Sideways", "Bull"]
}
```

### `POST /regimes`
```json
// Request
{
  "rows": [
    {"date": "2024-01-02", "open": 4700, "high": 4750, "low": 4680, "close": 4742, "volume": 3500000},
    ...
  ]
}

// Response
{
  "ticker": "^GSPC",
  "n_samples": 59,
  "best_n_states": 3,
  "regime_names": {"0": "Bear", "1": "Sideways", "2": "Bull"},
  "regimes": [
    {"date": "2024-02-01", "regime": "Bull", "state_id": 2,
     "prob_bull": 0.94, "prob_bear": 0.02, "prob_sideways": 0.04},
    ...
  ]
}
```

---

## Configuration

All hyperparameters live in `configs/config.yaml`:

```yaml
data:
  ticker: "^GSPC"
  start_date: "2000-01-01"
  end_date: null              # null → today

features:
  vol_window: 21              # ~1 trading month
  volume_window: 21

model:
  n_states_range: [2, 3, 4, 5]
  covariance_type: "full"
  n_iter: 100
  n_seeds: 10
  random_state: 42

mlflow:
  tracking_uri: "mlflow"
  experiment_name: "market-regime-hmm"

api:
  host: "0.0.0.0"
  port: 8000
  model_path: "models/hmm_3state.pkl"
  title: "Market Regime HMM API"
  version: "1.0.0"
```

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data pipeline + feature engineering | ✅ Complete |
| 2 | Exploratory data analysis notebook | ✅ Complete |
| 3 | HMM training, Viterbi decoding, BIC model selection | ✅ Complete |
| 4 | MLflow experiment tracking | ✅ Complete |
| 5 | FastAPI inference API | ✅ Complete |
| 6 | Streamlit dashboard | ✅ Complete |
| 7 | Docker + docker-compose | ✅ Complete |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `hmmlearn` | 0.3.3 | GaussianHMM — core model |
| `yfinance` | 0.2.54 | S&P 500 OHLCV data |
| `curl-cffi` | 0.14.0 | Chrome TLS impersonation for yfinance |
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.4.3 | Numerical computing |
| `scipy` | 1.17.1 | Statistical analysis |
| `mlflow` | 3.10.1 | Experiment tracking |
| `fastapi` | 0.135.1 | Inference API |
| `uvicorn` | 0.42.0 | ASGI server |
| `streamlit` | 1.55.0 | Dashboard |
| `joblib` | 1.5.3 | Model serialisation |
| `pytest` | 9.0.2 | Testing |
| `httpx` | 0.28.1 | FastAPI TestClient |
