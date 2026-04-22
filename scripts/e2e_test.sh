#!/usr/bin/env bash
# e2e_test.sh — end-to-end verification of the Market Regime HMM stack
#
# Sections:
#   1. Unit tests (pytest, synthetic data only)
#   2. Train model on real data
#   3. Start API, test all 4 endpoints
#   4. Docker build + compose up, re-test endpoints
#
# Usage:
#   chmod +x scripts/e2e_test.sh
#   ./scripts/e2e_test.sh
#
# Requirements: venv activated, Docker + docker-compose installed, network access for yfinance

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

FAILURES=0
API_PID=""

# ── Cleanup trap ───────────────────────────────────────────────────────────
cleanup() {
    if [[ -n "$API_PID" ]]; then
        info "Stopping API (pid $API_PID) ..."
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    if docker compose ps --quiet 2>/dev/null | grep -q .; then
        info "Bringing down Docker Compose ..."
        docker compose down --remove-orphans 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── 1. Unit tests ──────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════"
echo "  Section 1 — Unit tests (pytest)"
echo "════════════════════════════════════════════"

if pytest tests/ -v --tb=short -q 2>&1 | tee /tmp/pytest_out.txt; then
    pass "All pytest tests passed"
else
    fail "pytest suite failed — check /tmp/pytest_out.txt"
fi

# ── 2. Train model ─────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════"
echo "  Section 2 — Train model (real data)"
echo "════════════════════════════════════════════"

info "Training HMM on ^GSPC (2000-01-01 → today) ..."
python - <<'PYEOF'
import yaml
from src.data.fetch import fetch_ohlcv
from src.features.engineer import build_feature_matrix
from src.model.train import model_selection, save_model
import pathlib

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

df       = fetch_ohlcv(config["data"]["ticker"], start=config["data"]["start_date"])
features = build_feature_matrix(df, config)
results  = model_selection(features, config)

pathlib.Path("models").mkdir(exist_ok=True)
model_path = f"models/hmm_{results['best_n_states']}state.pkl"
save_model(results["best_model"], model_path)
print(f"Saved model → {model_path}  (best_k={results['best_n_states']}  BIC={results['bic_scores'][results['best_n_states']]:.2f})")
PYEOF

if [[ -f "models/hmm_3state.pkl" ]] || ls models/hmm_*.pkl 1>/dev/null 2>&1; then
    pass "Model trained and saved"
else
    fail "No model file found in models/"
fi

# ── 3. API tests ───────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════"
echo "  Section 3 — FastAPI endpoint tests"
echo "════════════════════════════════════════════"

info "Starting API server ..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level warning &
API_PID=$!

# Wait for API to be ready (up to 30s)
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    fail "API did not start within 30s"
else
    pass "API is up"
fi

# GET /health
HEALTH=$(curl -sf http://localhost:8000/health)
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok', d" 2>/dev/null; then
    pass "GET /health → status=ok"
else
    fail "GET /health returned unexpected response: $HEALTH"
fi

# GET /model-info
if curl -sf http://localhost:8000/model-info | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'n_states' in d, d" 2>/dev/null; then
    pass "GET /model-info → n_states present"
else
    fail "GET /model-info failed"
fi

# GET /transition-matrix
if curl -sf http://localhost:8000/transition-matrix | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'matrix' in d, d" 2>/dev/null; then
    pass "GET /transition-matrix → matrix present"
else
    fail "GET /transition-matrix failed"
fi

# POST /regimes — minimal 35-row OHLCV payload
PAYLOAD=$(python3 - <<'PYEOF'
import json, datetime
rows = []
price = 4000.0
for i in range(35):
    d = (datetime.date(2024, 1, 2) + datetime.timedelta(days=i)).isoformat()
    rows.append({"date": d, "open": price, "high": price * 1.005, "low": price * 0.995,
                 "close": price * (1 + 0.001 * (i % 3 - 1)), "volume": 3_000_000_000})
    price = rows[-1]["close"]
print(json.dumps({"rows": rows}))
PYEOF
)

if curl -sf -X POST http://localhost:8000/regimes \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'regimes' in d, d" 2>/dev/null; then
    pass "POST /regimes → regimes present"
else
    fail "POST /regimes failed"
fi

# Stop local API
kill "$API_PID" 2>/dev/null || true
wait "$API_PID" 2>/dev/null || true
API_PID=""

# ── 4. Docker tests ────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════"
echo "  Section 4 — Docker Compose stack"
echo "════════════════════════════════════════════"

if ! command -v docker &>/dev/null; then
    info "Docker not found — skipping Docker section"
else
    info "Building Docker image ..."
    if docker compose build --quiet 2>&1 | tail -5; then
        pass "Docker image built"
    else
        fail "Docker build failed"
    fi

    info "Starting Docker Compose stack ..."
    docker compose up -d

    # Wait for API healthcheck (up to 60s)
    for i in $(seq 1 60); do
        STATUS=$(docker compose ps --format json api 2>/dev/null \
            | python3 -c "import sys,json; rows=sys.stdin.read().strip().splitlines(); print(json.loads(rows[0]).get('Health','') if rows else '')" 2>/dev/null || true)
        if [[ "$STATUS" == "healthy" ]]; then
            break
        fi
        sleep 2
    done

    if [[ "$STATUS" == "healthy" ]]; then
        pass "Docker API container is healthy"
    else
        fail "Docker API container did not become healthy within 120s"
    fi

    # Re-run endpoint checks via Docker
    if curl -sf http://localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
        pass "Docker GET /health → ok"
    else
        fail "Docker GET /health failed"
    fi

    if curl -sf http://localhost:8000/model-info | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'n_states' in d" 2>/dev/null; then
        pass "Docker GET /model-info → ok"
    else
        fail "Docker GET /model-info failed"
    fi

    info "Bringing Docker Compose stack down ..."
    docker compose down --remove-orphans
    pass "Docker Compose down"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════"
if [[ $FAILURES -eq 0 ]]; then
    echo -e "  ${GREEN}ALL SECTIONS PASSED${NC}"
else
    echo -e "  ${RED}$FAILURES SECTION(S) FAILED${NC}"
fi
echo "════════════════════════════════════════════"

exit $FAILURES
