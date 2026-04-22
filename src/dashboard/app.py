"""Streamlit dashboard — Market Regime HMM.

Layout
------
Sidebar  : ticker, date range, model path, n_states
Main     : regime timeline (price + posteriors), transition matrix heatmap,
           emission parameters table, regime statistics table
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from src.data.fetch import fetch_ohlcv
from src.features.engineer import build_feature_matrix
from src.model.evaluate import regime_statistics, transition_matrix_display
from src.model.predict import decode_regimes, label_regimes, predict_probabilities
from src.model.train import load_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path("configs/config.yaml")

_REGIME_COLOURS: dict[str, str] = {
    "Bull":     "#2ecc71",
    "Bear":     "#e74c3c",
    "Sideways": "#95a5a6",
}
_DEFAULT_COLOUR = "#bdc3c7"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_config() -> dict:
    """Load config.yaml once per session."""
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


@st.cache_resource
def _load_model(model_path: str):
    """Load the trained GaussianHMM once per session.

    Returns None if the model file does not exist.
    """
    p = Path(model_path)
    if not p.exists():
        return None
    return load_model(p)


@st.cache_data
def _fetch_features(ticker: str, start_date: str, end_date: str | None, _config: dict):
    """Fetch OHLCV and compute feature matrix. Cached by (ticker, start, end)."""
    df = fetch_ohlcv(ticker, start=start_date, end=end_date)
    features = build_feature_matrix(df, _config)
    return df, features


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _regime_segments(
    labels: pd.Series,
    regime_names: dict[int, str],
) -> list[dict]:
    """Convert a label series to a list of contiguous regime segments.

    Returns
    -------
    list[dict]  Each dict has keys: start, end, regime (str)
    """
    segments: list[dict] = []
    if labels.empty:
        return segments

    dates = labels.index
    prev_regime = regime_names[int(labels.iloc[0])]
    seg_start = dates[0]

    for i in range(1, len(labels)):
        curr_regime = regime_names[int(labels.iloc[i])]
        if curr_regime != prev_regime:
            segments.append({"start": seg_start, "end": dates[i - 1], "regime": prev_regime})
            seg_start = dates[i]
            prev_regime = curr_regime

    segments.append({"start": seg_start, "end": dates[-1], "regime": prev_regime})
    return segments


def _build_timeline_figure(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    probs: pd.DataFrame,
    regime_names: dict[int, str],
) -> go.Figure:
    """Build a two-subplot Plotly figure: price timeline + posterior probs."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=("S&P 500 Close — Regime Overlay", "Posterior Probabilities"),
    )

    # ── Top: Close price ──────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=ohlcv["Close"],
            mode="lines",
            line={"color": "#2c3e50", "width": 1.2},
            name="Close",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # ── Regime background bands ────────────────────────────────────────────
    already_shown: set[str] = set()
    for seg in _regime_segments(labels, regime_names):
        colour = _REGIME_COLOURS.get(seg["regime"], _DEFAULT_COLOUR)
        show = seg["regime"] not in already_shown
        already_shown.add(seg["regime"])
        fig.add_vrect(
            x0=seg["start"],
            x1=seg["end"],
            fillcolor=colour,
            opacity=0.18,
            layer="below",
            line_width=0,
            annotation_text=seg["regime"] if show else "",
            annotation_position="top left",
            annotation_font_size=9,
            row=1,
            col=1,
        )

    # ── Bottom: stacked posterior probabilities ────────────────────────────
    # Build one trace per regime, sorted Bull / Sideways / Bear so Bull is on top
    ordered = sorted(regime_names.items(), key=lambda kv: kv[1])  # alphabetical fallback
    # Prefer explicit ordering
    _pref = {"Bull": 0, "Sideways": 1, "Bear": 2}
    ordered = sorted(regime_names.items(), key=lambda kv: _pref.get(kv[1], 99))

    cumulative = np.zeros(len(probs))
    for state_id, rname in ordered:
        col_name = f"state_{state_id}"
        if col_name not in probs.columns:
            continue
        vals = probs[col_name].values
        colour = _REGIME_COLOURS.get(rname, _DEFAULT_COLOUR)
        fig.add_trace(
            go.Scatter(
                x=probs.index,
                y=cumulative + vals,
                mode="lines",
                line={"width": 0},
                fill="tonexty" if state_id != ordered[0][0] else "tozeroy",
                fillcolor=colour,
                name=f"P({rname})",
                opacity=0.7,
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        cumulative = cumulative + vals

    fig.update_layout(
        height=620,
        margin={"l": 50, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hovermode="x unified",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def _build_heatmap(trans_df: pd.DataFrame) -> go.Figure:
    """Build a Plotly heatmap for the transition matrix."""
    labels = list(trans_df.columns)
    z = trans_df.values

    text = [[f"{v:.4f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar={"title": "Probability"},
        )
    )
    fig.update_layout(
        title="Transition Matrix",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        height=380,
        margin={"l": 80, "r": 20, "t": 60, "b": 60},
    )
    return fig


# ---------------------------------------------------------------------------
# Emission parameters table
# ---------------------------------------------------------------------------


def _emission_table(model, regime_names: dict[int, str]) -> pd.DataFrame:
    """Build a tidy DataFrame of per-state emission statistics."""
    rows = []
    for state_id, rname in sorted(regime_names.items()):
        mean_ret = float(model.means_[state_id, 0])
        mean_vol = float(model.means_[state_id, 1])
        mean_vr  = float(model.means_[state_id, 2])
        rows.append(
            {
                "Regime":             rname,
                "Mean Log Return":    f"{mean_ret:.6f}",
                "Ann. Return (%)":    f"{mean_ret * 252 * 100:.2f}%",
                "Ann. Vol (%)":       f"{mean_vol * 100:.2f}%",
                "Mean Volume Ratio":  f"{mean_vr:.4f}",
                "Start Prob":         f"{float(model.startprob_[state_id]):.4f}",
            }
        )
    return pd.DataFrame(rows).set_index("Regime")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Market Regime HMM",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    st.title("Market Regime Detection — Gaussian HMM")
    st.caption("Unsupervised regime classification of S&P 500 daily returns via Hidden Markov Model")

    cfg = _load_config()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        ticker     = cfg["data"]["ticker"]
        start_date = cfg["data"]["start_date"]
        end_date   = cfg["data"].get("end_date") or "today"
        model_path = cfg["api"]["model_path"]

        st.metric("Ticker",      ticker)
        st.metric("Start Date",  start_date)
        st.metric("End Date",    end_date)
        st.metric("Model Path",  model_path)
        st.metric("Vol Window",  f'{cfg["features"]["vol_window"]}d')
        st.metric("Vol+Volume Window", f'{cfg["features"]["volume_window"]}d')

    # ── Load model ────────────────────────────────────────────────────────
    model = _load_model(model_path)

    if model is None:
        st.error(
            f"**Model not found at `{model_path}`.**  "
            "Run `notebooks/02_model_selection.ipynb` or "
            "`notebooks/03_mlflow_experiment.ipynb` to train and save a model first."
        )
        st.stop()

    regime_names = label_regimes(model)

    with st.sidebar:
        st.divider()
        st.metric("N States",       model.n_components)
        st.metric("Covariance Type", model.covariance_type)
        for sid, rname in sorted(regime_names.items()):
            colour = _REGIME_COLOURS.get(rname, _DEFAULT_COLOUR)
            st.markdown(
                f'<span style="color:{colour}; font-weight:bold;">'
                f"State {sid} → {rname}</span>",
                unsafe_allow_html=True,
            )

    # ── Fetch data & compute features ─────────────────────────────────────
    with st.spinner("Loading market data and computing features…"):
        try:
            ohlcv, features = _fetch_features(
                ticker,
                start_date,
                cfg["data"].get("end_date"),
                cfg,
            )
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            st.stop()

    # ── Decode regimes ────────────────────────────────────────────────────
    labels = decode_regimes(model, features)
    probs  = predict_probabilities(model, features)

    # ── Section 1: Regime Timeline ─────────────────────────────────────────
    st.subheader("Regime Timeline")
    st.plotly_chart(
        _build_timeline_figure(ohlcv, features, labels, probs, regime_names),
        use_container_width=True,
    )

    # ── Section 2: Transition Matrix + Emission Params side by side ────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Transition Matrix")
        trans_df = transition_matrix_display(model, regime_names)
        st.plotly_chart(_build_heatmap(trans_df), use_container_width=True)

    with col_right:
        st.subheader("Emission Parameters")
        st.dataframe(
            _emission_table(model, regime_names),
            use_container_width=True,
        )

    # ── Section 3: Regime Statistics ──────────────────────────────────────
    st.subheader("Regime Statistics")
    stats = regime_statistics(labels, regime_names)
    stats_display = stats.copy()
    stats_display["pct_time"] = stats_display["pct_time"].map(lambda x: f"{x:.1f}%")
    stats_display["avg_duration_days"] = stats_display["avg_duration_days"].map(
        lambda x: f"{x:.1f}d"
    )
    stats_display.columns = ["Total Days", "% Time", "# Episodes", "Avg Duration"]
    st.dataframe(stats_display, use_container_width=True)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Data: {ticker} · {start_date} → {end_date} · "
        f"{len(features):,} trading days · "
        f"Model: {model.n_components}-state GaussianHMM ({model.covariance_type} cov)"
    )


if __name__ == "__main__":
    main()
