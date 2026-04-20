# Market Regime Detection using Hidden Markov Models

Unsupervised probabilistic classification of market states from price action using a Hidden Markov Model (HMM) with Gaussian emissions. 

Financial markets cycle through distinct behavioral regimes (trending up, trending down, consolidating). Traditional technical indicators react to regime changes *after* the fact. This system models these as latent states and infers them from observable market signals to provide earlier regime awareness.

*Note: This is a regime classifier, not a price predictor. It classifies the current state of the market.*

## Architecture & Methodology

* **Hidden States (3):** Bull, Bear, Sideways (learned unsupervised, not manually labeled)
* **Observations (Daily):** * Log returns
    * Rolling 21-day realized volatility
    * Rolling 21-day volume change ratio
* **Training:** Baum-Welch (Expectation-Maximization algorithm)
* **Decoding:** Viterbi algorithm for optimal state sequence inference

## Results & Visualizations

![Regime Overlay](assets/image_ea5209.png)
*Price action segmented by inferred hidden states.*

![Observation Distributions](assets/image_ea560b.png)
*Feature distributions across the different learned regimes.*

![Model Evaluation](assets/image_ea5663.png)
*HMM convergence and transition metrics.*

## Project Structure

```text
market-regime-aml/
├── assets/             # Visual outputs and model evaluation plots
├── configs/            # Pipeline configurations (config.yaml)
├── notebooks/          # EDA, model selection, and regime detection experiments
├── src/
│   ├── data/           # Market data ingestion (fetch.py)
│   ├── features/       # Rolling windows and feature engineering (engineer.py)
│   └── model/          # HMM training, evaluation, and inference scripts
└── tests/              # Unit tests for feature extraction and model validation
```
## Key Technical Implementations
* **Robust Data Ingestion:** Bypasses basic TLS fingerprinting blocks using curl_cffi to ensure reliable data scraping from financial APIs.

* **Mathematical Rigor:** Proper handling of Log-Likelihood and Bayesian Information Criterion (BIC) evaluation for continuous Gaussian densities.

* **Production-Ready Pipeline:** Modularized architecture with separated data fetching, feature engineering, and modeling layers, backed by unit testing.
