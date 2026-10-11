# Crop Yield Federated Learning

A federated learning system for predicting crop yield across multiple, non-colocated data sources (clients), without centralizing raw agricultural data. Includes an encrypted FL variant, performance/latency benchmarking, and a monitoring dashboard.

## Overview

Traditional ML pipelines require pooling all data centrally, which isn't always feasible or desirable for distributed agricultural data (privacy, bandwidth, ownership). This project trains a shared crop-yield model across multiple simulated clients using **Federated Averaging (FedAvg)**, comparing it against a centralized baseline, and optionally applying **homomorphic encryption** to protect model updates in transit.

## Project Structure

```
config/        # YAML configs for model, clients, FL rounds, security
data/          # raw -> interim -> cleaned -> processed, plus per-client splits
notebooks/     # exploratory analysis, from EDA through FL experiments
src/           # core library code
  data/            data loading, cleaning, validation, splitting
  preprocessing/   encoding, scaling, feature transforms
  models/          baseline + trainable model, train/predict/evaluate
  clients/         client simulation, local training
  federated/       FL server, FedAvg, round orchestration
  security/        encryption/decryption, key management
  performance/     timing, latency, energy metrics
  evaluation/      metrics, comparisons, reporting
experiments/   # one folder per experiment axis (centralized, edge, federated,
               # encryption, iid vs non-iid, client count, round count)
results/       # metrics, latency, energy, trained models, figures, tables
dashboard/     # Streamlit app for visualizing training/results
scripts/       # CLI entry points to run each stage end-to-end
tests/         # unit tests, mirrors src/ layout
docs/          # architecture, dataset, ML, FL, security, performance write-ups
```

## Getting Started

```bash
# 1. Clone and enter the repo
git clone https://github.com/Navya-Siyyadula/crop-yield-federated-learning.git
cd crop-yield-federated-learning

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env
```

## Typical Workflow

```bash
python scripts/prepare_data.py        # clean + preprocess raw data
python scripts/create_clients.py      # partition data across clients
python scripts/train_centralized.py   # baseline model
python scripts/run_federated.py       # federated training (FedAvg)
python scripts/run_encrypted_fl.py    # federated training with encryption
python scripts/generate_results.py    # aggregate metrics into results/
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Centralized crop-yield regression

The centralized prediction task is **regression**: it estimates the continuous
`yield_kg_per_hectare` target from 38 processed tabular features. The frozen
processed splits contain 350 training rows, 75 validation rows, and 75 test
rows. The test split is held out for final evaluation; candidate selection in
`src/ml/train_lstm.py` uses validation MAE (RMSE breaks ties), then fits the
selected model using train plus validation data.

```bash
python -m pip install -r requirements.txt
python src/ml/train_lstm.py
python scripts/predict_yield.py
```

The training script compares the saved legacy one-timestep `LSTM(64) →
Dense(32) → Dense(1)` with a regularized 38-value feed-forward Keras MLP and
tabular baselines. A row of unrelated features is not a temporal sequence.
On the available runtime, a shallow `GradientBoostingRegressor` (40 stages,
depth 3, Huber loss) was selected by validation MAE and refit on train plus
validation rows. It achieves test MAE 1088.96 kg/ha, RMSE 1206.27 kg/ha and
R² 0.0154 over all 75 held-out samples. This is a small improvement over the
mean predictor, not a high-accuracy forecaster. The original
`build_lstm_model()` remains unchanged for federated clients and weight
aggregation. See [the model results](docs/ml/model_results.md) for metrics
and limitations.

The saved selected model and metadata live in `models/lstm/`. The final test
evaluation uses all 75 rows and writes:

- `results/predictions_actual_vs_predicted.csv` — sample, actual, predicted,
  signed error, and absolute error for every test row.
- `results/model_metrics.json` — validation model comparisons and final MAE,
  RMSE, R², and prediction/actual distribution summaries.
- `results/actual_vs_predicted.png` and `results/prediction_scatter.png` —
  test-set plots; the scatter includes the ideal `y = x` line.

MAE and RMSE are measured in kg/hectare. R² is computed as the standard
coefficient of determination by scikit-learn; it is not classification
accuracy. No project-specific formula or source for the requested 63.66% was
found in the repository or its history, so no custom accuracy percentage is
reported. R² = 1.0 denotes perfect predictions and is reported only if
achieved from held-out labels and predictions. Neither target was achieved.

The federated LSTM and Keras candidates require a working TensorFlow 2.16.1
runtime. If TensorFlow's native runtime cannot load, training reports the
reason and continues with the tabular candidates; it does not retrain the
LSTM/MLP comparison in that fallback. The frozen split CSVs and fitted
preprocessor are local data artifacts ignored by Git; place them in
`data/processed/` before running the commands above.

Run tests:

```bash
pytest tests/ -v
```

## Team

See `docs/team/responsibility_matrix.md` for role breakdown and `docs/team/execution_book.md` for the project plan.

## License

See `LICENSE`.
