# Crop-yield regression results

## Data and diagnosis

- Target: `yield_kg_per_hectare`; 38 processed tabular features.
- Frozen split sizes: 350 train, 75 validation, 75 test.
- Every processed feature varies, and feature column names/order match across
  the three splits.
- The largest absolute train-only Pearson correlation with the target is
  0.120 (feature `25`). The strongest feature alone therefore has little
  linear association with yield.
- All 350 federated client feature rows match a training row, and their 350
  targets match the target paired with that row.
- The raw source data are not present in this checkout; `data/processed/` is
  locally supplied and ignored by Git. The fitted preprocessor pickle loads
  with a scikit-learn version warning (serialized as 1.9.0, run with 1.5.2).
  Centralized training and evaluation use the already processed split files;
  they do not refit or apply that preprocessor. Because raw data and fit
  provenance are unavailable, preprocessing leakage cannot be ruled out.

## Model selection

The one-timestep LSTM consumes `(batch, 1, 38)`, so it has no temporal order to
learn. Its existing architecture was retrained on the training split for a
validation comparison. The old `results/Sreenidhi_LSTM_best.keras` weights
could not be loaded in this environment because the application-control policy
blocked h5py's native extension.

Validation results (75 rows; model selection used MAE, with RMSE as tie-break):

| Model | MAE (kg/ha) | RMSE (kg/ha) | R² |
|---|---:|---:|---:|
| Existing LSTM architecture, retrained | 1385.15 | 1728.47 | -1.3112 |
| Regularized vector Keras MLP | 1045.91 | 1241.51 | -0.1924 |
| Random Forest, leaf size 3 | 1038.98 | 1210.45 | -0.1335 |
| Gradient Boosting | 1011.79 | 1195.52 | -0.1057 |
| RBF SVR | 1010.10 | 1178.29 | -0.0741 |
| Training-mean baseline | **1003.55** | **1169.56** | **-0.0582** |

The training-mean baseline was selected because it had the lowest validation
MAE and the best validation R² among these candidates. More complex models
were not supported by the available validation signal.

## Final held-out test result

The selected baseline was refit on train plus validation data and evaluated
once on all 75 test rows:

| Metric | Result |
|---|---:|
| MAE | 1097.3908 kg/ha |
| RMSE | 1218.1706 kg/ha |
| R² | -0.0041 |
| Actual mean / standard deviation | 4099.0915 / 1215.6811 kg/ha |
| Prediction mean / standard deviation | 4021.2508 / 0.0000 kg/ha |

The near-zero R² and constant predictions indicate that these processed
features do not support useful generalization in the provided frozen splits.
The model should not be presented as an accurate yield forecaster until the
feature/label provenance and source data can be checked and a stronger
feature-target signal is established.

No project-specific regression accuracy formula or source for 63.66% was
found in the repository, its checked-in notebooks/docs/scripts, or Git history.
Therefore this report does not invent an accuracy percentage. R² is a
regression metric, not classification accuracy. Neither 63.66% nor R² = 1.0
was achieved.

## Artifacts

- Selected predictor: `models/lstm/tabular_crop_yield_best.joblib`
- Inference metadata: `models/lstm/model_metadata.pkl`
- All predictions: `results/predictions_actual_vs_predicted.csv`
- Metrics and validation comparison: `results/model_metrics.json`
- Figures: `results/actual_vs_predicted.png`,
  `results/prediction_scatter.png`
- Reproduce training/model selection: `python src/ml/train_lstm.py`
- Reproduce final test inference: `python scripts/predict_yield.py`

The federated LSTM factory, local training, FedAvg implementation, and
security/aggregation code remain separate and unchanged in behavior.
