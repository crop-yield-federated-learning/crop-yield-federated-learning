# Crop-yield regression results

## Data and diagnostics

- Target: `yield_kg_per_hectare`; 38 processed tabular features.
- Frozen split sizes: 350 train, 75 validation, and 75 test rows.
- Feature names and order match across all three splits; all 38 features vary
  and the matrices and targets contain only finite values.
- Target ranges are compatible across the train and test splits. The largest
  absolute train-only Pearson correlation is 0.120 (feature `25`), indicating
  weak individual linear signal.
- All 350 federated client rows match a training feature row and paired target.
- Raw source records are not present. The processed matrices and fitted
  preprocessor are local/ignored artifacts. The preprocessor loads with a
  scikit-learn version warning (serialized with 1.9.0, run here with 1.5.2),
  and its fit provenance cannot be checked. Leakage therefore cannot be ruled
  out.

## Model selection

The one-timestep LSTM consumes `(batch, 1, 38)`. These rows contain independent
tabular features, not ordered timesteps, so this input shape gives the LSTM no
temporal sequence to learn. The federated LSTM remains unchanged. A separate
feed-forward MLP and classical regressors are considered for centralized
prediction.

Validation results (75 rows; selected by MAE, then RMSE):

| Candidate | Validation MAE (kg/ha) | RMSE (kg/ha) | R² | Source |
|---|---:|---:|---:|---|
| Existing LSTM architecture, retrained | 1385.15 | 1728.47 | -1.3112 | Previously recorded validation run |
| GradientBoostingRegressor, 40 stages, depth 3, Huber | **989.54** | **1155.39** | -0.0327 | Reproduced in this run |
| Training-mean baseline | 1003.55 | 1169.56 | -0.0582 | Reproduced in this run |
| Gradient Boosting, depth 2 | 1011.79 | 1195.52 | -0.1057 | Reproduced in this run |
| RBF SVR | 1010.10 | 1178.29 | -0.0741 | Reproduced in this run |

The shallow Huber Gradient Boosting model was selected and refit on train plus
validation data. Its validation improvement over the training-mean predictor
is modest. In this execution environment TensorFlow could not load because the
native runtime requires missing `msvcp140_1.dll`; the Keras MLP and LSTM
comparison were therefore not rerun. The historical LSTM metric above is
clearly retained as prior-run context and was excluded from selection.

## Final test result

The selected model was evaluated on all 75 untouched test rows:

| Metric | Result |
|---|---:|
| MAE | 1088.9589 kg/ha |
| RMSE | 1206.2666 kg/ha |
| R² | 0.015428 |
| Actual mean / standard deviation | 4099.0915 / 1215.6811 kg/ha |
| Prediction mean / standard deviation | 4022.0095 / 154.9193 kg/ha |
| Prediction minimum / maximum | 3610.9639 / 4445.8997 kg/ha |

These metrics come directly from the predictions and all 75 test labels. The
near-zero R² and narrow prediction range show that the current features do not
support accurate yield estimates. The model improves slightly over a
constant-mean predictor, but it is not a strong forecaster.

The repository and its history contain no supported project-specific
regression-accuracy formula or source for 63.66%. No custom percentage is
reported. R² is a regression metric, not accuracy. Neither 63.66% nor R² = 1.0
was achieved.

## Artifacts and reproduction

- Selected model: `models/lstm/tabular_crop_yield_best.joblib`
- Inference metadata: `models/lstm/model_metadata.pkl`
- All test predictions: `results/predictions_actual_vs_predicted.csv`
- Metrics and validation comparison: `results/model_metrics.json`
- Figures: `results/actual_vs_predicted.png`, `results/prediction_scatter.png`
- Train/select/refit: `python src/ml/train_lstm.py`
- Predict/evaluate all test rows: `python scripts/predict_yield.py`

The federated LSTM factory, local training, FedAvg, round management, and
security/update serialization remain separate. The centralized tabular model
does not change their input shape or weight layout.
