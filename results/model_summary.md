# Crop Yield Regression Results

Final evaluation uses all 75 held-out test samples. Metrics are calculated from the saved predictions in `predictions_actual_vs_predicted.csv`.

| Metric | Result |
| --- | ---: |
| MAE | 1088.9589 kg/hectare |
| RMSE | 1206.2666 kg/hectare |
| R² | 0.0154284 |

The R² value is the coefficient of determination from `r2_score(y_test, predictions)`. It is a regression metric, not classification accuracy. The full precision metrics are in [`model_metrics.json`](model_metrics.json), under `test_metrics`.

See [`actual_vs_predicted.png`](actual_vs_predicted.png) and [`prediction_scatter.png`](prediction_scatter.png) for plots of all 75 predictions.
