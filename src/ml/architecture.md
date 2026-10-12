# Centralized and federated model architecture

## Centralized crop-yield regression

The centralized data contains 38 tabular features and no known sequence
dimension. The historical centralized LSTM consumes `(batch, 1, 38)`, which
does not provide a temporal sequence. A regularized vector MLP and classical
regressors are compared using the validation split. The selected model for the
current frozen data is `GradientBoostingRegressor` with 40 boosting stages,
depth 3, learning rate 0.03, and Huber loss. It is refit on train plus
validation rows and saved as a scikit-learn joblib model.

The training script also compares the Keras LSTM and MLP when TensorFlow's
native runtime loads. If TensorFlow is unavailable, it reports the import
failure and continues with tabular baselines; any prior-run LSTM metrics are
labelled as historical context and are excluded from model selection.

## Federated learning

`build_lstm_model(input_features=38)` retains its existing
`LSTM(64) -> Dense(32, ReLU) -> Dense(1)` architecture and `(batch, 1, 38)`
input contract. `src/clients/local_training.py`, FedAvg, round management, and
encrypted update serialization continue using the same model weights and
layer shapes. Centralized tabular training does not alter federated model
aggregation or client training.

See `src/ml/train_lstm.py` for validation-based comparison and
`src/ml/model_interface.py` for centralized inference.
