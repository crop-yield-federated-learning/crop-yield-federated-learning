# Centralized and federated model architecture

## Centralized crop-yield prediction

The centralized dataset has 38 processed tabular features and no genuine
ordered sequence dimension. The historical one-timestep model consumes a row
as `(batch, 1, 38)`; this is retained as a comparator but does not learn
temporal dependencies.

The candidate neural model treats each row as a feature vector:

```text
Input (38)
Dense(64, ReLU, L2=0.001)
Dropout(0.15)
Dense(32, ReLU, L2=0.001)
Dense(1, linear)
```

The target is standardized using training targets during validation selection.
After selection, the model and target scaler are refit on training plus
validation rows. The validation split selects between this model and the
classical baselines. Final test metrics use every held-out test row once.

## Federated learning

`build_lstm_model(input_features=38)` retains its existing
`LSTM(64) -> Dense(32, ReLU) -> Dense(1)` architecture and `(batch, 1, 38)`
input contract. `src/clients/local_training.py`, FedAvg, and encrypted update
serialization continue using the same model weights and layer shapes. The
centralized vector MLP is a separate factory and does not change FL model
aggregation or client training.

See `src/ml/train_lstm.py` for model comparison and `src/ml/model_interface.py`
for centralized inference.
