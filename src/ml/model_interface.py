"""Inference interface for the selected centralized crop-yield model."""

from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "models" / "lstm" / "model_metadata.pkl"


def load_model():
    """Load the selected model and its feature/target transformation data."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Trained model metadata not found: {METADATA_PATH}. Run src/ml/train_lstm.py first.")
    metadata = joblib.load(METADATA_PATH)
    model_path = Path(metadata["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if metadata["kind"] == "keras_mlp":
        from tensorflow import keras

        model = keras.models.load_model(model_path)
        scaler_path = Path(metadata["target_scaler_path"])
        if not scaler_path.is_absolute():
            scaler_path = PROJECT_ROOT / scaler_path
        scaler = joblib.load(scaler_path)
        estimator = model
    elif metadata["kind"] == "sklearn":
        saved = joblib.load(model_path)
        estimator = saved["estimator"]
        scaler = None
    else:
        raise ValueError(f"Unknown saved model kind: {metadata['kind']}")
    return {"kind": metadata["kind"], "estimator": estimator, "target_scaler": scaler, "feature_names": metadata["feature_names"]}


def predict(model, features):
    """Predict yield from one or more rows of the 38 processed features."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        if values.size != 38:
            raise ValueError("Expected exactly 38 input features for one sample.")
        values = values.reshape(1, 38)
    if values.ndim != 2 or values.shape[1] != 38:
        raise ValueError(f"Expected features with shape (n_samples, 38), got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Input features must be finite.")
    if model["kind"] == "keras_mlp":
        scaled_predictions = model["estimator"].predict(values, verbose=0).reshape(-1, 1)
        return model["target_scaler"].inverse_transform(scaled_predictions).reshape(-1)
    return np.asarray(model["estimator"].predict(values)).reshape(-1)
