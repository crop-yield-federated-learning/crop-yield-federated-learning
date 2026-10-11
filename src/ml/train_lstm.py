"""Train and compare tabular crop-yield regressors using the frozen splits.

The historical one-timestep LSTM is evaluated as a comparator. Since the
processed rows are tabular rather than temporal sequences, a vector MLP and
classical regressors are also considered. Candidate selection uses validation
MAE only; the test set is evaluated once after selection.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.lstm_model import build_lstm_model, build_tabular_mlp_model


DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models" / "lstm"
OLD_MODEL_PATH = RESULTS_DIR / "Sreenidhi_LSTM_best.keras"
SEED = 42


def load_splits():
    X = {split: pd.read_csv(DATA_DIR / f"X_{split}.csv") for split in ("train", "val", "test")}
    y = {split: pd.read_csv(DATA_DIR / f"y_{split}.csv").iloc[:, 0] for split in ("train", "val", "test")}
    expected = list(X["train"].columns)
    for split in X:
        if list(X[split].columns) != expected:
            raise ValueError(f"Feature names/order differ in {split} split")
        if y[split].name != "yield_kg_per_hectare":
            raise ValueError(f"Unexpected target column in {split} split: {y[split].name}")
        if X[split].shape[1] != 38 or len(X[split]) != len(y[split]):
            raise ValueError(f"Invalid {split} split shape: X={X[split].shape}, y={y[split].shape}")
        if not np.isfinite(X[split].to_numpy()).all() or not np.isfinite(y[split].to_numpy()).all():
            raise ValueError(f"Non-finite values found in {split} split")
    return X, y


def metrics(y_true, y_pred):
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "sample_count": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "actual_mean": float(np.mean(y_true)),
        "actual_std": float(np.std(y_true)),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }


def build_candidates():
    return {
        "RandomForest_leaf1": RandomForestRegressor(n_estimators=500, max_features=1.0, min_samples_leaf=1, random_state=SEED, n_jobs=-1),
        "RandomForest_leaf3": RandomForestRegressor(n_estimators=500, max_features=0.8, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "ExtraTrees_leaf1": ExtraTreesRegressor(n_estimators=500, max_features=1.0, min_samples_leaf=1, random_state=SEED, n_jobs=-1),
        "ExtraTrees_leaf3": ExtraTreesRegressor(n_estimators=500, max_features=0.8, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.03, max_depth=2, loss="huber", random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, l2_regularization=1.0, max_leaf_nodes=7, random_state=SEED),
        "RBF_SVR": make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1, gamma="scale")),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=8, weights="distance", p=2)),
        "MLP_sklearn": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01, early_stopping=True, validation_fraction=0.15, max_iter=1200, random_state=SEED)),
        "Dummy_mean": DummyRegressor(strategy="mean"),
        "Dummy_median": DummyRegressor(strategy="median"),
        "RBF_SVR_high_gamma": make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1, gamma=0.1)),
        "ExtraTrees_leaf10": ExtraTreesRegressor(n_estimators=500, max_features=0.75, min_samples_leaf=10, random_state=SEED, n_jobs=-1),
        "GradientBoosting_shallow": GradientBoostingRegressor(n_estimators=80, learning_rate=0.02, max_depth=1, loss="huber", random_state=SEED),
    }


def train_dense_model(X_train, y_train, X_val, y_val, final=False):
    """Fit a regularized vector MLP with train-only target scaling."""
    target_scaler = StandardScaler().fit(np.asarray(y_train).reshape(-1, 1))
    scaled_train = target_scaler.transform(np.asarray(y_train).reshape(-1, 1)).reshape(-1)
    scaled_val = None if final else target_scaler.transform(np.asarray(y_val).reshape(-1, 1)).reshape(-1)
    model = build_tabular_mlp_model(input_features=X_train.shape[1], seed=SEED)
    callbacks = [] if final else [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=35, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5),
    ]
    validation_data = None if final else (
        np.asarray(X_val, dtype=np.float32), scaled_val
    )
    model.fit(
        np.asarray(X_train, dtype=np.float32), scaled_train,
        validation_data=validation_data,
        epochs=400 if not final else 260,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    return model, target_scaler


def predict_dense(model, scaler, X):
    scaled = model.predict(np.asarray(X, dtype=np.float32), verbose=0).reshape(-1, 1)
    return scaler.inverse_transform(scaled).reshape(-1)


def legacy_validation_prediction(X_train, y_train, X_val):
    """Evaluate saved legacy weights when readable, else retrain that architecture."""
    current = build_lstm_model(input_features=38)
    label = "Current_LSTM_architecture_retrained"
    if OLD_MODEL_PATH.exists():
        try:
            current.load_weights(OLD_MODEL_PATH)
            label = "Current_saved_LSTM"
            prediction = current.predict(np.asarray(X_val, dtype=np.float32).reshape(-1, 1, 38), verbose=0).reshape(-1)
            return label, prediction
        except (ImportError, ValueError, OSError) as exc:
            print(f"Saved LSTM weights could not be loaded; evaluating the existing architecture retrained from scratch: {exc}")
    current = build_lstm_model(input_features=38)
    current.fit(
        np.asarray(X_train, dtype=np.float32).reshape(-1, 1, 38),
        np.asarray(y_train, dtype=np.float32),
        epochs=100,
        batch_size=32,
        verbose=0,
    )
    prediction = current.predict(np.asarray(X_val, dtype=np.float32).reshape(-1, 1, 38), verbose=0).reshape(-1)
    return label, prediction


def main():
    tf.keras.utils.set_random_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_splits()
    print("DATA DIAGNOSTICS")
    for split in X:
        print(f"{split}: X={X[split].shape}, y={y[split].shape}, target mean/std={y[split].mean():.2f}/{y[split].std():.2f}")
    print(f"feature columns match; varying columns={int((X['train'].std() > 0).sum())}/38")
    train_corr = X["train"].corrwith(y["train"]).abs().sort_values(ascending=False)
    print("largest absolute train feature/target correlations:")
    print(train_corr.head(8).to_string())
    preprocessor = joblib.load(DATA_DIR / "preprocessor.pkl")
    preprocessor_steps = [name for name, _ in preprocessor.steps] if hasattr(preprocessor, "steps") else []
    print(f"fitted preprocessor type={type(preprocessor).__name__}; steps={preprocessor_steps or 'not a pipeline'}")
    client_check = {"checked": False}
    client_dirs = sorted((ROOT / "data" / "client_data").glob("client_*"))
    if client_dirs:
        client_X, client_y = [], []
        for client_dir in client_dirs:
            client_id = int(client_dir.name.split("_")[-1])
            client_X.append(pd.read_csv(client_dir / f"client_{client_id}_X.csv"))
            client_y.append(pd.read_csv(client_dir / f"client_{client_id}_y.csv").iloc[:, 0].to_numpy())
        all_client_X = pd.concat(client_X, ignore_index=True)
        all_client_y = np.concatenate(client_y)
        key = lambda row: tuple(np.round(row, 8))
        train_target_by_features = dict(zip((key(row) for row in X["train"].to_numpy()), y["train"].to_numpy()))
        matched = [key(row) in train_target_by_features for row in all_client_X.to_numpy()]
        aligned = [ok and np.isclose(train_target_by_features[key(row)], target) for ok, row, target in zip(matched, all_client_X.to_numpy(), all_client_y)]
        client_check = {"checked": True, "client_rows": int(len(all_client_X)), "feature_rows_matching_train": int(sum(matched)), "targets_matching_feature_rows": int(sum(aligned))}
        print(f"federated client/train alignment: {client_check}")

    validation_results = {}
    fitted = {}
    current_name, old_pred = legacy_validation_prediction(X["train"], y["train"], X["val"])
    validation_results[current_name] = metrics(y["val"], old_pred)
    print(f"{current_name} validation:", validation_results[current_name])

    dense_model, dense_scaler = train_dense_model(X["train"], y["train"], X["val"], y["val"])
    dense_pred = predict_dense(dense_model, dense_scaler, X["val"])
    validation_results["Dense_MLP_Keras"] = metrics(y["val"], dense_pred)
    fitted["Dense_MLP_Keras"] = (dense_model, dense_scaler)
    print("Dense_MLP_Keras validation:", validation_results["Dense_MLP_Keras"])

    for name, estimator in build_candidates().items():
        estimator.fit(X["train"], y["train"])
        pred = estimator.predict(X["val"])
        validation_results[name] = metrics(y["val"], pred)
        fitted[name] = estimator
        print(f"{name} validation: MAE={validation_results[name]['MAE']:.3f}, RMSE={validation_results[name]['RMSE']:.3f}, R2={validation_results[name]['R2']:.4f}")

    eligible = {name: m for name, m in validation_results.items() if name not in {"Current_saved_LSTM", "Current_LSTM_architecture_retrained"}}
    selected_name = min(eligible, key=lambda name: (eligible[name]["MAE"], eligible[name]["RMSE"]))
    print(f"Selected by validation MAE: {selected_name}")

    X_final = pd.concat([X["train"], X["val"]], ignore_index=True)
    y_final = pd.concat([y["train"], y["val"]], ignore_index=True)
    if selected_name == "Dense_MLP_Keras":
        final_model, final_scaler = train_dense_model(X_final, y_final, None, None, final=True)
        test_pred = predict_dense(final_model, final_scaler, X["test"])
        model_path = MODEL_DIR / "lstm_crop_yield_best.keras"
        final_model.save(model_path)
        joblib.dump(final_scaler, MODEL_DIR / "target_scaler.pkl")
        joblib.dump({"kind": "keras_mlp", "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"), "target_scaler_path": str((MODEL_DIR / "target_scaler.pkl").relative_to(ROOT)).replace("\\", "/"), "feature_names": list(X["train"].columns)}, MODEL_DIR / "model_metadata.pkl")
    else:
        final_model = clone(fitted[selected_name])
        final_model.fit(X_final, y_final)
        test_pred = np.asarray(final_model.predict(X["test"])).reshape(-1)
        model_path = MODEL_DIR / "tabular_crop_yield_best.joblib"
        joblib.dump({"kind": "sklearn", "estimator": final_model, "feature_names": list(X["train"].columns)}, model_path)
        joblib.dump({"kind": "sklearn", "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"), "feature_names": list(X["train"].columns)}, MODEL_DIR / "model_metadata.pkl")

    # Final test metrics are computed once from the entire untouched test split.
    test_metrics = metrics(y["test"], test_pred)
    print("\nFINAL TEST RESULTS (all rows)")
    print(json.dumps(test_metrics, indent=2))
    print(f"selected model={selected_name}; test sample count={len(y['test'])}")

    metrics_report = {
        "task": "regression",
        "target": "yield_kg_per_hectare",
        "n_features": int(X["train"].shape[1]),
        "feature_names": list(X["train"].columns),
        "validation_selection_metric": "MAE (tie-break by RMSE)",
        "selected_model": selected_name,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "train_samples": int(len(X["train"])),
        "validation_samples": int(len(X["val"])),
        "test_samples": int(len(X["test"])),
        "validation_model_metrics": validation_results,
        "test_metrics": test_metrics,
        "accuracy_percentage": None,
        "accuracy_note": "No project-specific regression accuracy formula was found; accuracy percentage is not reported.",
        "evaluation_note": "Test metrics are calculated directly from r2_score/MAE/RMSE using every test row, after validation-based model selection.",
        "environment": {"python": sys.version.split()[0], "tensorflow": tf.__version__, "scikit_learn": sklearn.__version__},
        "feature_signal_diagnostics": {"largest_absolute_train_pearson_correlation": float(train_corr.iloc[0]), "largest_correlated_feature": str(train_corr.index[0])},
        "preprocessor_diagnostic": {
            "type": type(preprocessor).__name__,
            "n_features_in": int(getattr(preprocessor, "n_features_in_", 0)),
            "pipeline_code_available": (ROOT / "src" / "preprocessing" / "pipeline.py").exists(),
            "raw_source_data_available": any(
                path.is_file() and path.stat().st_size > 0 and path.name != ".gitkeep"
                for path in (ROOT / "data" / "raw").rglob("*")
            ) if (ROOT / "data" / "raw").exists() else False,
            "fit_provenance_verified": False,
        },
        "preprocessing_and_leakage_note": "Frozen processed matrices are used. The raw source data and preprocessor fit provenance are unavailable, so preprocessing leakage cannot be independently ruled out.",
        "federated_client_alignment_check": client_check,
    }
    (RESULTS_DIR / "model_metrics.json").write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")

    errors = np.asarray(test_pred) - np.asarray(y["test"])
    predictions = pd.DataFrame({
        "sample": np.arange(1, len(test_pred) + 1),
        "actual_yield": np.asarray(y["test"]),
        "predicted_yield": test_pred,
        "error": errors,
        "absolute_error": np.abs(errors),
    })
    predictions.to_csv(RESULTS_DIR / "predictions_actual_vs_predicted.csv", index=False, float_format="%.8f")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(predictions["sample"], predictions["actual_yield"], label="Actual", marker="o", linewidth=1.4)
    ax.plot(predictions["sample"], predictions["predicted_yield"], label="Predicted", marker="x", linewidth=1.2)
    ax.set(xlabel="Test sample", ylabel="Yield (kg/hectare)", title=f"Actual vs Predicted Yield — all {len(predictions)} Test Samples")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(predictions["actual_yield"], predictions["predicted_yield"], alpha=0.8, edgecolor="black", linewidth=0.3)
    low = float(min(predictions["actual_yield"].min(), predictions["predicted_yield"].min()))
    high = float(max(predictions["actual_yield"].max(), predictions["predicted_yield"].max()))
    ax.plot([low, high], [low, high], linestyle="--", color="tab:red", label="Ideal y = x")
    ax.set(xlabel="Actual yield (kg/hectare)", ylabel="Predicted yield (kg/hectare)", title="Actual vs Predicted Yield")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "prediction_scatter.png", dpi=160)
    plt.close(fig)
    print(f"Saved {len(predictions)} predictions, metrics, plots, and selected model.")


if __name__ == "__main__":
    main()
