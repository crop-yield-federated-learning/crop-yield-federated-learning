"""Evaluate the saved model on every row of the frozen test split."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.model_interface import load_model, predict


DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def main():
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0].to_numpy(dtype=float)
    model = load_model()
    if list(X_test.columns) != model["feature_names"]:
        raise ValueError("Test feature columns/order do not match the saved model.")
    predicted = predict(model, X_test.to_numpy(dtype=np.float32))
    if len(predicted) != len(y_test):
        raise ValueError(f"Expected predictions for all {len(y_test)} test samples, got {len(predicted)}")

    error = predicted - y_test
    frame = pd.DataFrame({
        "sample": np.arange(1, len(y_test) + 1),
        "actual_yield": y_test,
        "predicted_yield": predicted,
        "error": error,
        "absolute_error": np.abs(error),
    })
    frame.to_csv(RESULTS_DIR / "predictions_actual_vs_predicted.csv", index=False, float_format="%.8f")
    report_path = RESULTS_DIR / "model_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["test_metrics"] = {
        "sample_count": int(len(y_test)),
        "MAE": float(mean_absolute_error(y_test, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "R2": float(r2_score(y_test, predicted)),
        "actual_mean": float(y_test.mean()),
        "actual_std": float(y_test.std()),
        "prediction_mean": float(predicted.mean()),
        "prediction_std": float(predicted.std()),
        "prediction_min": float(predicted.min()),
        "prediction_max": float(predicted.max()),
    }
    report["test_samples"] = int(len(y_test))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(frame["sample"], frame["actual_yield"], marker="o", linewidth=1.4, label="Actual")
    ax.plot(frame["sample"], frame["predicted_yield"], marker="x", linewidth=1.2, label="Predicted")
    ax.set(xlabel="Test sample", ylabel="Yield (kg/hectare)", title=f"Actual vs Predicted Yield — all {len(frame)} Test Samples")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_test, predicted, alpha=0.8, edgecolor="black", linewidth=0.3)
    low = float(min(y_test.min(), predicted.min()))
    high = float(max(y_test.max(), predicted.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="tab:red", label="Ideal y = x")
    ax.set(xlabel="Actual yield (kg/hectare)", ylabel="Predicted yield (kg/hectare)", title="Actual vs Predicted Yield")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "prediction_scatter.png", dpi=160)
    plt.close(fig)

    print("TEST EVALUATION — all samples")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"Prediction CSV: {RESULTS_DIR / 'predictions_actual_vs_predicted.csv'}")


if __name__ == "__main__":
    main()
