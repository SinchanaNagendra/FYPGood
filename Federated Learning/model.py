from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd


TARGET_COLUMN = "CPU usage [%]"
CPU_SCALE = 100.0


def load_cpu_series(frame: pd.DataFrame) -> np.ndarray:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset is missing required column: {TARGET_COLUMN}")

    values = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{TARGET_COLUMN} contains missing or non-finite values")
    return values / CPU_SCALE


def create_windows(series: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if len(series) <= window_size:
        raise ValueError(
            f"Need more than {window_size} CPU samples to create a training window"
        )
    windows = np.lib.stride_tricks.sliding_window_view(series, window_size + 1)
    return windows[:, :-1].copy(), windows[:, -1].copy()


def initial_parameters(window_size: int) -> list[np.ndarray]:
    return [
        np.zeros(window_size, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    ]


def set_parameters(parameters: Sequence[np.ndarray], window_size: int) -> tuple[np.ndarray, float]:
    if len(parameters) != 2:
        raise ValueError("A model must contain the lag coefficient vector and intercept")
    coefficients = np.asarray(parameters[0], dtype=np.float64)
    intercept_array = np.asarray(parameters[1], dtype=np.float64)
    if coefficients.shape != (window_size,) or intercept_array.shape != (1,):
        raise ValueError("Model parameter dimensions do not match the configured window size")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept_array).all():
        raise ValueError("Model parameters contain non-finite values")
    return coefficients.copy(), float(intercept_array[0])


def predict_values(features: np.ndarray, parameters: Sequence[np.ndarray]) -> np.ndarray:
    coefficients, intercept = set_parameters(parameters, features.shape[1])
    return features @ coefficients + intercept


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 1 or not targets.size:
        raise ValueError("Targets and predictions must be non-empty one-dimensional arrays")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Targets and predictions must contain only finite values")

    residuals = targets - predictions
    squared_error = float(np.dot(residuals, residuals))
    total_variance = float(np.sum((targets - targets.mean()) ** 2))
    return {
        "mse": squared_error / len(targets),
        "rmse": float(np.sqrt(squared_error / len(targets))),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": 1.0 - squared_error / total_variance if total_variance else 0.0,
        "sse": squared_error,
        "sae": float(np.sum(np.abs(residuals))),
        "sum_y": float(targets.sum()),
        "sum_y_sq": float(np.dot(targets, targets)),
    }


def calculate_model_hash(parameters: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        canonical = np.ascontiguousarray(np.asarray(parameter, dtype="<f8"))
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()
