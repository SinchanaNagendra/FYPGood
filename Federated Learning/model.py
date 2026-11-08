from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


FEATURE_COUNT = 128


def initial_parameters(feature_count: int = FEATURE_COUNT) -> list[np.ndarray]:
    return [
        np.zeros(feature_count, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    ]


def set_parameters(
    parameters: Sequence[np.ndarray],
    feature_count: int = FEATURE_COUNT,
) -> tuple[np.ndarray, float]:
    if len(parameters) != 2:
        raise ValueError("A model must contain the feature weights and intercept")
    coefficients = np.asarray(parameters[0], dtype=np.float64)
    intercept_array = np.asarray(parameters[1], dtype=np.float64)
    if coefficients.shape != (feature_count,) or intercept_array.shape != (1,):
        raise ValueError("Model parameter dimensions do not match the feature count")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept_array).all():
        raise ValueError("Model parameters contain non-finite values")
    return coefficients.copy(), float(intercept_array[0])


def predict_logits(features: np.ndarray, parameters: Sequence[np.ndarray]) -> np.ndarray:
    coefficients, intercept = set_parameters(parameters, features.shape[1])
    return features @ coefficients + intercept


def calculate_metrics(targets: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if targets.shape != logits.shape or targets.ndim != 1 or not targets.size:
        raise ValueError("Targets and logits must be non-empty one-dimensional arrays")
    if (
        not np.isfinite(targets).all()
        or not np.isin(targets, [0.0, 1.0]).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Targets must be binary and logits must be finite")

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
    predictions = probabilities >= 0.5
    positive_targets = targets == 1.0
    true_positives = int(np.sum(predictions & positive_targets))
    false_positives = int(np.sum(predictions & ~positive_targets))
    false_negatives = int(np.sum(~predictions & positive_targets))
    correct = int(np.sum(predictions == positive_targets))
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    loss_sum = float(np.sum(np.logaddexp(0.0, logits) - targets * logits))
    return {
        "loss": loss_sum / len(targets),
        "loss_sum": loss_sum,
        "accuracy": correct / len(targets),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "correct": float(correct),
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
    }


def calculate_model_hash(parameters: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        canonical = np.ascontiguousarray(np.asarray(parameter, dtype="<f8"))
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()
