from __future__ import annotations

import os

import flwr as fl
import numpy as np

from config import (
    BATCH_SIZE,
    DATA_DIR,
    FEATURE_COUNT,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    SERVER_ADDRESS,
)
from model import (
    calculate_metrics,
    initial_parameters,
    predict_logits,
    set_parameters,
)


class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        features_path = DATA_DIR / f"client_{client_id}_features.npy"
        labels_path = DATA_DIR / f"client_{client_id}_labels.npy"
        if not features_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Prepared data for client {client_id} was not found in {DATA_DIR}. "
                "Place train_transaction.csv and train_identity.csv in data/, "
                "then run prepare_dataset.py."
            )

        features = np.load(features_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if features.shape != (len(labels), FEATURE_COUNT):
            raise ValueError(f"{features_path} has an invalid feature shape")
        if len(labels) < 3 or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{labels_path} must contain at least 3 binary labels")
        if not np.isfinite(features).all():
            raise ValueError(f"{features_path} contains non-finite feature values")

        rng = np.random.default_rng(42 + client_id)
        indices = rng.permutation(len(labels))
        split_index = int(len(indices) * 0.8)
        if split_index < 1 or len(indices) - split_index < 1:
            raise ValueError(f"{features_path} needs a non-empty train and test split")
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        self.x_train = np.asarray(features[train_indices], dtype=np.float64)
        self.y_train = np.asarray(labels[train_indices], dtype=np.float64)
        self.x_test = np.asarray(features[test_indices], dtype=np.float64)
        self.y_test = np.asarray(labels[test_indices], dtype=np.float64)
        self.parameters = initial_parameters(FEATURE_COUNT)

    def get_parameters(self, config: dict[str, object]) -> list[np.ndarray]:
        return [parameter.copy() for parameter in self.parameters]

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, object],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        coefficients, intercept = set_parameters(parameters, FEATURE_COUNT)
        self.parameters = self._train(coefficients, intercept)
        metrics = calculate_metrics(
            self.y_train,
            predict_logits(self.x_train, self.parameters),
        )
        print(f"Client {self.client_id}: local fraud-model training finished")
        return self.get_parameters({}), len(self.y_train), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, object],
    ) -> tuple[float, int, dict[str, float]]:
        self.parameters = [
            np.asarray(parameter, dtype=np.float64) for parameter in parameters
        ]
        metrics = calculate_metrics(
            self.y_test,
            predict_logits(self.x_test, self.parameters),
        )
        return metrics["loss"], len(self.y_test), metrics

    def _train(self, coefficients: np.ndarray, intercept: float) -> list[np.ndarray]:
        positives = float(np.sum(self.y_train))
        negatives = float(len(self.y_train) - positives)
        positive_weight = negatives / positives if positives else 1.0
        rng = np.random.default_rng(42 + self.client_id)
        for _ in range(LOCAL_EPOCHS):
            indices = rng.permutation(len(self.y_train))
            for start in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[start : start + BATCH_SIZE]
                batch_x = self.x_train[batch_indices]
                batch_y = self.y_train[batch_indices]
                logits = np.clip(batch_x @ coefficients + intercept, -35.0, 35.0)
                probabilities = 1.0 / (1.0 + np.exp(-logits))
                sample_weights = np.where(batch_y == 1.0, positive_weight, 1.0)
                weighted_errors = sample_weights * (probabilities - batch_y)
                normalizer = max(float(sample_weights.sum()), 1.0)
                coefficients -= (
                    LEARNING_RATE * (batch_x.T @ weighted_errors) / normalizer
                )
                intercept -= LEARNING_RATE * float(weighted_errors.sum()) / normalizer

        return [coefficients, np.asarray([intercept], dtype=np.float64)]


def main() -> None:
    client_id = int(os.environ["CLIENT_ID"])
    client = FraudDetectionClient(client_id)
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
