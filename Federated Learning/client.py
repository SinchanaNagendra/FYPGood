from __future__ import annotations

import os

import flwr as fl
import numpy as np
import pandas as pd

from config import (
    BATCH_SIZE,
    DATA_DIR,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    SERVER_ADDRESS,
    WINDOW_SIZE,
)
from model import (
    calculate_metrics,
    create_windows,
    initial_parameters,
    load_cpu_series,
    predict_values,
    set_parameters,
)


class CPUUsageClient(fl.client.NumPyClient):
    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        dataset_path = DATA_DIR / f"terraform_worker_{client_id}.csv"
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Worker {client_id} dataset not found at {dataset_path}. "
                "Download the three worker CSVs and run prepare_dataset.py."
            )

        series = load_cpu_series(pd.read_csv(dataset_path))
        split_index = int(len(series) * 0.8)
        if split_index <= WINDOW_SIZE or len(series) - split_index < 2:
            raise ValueError(
                f"{dataset_path} needs enough ordered samples for the configured "
                f"{WINDOW_SIZE}-sample window and a test split"
            )

        self.x_train, self.y_train = create_windows(series[:split_index], WINDOW_SIZE)
        self.x_test, self.y_test = create_windows(
            series[split_index - WINDOW_SIZE :],
            WINDOW_SIZE,
        )
        self.parameters = initial_parameters(WINDOW_SIZE)

    def get_parameters(self, config: dict[str, object]) -> list[np.ndarray]:
        return [parameter.copy() for parameter in self.parameters]

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, object],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        coefficients, intercept = set_parameters(parameters, WINDOW_SIZE)
        self.parameters = self._train(coefficients, intercept)
        predictions = predict_values(self.x_train, self.parameters)
        metrics = calculate_metrics(self.y_train, predictions)
        print(f"Worker {self.client_id}: local training finished")
        return self.get_parameters({}), len(self.y_train), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, object],
    ) -> tuple[float, int, dict[str, float]]:
        self.parameters = [np.asarray(parameter, dtype=np.float64) for parameter in parameters]
        metrics = calculate_metrics(
            self.y_test,
            predict_values(self.x_test, self.parameters),
        )
        return metrics["mse"], len(self.y_test), metrics

    def _train(self, coefficients: np.ndarray, intercept: float) -> list[np.ndarray]:
        rng = np.random.default_rng(42 + self.client_id)
        for _ in range(LOCAL_EPOCHS):
            indices = rng.permutation(len(self.y_train))
            for start in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[start : start + BATCH_SIZE]
                batch_x = self.x_train[batch_indices]
                batch_y = self.y_train[batch_indices]
                errors = batch_x @ coefficients + intercept - batch_y
                coefficients -= LEARNING_RATE * (2.0 * batch_x.T @ errors / len(batch_y))
                intercept -= LEARNING_RATE * 2.0 * float(errors.mean())

        return [coefficients, np.asarray([intercept], dtype=np.float64)]


def main() -> None:
    client_id = int(os.environ["CLIENT_ID"])
    client = CPUUsageClient(client_id)
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
