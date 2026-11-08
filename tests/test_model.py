from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FL_DIR = PROJECT_ROOT / "Federated Learning"
sys.path.insert(0, str(FL_DIR))

from model import (  # noqa: E402
    calculate_metrics,
    calculate_model_hash,
    create_windows,
    initial_parameters,
    load_cpu_series,
)


class TimeSeriesModelTests(unittest.TestCase):
    def test_cpu_series_is_scaled_and_windows_keep_temporal_order(self) -> None:
        series = load_cpu_series(pd.DataFrame({"CPU usage [%]": [10, 20, 30, 40]}))
        features, targets = create_windows(series, 2)

        np.testing.assert_allclose(features, [[0.1, 0.2], [0.2, 0.3]])
        np.testing.assert_allclose(targets, [0.3, 0.4])

    def test_invalid_series_and_window_sizes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            load_cpu_series(pd.DataFrame({"CPU usage [%]": [10, np.nan]}))
        with self.assertRaisesRegex(ValueError, "more than 2"):
            create_windows(np.array([0.1, 0.2]), 2)

    def test_regression_metrics_and_model_hash_are_stable(self) -> None:
        metrics = calculate_metrics(np.array([1.0, 2.0]), np.array([1.0, 1.0]))

        self.assertAlmostEqual(metrics["mse"], 0.5)
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["r2"], -1.0)
        parameters = initial_parameters(3)
        self.assertEqual(calculate_model_hash(parameters), calculate_model_hash(parameters))
        self.assertEqual(len(calculate_model_hash(parameters)), 64)


class DatasetValidationTests(unittest.TestCase):
    def test_validation_requires_three_cpu_worker_csvs(self) -> None:
        from prepare_dataset import validate_datasets

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            for worker_id in range(1, 4):
                pd.DataFrame(
                    {"CPU usage [%]": np.linspace(10, 80, 60)}
                ).to_csv(data_dir / f"terraform_worker_{worker_id}.csv", index=False)

            paths = validate_datasets(data_dir)

            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() for path in paths))

    def test_validation_reports_missing_worker_files(self) -> None:
        from prepare_dataset import validate_datasets

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "terraform_worker_1.csv"):
                validate_datasets(Path(temporary_directory))


class ClientTrainingTests(unittest.TestCase):
    def test_client_trains_and_evaluates_on_its_local_time_series(self) -> None:
        from client import CPUUsageClient

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            values = 50 + 25 * np.sin(np.linspace(0, 8 * np.pi, 120))
            pd.DataFrame({"CPU usage [%]": values}).to_csv(
                data_dir / "terraform_worker_1.csv",
                index=False,
            )
            with patch("client.DATA_DIR", data_dir), patch("client.WINDOW_SIZE", 8), patch(
                "client.LOCAL_EPOCHS", 2
            ):
                client = CPUUsageClient(1)
                parameters, count, fit_metrics = client.fit(
                    client.get_parameters({}),
                    {},
                )
                loss, test_count, evaluation_metrics = client.evaluate(parameters, {})

        self.assertGreater(count, 0)
        self.assertGreater(test_count, 0)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(fit_metrics["mse"]))
        self.assertTrue(np.isfinite(evaluation_metrics["mae"]))


class AggregationTests(unittest.TestCase):
    def test_regression_metrics_are_pooled_from_error_sufficient_statistics(self) -> None:
        from server import aggregate_regression_metrics

        client_metrics = [
            (
                2,
                {
                    "sse": 1.0,
                    "sae": 1.0,
                    "sum_y": 3.0,
                    "sum_y_sq": 5.0,
                },
            ),
            (
                2,
                {
                    "sse": 2.0,
                    "sae": 2.0,
                    "sum_y": 7.0,
                    "sum_y_sq": 25.0,
                },
            ),
        ]

        metrics = aggregate_regression_metrics(client_metrics)

        self.assertAlmostEqual(metrics["mse"], 0.75)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.75))
        self.assertAlmostEqual(metrics["mae"], 0.75)
        self.assertAlmostEqual(metrics["r2"], 0.4)


if __name__ == "__main__":
    unittest.main()
