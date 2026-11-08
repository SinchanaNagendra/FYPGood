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
    FEATURE_COUNT,
    calculate_metrics,
    calculate_model_hash,
    initial_parameters,
    predict_logits,
)


class FraudModelTests(unittest.TestCase):
    def test_binary_metrics_are_calculated_from_logits(self) -> None:
        metrics = calculate_metrics(
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([-2.0, 2.0, -1.0, 1.0]),
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertGreater(metrics["loss"], 0.0)
        self.assertEqual(metrics["true_positives"], 1.0)
        self.assertEqual(metrics["false_positives"], 1.0)
        self.assertEqual(metrics["false_negatives"], 1.0)

    def test_invalid_labels_and_parameter_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary"):
            calculate_metrics(np.array([0.0, 2.0]), np.array([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "feature count"):
            predict_logits(np.zeros((2, 4)), initial_parameters(FEATURE_COUNT))

    def test_model_hash_is_stable(self) -> None:
        parameters = initial_parameters(FEATURE_COUNT)
        self.assertEqual(calculate_model_hash(parameters), calculate_model_hash(parameters))
        self.assertEqual(len(calculate_model_hash(parameters)), 64)


class DatasetPreparationTests(unittest.TestCase):
    def test_ieee_files_are_merged_and_partitioned_for_three_clients(self) -> None:
        from prepare_dataset import (
            FEATURE_COUNT as PREPARED_FEATURE_COUNT,
            prepare_datasets,
            validate_datasets,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "clients"
            pd.DataFrame(
                {
                    "TransactionID": np.arange(100, 160),
                    "TransactionAmt": np.linspace(1.0, 60.0, 60),
                    "ProductCD": ["W", "C", "R"] * 20,
                    "isFraud": [0, 1] * 30,
                }
            ).to_csv(root / "train_transaction.csv", index=False)
            pd.DataFrame(
                {
                    "TransactionID": np.arange(100, 160, 2),
                    "id_01": np.linspace(-10.0, 10.0, 30),
                    "DeviceType": ["mobile", "desktop"] * 15,
                }
            ).to_csv(root / "train_identity.csv", index=False)

            paths = prepare_datasets(root, output_dir, chunk_size=11)
            valid_paths = validate_datasets(output_dir)

            self.assertEqual(len(paths), 3)
            self.assertEqual(paths, valid_paths)
            for client_id in range(1, 4):
                features = np.load(output_dir / f"client_{client_id}_features.npy")
                labels = np.load(output_dir / f"client_{client_id}_labels.npy")
                self.assertEqual(features.shape, (len(labels), PREPARED_FEATURE_COUNT))
                self.assertGreaterEqual(len(labels), 3)
                self.assertTrue(np.isfinite(features).all())
                self.assertTrue(np.isin(labels, [0, 1]).all())

    def test_missing_ieee_transaction_file_is_reported(self) -> None:
        from prepare_dataset import prepare_datasets

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "train_transaction.csv"):
                prepare_datasets(Path(temporary_directory), Path(temporary_directory))


class ClientTrainingTests(unittest.TestCase):
    def test_client_trains_and_evaluates_on_its_partition(self) -> None:
        from client import FraudDetectionClient

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            labels = np.tile(np.array([0, 1], dtype=np.uint8), 60)
            features = np.zeros((len(labels), FEATURE_COUNT), dtype=np.float32)
            features[:, 0] = labels * 2.0 - 1.0
            np.save(data_dir / "client_1_features.npy", features)
            np.save(data_dir / "client_1_labels.npy", labels)

            with patch("client.DATA_DIR", data_dir), patch(
                "client.LOCAL_EPOCHS", 2
            ), patch("client.BATCH_SIZE", 16):
                client = FraudDetectionClient(1)
                parameters, count, fit_metrics = client.fit(
                    client.get_parameters({}),
                    {},
                )
                loss, test_count, evaluation_metrics = client.evaluate(parameters, {})

        self.assertGreater(count, 0)
        self.assertGreater(test_count, 0)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(fit_metrics["loss"]))
        self.assertLess(fit_metrics["loss"], np.log(2.0))
        self.assertTrue(np.isfinite(evaluation_metrics["f1"]))


class AggregationTests(unittest.TestCase):
    def test_classification_metrics_are_pooled_from_client_counts(self) -> None:
        from server import aggregate_classification_metrics

        client_metrics = [
            (
                2,
                {
                    "loss_sum": 1.0,
                    "correct": 1.0,
                    "true_positives": 1.0,
                    "false_positives": 0.0,
                    "false_negatives": 0.0,
                },
            ),
            (
                2,
                {
                    "loss_sum": 3.0,
                    "correct": 1.0,
                    "true_positives": 0.0,
                    "false_positives": 1.0,
                    "false_negatives": 1.0,
                },
            ),
        ]

        metrics = aggregate_classification_metrics(client_metrics)

        self.assertAlmostEqual(metrics["loss"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
