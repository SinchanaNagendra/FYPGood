from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from config import (
    BLOCKCHAIN_ENABLED,
    FEATURE_COUNT,
    NUM_CLIENTS,
    NUM_ROUNDS,
    RESULTS_DIR,
    SERVER_ADDRESS,
)
from model import calculate_model_hash, initial_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("federated-server")


def aggregate_classification_metrics(
    client_metrics: list[tuple[int, dict[str, Any]]],
) -> dict[str, float]:
    total_examples = sum(count for count, _ in client_metrics)
    if not total_examples:
        return {}

    loss_sum = sum(float(metrics["loss_sum"]) for _, metrics in client_metrics)
    correct = sum(float(metrics["correct"]) for _, metrics in client_metrics)
    true_positives = sum(
        float(metrics["true_positives"]) for _, metrics in client_metrics
    )
    false_positives = sum(
        float(metrics["false_positives"]) for _, metrics in client_metrics
    )
    false_negatives = sum(
        float(metrics["false_negatives"]) for _, metrics in client_metrics
    )
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
    return {
        "loss": loss_sum / total_examples,
        "accuracy": correct / total_examples,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def aggregate_fit_metrics(
    client_metrics: list[tuple[int, dict[str, Any]]],
) -> dict[str, float]:
    return {
        f"train_{name}": value
        for name, value in aggregate_classification_metrics(client_metrics).items()
    }


class AuditedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, recorder: Any | None, **kwargs: Any) -> None:
        self.recorder = recorder
        self.round_history: list[dict[str, Any]] = []
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round: int, results: list[Any], failures: list[Any]):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            LOGGER.warning("Round %s did not produce aggregated parameters", server_round)
            return parameters, metrics

        arrays = parameters_to_ndarrays(parameters)
        model_hash = calculate_model_hash(arrays)
        model_path = RESULTS_DIR / f"global_model_round_{server_round}.npz"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(model_path, *arrays)
        fit_metrics = dict(metrics or {})
        self.round_history.append(
            {
                "round": server_round,
                "model_hash": model_hash,
                "model_file": str(model_path.relative_to(PROJECT_ROOT)),
                "train_metrics": fit_metrics,
            }
        )
        self._write_history()
        LOGGER.info(
            "Round %s aggregated; model hash %s; saved %s",
            server_round,
            model_hash,
            model_path,
        )

        if self.recorder is not None:
            try:
                from Blockchain.blockchain_utils import queue_pending_record

                transaction_hash = self.recorder.record_model(server_round, model_hash)
                LOGGER.info(
                    "Round %s recorded on-chain; transaction %s",
                    server_round,
                    transaction_hash,
                )
            except Exception:
                LOGGER.exception(
                    "Blockchain recording failed for round %s; preserving a retry record",
                    server_round,
                )
                try:
                    queue_pending_record(server_round, model_hash)
                except OSError:
                    LOGGER.exception(
                        "Could not persist the pending blockchain record for round %s",
                        server_round,
                    )
        return parameters, metrics

    def aggregate_evaluate(self, server_round: int, results: list[Any], failures: list[Any]):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if loss is not None:
            entry = next(
                (record for record in self.round_history if record["round"] == server_round),
                None,
            )
            if entry is not None:
                entry["evaluation_loss"] = float(loss)
                entry["evaluation_metrics"] = dict(metrics or {})
                self._write_history()
            LOGGER.info(
                "Round %s evaluation: loss=%s metrics=%s",
                server_round,
                loss,
                metrics,
            )
        return loss, metrics

    def _write_history(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        history_path = RESULTS_DIR / "metrics_history.json"
        temporary_path = history_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(self.round_history, indent=2), encoding="utf-8")
        temporary_path.replace(history_path)


def main() -> None:
    if BLOCKCHAIN_ENABLED:
        from Blockchain.blockchain_utils import BlockchainRecorder

        recorder = BlockchainRecorder.from_environment()
    else:
        recorder = None
    if recorder is None:
        LOGGER.info("Blockchain recording disabled; set BLOCKCHAIN_ENABLED=true to enable it")

    strategy = AuditedFedAvg(
        recorder=recorder,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=ndarrays_to_parameters(initial_parameters(FEATURE_COUNT)),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_classification_metrics,
    )
    LOGGER.info("Starting Flower server at %s for %s rounds", SERVER_ADDRESS, NUM_ROUNDS)
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
