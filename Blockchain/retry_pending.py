from __future__ import annotations

import json
from pathlib import Path

from blockchain_utils import BlockchainRecorder, PENDING_PATH


def main() -> None:
    if not PENDING_PATH.exists():
        print("No pending blockchain records.")
        return

    recorder = BlockchainRecorder.from_environment()
    records = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Pending record file must contain a JSON list: {PENDING_PATH}")

    remaining = []
    for record in sorted(records, key=lambda item: item["round"]):
        try:
            transaction_hash = recorder.record_model(
                int(record["round"]),
                str(record["model_hash"]),
            )
        except Exception as error:
            print(f"Could not record round {record.get('round')}: {error}")
            remaining.append(record)
            continue
        print(f"Round {record['round']} recorded: {transaction_hash}")

    temporary_path = PENDING_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
    temporary_path.replace(PENDING_PATH)
    if remaining:
        raise RuntimeError(f"{len(remaining)} blockchain record(s) remain pending")


if __name__ == "__main__":
    main()
