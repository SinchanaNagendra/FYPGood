from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "clients"
CLIENT_COUNT = 3
TARGET_COLUMN = "CPU usage [%]"


def validate_datasets(data_dir: Path = DATA_DIR) -> list[Path]:
    datasets = [
        data_dir / f"terraform_worker_{client_id}.csv"
        for client_id in range(1, CLIENT_COUNT + 1)
    ]
    missing = [path for path in datasets if not path.is_file()]
    if missing:
        expected = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing worker CSV file(s) in {data_dir}: {expected}. "
            "Download the three CSVs from the linked repository's data1/ folder."
        )

    for path in datasets:
        frame = pd.read_csv(path, usecols=[TARGET_COLUMN])
        values = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{path} contains missing or non-finite CPU usage values")
        if len(values) < 32:
            raise ValueError(f"{path} needs at least 32 rows for the default windowed split")
        print(f"Validated {len(values)} CPU usage samples in {path}")
    return datasets


if __name__ == "__main__":
    validate_datasets()
