from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CLIENT_DATA_DIR = DATA_DIR / "clients"
CLIENT_COUNT = 3
FEATURE_COUNT = 128
TARGET_COLUMN = "isFraud"
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
CHUNK_SIZE = 50_000


def _column_hash(column: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(column.encode("utf-8"), digest_size=8).digest(),
        byteorder="little",
    )


def _read_chunks(
    transaction_path: Path,
    identity: pd.DataFrame | None,
    chunk_size: int,
):
    with pd.read_csv(
        transaction_path,
        chunksize=chunk_size,
        low_memory=False,
    ) as reader:
        for chunk in reader:
            if identity is not None:
                chunk = chunk.merge(identity, on="TransactionID", how="left", sort=False)
            yield chunk


def _encode_features(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> np.ndarray:
    rows = len(frame)
    encoded = np.zeros((rows, FEATURE_COUNT), dtype=np.float32)
    row_indices = np.arange(rows)

    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        values[~np.isfinite(values)] = means[column]
        values = np.clip((values - means[column]) / scales[column], -8.0, 8.0)
        salt = _column_hash(column)
        bucket = salt % FEATURE_COUNT
        sign = -1.0 if salt >> 63 else 1.0
        encoded[:, bucket] += (values * sign).astype(np.float32)

    for column in categorical_columns:
        values = frame[column].astype("string").fillna("<MISSING>")
        hashes = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
        hashes ^= np.uint64(_column_hash(column))
        buckets = (hashes % FEATURE_COUNT).astype(np.intp)
        signs = np.where((hashes >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)
        np.add.at(encoded, (row_indices, buckets), signs)

    feature_scale = math.sqrt(max(1, len(numeric_columns) + len(categorical_columns)))
    encoded /= feature_scale
    return encoded


def prepare_datasets(
    input_dir: Path = DATA_DIR,
    output_dir: Path = CLIENT_DATA_DIR,
    chunk_size: int = CHUNK_SIZE,
) -> list[Path]:
    transaction_path = input_dir / TRANSACTION_FILE
    identity_path = input_dir / IDENTITY_FILE
    if not transaction_path.is_file():
        raise FileNotFoundError(
            f"Missing {transaction_path}. Download IEEE-CIS train_transaction.csv "
            "from Kaggle and place it in the input directory."
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    identity = None
    if identity_path.is_file():
        identity = pd.read_csv(identity_path, low_memory=False)
        if "TransactionID" not in identity.columns:
            raise ValueError(f"{identity_path} is missing the TransactionID column")
        if identity["TransactionID"].duplicated().any():
            raise ValueError(f"{identity_path} contains duplicate TransactionID values")

    feature_columns: list[str] | None = None
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    sums: dict[str, float] = {}
    squared_sums: dict[str, float] = {}
    value_counts: dict[str, int] = {}
    client_counts = np.zeros(CLIENT_COUNT, dtype=np.int64)
    total_rows = 0

    for frame in _read_chunks(transaction_path, identity, chunk_size):
        if TARGET_COLUMN not in frame.columns or "TransactionID" not in frame.columns:
            raise ValueError(
                f"{transaction_path} must contain TransactionID and {TARGET_COLUMN}"
            )
        current_features = [
            column
            for column in frame.columns
            if column not in {"TransactionID", TARGET_COLUMN}
        ]
        if feature_columns is None:
            feature_columns = current_features
            numeric_columns = [
                column for column in feature_columns if is_numeric_dtype(frame[column])
            ]
            categorical_columns = [
                column for column in feature_columns if column not in numeric_columns
            ]
            sums = dict.fromkeys(numeric_columns, 0.0)
            squared_sums = dict.fromkeys(numeric_columns, 0.0)
            value_counts = dict.fromkeys(numeric_columns, 0)
        elif current_features != feature_columns:
            raise ValueError("IEEE-CIS feature columns changed between input chunks")

        labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError(f"{TARGET_COLUMN} must contain only finite 0/1 labels")

        for column in numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            values = values[np.isfinite(values)]
            sums[column] += float(values.sum())
            squared_sums[column] += float(np.dot(values, values))
            value_counts[column] += len(values)

        partitions = (
            pd.util.hash_pandas_object(frame["TransactionID"], index=False)
            .to_numpy(dtype=np.uint64)
            % CLIENT_COUNT
        )
        client_counts += np.bincount(
            partitions.astype(np.intp),
            minlength=CLIENT_COUNT,
        )
        total_rows += len(frame)

    if not total_rows or feature_columns is None:
        raise ValueError(f"{transaction_path} contains no training rows")
    if (client_counts < 3).any():
        raise ValueError(
            "Each client needs at least 3 transactions for a train/test split"
        )

    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in numeric_columns:
        count = value_counts[column]
        mean = sums[column] / count if count else 0.0
        variance = squared_sums[column] / count - mean * mean if count else 0.0
        means[column] = mean
        scales[column] = math.sqrt(max(variance, 0.0)) or 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_maps = []
    label_maps = []
    feature_paths = []
    label_paths = []
    for client_index, count in enumerate(client_counts, start=1):
        features_path = output_dir / f"client_{client_index}_features.npy"
        labels_path = output_dir / f"client_{client_index}_labels.npy"
        feature_paths.append(features_path)
        label_paths.append(labels_path)
        feature_maps.append(
            np.lib.format.open_memmap(
                features_path,
                mode="w+",
                dtype=np.float32,
                shape=(int(count), FEATURE_COUNT),
            )
        )
        label_maps.append(
            np.lib.format.open_memmap(
                labels_path,
                mode="w+",
                dtype=np.uint8,
                shape=(int(count),),
            )
        )

    cursors = np.zeros(CLIENT_COUNT, dtype=np.int64)
    for frame in _read_chunks(transaction_path, identity, chunk_size):
        labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(
            dtype=np.uint8
        )
        partitions = (
            pd.util.hash_pandas_object(frame["TransactionID"], index=False)
            .to_numpy(dtype=np.uint64)
            % CLIENT_COUNT
        )
        encoded = _encode_features(
            frame,
            numeric_columns,
            categorical_columns,
            means,
            scales,
        )
        for client_index in range(CLIENT_COUNT):
            mask = partitions == client_index
            count = int(mask.sum())
            if count:
                start = int(cursors[client_index])
                stop = start + count
                feature_maps[client_index][start:stop] = encoded[mask]
                label_maps[client_index][start:stop] = labels[mask]
                cursors[client_index] = stop

    for feature_map, label_map in zip(feature_maps, label_maps):
        feature_map.flush()
        label_map.flush()
    del feature_maps, label_maps

    for feature_path, label_path, count in zip(
        feature_paths, label_paths, client_counts
    ):
        print(f"Prepared {count} IEEE-CIS transactions for {feature_path.stem}")
    if identity is None:
        print("train_identity.csv not found; preparing transaction features only")
    return feature_paths


def validate_datasets(data_dir: Path = CLIENT_DATA_DIR) -> list[Path]:
    datasets = [
        (
            data_dir / f"client_{client_id}_features.npy",
            data_dir / f"client_{client_id}_labels.npy",
        )
        for client_id in range(1, CLIENT_COUNT + 1)
    ]
    missing = [
        path.name
        for features_path, labels_path in datasets
        for path in (features_path, labels_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared client data in {data_dir}: {', '.join(missing)}. "
            "Run prepare_dataset.py with the IEEE-CIS training files."
        )

    feature_paths = []
    for features_path, labels_path in datasets:
        features = np.load(features_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if features.shape != (len(labels), FEATURE_COUNT):
            raise ValueError(f"{features_path} has an invalid feature shape")
        if len(labels) < 3:
            raise ValueError(f"{labels_path} needs at least 3 transactions")
        for start in range(0, len(labels), CHUNK_SIZE):
            if not np.isfinite(features[start : start + CHUNK_SIZE]).all():
                raise ValueError(f"{features_path} contains non-finite feature values")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{labels_path} contains labels other than 0/1")
        feature_paths.append(features_path)
    return feature_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare IEEE-CIS training data for three federated clients."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing train_transaction.csv and optional train_identity.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CLIENT_DATA_DIR,
        help="Directory where prepared client arrays will be written",
    )
    args = parser.parse_args()
    prepare_datasets(args.input_dir, args.output_dir)
    validate_datasets(args.output_dir)


if __name__ == "__main__":
    main()
