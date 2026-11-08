from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "clients"
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_CLIENTS = 3
FEATURE_COUNT = 128
SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8081")
NUM_ROUNDS = int(os.environ.get("FL_NUM_ROUNDS", "1"))
LOCAL_EPOCHS = int(os.environ.get("FL_LOCAL_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("FL_BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("FL_LEARNING_RATE", "0.05"))
_blockchain_setting = os.environ.get("BLOCKCHAIN_ENABLED", "false").strip().lower()
if _blockchain_setting not in {"true", "false"}:
    raise ValueError("BLOCKCHAIN_ENABLED must be either 'true' or 'false'")
BLOCKCHAIN_ENABLED = _blockchain_setting == "true"

