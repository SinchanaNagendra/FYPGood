from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from web3 import Web3


BLOCKCHAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BLOCKCHAIN_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PENDING_PATH = RESULTS_DIR / "pending_blockchain_records.json"


class BlockchainRecorder:
    def __init__(self, rpc_url: str, contract_address: str, account: str) -> None:
        self.web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
        if not self.web3.is_connected():
            raise ConnectionError(f"Could not connect to blockchain RPC at {rpc_url}")
        if not self.web3.is_address(contract_address):
            raise ValueError("BLOCKCHAIN_CONTRACT_ADDRESS is not a valid EVM address")
        if not self.web3.is_address(account):
            raise ValueError("BLOCKCHAIN_ACCOUNT is not a valid EVM address")

        self.account = self.web3.to_checksum_address(account)
        self.contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(contract_address),
            abi=json.loads((BLOCKCHAIN_DIR / "contract_abi.json").read_text(encoding="utf-8")),
        )

    @classmethod
    def from_environment(cls) -> BlockchainRecorder:
        rpc_url = os.environ.get("BLOCKCHAIN_RPC_URL", "").strip()
        contract_address = os.environ.get("BLOCKCHAIN_CONTRACT_ADDRESS", "").strip()
        account = os.environ.get("BLOCKCHAIN_ACCOUNT", "").strip()
        missing = [
            name
            for name, value in (
                ("BLOCKCHAIN_RPC_URL", rpc_url),
                ("BLOCKCHAIN_CONTRACT_ADDRESS", contract_address),
                ("BLOCKCHAIN_ACCOUNT", account),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Blockchain is enabled but these settings are missing: " + ", ".join(missing)
            )
        return cls(rpc_url, contract_address, account)

    def record_model(self, round_number: int, model_hash: str) -> str:
        if len(model_hash) != 64:
            raise ValueError("Model hash must be a 32-byte SHA-256 hex string")
        hash_bytes = bytes.fromhex(model_hash)
        latest_round = int(self.contract.functions.latestRound().call())

        if round_number <= latest_round:
            existing_hash = self.contract.functions.modelHashes(round_number).call()
            if bytes(existing_hash) == hash_bytes:
                return "already-recorded"
            raise ValueError(f"Round {round_number} is already recorded with a different model hash")
        if round_number != latest_round + 1:
            raise ValueError(
                f"Contract expects round {latest_round + 1}, cannot record round {round_number}"
            )

        transaction_hash = self.contract.functions.recordModel(
            round_number,
            hash_bytes,
        ).transact({"from": self.account})
        receipt = self.web3.eth.wait_for_transaction_receipt(transaction_hash)
        if receipt["status"] != 1:
            raise RuntimeError(f"Blockchain transaction reverted: {transaction_hash.hex()}")
        return transaction_hash.hex()


def queue_pending_record(round_number: int, model_hash: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if PENDING_PATH.exists():
        records = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Pending record file must contain a JSON list: {PENDING_PATH}")

    new_record = {"round": round_number, "model_hash": model_hash}
    if new_record not in records:
        records.append(new_record)
    records.sort(key=lambda record: record["round"])
    temporary_path = PENDING_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary_path.replace(PENDING_PATH)
