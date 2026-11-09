from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from web3 import Web3


BLOCKCHAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BLOCKCHAIN_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PENDING_PATH = RESULTS_DIR / "pending_blockchain_records.json"
load_dotenv(PROJECT_ROOT / ".env")


class BlockchainRecorder:
    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        account: str | None = None,
        private_key: str | None = None,
    ) -> None:
        self.web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
        if not self.web3.is_connected():
            raise ConnectionError(f"Could not connect to blockchain RPC at {rpc_url}")
        if not self.web3.is_address(contract_address):
            raise ValueError("BLOCKCHAIN_CONTRACT_ADDRESS is not a valid EVM address")

        self.signing_account: Any | None = None
        if private_key:
            try:
                self.signing_account = self.web3.eth.account.from_key(private_key)
            except (TypeError, ValueError) as error:
                raise ValueError("BLOCKCHAIN_PRIVATE_KEY is not a valid EVM private key") from error
            self.account = self.web3.to_checksum_address(self.signing_account.address)
            if account and (
                not self.web3.is_address(account)
                or self.web3.to_checksum_address(account) != self.account
            ):
                raise ValueError(
                    "BLOCKCHAIN_ACCOUNT does not match BLOCKCHAIN_PRIVATE_KEY"
                )
        else:
            if not account or not self.web3.is_address(account):
                raise ValueError(
                    "Set BLOCKCHAIN_PRIVATE_KEY or a valid BLOCKCHAIN_ACCOUNT "
                    "for an unlocked RPC account"
                )
            self.account = self.web3.to_checksum_address(account)

        self.contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(contract_address),
            abi=json.loads((BLOCKCHAIN_DIR / "contract_abi.json").read_text(encoding="utf-8")),
        )

    @classmethod
    def from_environment(cls) -> BlockchainRecorder:
        load_dotenv(PROJECT_ROOT / ".env")
        rpc_url = os.environ.get("BLOCKCHAIN_RPC_URL", "").strip()
        contract_address = os.environ.get("BLOCKCHAIN_CONTRACT_ADDRESS", "").strip()
        account = os.environ.get("BLOCKCHAIN_ACCOUNT", "").strip()
        private_key = os.environ.get("BLOCKCHAIN_PRIVATE_KEY", "").strip()
        missing = [
            name
            for name, value in (
                ("BLOCKCHAIN_RPC_URL", rpc_url),
                ("BLOCKCHAIN_CONTRACT_ADDRESS", contract_address),
            )
            if not value
        ]
        if not account and not private_key:
            missing.append("BLOCKCHAIN_ACCOUNT or BLOCKCHAIN_PRIVATE_KEY")
        if missing:
            raise ValueError(
                "Blockchain is enabled but these settings are missing: " + ", ".join(missing)
            )
        return cls(rpc_url, contract_address, account or None, private_key or None)

    def record_model(self, round_number: int, model_hash: str) -> str:
        if len(model_hash) != 64:
            raise ValueError("Model hash must be a 32-byte SHA-256 hex string")
        hash_bytes = bytes.fromhex(model_hash)
        if round_number < 1:
            raise ValueError("Round number must be a positive integer")
        latest_round = self.latest_round()

        if round_number <= latest_round:
            existing_hash = self.contract.functions.modelHashes(round_number).call()
            if bytes(existing_hash) == hash_bytes:
                return "already-recorded"
            raise ValueError(f"Round {round_number} is already recorded with a different model hash")
        if round_number != latest_round + 1:
            raise ValueError(
                f"Contract expects round {latest_round + 1}, cannot record round {round_number}"
            )

        record_function = self.contract.functions.recordModel(round_number, hash_bytes)
        if self.signing_account is None:
            transaction_hash = record_function.transact({"from": self.account})
        else:
            transaction = record_function.build_transaction(
                {
                    "from": self.account,
                    "nonce": self.web3.eth.get_transaction_count(self.account, "pending"),
                    "chainId": self.web3.eth.chain_id,
                }
            )
            signed_transaction = self.signing_account.sign_transaction(transaction)
            raw_transaction = getattr(signed_transaction, "raw_transaction", None)
            if raw_transaction is None:
                raw_transaction = signed_transaction.rawTransaction
            transaction_hash = self.web3.eth.send_raw_transaction(raw_transaction)

        receipt = self.web3.eth.wait_for_transaction_receipt(transaction_hash)
        if receipt["status"] != 1:
            raise RuntimeError(f"Blockchain transaction reverted: {transaction_hash.hex()}")
        return transaction_hash.hex()

    def latest_round(self) -> int:
        return int(self.contract.functions.latestRound().call())


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
