from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Blockchain.blockchain_utils import BlockchainRecorder


ACCOUNT = "0x" + "1" * 40
CONTRACT_ADDRESS = "0x" + "2" * 40


class FakeTransactionHash:
    def hex(self) -> str:
        return "0xtransaction"


class FakeRecordFunction:
    def __init__(self) -> None:
        self.transaction: dict[str, object] | None = None
        self.transact_arguments: dict[str, str] | None = None

    def build_transaction(self, transaction: dict[str, object]) -> dict[str, object]:
        self.transaction = transaction
        return transaction

    def transact(self, arguments: dict[str, str]) -> FakeTransactionHash:
        self.transact_arguments = arguments
        return FakeTransactionHash()


class FakeContract:
    def __init__(self, latest_round: int = 0) -> None:
        self.record_function = FakeRecordFunction()
        self.current_round = latest_round
        self.functions = SimpleNamespace(
            latestRound=lambda: SimpleNamespace(call=lambda: self.current_round),
            recordModel=self._record_model,
        )
        self.recorded_model: tuple[int, bytes] | None = None

    def _record_model(self, round_number: int, model_hash: bytes) -> FakeRecordFunction:
        self.recorded_model = (round_number, model_hash)
        return self.record_function


class FakeSigningAccount:
    address = ACCOUNT

    def __init__(self) -> None:
        self.signed_transaction: dict[str, object] | None = None

    def sign_transaction(self, transaction: dict[str, object]) -> SimpleNamespace:
        self.signed_transaction = transaction
        return SimpleNamespace(raw_transaction=b"signed")


class FakeWeb3:
    instance: FakeWeb3 | None = None

    @staticmethod
    def HTTPProvider(url: str, request_kwargs: dict[str, int]) -> tuple[str, dict[str, int]]:
        return url, request_kwargs

    def __init__(self, provider: tuple[str, dict[str, int]]) -> None:
        self.signing_account = FakeSigningAccount()
        self.contract_instance = FakeContract()
        self.provider = provider
        self.contract_arguments: tuple[str, list[dict[str, object]]] | None = None
        self.configured_private_key: str | None = None
        self.nonce_request: tuple[str, str] | None = None
        self.sent_transaction: bytes | None = None
        self.receipt_transaction: FakeTransactionHash | None = None
        self.eth = SimpleNamespace(
            account=SimpleNamespace(from_key=self._from_key),
            chain_id=31337,
            contract=self._contract,
            get_transaction_count=self._get_transaction_count,
            send_raw_transaction=self._send_raw_transaction,
            wait_for_transaction_receipt=self._wait_for_transaction_receipt,
        )
        FakeWeb3.instance = self

    def is_connected(self) -> bool:
        return True

    @staticmethod
    def is_address(address: str) -> bool:
        return address.startswith("0x") and len(address) == 42

    @staticmethod
    def to_checksum_address(address: str) -> str:
        return address.lower()

    def _from_key(self, private_key: str) -> FakeSigningAccount:
        self.configured_private_key = private_key
        return self.signing_account

    def _contract(
        self,
        address: str,
        abi: list[dict[str, object]],
    ) -> FakeContract:
        self.contract_arguments = (address, abi)
        return self.contract_instance

    def _get_transaction_count(self, account: str, state: str) -> int:
        self.nonce_request = (account, state)
        return 4

    def _send_raw_transaction(self, transaction: bytes) -> FakeTransactionHash:
        self.sent_transaction = transaction
        return FakeTransactionHash()

    def _wait_for_transaction_receipt(
        self,
        transaction_hash: FakeTransactionHash,
    ) -> dict[str, int]:
        self.receipt_transaction = transaction_hash
        return {"status": 1}


class BlockchainRecorderTests(unittest.TestCase):
    def test_private_key_signs_and_submits_a_transaction(self) -> None:
        with patch("Blockchain.blockchain_utils.Web3", FakeWeb3):
            recorder = BlockchainRecorder(
                "http://localhost:8545",
                CONTRACT_ADDRESS,
                private_key="0x" + "3" * 64,
            )
            result = recorder.record_model(1, "ab" * 32)

        fake_web3 = FakeWeb3.instance
        self.assertEqual(result, "0xtransaction")
        self.assertEqual(fake_web3.sent_transaction, b"signed")
        self.assertEqual(fake_web3.nonce_request, (ACCOUNT.lower(), "pending"))
        self.assertEqual(
            fake_web3.signing_account.signed_transaction,
            {"from": ACCOUNT.lower(), "nonce": 4, "chainId": 31337},
        )
        self.assertEqual(fake_web3.contract_instance.recorded_model, (1, bytes.fromhex("ab" * 32)))

    def test_unlocked_account_uses_rpc_transaction_submission(self) -> None:
        with patch("Blockchain.blockchain_utils.Web3", FakeWeb3):
            recorder = BlockchainRecorder(
                "http://localhost:8545",
                CONTRACT_ADDRESS,
                account=ACCOUNT,
            )
            result = recorder.record_model(1, "cd" * 32)

        self.assertEqual(result, "0xtransaction")
        self.assertEqual(
            FakeWeb3.instance.contract_instance.record_function.transact_arguments,
            {"from": ACCOUNT.lower()},
        )

    def test_private_key_account_must_match_configured_account(self) -> None:
        with patch("Blockchain.blockchain_utils.Web3", FakeWeb3):
            with self.assertRaisesRegex(ValueError, "does not match"):
                BlockchainRecorder(
                    "http://localhost:8545",
                    CONTRACT_ADDRESS,
                    account=CONTRACT_ADDRESS,
                    private_key="0x" + "3" * 64,
                )

    def test_latest_round_reads_existing_contract_state(self) -> None:
        with patch("Blockchain.blockchain_utils.Web3", FakeWeb3):
            recorder = BlockchainRecorder(
                "http://localhost:8545",
                CONTRACT_ADDRESS,
                account=ACCOUNT,
            )
            FakeWeb3.instance.contract_instance.current_round = 1

        self.assertEqual(recorder.latest_round(), 1)


if __name__ == "__main__":
    unittest.main()
