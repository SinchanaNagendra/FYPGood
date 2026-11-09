# Blockchain-Secured Federated Learning

This project trains a federated fraud-detection model and can record each
aggregated model's SHA-256 hash in the `FederatedModelRegistry` smart contract.
The model itself stays in the local `results/` directory; only its hash and
training round are written on-chain.

## Run against Ganache

1. Open Ganache's **Quickstart** workspace and start the local blockchain.
   The project defaults to the screenshot's RPC URL,
   `http://127.0.0.1:7545`.
2. In Remix, open
   [`Blockchain/FederatedModelRegistry.sol`](Blockchain/FederatedModelRegistry.sol),
   compile it with Solidity 0.8.20 or newer, and select **Custom - External
   Http Provider** with `http://127.0.0.1:7545`. Confirm the connection and
   select a Ganache account.
3. In Remix's **Deploy & Run Transactions** tab, deploy
   `FederatedModelRegistry`. The deploying Ganache account becomes the
   contract owner. Copy the deployed contract address from Remix.
4. Copy `.env.example` to `.env`. Keep the local-development settings and set:
   - `BLOCKCHAIN_CONTRACT_ADDRESS` to the deployed address from Remix
   - `BLOCKCHAIN_ACCOUNT` to the same Ganache account selected in Remix
   - Leave `BLOCKCHAIN_PRIVATE_KEY` empty; Ganache unlocks its local accounts
     for transactions submitted through its RPC.

   Never commit `.env` or use a wallet holding valuable assets for development.
   The `.env` file is ignored by git.
5. Install dependencies with `pip install -r requirements.txt`.
6. From the repository root, start the Flower server:

   ```powershell
   python "Federated Learning\server.py"
   ```

7. In a second terminal, start the three Flower clients:

   ```powershell
   python "Federated Learning\start.py"
   ```

Ganache should show the contract deployment and a contract-call transaction for
each federated training round. The server logs each transaction hash. The model
hash is recorded on-chain; model weights remain in `results/`. On restarts, the
server reads `latestRound()` and continues at the next on-chain round, so an
existing registry (such as one already at round 1) does not conflict with a
new Flower run starting at training round 1.

If the RPC or transaction is unavailable, the server saves a retry record in
`results/pending_blockchain_records.json`. Once connectivity is restored, run:

```powershell
python Blockchain\retry_pending.py
```

Set `BLOCKCHAIN_ENABLED=false` in `.env` to run without blockchain recording.
