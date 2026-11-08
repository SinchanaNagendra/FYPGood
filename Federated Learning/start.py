from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import NUM_CLIENTS


def start_clients() -> None:
    script = Path(__file__).with_name("client.py")
    processes: list[subprocess.Popen[bytes]] = []
    for client_id in range(1, NUM_CLIENTS + 1):
        environment = os.environ.copy()
        environment["CLIENT_ID"] = str(client_id)
        print(f"Starting CPU worker client {client_id}")
        processes.append(
            subprocess.Popen(
                [sys.executable, str(script)],
                cwd=script.parent,
                env=environment,
            )
        )

    exit_codes = [process.wait() for process in processes]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"One or more Flower clients failed: exit codes {exit_codes}")


if __name__ == "__main__":
    start_clients()
