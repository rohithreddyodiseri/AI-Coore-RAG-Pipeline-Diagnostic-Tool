"""Start the local diagnostic backend and Next.js frontend with shared credentials."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-port", type=int, default=4318)
    parser.add_argument("--frontend-port", type=int, default=3000)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/otlp")
    args = parser.parse_args()
    env = {
        **{k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None},
        **{k: v for k, v in dotenv_values(ROOT / "frontend/.env.local").items() if v is not None},
        **os.environ,
    }
    configured = env.get("RAG_DIAGNOSTIC_REGISTRY_TOKENS")
    if configured:
        credentials = json.loads(configured)
        token = env.get("RAG_REGISTRY_TOKEN")
        if not token:
            raise SystemExit(
                "Set RAG_REGISTRY_TOKEN to an admin token from RAG_DIAGNOSTIC_REGISTRY_TOKENS."
            )
        if token not in [roles.get("admin") for roles in credentials.values()]:
            raise SystemExit("RAG_REGISTRY_TOKEN must match a configured backend admin token.")
    else:
        token = env.get("RAG_REGISTRY_TOKEN") or secrets.token_urlsafe(32)
        env["RAG_DIAGNOSTIC_REGISTRY_TOKENS"] = json.dumps({"local": {"admin": token}})
    # Both child servers use one local backend address, regardless of standalone settings.
    env["RAG_REGISTRY_TOKEN"] = token
    env["RAG_API_BASE_URL"] = f"http://127.0.0.1:{args.backend_port}"
    env["PYTHONPATH"] = str(ROOT / "backend/src") + os.pathsep + env.get("PYTHONPATH", "")
    if not (ROOT / "frontend/node_modules/next").exists():
        raise SystemExit("Install frontend dependencies first: npm --prefix frontend ci")
    processes = []
    try:
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "rag_diagnostic.server",
                    "--public-url",
                    env["RAG_API_BASE_URL"],
                    "--port",
                    str(args.backend_port),
                    "--data-dir",
                    str(args.data_dir),
                ],
                cwd=ROOT,
                env=env,
                start_new_session=True,
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--hostname",
                    "127.0.0.1",
                    "--port",
                    str(args.frontend_port),
                ],
                cwd=ROOT / "frontend",
                env=env,
                start_new_session=True,
            )
        )
        print(
            f"Diagnostic UI: http://127.0.0.1:{args.frontend_port} "
            f"| Backend: {env['RAG_API_BASE_URL']}",
            flush=True,
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.25)
        return next(process.returncode for process in processes if process.returncode is not None)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
