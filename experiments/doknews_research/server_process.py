"""Manage an isolated child worker from the existing ai-service lifespan."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess

from .server_config import ResearchError, load_config
from .server_control import Controller
from .signal_ledger import LedgerError


def start_research(config_path: str, *, service_token_configured: bool):
    if not config_path:
        return None, None
    cfg = load_config(Path(config_path))
    try:
        controller = Controller(cfg)
    except (LedgerError, OSError) as exc:
        raise ResearchError("state_unavailable", "Research state is unavailable; existing AI routes are unaffected.") from exc
    if not cfg.api_enabled:
        return controller, None
    if not service_token_configured:
        raise ResearchError("service_token_missing", "Research requires configured ai-service authentication.")
    if not cfg.worker_python.is_file() or not os.access(cfg.worker_python, os.X_OK):
        raise ResearchError("runtime_unavailable", "The isolated research Python runtime is unavailable.")
    env_names = {
        "PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy", "OPENAI_API_KEY",
    }
    env = {name: value for name, value in os.environ.items() if name in env_names}
    script = Path(__file__).with_name("server_runner.py").resolve()
    try:
        process = subprocess.Popen(
            [str(cfg.worker_python), "-I", "-X", "utf8", str(script), "worker", "--config", str(Path(config_path).absolute())],
            cwd=str(script.parent), env=env, stdin=subprocess.DEVNULL, close_fds=True)
    except OSError as exc:
        raise ResearchError("runtime_unavailable", "The research worker could not be started.") from exc
    return controller, process


async def stop_research(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        await asyncio.to_thread(process.wait, timeout=65)
    except subprocess.TimeoutExpired:
        # Local process death does not prove remote cancellation. Its job stays
        # unresolved; a later worker reconciles saved IDs instead of resubmitting.
        process.kill()
        await asyncio.to_thread(process.wait)
