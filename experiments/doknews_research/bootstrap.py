"""Prepare the local pilot venv before credentials are requested (stdlib only)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

CAPTURE_BYTES = 64 * 1024
LOG_CHARS = 6000
DIAGNOSTICS = "bootstrap-diagnostics.json"


def redact(text: str) -> str:
    def clean_url(match: re.Match[str]) -> str:
        try:
            parts = urlsplit(match.group(0))
            netloc = parts.netloc
            if "@" in netloc:
                netloc = "[REDACTED]@" + netloc.rsplit("@", 1)[1]
            return urlunsplit((parts.scheme, netloc, parts.path,
                               "[REDACTED]" if parts.query else "",
                               "[REDACTED]" if parts.fragment else ""))
        except ValueError:
            return "[REDACTED_URL]"

    text = re.sub(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", clean_url, text)
    text = re.sub(r"(?i)\bBearer\s+[^\s\"'<>,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED_KEY]", text)
    # Also cover query fragments when a diagnostic omits the URL prefix.
    text = re.sub(r"([?&][^=\s&]+)=([^&\s]*)", r"\1=[REDACTED]", text)
    return text


def child_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    blocked = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT"}
    env = {key: value for key, value in source.items()
           if not key.upper().startswith("OPENAI_") and key.upper() not in blocked}
    env.update(PYTHONUTF8="1", PYTHONNOUSERSITE="1")
    return env


class Capture:
    """Drain a pipe fully while retaining a bounded prefix in memory."""
    def __init__(self) -> None:
        self.data = bytearray()
        self.truncated = False
        self.error = None

    def read(self, pipe) -> None:
        try:
            while chunk := pipe.read(8192):
                remaining = CAPTURE_BYTES - len(self.data)
                self.data.extend(chunk[:remaining])
                self.truncated |= len(chunk) > remaining
        except Exception as exc:
            self.error = {"type": type(exc).__name__, "winerror": getattr(exc, "winerror", None)}
        finally:
            try:
                pipe.close()
            except OSError as exc:
                self.error = self.error or {"type": type(exc).__name__,
                                           "winerror": getattr(exc, "winerror", None)}

    def result(self) -> tuple[str, bool]:
        data = bytes(self.data)
        if self.truncated:
            # A truncated URL/key must never leave a suffix without its marker.
            data = data.rsplit(b"\n", 1)[0] if b"\n" in data else b""
        clean = redact(data.decode("utf-8", errors="replace"))
        return clean[:LOG_CHARS], self.truncated or len(clean) > LOG_CHARS


def run_stage(stage: str, argv: list[str], cwd: Path, env: dict[str, str]) -> dict:
    record = {"stage": stage, "exit_code": None, "exception": None}
    out, err = Capture(), Capture()
    process = None
    readers = []
    try:
        process = subprocess.Popen(argv, cwd=str(cwd), env=env, shell=False,
                                   stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for capture, pipe in ((out, process.stdout), (err, process.stderr)):
            thread = threading.Thread(target=capture.read, args=(pipe,), daemon=True)
            thread.start()
            readers.append(thread)
        record["exit_code"] = process.wait()
    except (Exception, KeyboardInterrupt) as exc:
        record["exception"] = {"type": type(exc).__name__,
                               "winerror": getattr(exc, "winerror", None),
                               "errno": getattr(exc, "errno", None)}
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait()
            except OSError:
                pass
    finally:
        for reader in readers:
            reader.join(timeout=5)
        incomplete = any(reader.is_alive() for reader in readers)
        if incomplete:
            out.truncated = err.truncated = True
        record["stdout"], record["stdout_truncated"] = out.result()
        record["stderr"], record["stderr_truncated"] = err.result()
        if not record["exception"]:
            record["exception"] = out.error or err.error
        if incomplete:
            record["exception"] = record["exception"] or {"type": "CaptureIncomplete", "winerror": None}
    return record


ISOLATION_PROBE = """import importlib.util,json,os,sys
norm=lambda p: os.path.normcase(os.path.realpath(p))
info={'version':sys.version.split()[0], 'executable':sys.executable,
      'prefix':sys.prefix, 'base_prefix':sys.base_prefix,
      'ensurepip_available':importlib.util.find_spec('ensurepip') is not None}
print(json.dumps(info,ensure_ascii=True))
valid=(sys.version_info >= (3,11) and norm(sys.prefix)==norm(sys.argv[1])
       and norm(sys.prefix)!=norm(sys.base_prefix))
sys.exit(0 if valid else 3)
"""
SDK_PROBE = ('import sys,openai,socksio,importlib.metadata as m; '
             'sys.exit(0 if m.version("openai")=="3.49.0" else 4)')


def write_diagnostics(root: Path, report: dict) -> bool:
    try:
        # Only the fixed bootstrap report is saved; never argv, env or a key file.
        (root / DIAGNOSTICS).write_text(json.dumps(report, indent=2, ensure_ascii=True)
                                        + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def bootstrap(root: Path, runner=None) -> int:
    root = root.resolve()
    runner = run_stage if runner is None else runner
    venv = root / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "completed": False, "stages": []}
    env = child_environment()
    steps = [
        ("venv_without_pip", [sys.executable, "-m", "venv", "--without-pip", str(venv)]),
        ("verify_venv", [str(python), "-c", ISOLATION_PROBE, str(venv)]),
        ("ensurepip", [str(python), "-m", "ensurepip", "--upgrade"]),
        ("install_dependencies", [str(python), "-m", "pip", "--isolated",
                                  "--disable-pip-version-check", "--no-input", "install",
                                  "--require-virtualenv", "--index-url", "https://pypi.org/simple",
                                  "-r", str(root / "requirements-pilot.txt")]),
        ("verify_sdk", [str(python), "-c", SDK_PROBE]),
    ]
    if venv.is_symlink() or (hasattr(venv, "is_junction") and venv.is_junction()):
        report["stages"].append({"stage": "check_directory", "exit_code": None,
                                  "exception": {"type": "LinkedVenvRejected", "winerror": None}})
    else:
        for stage, argv in steps:
            record = runner(stage, argv, root, env)
            report["stages"].append(record)
            write_diagnostics(root, report)
            if record["exit_code"] != 0 or record.get("exception"):
                break
        else:
            report["completed"] = True
    saved = write_diagnostics(root, report)
    if report["completed"]:
        return 0
    failed = report["stages"][-1]
    print(f"Bootstrap failed at {failed['stage']} (exit_code={failed['exit_code']}).")
    if saved:
        print(f"Sanitized diagnostics: {root / DIAGNOSTICS}")
    else:
        print("Diagnostics could not be saved; the bootstrap failure is unchanged.")
    return 2


if __name__ == "__main__":
    raise SystemExit(bootstrap(Path(__file__).resolve().parent))
