"""Run the pilot with project-local packages and an isolated base interpreter."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import json
import os
from pathlib import Path
import runpy
import struct
import sys
import sysconfig
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
BASE_PATHS = tuple(sys.path)
MARKER = ".runtime-ready.json"
DIAGNOSTICS = "runtime-diagnostics.json"
PYTHON_FLAGS = ["-I", "-S", "-X", "utf8", "-B"]


class RuntimeFailure(Exception):
    """Messages are authored constants, never external exception text."""


def require_isolation() -> None:
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeFailure("Start local_runtime.py with Python -I -S -X utf8.")


def is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def runtime_spec(root: Path = ROOT) -> dict:
    root = root.resolve()
    requirements = root / "requirements-pilot.txt"
    if is_link(requirements) or not requirements.is_file() or requirements.stat().st_size > 65536:
        raise RuntimeFailure("The local requirements file is missing or invalid.")
    identity = {
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "executable": os.path.normcase(os.path.realpath(sys.executable)),
        "python_version": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "bits": struct.calcsize("P") * 8,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    target = root / (".pilot-packages-" + fingerprint[:24])
    if is_link(target) or (target.exists() and not target.is_dir()):
        raise RuntimeFailure("The local package directory is not a regular directory.")
    return {"root": root, "requirements": requirements, "identity": identity,
            "fingerprint": fingerprint, "target": target}


def marker_payload(spec: dict) -> dict:
    return {"schema_version": 1, "identity": spec["identity"],
            "fingerprint": spec["fingerprint"], "openai_version": "3.49.0"}


def marker_valid(spec: dict) -> bool:
    path = spec["target"] / MARKER
    if is_link(path) or not path.is_file() or path.stat().st_size > 16384:
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) == marker_payload(spec)
    except (OSError, ValueError):
        return False


def write_marker(spec: dict) -> None:
    path = spec["target"] / MARKER
    if is_link(path):
        raise RuntimeFailure("The runtime marker is a link.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=spec["target"],
                                         prefix=".runtime-ready-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(marker_payload(spec), stream, ensure_ascii=True, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def configure_paths(root: Path, target: Path | None = None) -> None:
    require_isolation()
    for path in BASE_PATHS:
        if not path or any(part.lower() in {"site-packages", "dist-packages"} for part in Path(path).parts):
            raise RuntimeFailure("Unexpected external package path in the isolated interpreter.")
    sys.path[:] = list(BASE_PATHS)
    # Keep the interpreter's standard library ahead of application packages.
    for path in (root, target):
        if path is not None and str(path) not in sys.path:
            sys.path.append(str(path))
    importlib.invalidate_caches()


def local_module(name: str, root: Path = ROOT):
    if str(root) not in sys.path:
        sys.path.append(str(root))
    expected = (root / (name + ".py")).resolve()
    found = importlib.util.find_spec(name)
    if found is None or not found.origin or Path(found.origin).resolve() != expected:
        raise RuntimeFailure("A local pilot module was found in an unexpected location.")
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != (root / (name + ".py")).resolve():
        raise RuntimeFailure("A local pilot module was loaded from an unexpected location.")
    return module


def verify_sdk(spec: dict) -> dict:
    configure_paths(spec["root"], spec["target"])
    target = spec["target"].resolve()
    origins = {}
    for name in ("openai", "socksio"):
        module_spec = importlib.util.find_spec(name)
        if (module_spec is None or not module_spec.origin
                or not Path(module_spec.origin).resolve().is_relative_to(target)):
            raise RuntimeFailure("A required SDK module is missing from the local package directory.")
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(target):
            raise RuntimeFailure("An SDK module was loaded from outside the local package directory.")
        origins[name] = str(origin)
    distribution = importlib.metadata.distribution("openai")
    if (distribution.version != "3.49.0"
            or not Path(distribution.locate_file("")).resolve().is_relative_to(target)):
        raise RuntimeFailure("The local OpenAI SDK does not match version 3.49.0.")
    return {"openai_version": distribution.version, "origins": origins,
            "isolated": bool(sys.flags.isolated), "no_site": bool(sys.flags.no_site)}


def bundled_pip_wheel() -> Path:
    directory = importlib.resources.files("ensurepip") / "_bundled"
    wheels = [path for path in directory.iterdir()
              if path.name.startswith("pip-") and path.name.endswith(".whl") and path.is_file()]
    # This local resource layout is an implementation detail of CPython ensurepip.
    if len(wheels) != 1:
        raise RuntimeFailure("Exactly one local ensurepip pip wheel is required; no global fallback is used.")
    wheel = Path(str(wheels[0]))
    if is_link(wheel) or not wheel.is_file():
        raise RuntimeFailure("The bundled pip wheel must be an ordinary local file.")
    return wheel


def clean_install_environment() -> dict:
    bootstrap = local_module("bootstrap")
    env = {key: value for key, value in bootstrap.child_environment().items()
           if not key.upper().startswith("PIP_")}
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def install_packages(spec: dict) -> int:
    require_isolation()
    configure_paths(spec["root"])
    wheel = bundled_pip_wheel()
    # This process is only the credential-free installer, never execute/check.
    cleaned = clean_install_environment()
    os.environ.clear()
    os.environ.update(cleaned)
    sys.path.insert(0, str(wheel))
    sys.argv = ["pip", "--isolated", "--disable-pip-version-check", "--no-input", "--no-cache-dir",
                "install", "--target", str(spec["target"]), "--upgrade", "--ignore-installed",
                "--only-binary=:all:", "--no-compile", "--index-url", "https://pypi.org/simple",
                "-r", str(spec["requirements"])]
    try:
        runpy.run_module("pip", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        return exc.code if type(exc.code) is int else 2
    return 0


def save_diagnostics(root: Path, report: dict) -> bool:
    bootstrap = local_module("bootstrap")

    def clean(value):
        if isinstance(value, str):
            return bootstrap.redact(value)[:bootstrap.LOG_CHARS]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    try:
        (root / DIAGNOSTICS).write_text(json.dumps(clean(report), indent=2, ensure_ascii=True)
                                        + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def prepare(root: Path = ROOT, runner=None) -> int:
    spec = runtime_spec(root)
    bootstrap = local_module("bootstrap")
    runner = bootstrap.run_stage if runner is None else runner
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "identity": spec["identity"], "completed": False, "stages": []}
    spec["target"].mkdir(exist_ok=True)
    marker = spec["target"] / MARKER
    if is_link(marker):
        raise RuntimeFailure("The runtime marker is a link.")
    cached = marker_valid(spec)
    if marker.exists() and not cached:
        raise RuntimeFailure("The existing runtime marker does not match this interpreter and requirements.")
    report["cached"] = cached
    stages = [("verify_sdk", "_probe")] if cached else [
        ("install_packages", "_install"), ("verify_sdk", "_probe")]
    env = clean_install_environment()
    for stage, command in stages:
        argv = [sys.executable, *PYTHON_FLAGS, str(spec["root"] / "local_runtime.py"), command, spec["fingerprint"]]
        record = runner(stage, argv, spec["root"], env.copy())
        report["stages"].append(record)
        saved = save_diagnostics(spec["root"], report)
        if record["exit_code"] != 0 or record.get("exception"):
            if cached and marker.exists():
                try:
                    marker.unlink()
                except OSError as exc:
                    report["marker_invalidation_error"] = {"type": type(exc).__name__}
                    saved = save_diagnostics(spec["root"], report)
            print(f"Runtime preparation failed at {stage} (exit_code={record['exit_code']}).")
            print(f"Sanitized diagnostics: {spec['root'] / DIAGNOSTICS}" if saved
                  else "Runtime diagnostics could not be saved; the original failure is unchanged.")
            return 2
    if runtime_spec(root)["fingerprint"] != spec["fingerprint"]:
        raise RuntimeFailure("Runtime requirements changed during preparation.")
    write_marker(spec)
    report["completed"] = True
    save_diagnostics(spec["root"], report)
    print("Local runtime ready: OpenAI SDK 3.49.0; no API request was made.")
    return 0


def forward(command: str, arguments: list[str], root: Path = ROOT) -> int:
    # Validate parser inputs without echoing unknown arguments or credential-like values.
    options = {"--action": str, "--run-dir": str, "--max-seconds": int, "--max-snapshot-hours": float}
    if len(arguments) % 2 or "--run-dir" not in arguments:
        raise RuntimeFailure("Provide the prepared run directory and supported launcher options.")
    for option, value in zip(arguments[::2], arguments[1::2]):
        if option not in options:
            raise RuntimeFailure("Unsupported launcher option.")
        try:
            options[option](value)
        except (TypeError, ValueError):
            raise RuntimeFailure("Invalid launcher option value.") from None
        if option == "--action" and value not in {"run", "status", "resume", "cancel", "inspect"}:
            raise RuntimeFailure("Unsupported pilot action.")
    return local_module("launcher", root).main([command, *arguments])


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        require_isolation()
        if not arguments:
            raise RuntimeFailure("Use check, prepare or execute.")
        command, rest = arguments[0], arguments[1:]
        if command == "check":
            configure_paths(ROOT)
            return forward("check", rest)
        if command == "prepare":
            configure_paths(ROOT)
            if rest:
                code = forward("check", rest)
                if code:
                    return code
            return prepare()
        if command in {"_install", "_probe"}:
            spec = runtime_spec()
            if rest != [spec["fingerprint"]]:
                raise RuntimeFailure("The preparation process does not match the expected runtime.")
            if command == "_install":
                return install_packages(spec)
            print(json.dumps(verify_sdk(spec), ensure_ascii=True))
            return 0
        if command == "execute":
            spec = runtime_spec()
            if not marker_valid(spec):
                raise RuntimeFailure("Prepare this interpreter and requirements before executing the pilot.")
            verify_sdk(spec)
            return forward("execute", rest)
        raise RuntimeFailure("Unsupported runtime command.")
    except RuntimeFailure as exc:
        print(f"Runtime: {exc}", file=sys.stderr)
        return 2
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Runtime stopped ({type(exc).__name__}); no exception details were printed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
