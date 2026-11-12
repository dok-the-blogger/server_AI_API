"""Isolated runtime tests: fake packages/installers, real local Python, no API."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import local_runtime


class LocalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "pilot"
        self.root.mkdir()
        for name in ("local_runtime.py", "bootstrap.py", "requirements-pilot.txt"):
            shutil.copyfile(local_runtime.ROOT / name, self.root / name)
        (self.root / "launcher.py").write_text(
            "import importlib.util,json,sys\nfrom pathlib import Path\n"
            "def main(argv):\n"
            " if argv[0]=='execute': Path(__file__).with_name('execution.flag').write_text('executed')\n"
            " print(json.dumps({'command':argv[0], 'isolated':bool(sys.flags.isolated), "
            "'project_limit_confirmed':'--project-limit-confirmed' in argv, "
            "'no_site':bool(sys.flags.no_site), 'injected':importlib.util.find_spec('injected_module') is not None, "
            "'paths':sys.path}))\n"
            " return 0\n", encoding="utf-8")
        self.spec = local_runtime.runtime_spec(self.root)
        self.spec["target"].mkdir()

    def run_cli(self, command, *args, env=None, isolated=True):
        flags = local_runtime.PYTHON_FLAGS if isolated else []
        return subprocess.run([sys.executable, *flags, str(self.root / "local_runtime.py"), command, *args],
                              cwd=self.root, env=env, capture_output=True, text=True,
                              encoding="utf-8", timeout=15)

    def fake_sdk(self, target=None, version="3.49.0"):
        target = self.spec["target"] if target is None else target
        target.mkdir(exist_ok=True)
        for name in ("openai", "socksio"):
            package = target / name
            package.mkdir(exist_ok=True)
            (package / "__init__.py").write_text(f"__version__ = {version!r}\n", encoding="utf-8")
        info = target / f"openai-{version}.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: openai\nVersion: {version}\n",
                                      encoding="utf-8")

    def test_check_ignores_python_environment_and_global_injection(self):
        injected = self.root.parent / "foreign-site-packages"
        injected.mkdir()
        (injected / "injected_module.py").write_text("VALUE = True\n")
        sentinel = self.root / "sitecustomize.flag"
        (injected / "sitecustomize.py").write_text(
            f"from pathlib import Path; Path({str(sentinel)!r}).write_text('unsafe')\n")
        env = os.environ.copy()
        env.update(PYTHONPATH=str(injected), PYTHONHOME=str(injected), PYTHONUSERBASE=str(injected))
        result = self.run_cli("check", "--run-dir", str(self.root / "prepared-run"), env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertTrue(value["isolated"] and value["no_site"])
        self.assertFalse(value["injected"])
        self.assertFalse(sentinel.exists())
        self.assertFalse(any("site-packages" in path for path in value["paths"]))

    def test_nonisolated_entrypoint_fails_before_launcher(self):
        result = self.run_cli("execute", "--run-dir", str(self.root), isolated=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("-I -S", result.stderr)
        self.assertFalse((self.root / "execution.flag").exists())

    def test_explicit_project_confirmation_survives_isolated_forwarding(self):
        result = self.run_cli("check", "--run-dir", str(self.root / "prepared-run"),
                              "--project-limit-confirmed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["project_limit_confirmed"])
        duplicate = self.run_cli("check", "--run-dir", str(self.root / "prepared-run"),
                                 "--project-limit-confirmed", "--project-limit-confirmed")
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("Duplicate project-limit confirmation", duplicate.stderr)
        self.assertFalse((self.root / "execution.flag").exists())

    def test_install_failure_never_creates_marker_or_executes(self):
        calls = []
        def fail(stage, argv, cwd, env):
            calls.append((stage, argv, env))
            return {"stage": stage, "exit_code": 9, "exception": None, "stderr": "safe failure"}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-real-key", "OPENAI_LOG": "debug"}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(local_runtime.prepare(self.root, fail), 2)
        self.assertEqual([call[0] for call in calls], ["install_packages"])
        self.assertEqual(calls[0][1][1:6], local_runtime.PYTHON_FLAGS)
        self.assertFalse(any(key.upper().startswith("OPENAI_") for key in calls[0][2]))
        self.assertFalse((self.spec["target"] / local_runtime.MARKER).exists())
        self.assertFalse((self.root / "execution.flag").exists())
        self.assertIn("exit_code=9", output.getvalue())

    def test_cached_marker_still_probes_and_failure_invalidates_it(self):
        local_runtime.write_marker(self.spec)
        calls = []
        def fail(stage, argv, cwd, env):
            calls.append(stage)
            return {"stage": stage, "exit_code": 7, "exception": None}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(local_runtime.prepare(self.root, fail), 2)
        self.assertEqual(calls, ["verify_sdk"])
        self.assertFalse(local_runtime.marker_valid(self.spec))
        self.assertFalse((self.root / "execution.flag").exists())

    def test_target_sdk_and_matching_marker_allow_local_launcher(self):
        self.fake_sdk()
        local_runtime.write_marker(self.spec)
        result = self.run_cli("execute", "--run-dir", str(self.root / "prepared-run"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "execution.flag").exists())
        self.assertEqual(json.loads(result.stdout)["command"], "execute")

    def test_sdk_outside_target_is_rejected_before_module_executes(self):
        outside = self.root / "openai.py"
        outside.write_text("raise RuntimeError('foreign module was executed')\n")
        self.fake_sdk()
        local_runtime.write_marker(self.spec)
        result = self.run_cli("execute", "--run-dir", str(self.root))
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing from the local package", result.stderr)
        self.assertNotIn("foreign module was executed", result.stderr)
        self.assertFalse((self.root / "execution.flag").exists())

    def test_target_cannot_shadow_local_launcher_or_pilot_modules(self):
        self.fake_sdk()
        (self.spec["target"] / "launcher.py").write_text("raise RuntimeError('foreign launcher')\n")
        local_runtime.write_marker(self.spec)
        result = self.run_cli("execute", "--run-dir", str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "execution.flag").exists())

    def test_marker_and_requirements_changes_block_execution(self):
        self.fake_sdk()
        local_runtime.write_marker(self.spec)
        marker = self.spec["target"] / local_runtime.MARKER
        original = marker.read_text()
        altered = json.loads(original)
        altered["identity"]["bits"] = 1
        marker.write_text(json.dumps(altered))
        result = self.run_cli("execute", "--run-dir", str(self.root))
        self.assertEqual(result.returncode, 2)
        marker.write_text(original)
        with (self.root / "requirements-pilot.txt").open("a") as stream:
            stream.write("\n# changed requirements identity\n")
        result = self.run_cli("execute", "--run-dir", str(self.root))
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "execution.flag").exists())


if __name__ == "__main__":
    unittest.main()
