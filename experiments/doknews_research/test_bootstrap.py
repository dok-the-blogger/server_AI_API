"""Bootstrap tests use temporary directories and fake installers; no network/key."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []

    def runner(self, stage, argv, cwd, env):
        self.calls.append((stage, argv, env.copy()))
        return {"stage": stage, "exit_code": 0, "exception": None,
                "stdout": "", "stderr": ""}

    def test_redacts_key_bearer_proxy_and_query_values(self):
        raw = ("sk-proj-abcdef0123456789 Bearer opaque-secret "
               "https://user:proxy-secret@host/path?token=query-secret&x=other-secret "
               "socks5://login:password@localhost:1080 ?api_key=fragment-secret")
        clean = bootstrap.redact(raw)
        for secret in ("sk-proj-", "opaque-secret", "proxy-secret", "query-secret",
                       "other-secret", "password", "fragment-secret"):
            self.assertNotIn(secret, clean)
        self.assertIn("host/path", clean)
        self.assertIn("localhost:1080", clean)

    def test_bounded_capture_discards_a_truncated_secret_line(self):
        capture = bootstrap.Capture()
        raw = b"useful first line\nhttps://user:" + b"s" * (bootstrap.CAPTURE_BYTES * 2)
        capture.read(io.BytesIO(raw))
        clean, truncated = capture.result()
        self.assertEqual(clean, "useful first line")
        self.assertTrue(truncated)
        self.assertLessEqual(len(capture.data), bootstrap.CAPTURE_BYTES)

    def test_real_process_drains_both_pipes_and_preserves_failure(self):
        script = (
            "import sys; "
            "sys.stdout.write('Bearer fake-bearer-secret\\n' + "
            "'https://user:fake-proxy-secret@host/path?token=fake-query-secret\\n' + "
            "'public stdout line\\n' * 20000); "
            "sys.stderr.write('sk-proj-fake-test-key-0123456789\\n' + "
            "'public stderr line\\n' * 20000); sys.exit(9)"
        )
        result = bootstrap.run_stage("diagnostic_probe", [sys.executable, "-c", script],
                                     self.root, bootstrap.child_environment())
        self.assertEqual(result["exit_code"], 9)
        self.assertIsNone(result["exception"])
        for stream in ("stdout", "stderr"):
            self.assertTrue(result[stream + "_truncated"])
            self.assertLessEqual(len(result[stream]), bootstrap.LOG_CHARS)
        saved = json.dumps(result)
        for secret in ("fake-bearer-secret", "fake-proxy-secret", "fake-query-secret",
                       "sk-proj-fake-test-key-0123456789"):
            self.assertNotIn(secret, saved)

    def test_partial_venv_is_preserved_and_completed_in_order(self):
        scripts = self.root / ".venv" / "Scripts"
        scripts.mkdir(parents=True)
        sentinel = scripts / "python.exe"
        sentinel.write_bytes(b"existing interpreter")
        self.assertEqual(bootstrap.bootstrap(self.root, self.runner), 0)
        self.assertEqual([call[0] for call in self.calls],
                         ["venv_without_pip", "verify_venv", "ensurepip",
                          "install_dependencies", "verify_sdk"])
        self.assertEqual(self.calls[0][1][:5],
                         [sys.executable, "-m", "venv", "--without-pip", str(self.root / ".venv")])
        for _, argv, _ in self.calls:
            self.assertNotIn("--clear", argv)
        self.assertEqual(sentinel.read_bytes(), b"existing interpreter")
        self.assertIn("--require-virtualenv", self.calls[3][1])

    def test_no_install_until_isolation_verified(self):
        def fail_isolation(*args):
            record = self.runner(*args)
            if record["stage"] == "verify_venv":
                record["exit_code"] = 3
            return record
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bootstrap.bootstrap(self.root, fail_isolation), 2)
        self.assertEqual([call[0] for call in self.calls], ["venv_without_pip", "verify_venv"])
        saved = json.loads((self.root / bootstrap.DIAGNOSTICS).read_text())
        self.assertFalse(saved["completed"])
        self.assertEqual(saved["stages"][-1]["exit_code"], 3)

    def test_credentials_never_reach_children_but_proxy_is_preserved(self):
        environment = {"OPENAI_API_KEY": "sentinel-key", "openai_admin_key": "sentinel-admin",
                       "OPENAI_LOG": "debug", "OPENAI_FUTURE_SECRET": "sentinel-future",
                       "PYTHONPATH": "untrusted", "pythonhome": "untrusted",
                       "HTTPS_PROXY": "http://proxy.example:8080", "PATH": "path-value"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(bootstrap.bootstrap(self.root, self.runner), 0)
        for _, _, env in self.calls:
            self.assertFalse(any(key.upper().startswith("OPENAI_") for key in env))
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("pythonhome", env)
            self.assertEqual(env["HTTPS_PROXY"], environment["HTTPS_PROXY"])
        text = (self.root / bootstrap.DIAGNOSTICS).read_text()
        self.assertNotIn("sentinel", text)
        self.assertNotIn("argv", text)

    def test_diagnostics_write_failure_preserves_original_failure(self):
        def fail(*args):
            record = self.runner(*args)
            record["exit_code"] = 9
            return record
        with patch.object(Path, "write_text", side_effect=OSError("unwritable")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(bootstrap.bootstrap(self.root, fail), 2)
        self.assertIn("exit_code=9", output.getvalue())
        self.assertIn("could not be saved", output.getvalue())


if __name__ == "__main__":
    unittest.main()
