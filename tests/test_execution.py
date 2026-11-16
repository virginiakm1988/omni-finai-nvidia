import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from project_demo.common import Run, evaluation_source


class ExecutionEvidenceTests(unittest.TestCase):
    def test_evaluate_without_results_never_runs_model(self):
        process = subprocess.run(
            [sys.executable, "-m", "project_demo", "evaluate", "--mode", "live"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("evaluate requires --results", process.stderr)

    def test_reject_failed_evaluation_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            path.write_text(
                json.dumps({"status": "failed", "mode": "live", "config": {}})
            )
            with self.assertRaises(ValueError):
                evaluation_source(path)

    def test_evaluation_source_preserves_mode_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "succeeded",
                        "mode": "fixture",
                        "config": {"model": "authored"},
                    }
                )
            )
            source, proof = evaluation_source(path)
            self.assertEqual(source["mode"], "fixture")
            self.assertEqual(proof["source_mode"], "fixture")
            self.assertEqual(len(proof["source_sha256"]), 64)
            self.assertFalse(proof["model_called"])

    def test_run_directory_is_unique(self):
        with tempfile.TemporaryDirectory() as temp:
            a = Run(temp, {}, "fixture", "demo")
            b = Run(temp, {}, "fixture", "demo")
            self.assertNotEqual(a.path, b.path)
