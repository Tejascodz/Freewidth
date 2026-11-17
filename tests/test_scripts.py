"""Run the Kaggle shell scripts end to end on the simulated device, with GPU tools stubbed.
These catch script-level bugs (argument defaults, exit codes, output layout) that unit tests miss."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _env(tmp_path, **extra):
    stub = tmp_path / "stub"
    stub.mkdir()
    for tool in ("nvidia-smi", "pip"):
        (stub / tool).write_text("#!/bin/sh\necho stub\n")
        (stub / tool).chmod(0o755)
    (stub / "python").symlink_to(sys.executable)
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", **extra}
    return env


def _mock_engine(tmp_path, policy_extra=""):
    cfg = tmp_path / "engine.yaml"
    cfg.write_text(
        "engine:\n  ladder:\n"
        "    - {tag: sim-0.8b, backend: mock, params_b: 0.8}\n"
        "    - {tag: sim-4b, backend: mock, params_b: 4.0}\n"
        "  sampling: {max_new_tokens: 512}\n"
        f"  calibration: {{dir: {tmp_path / 'calib'}}}\n"
        f"  policy:\n    name: freewidth\n{policy_extra}")
    return cfg


def test_experiment_script_without_primary_builds_report(tmp_path):
    out = tmp_path / "out"
    env = _env(tmp_path, CONFIG=str(_mock_engine(tmp_path, "    quorum_accept: false\n")), DATASET="toy",
               LIMIT="12", POLICIES="overlap", PRIMARY="", SESSION_HOURS="1", OUT=str(out))
    r = subprocess.run(["bash", str(ROOT / "scripts/kaggle_experiment.sh")], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "report failed" not in r.stdout
    assert (out / "report" / "REPORT.md").exists() and (out / "bench_rows.jsonl").exists()


def test_experiment_script_with_primary(tmp_path):
    out = tmp_path / "out"
    env = _env(tmp_path, CONFIG=str(_mock_engine(tmp_path)), DATASET="toy", LIMIT="8",
               POLICIES="overlap,big_sc", PRIMARY="overlap:big_sc", SESSION_HOURS="1", OUT=str(out))
    r = subprocess.run(["bash", str(ROOT / "scripts/kaggle_experiment.sh")], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "**primary**" in (out / "report" / "REPORT.md").read_text()


def test_atlas_script_two_workers(tmp_path):
    out = tmp_path / "nested" / "atlas_t4"            # parent does not exist yet
    env = _env(tmp_path, CONFIG="configs/atlas_mock.yaml", SESSION_HOURS="0.5", OUT=str(out))
    r = subprocess.run(["bash", str(ROOT / "scripts/kaggle_atlas_t4.sh")], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "analysis" / "ATLAS.md").exists()
    assert {p.name for p in out.glob("atlas*.jsonl")} == {"atlas.cuda0.jsonl", "atlas.cuda1.jsonl"}
